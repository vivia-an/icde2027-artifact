"""M1 gate: for each dsl_native rule, the DSL predicate and the hand-written
Python rule must produce identical violation `event_id` sets on the same
trace, both on a buggy synthetic trace (>=1 violation) and on a clean trace
(0 violations).

Synthetic traces stand in for the live `trainaudit_run.sh` traces from the
5 paper bugs (B11, M-014, O-NEW-1, O-005, B12). The shape of each event
mirrors what the corresponding hook actually emits at runtime; using
synthetic events makes the test reproducible without GPU and lets us
exercise both buggy and clean cases in one process. The same equivalence
check can later be re-run on captured `.duckdb` files (see M1 closeout
notes in MAPPING.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from trainaudit.dsl import (load_predicate, load_predicates_dir,
                            violation_event_ids)
from trainaudit.rules import all_rules
from trainaudit.store import TraceStore

REGISTRY = Path(__file__).resolve().parents[2] / "trainaudit" / "dsl" / "registry"


def _python_violation_event_ids(store: TraceStore, rule_id: str) -> List[int]:
    """Run the registered Python rule whose id matches `rule_id` and pull
    the full violation event_id list out of evidence (rules in the 7-rule
    dsl_native batch have been augmented to include `violation_event_ids`)."""
    store.flush()
    matching = [r for r in all_rules() if r.rule_id == rule_id]
    assert len(matching) == 1, f"expected 1 Python rule for {rule_id}, got {len(matching)}"
    res = matching[0].check(store.conn)
    if not isinstance(res, list):
        res = [res]
    ids: List[int] = []
    for r in res:
        if r.violated and "violation_event_ids" in (r.evidence or {}):
            ids.extend(r.evidence["violation_event_ids"])
    return sorted(ids)


def _dsl_violation_event_ids(store: TraceStore, predicate_id: str) -> List[int]:
    p = next(p for p in load_predicates_dir(REGISTRY) if p.id == predicate_id)
    return sorted(violation_event_ids(store, p))


def _assert_equivalent(store: TraceStore, rule_id: str,
                       expected_violations: List[int]) -> None:
    py_ids = _python_violation_event_ids(store, rule_id)
    dsl_ids = _dsl_violation_event_ids(store, rule_id)
    assert dsl_ids == py_ids == sorted(expected_violations), (
        f"{rule_id}: DSL={dsl_ids} Python={py_ids} expected={expected_violations}")


@pytest.fixture
def store():
    s = TraceStore(":memory:")
    yield s
    s.close()


# === T0-clip-grad-bounded (B11 surrogate) ================================


def test_clip_grad_equivalence_buggy(store):
    e1 = store.emit("utils.clip_grad.post",
                    {"max_norm": 1.0, "pre_norm": 5.0,
                     "post_norm": 5.0, "ratio": 5.0})  # buggy: no clipping
    store.emit("utils.clip_grad.post",
               {"max_norm": 1.0, "pre_norm": 1.5,
                "post_norm": 1.0, "ratio": 0.66})     # clean
    _assert_equivalent(store, "T0-clip-grad-bounded", [e1])


def test_clip_grad_equivalence_clean(store):
    store.emit("utils.clip_grad.post",
               {"max_norm": 1.0, "pre_norm": 1.5,
                "post_norm": 1.0, "ratio": 0.66})
    _assert_equivalent(store, "T0-clip-grad-bounded", [])


# === T0-no-nan-inf (sanity rule, NaN/Inf scan) ===========================


def test_no_nan_inf_equivalence_buggy(store):
    store.emit("module.fwd.post", {
        "module_class": "Linear",
        "output": {"dtype": "torch.float32", "shape": [4],
                   "l2_norm": 1.0, "has_nan": False, "has_inf": False}})
    e2 = store.emit("module.fwd.post", {
        "module_class": "Linear",
        "outputs": [
            {"dtype": "torch.float32", "shape": [4],
             "l2_norm": 1.0, "has_nan": True, "has_inf": False}]})
    e3 = store.emit("optim.step.post", {
        "params": [
            {"dtype": "torch.bfloat16", "shape": [16],
             "l2_norm": 0.0, "has_nan": False, "has_inf": True}]})
    _assert_equivalent(store, "T0-no-nan-inf", [e2, e3])


def test_no_nan_inf_equivalence_clean(store):
    store.emit("module.fwd.post", {
        "module_class": "Linear",
        "output": {"dtype": "torch.float32", "shape": [4],
                   "l2_norm": 1.0, "has_nan": False, "has_inf": False}})
    _assert_equivalent(store, "T0-no-nan-inf", [])


# === T0-optim-lr-positive ================================================


def test_optim_lr_positive_equivalence_buggy(store):
    e1 = store.emit("build.snapshot", {
        "optimizer": {"param_groups": [
            {"index": 0, "lr": 1e-4},
            {"index": 1, "lr": 0.0}]}})  # group 1 violates
    store.emit("build.snapshot", {
        "optimizer": {"param_groups": [
            {"index": 0, "lr": 1e-4}]}})  # clean
    # Both Python and DSL should report event_id e1 once.
    # (Python returns 1 violation entry per offending group, DSL returns
    # 1 row per failing list element — both keyed by event_id.)
    py = _python_violation_event_ids(store, "T0-optim-lr-positive")
    ds = _dsl_violation_event_ids(store, "T0-optim-lr-positive")
    assert py == ds == [e1]


def test_optim_lr_positive_equivalence_clean(store):
    store.emit("build.snapshot", {
        "optimizer": {"param_groups": [{"index": 0, "lr": 1e-4}]}})
    _assert_equivalent(store, "T0-optim-lr-positive", [])


# === T0-build-has-modules (M-020 surrogate) ==============================


def test_build_has_modules_equivalence_buggy(store):
    e1 = store.emit("build.snapshot",
                    {"model": {"n_parameters": 0, "n_modules": 5}})
    store.emit("build.snapshot",
               {"model": {"n_parameters": 12, "n_modules": 5}})
    _assert_equivalent(store, "T0-build-has-modules", [e1])


def test_build_has_modules_equivalence_clean(store):
    store.emit("build.snapshot",
               {"model": {"n_parameters": 12, "n_modules": 5}})
    _assert_equivalent(store, "T0-build-has-modules", [])


# === T0-initial-lr-present (B12 surrogate) ===============================


def test_initial_lr_present_equivalence_buggy(store):
    # fresh start: missing initial_lr is fine
    store.emit("scheduler.init", {
        "scheduler_class": "CosineLR", "last_epoch": -1,
        "param_groups": [{"index": 0, "has_initial_lr": False, "lr": 1e-4}]})
    # resume: missing initial_lr is the bug
    e2 = store.emit("scheduler.init", {
        "scheduler_class": "CosineLR", "last_epoch": 10,
        "param_groups": [{"index": 0, "has_initial_lr": False, "lr": 1e-4}]})
    _assert_equivalent(store, "T0-initial-lr-present", [e2])


def test_initial_lr_present_equivalence_clean(store):
    store.emit("scheduler.init", {
        "scheduler_class": "CosineLR", "last_epoch": 10,
        "param_groups": [{"index": 0, "has_initial_lr": True, "lr": 1e-4}]})
    _assert_equivalent(store, "T0-initial-lr-present", [])


# === T0-token-id-in-vocab (O-NEW-9 surrogate) ============================


def test_token_id_in_vocab_equivalence_buggy(store):
    store.emit("dataloader.batch", {
        "input_ids": {"dtype": "torch.int64", "shape": [2, 16],
                      "min": 0, "max": 50000}})
    e2 = store.emit("dataloader.batch", {
        "input_ids": {"dtype": "torch.int64", "shape": [2, 16],
                      "min": 0, "max": 5_000_000}})  # absurd → bug
    _assert_equivalent(store, "T0-token-id-in-vocab", [e2])


def test_token_id_in_vocab_equivalence_clean(store):
    store.emit("dataloader.batch", {
        "input_ids": {"dtype": "torch.int64", "shape": [2, 16],
                      "min": 0, "max": 50000}})
    _assert_equivalent(store, "T0-token-id-in-vocab", [])


# === T1-replica-cksum-equal (B1 / M-005 surrogate) =======================


def test_replica_cksum_equal_equivalence_buggy(store):
    e1 = store.emit("build.snapshot", {
        "cross_rank_cksums": [
            {"name": "weight.foo", "group_size": 1,
             "local_cksum": 1, "all_equal": True},  # size-1: skipped
            {"name": "router.weight", "group_size": 2,
             "local_cksum": 1, "gathered_cksums": [1, 2],
             "all_equal": False},                    # buggy
            {"name": "shared.bias", "group_size": 4,
             "local_cksum": 5, "gathered_cksums": [5, 5, 5, 5],
             "all_equal": True}],                    # clean
    })
    _assert_equivalent(store, "T1-replica-cksum-equal", [e1])


def test_replica_cksum_equal_equivalence_clean(store):
    store.emit("build.snapshot", {
        "cross_rank_cksums": [
            {"name": "router.weight", "group_size": 2,
             "local_cksum": 1, "gathered_cksums": [1, 1],
             "all_equal": True}],
    })
    _assert_equivalent(store, "T1-replica-cksum-equal", [])


# === T0-norm-output-unit-rms (O-NEW-1 surrogate, A3 tensor_signature) ===


def test_norm_output_rms_equivalence_buggy(store):
    # Buggy: RMS ≈ 0.33 (the O-NEW-1 fingerprint)
    # l2² / numel = (0.33 * sqrt(numel))² = 0.109 * numel; pick l2 = 0.33 * sqrt(8*4) = 1.866
    e1 = store.emit("module.fwd.post", {
        "module_class": "RMSLayerNorm", "is_normalizer": True,
        "output": {"dtype": "torch.float32", "shape": [8, 4],
                   "l2_norm": 1.866}})  # rms = 1.866 / sqrt(32) ≈ 0.33 → bug
    # Clean: rms ≈ 1.0
    store.emit("module.fwd.post", {
        "module_class": "RMSLayerNorm", "is_normalizer": True,
        "output": {"dtype": "torch.float32", "shape": [8, 4],
                   "l2_norm": 5.66}})   # rms ≈ 1.0
    # Non-normalizer event must not be checked
    store.emit("module.fwd.post", {
        "module_class": "Linear", "is_normalizer": False,
        "output": {"dtype": "torch.float32", "shape": [8, 4],
                   "l2_norm": 100.0}})
    _assert_equivalent(store, "T0-norm-output-unit-rms", [e1])


def test_norm_output_rms_equivalence_clean(store):
    store.emit("module.fwd.post", {
        "module_class": "LayerNorm", "is_normalizer": True,
        "output": {"dtype": "torch.float32", "shape": [4, 16],
                   "l2_norm": 8.0}})    # rms = 8.0/sqrt(64) = 1.0
    _assert_equivalent(store, "T0-norm-output-unit-rms", [])


# === T0-softmax-degenerate (M-014 surrogate, A3 tensor_signature) =======


def test_softmax_degenerate_equivalence_buggy(store):
    # Trivial size-1 softmax: every row picks the only option (the M-014 bug)
    # shape (4, 1): n_rows = 4, l2² = 4, abs_max = 1.0 → degenerate
    e1 = store.emit("module.fwd.post", {
        "module_class": "TopKRouter",
        "output": {"dtype": "torch.float32", "shape": [4, 1],
                   "l2_norm": 2.0, "abs_max": 1.0}})  # bug
    # Clean: same router, but softmax output is a real distribution
    # shape (4, 8), l2² < n_rows → not one-hot
    store.emit("module.fwd.post", {
        "module_class": "TopKRouter",
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 0.5, "abs_max": 0.3}})  # not degenerate
    # Non-router class (Linear) — degenerate values but ignored by precondition
    store.emit("module.fwd.post", {
        "module_class": "Linear",
        "output": {"dtype": "torch.float32", "shape": [4, 1],
                   "l2_norm": 2.0, "abs_max": 1.0}})
    _assert_equivalent(store, "T0-softmax-degenerate", [e1])


def test_softmax_degenerate_equivalence_clean(store):
    store.emit("module.fwd.post", {
        "module_class": "TopKRouter",
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 0.5, "abs_max": 0.3}})
    _assert_equivalent(store, "T0-softmax-degenerate", [])


# === T0-optim-step-counter-monotonic (OC-NEW-2 surrogate, A3 MONOTONIC) ===


def test_optim_step_counter_equivalence_buggy(store):
    # state_step_max stays the same → bug (SkipStepAdamW step.add_(...) commented out)
    store.emit("optim.step.post", {
        "optimizer_class": "SkipStepAdamW",
        "state_step_max": 5.0, "state_step_min": 5.0,
        "total_param_l2": 1.0})
    e2 = store.emit("optim.step.post", {
        "optimizer_class": "SkipStepAdamW",
        "state_step_max": 5.0, "state_step_min": 5.0,  # didn't move → bug
        "total_param_l2": 1.0})
    e3 = store.emit("optim.step.post", {
        "optimizer_class": "SkipStepAdamW",
        "state_step_max": 5.0, "state_step_min": 5.0,  # still didn't move
        "total_param_l2": 1.0})
    _assert_equivalent(store, "T0-optim-step-counter-monotonic", [e2, e3])


def test_optim_step_counter_equivalence_clean(store):
    for i in range(3):
        store.emit("optim.step.post", {
            "optimizer_class": "AdamW",
            "state_step_max": float(i + 1), "state_step_min": float(i + 1),
            "total_param_l2": 1.0})
    _assert_equivalent(store, "T0-optim-step-counter-monotonic", [])


# === T1-expert-bias-fp32 (M-012 surrogate) ===============================


def test_expert_bias_fp32_equivalence_buggy(store):
    e1 = store.emit("module.fwd.post", {
        "module_class": "TopKRouter",
        "semantic": {"is_router": True, "expert_bias_dtype": "bfloat16"},
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 1.0}})
    store.emit("module.fwd.post", {
        "module_class": "TopKRouter",
        "semantic": {"is_router": True, "expert_bias_dtype": "float32"},
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 1.0}})
    # Non-router event must be skipped by precondition
    store.emit("module.fwd.post", {
        "module_class": "Linear",
        "semantic": {"is_router": False, "expert_bias_dtype": None},
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 1.0}})
    _assert_equivalent(store, "T1-expert-bias-fp32", [e1])


def test_expert_bias_fp32_equivalence_clean(store):
    store.emit("module.fwd.post", {
        "module_class": "TopKRouter",
        "semantic": {"is_router": True, "expert_bias_dtype": "float32"}})
    _assert_equivalent(store, "T1-expert-bias-fp32", [])


# === T1-jitter-preserves-dtype (M-024 surrogate, probe_derived dsl_native) =


def test_jitter_dtype_equivalence_buggy(store):
    e1 = store.emit("jitter.probe", {
        "module_class": "TopKRouter",
        "input_dtype": "bfloat16", "output_dtype": "float32",
        "dtypes_match": False})
    store.emit("jitter.probe", {
        "module_class": "TopKRouter",
        "input_dtype": "bfloat16", "output_dtype": "bfloat16",
        "dtypes_match": True})
    _assert_equivalent(store, "T1-jitter-preserves-dtype", [e1])


def test_jitter_dtype_equivalence_clean(store):
    store.emit("jitter.probe", {
        "module_class": "TopKRouter",
        "input_dtype": "float32", "output_dtype": "float32",
        "dtypes_match": True})
    _assert_equivalent(store, "T1-jitter-preserves-dtype", [])


# === T1-residual-stream-preserved (B13 surrogate, probe_derived dsl_native) =


def test_residual_stream_equivalence_buggy(store):
    e1 = store.emit("residual.probe", {
        "block_class": "OLMoSequentialBlock",
        "d_to_original_input": 5.0,
        "d_to_normed_input": 0.5,  # output closer to normed → bug
    })
    store.emit("residual.probe", {
        "block_class": "OLMoSequentialBlock",
        "d_to_original_input": 0.4,
        "d_to_normed_input": 4.0,  # output closer to original → healthy
    })
    _assert_equivalent(store, "T1-residual-stream-preserved", [e1])


def test_residual_stream_equivalence_clean(store):
    store.emit("residual.probe", {
        "block_class": "OLMoSequentialBlock",
        "d_to_original_input": 0.4,
        "d_to_normed_input": 4.0})
    _assert_equivalent(store, "T1-residual-stream-preserved", [])
