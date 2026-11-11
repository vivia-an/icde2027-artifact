"""A2 smoke tests: each of the 7 dsl_native YAMLs compiles and produces the
expected violations on a synthetic TraceStore.

These are unit-level — A2's full equivalence test (M1 gate) lives in
test_compiler_equivalence.py and runs against live trainaudit_run.sh traces.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from trainaudit.dsl import (compile_predicate, load_predicate,
                            load_predicates_dir, run_compiled_one,
                            violation_event_ids)
from trainaudit.store import TraceStore

REGISTRY = Path(__file__).resolve().parents[2] / "trainaudit" / "dsl" / "registry"


@pytest.fixture
def store():
    s = TraceStore(":memory:")
    yield s
    s.close()


def _pred(pid: str):
    for p in load_predicates_dir(REGISTRY):
        if p.id == pid:
            return p
    raise KeyError(pid)


# --- T0-clip-grad-bounded -----------------------------------------------


def test_clip_grad_violates_when_post_exceeds_max(store):
    store.emit("utils.clip_grad.post", {
        "max_norm": 1.0, "pre_norm": 5.0, "post_norm": 5.0, "ratio": 1.0})
    store.emit("utils.clip_grad.post", {
        "max_norm": 1.0, "pre_norm": 5.0, "post_norm": 1.005, "ratio": 1.005})
    p = _pred("T0-clip-grad-bounded")
    ids = violation_event_ids(store, p)
    assert ids == [1], "only the post=5.0 row should violate (1.005 within 1% rel tol)"


def test_clip_grad_clean(store):
    store.emit("utils.clip_grad.post", {
        "max_norm": 1.0, "pre_norm": 1.5, "post_norm": 1.0, "ratio": 1.0})
    store.flush()
    p = _pred("T0-clip-grad-bounded")
    res = run_compiled_one(store.conn, compile_predicate(p))
    assert not res.violated


# --- T0-no-nan-inf ------------------------------------------------------


def test_no_nan_inf_violates_when_nested_tensor_has_nan(store):
    store.emit("module.fwd.post", {
        "module_class": "Linear",
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 1.0, "has_nan": False, "has_inf": False},
    })
    store.emit("module.fwd.post", {
        "module_class": "Linear",
        "outputs": [
            {"dtype": "torch.float32", "shape": [4, 8],
             "l2_norm": 1.0, "has_nan": True, "has_inf": False},
        ],
    })
    p = _pred("T0-no-nan-inf")
    ids = violation_event_ids(store, p)
    assert ids == [2]


# --- T0-optim-lr-positive ----------------------------------------------


def test_optim_lr_positive_flags_zero_lr(store):
    store.emit("build.snapshot", {
        "optimizer": {
            "param_groups": [
                {"index": 0, "lr": 1e-4},
                {"index": 1, "lr": 0.0},
            ],
        },
    })
    p = _pred("T0-optim-lr-positive")
    ids = violation_event_ids(store, p)
    assert ids == [1], "row should fire on the lr=0 entry"


# --- T0-build-has-modules ----------------------------------------------


def test_build_has_modules_flags_zero_count(store):
    store.emit("build.snapshot", {"model": {"n_parameters": 0, "n_modules": 5}})
    store.emit("build.snapshot", {"model": {"n_parameters": 12, "n_modules": 5}})
    p = _pred("T0-build-has-modules")
    ids = violation_event_ids(store, p)
    assert ids == [1]


# --- T0-initial-lr-present ---------------------------------------------


def test_initial_lr_required_when_resuming(store):
    # fresh start (last_epoch=-1): missing initial_lr is fine
    store.emit("scheduler.init", {
        "scheduler_class": "CosineLR", "last_epoch": -1,
        "param_groups": [{"index": 0, "has_initial_lr": False, "lr": 1e-4}],
    })
    # resume (last_epoch=10): missing initial_lr is a violation
    store.emit("scheduler.init", {
        "scheduler_class": "CosineLR", "last_epoch": 10,
        "param_groups": [{"index": 0, "has_initial_lr": False, "lr": 1e-4}],
    })
    p = _pred("T0-initial-lr-present")
    ids = violation_event_ids(store, p)
    assert ids == [2]


# --- T0-token-id-in-vocab ----------------------------------------------


def test_token_id_in_vocab_flags_truncation(store):
    store.emit("dataloader.batch", {
        "input_ids": {"dtype": "torch.int64", "shape": [2, 16],
                      "min": 0, "max": 50000}})
    store.emit("dataloader.batch", {
        "input_ids": {"dtype": "torch.int64", "shape": [2, 16],
                      "min": 0, "max": 5_000_000}})
    p = _pred("T0-token-id-in-vocab")
    ids = violation_event_ids(store, p)
    assert ids == [2]


# --- T1-replica-cksum-equal --------------------------------------------


def test_replica_cksum_violates_when_disagreement(store):
    # group_size=1 entries (size-1 group) should never violate even if
    # all_equal is missing/false
    store.emit("build.snapshot", {
        "cross_rank_cksums": [
            {"name": "weight.foo", "group_size": 1,
             "local_cksum": 1, "all_equal": True},
            {"name": "router.weight", "group_size": 2,
             "local_cksum": 1, "gathered_cksums": [1, 2], "all_equal": False},
            {"name": "shared.bias", "group_size": 4,
             "local_cksum": 5, "gathered_cksums": [5, 5, 5, 5],
             "all_equal": True},
        ],
    })
    p = _pred("T1-replica-cksum-equal")
    ids = violation_event_ids(store, p)
    assert ids == [1]
    # the violated entry should be flagged exactly once
    assert len(violation_event_ids(store, p)) == 1
