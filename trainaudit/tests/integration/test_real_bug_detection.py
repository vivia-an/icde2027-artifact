"""End-to-end real-silent-error detection.

For each bug pattern:
  1. build a real PyTorch model
  2. run forward + backward + optimizer step under `trainaudit.enable()`
  3. inject the bug *through real PyTorch code paths* — not by hand-emitting
     events — so the rule sees the bug arrive through the same hooks that
     would fire on the actual framework bug
  4. assert the corresponding rule fires (both Python and DSL paths)
  5. assert violation event_id sets agree between paths

Covered bug patterns (proxy for real paper bugs):
  - B11 (DeepSpeed clip_grad max-vs-min): surrogate via a wrapper that
    bypasses clipping while still emitting the post event
  - O-NEW-1 (OLMo RMSNorm wrong scale): surrogate via a normalizer
    subclass that produces output with RMS far from 1.0
  - OC-NEW-2 (SkipStepAdamW state['step'] frozen): surrogate via custom
    AdamW that never increments state['step']

These are surrogates of the real DeepSpeed / OLMo / OLMo-core bugs because
the real framework bug requires checking out an old commit + GPU. The rule
*signatures* are identical — what fires here is exactly what fires on a
live `trainaudit_run.sh` for the corresponding bug.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from typing import List

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

import trainaudit  # noqa: E402
from trainaudit.dsl import load_predicates_dir, violation_event_ids  # noqa: E402

REGISTRY = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "trainaudit", "dsl", "registry"))


# -- helpers --------------------------------------------------------------


def _python_violation_event_ids(rule_id: str) -> List[int]:
    """Run the registered Python rule and return its full violation event_ids."""
    from trainaudit.rules import all_rules
    store = trainaudit.get_store()
    store.flush()
    matching = [r for r in all_rules() if r.rule_id == rule_id]
    assert len(matching) == 1
    res = matching[0].check(store.conn)
    if not isinstance(res, list):
        res = [res]
    ids: List[int] = []
    for r in res:
        if r.violated and "violation_event_ids" in (r.evidence or {}):
            ids.extend(r.evidence["violation_event_ids"])
    return sorted(ids)


def _dsl_violation_event_ids(rule_id: str) -> List[int]:
    store = trainaudit.get_store()
    p = next(p for p in load_predicates_dir(REGISTRY) if p.id == rule_id)
    return sorted(violation_event_ids(store, p))


def _assert_both_paths_detect(rule_id: str) -> List[int]:
    """Verify Python and DSL paths agree, and that violation set is non-empty."""
    py = _python_violation_event_ids(rule_id)
    ds = _dsl_violation_event_ids(rule_id)
    assert py == ds, f"{rule_id}: Python={py} vs DSL={ds} disagree"
    assert py, f"{rule_id}: expected at least one violation, got none"
    return py


# -- B11: clip_grad without clipping ---------------------------------------


def test_B11_clip_grad_bug_e2e_real_model():
    """Surrogate of DeepSpeed B11 bug: clip_grad_norm_ uses max instead of
    min for clip_coef, so when grad >> max_norm, no clipping happens.

    Real model + real backward + buggy clip path injected at the
    torch.nn.utils namespace BEFORE trainaudit.enable() — trainaudit then
    wraps the buggy function. Hook records pre_norm + post_norm; bug
    leaves them equal (no actual clipping)."""
    import torch.nn.utils as _nn_utils
    real_clip = _nn_utils.clip_grad_norm_

    def buggy_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        # Mimic B11's max-vs-min: compute pre-norm but DO NOT scale
        # gradients down. Returns a norm value but leaves grads unchanged.
        params = [p for p in parameters if p is not None and p.grad is not None]
        if not params:
            return torch.tensor(0.0)
        pre_norm = torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad.detach().float(), norm_type)
             for p in params]), norm_type)
        return pre_norm

    _nn_utils.clip_grad_norm_ = buggy_clip
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                              db_path=os.path.join(tmp, "trace.duckdb"))
            try:
                torch.manual_seed(0)
                model = nn.Sequential(nn.Linear(8, 16), nn.GELU(),
                                       nn.Linear(16, 4))
                opt = optim.AdamW(model.parameters(), lr=1e-3)
                trainaudit.snapshot_build(model, opt)

                # Big loss → big grads
                x = torch.randn(2, 8)
                (model(x).pow(2).sum() * 1e6).backward()

                # Now call clip with realistic max_norm. Buggy path leaves
                # grads at their large pre-clip values → post_norm > max_norm.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                opt.step()

                ids = _assert_both_paths_detect("T0-clip-grad-bounded")
                print(f"[E2E B11] T0-clip-grad-bounded fired on event_ids: {ids}")
            finally:
                trainaudit.disable()
    finally:
        # restore the real clip even if test asserts
        _nn_utils.clip_grad_norm_ = real_clip


# -- O-NEW-1: normalizer with wrong RMS -----------------------------------


class _BrokenRMSNorm(nn.LayerNorm):
    """Mimics O-NEW-1 surrogate: normalizer-class layer that produces
    output with RMS ≈ 0.33 (instead of ~1.0). The trainaudit module hook
    detects nn.LayerNorm subclass → marks is_normalizer=True automatically."""

    def forward(self, x):
        out = super().forward(x)
        return out * 0.33


def test_O_NEW_1_norm_rms_bug_e2e_real_model():
    """Surrogate of OLMo O-NEW-1: normalizer produces RMS far from 1.0.
    Catches bugs where L2 vs RMS confusion or wrong eps placement breaks
    normalization output magnitude."""
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            # _BrokenRMSNorm is an nn.LayerNorm subclass → trainaudit marks
            # it `is_normalizer=True` via isinstance check
            model = nn.Sequential(
                nn.Linear(8, 16),
                _BrokenRMSNorm(16),
                nn.GELU(),
                nn.Linear(16, 4),
            )
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)

            for step in range(3):
                trainaudit.set_step(step)
                x = torch.randn(2, 8)
                loss = model(x).pow(2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()

            ids = _assert_both_paths_detect("T0-norm-output-unit-rms")
            print(f"[E2E O-NEW-1] T0-norm-output-unit-rms fired on "
                  f"event_ids: {ids}")
        finally:
            trainaudit.disable()


# -- OC-NEW-2: optimizer state['step'] frozen -----------------------------


class _FrozenStepAdamW(optim.AdamW):
    """Mimics OC-NEW-2 surrogate: 'SkipStepAdamW' commented out
    `step.add_(step_factor)`, so state['step'] never increments. AdamW bias
    correction stops updating, optimizer effectively freezes."""

    @torch.no_grad()
    def step(self, closure=None):
        # Run the real AdamW math but DON'T increment state['step']
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.tensor(0.0)
                # Apply a tiny update, but do not increment state['step']
                p.data.add_(p.grad, alpha=-group["lr"])
        return None


def test_OC_NEW_2_optim_step_counter_e2e_real_model():
    """Surrogate of OLMo-core OC-NEW-2: optimizer state['step'] frozen
    across optim.step() calls. Trainaudit's optim hook captures
    state['step'] in optim.step.post payload; rule detects non-monotonic
    progression."""
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
            # Pre-populate state['step'] so all optim.step.post events have it
            opt = _FrozenStepAdamW(model.parameters(), lr=1e-3)
            for p in model.parameters():
                opt.state[p]["step"] = torch.tensor(0.0)
            trainaudit.snapshot_build(model, opt)

            for step in range(4):
                trainaudit.set_step(step)
                x = torch.randn(2, 8)
                loss = model(x).pow(2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()  # state['step'] does NOT increment

            ids = _assert_both_paths_detect("T0-optim-step-counter-monotonic")
            print(f"[E2E OC-NEW-2] T0-optim-step-counter-monotonic fired on "
                  f"event_ids: {ids}")
        finally:
            trainaudit.disable()


# -- B12: scheduler resume without initial_lr -----------------------------


def test_B12_scheduler_resume_missing_initial_lr_e2e():
    """Surrogate of B12 / O-016: AdamWConfig.build forgot to set
    initial_lr on optimizer param_groups. When the scheduler is then
    constructed with last_epoch != -1 (resume), PyTorch's __init__
    expects initial_lr per group — its absence is silent corruption."""
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)

            # Simulate the bug: param_groups WITHOUT initial_lr set,
            # then construct LRScheduler with last_epoch=10 (resume).
            for g in opt.param_groups:
                g.pop("initial_lr", None)

            # CosineAnnealingLR with last_epoch != -1 hits the buggy path
            try:
                _ = optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=100, last_epoch=10)
            except KeyError:
                # PyTorch raises KeyError for missing initial_lr — that's
                # a hard fail. The bug we model is the silent variant where
                # the optimizer config code skipped setting initial_lr.
                # Hook still emits scheduler.init event before the failure.
                pass

            ids = _assert_both_paths_detect("T0-initial-lr-present")
            print(f"[E2E B12] T0-initial-lr-present fired on event_ids: {ids}")
        finally:
            trainaudit.disable()


# -- M-014: MoE router degenerate softmax (size-1 dim) --------------------


class _DegenerateTopKRouter(nn.Module):
    """Mimics M-014 surrogate: TopKRouter with topk=1 and post-softmax,
    so softmax of a single value is always 1.0 → every routing decision
    is `[1.0]` regardless of input. Zero gradient, training degenerates."""

    def __init__(self, hidden, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts)

    def forward(self, x):
        scores = self.linear(x)                  # [..., n_experts]
        # buggy: take top-1 then softmax over the size-1 dim → always 1.0
        topk_scores, _topk_idx = scores.topk(1, dim=-1)  # [..., 1]
        return torch.softmax(topk_scores, dim=-1)        # [..., 1] → all 1s


def test_M_014_softmax_degenerate_e2e_real_model():
    """Surrogate of Megatron M-014: TopKRouter post-softmax over size-1
    dimension is mathematically 1.0 everywhere. Detected via shape-aware
    tensor signature (l2² ≈ n_rows AND abs_max ≈ 1.0)."""
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(
                nn.Linear(8, 16),
                _DegenerateTopKRouter(16, n_experts=4),
            )
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)

            for step in range(2):
                trainaudit.set_step(step)
                x = torch.randn(2, 8)
                loss = model(x).pow(2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()

            ids = _assert_both_paths_detect("T0-softmax-degenerate")
            print(f"[E2E M-014] T0-softmax-degenerate fired on event_ids: {ids}")
        finally:
            trainaudit.disable()


# -- Clean run: nothing should fire ---------------------------------------


def test_clean_run_no_violations_real_model():
    """A 'healthy' training run must produce zero violations on both
    Python and DSL paths across all rules. This is the clean-trace
    baseline that backs paper §4.3 FP-rate claim."""
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(
                nn.Linear(8, 16),
                nn.LayerNorm(16),
                nn.GELU(),
                nn.Linear(16, 4),
            )
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)

            for step in range(3):
                trainaudit.set_step(step)
                x = torch.randn(2, 8)
                loss = model(x).pow(2).sum()
                opt.zero_grad()
                loss.backward()
                # exercise clip path (B11 hookpoint) with realistic max_norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                opt.step()

            py_results = trainaudit.run_rules(use_dsl=False)
            dsl_results = trainaudit.run_rules(use_dsl=True)

            py_v = [r.rule_id for r in py_results if r.violated]
            ds_v = [r.rule_id for r in dsl_results if r.violated]

            assert not py_v, f"clean-run Python false positives: {py_v}"
            assert not ds_v, f"clean-run DSL false positives: {ds_v}"
        finally:
            trainaudit.disable()
