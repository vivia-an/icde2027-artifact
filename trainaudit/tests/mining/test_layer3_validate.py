"""B2 acceptance: healthy-trace validator + end-to-end mining loop.

The acid test for paper §3.2: a Hypothesis (no hard-coded thresholds)
plus a small set of healthy traces yields a verified Predicate; that
same predicate then fires on a buggy trace.

Pipeline:
  Hypothesis → Layer 2 enumerate → candidate Predicate
            → Layer 3 validate against N healthy traces → accepted
            → run on buggy trace → violation event_ids
"""
from __future__ import annotations

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
from trainaudit.dsl import violation_event_ids  # noqa: E402
from trainaudit.mining import (Hypothesis, RelationType,
                                enumerate_predicates,
                                infer_tolerance, schema_introspect,
                                validate_against_healthy)


def _run_clean_trace(seed: int, db_path: str):
    """One clean training run; returns the closed-over store handle (caller
    must trainaudit.disable() at end of test)."""
    store = trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path=db_path)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16),
                           nn.GELU(), nn.Linear(16, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    trainaudit.snapshot_build(model, opt)
    for step in range(3):
        trainaudit.set_step(step)
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
    store.flush()
    return store


# ---- B2: validator ------------------------------------------------------


def test_validator_accepts_invariant_holding_on_all_healthy():
    """Mine clip_grad invariant on 3 healthy traces; predicate must be
    accepted (zero violations on all)."""
    healthy_stores = []
    try:
        for i in range(3):
            with tempfile.TemporaryDirectory() as tmp:
                # NOTE: each enable() creates a fresh in-mem store, but we
                # need persistence for cross-trace inspection. Use file dbs.
                pass
        # Build in a way each store is independent and survives disable()
        tmp = tempfile.mkdtemp()
        for i in range(3):
            store = _run_clean_trace(seed=i, db_path=os.path.join(tmp, f"healthy_{i}.duckdb"))
            healthy_stores.append(store)
            trainaudit.disable()  # end this run; store stays alive on disk
            # re-open for cross-validation outside trainaudit lifecycle
            from trainaudit.store import TraceStore
            healthy_stores[-1] = TraceStore(os.path.join(tmp, f"healthy_{i}.duckdb"))

        schema = schema_introspect(healthy_stores[0])
        hyp = Hypothesis(relation_type=RelationType.PAYLOAD_FIELD_COMPARE,
                          entities=["norm"])
        cands = enumerate_predicates(hyp, schema)
        clip_pred = next(
            (p for p in cands
             if p.scope.hookpoint == "utils.clip_grad.post"
             and p.bound.value_is_field and p.bound.value == "max_norm"), None)
        assert clip_pred is not None, "no canonical clip_grad predicate enumerated"

        result = validate_against_healthy(clip_pred, healthy_stores)
        assert result.accepted, (
            f"clip predicate should hold on healthy traces; got "
            f"{result.n_violations_per_trace}, reason={result.rejection_reason}")
        assert result.n_violations_per_trace == [0, 0, 0]
    finally:
        for s in healthy_stores:
            try:
                s.close()
            except Exception:
                pass


def test_validator_rejects_overstrict_predicate():
    """A made-up predicate that's stricter than reality (lr must be
    > 1.0) should be REJECTED on healthy traces."""
    healthy_stores = []
    try:
        tmp = tempfile.mkdtemp()
        for i in range(3):
            _run_clean_trace(seed=i, db_path=os.path.join(tmp, f"h{i}.duckdb"))
            trainaudit.disable()
            from trainaudit.store import TraceStore
            healthy_stores.append(TraceStore(os.path.join(tmp, f"h{i}.duckdb")))

        from trainaudit.dsl import (Bound, BoundKind, Predicate, Scope,
                                     Template)
        overstrict = Predicate(
            id="overstrict/lr_gt_1",
            template=Template.PAYLOAD_FIELD_COMPARE,
            scope=Scope(hookpoint="utils.clip_grad.post"),
            bound=Bound(kind=BoundKind.BOUND, field="post_norm",
                        op=">", value=1e9),  # absurdly high lower bound
        )
        result = validate_against_healthy(overstrict, healthy_stores)
        assert not result.accepted
        assert any(v > 0 for v in result.n_violations_per_trace)
    finally:
        for s in healthy_stores:
            try:
                s.close()
            except Exception:
                pass


# ---- end-to-end: healthy-validated predicate fires on buggy trace --------


def test_mined_predicate_fires_on_buggy_trace():
    """The full §3.2 loop: mine on 3 healthy → accept → re-run on buggy →
    detect."""
    import torch.nn.utils as _nn_utils
    real_clip = _nn_utils.clip_grad_norm_

    def buggy_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        params = [p for p in parameters if p is not None and p.grad is not None]
        if not params:
            return torch.tensor(0.0)
        return torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad.detach().float(), norm_type)
             for p in params]), norm_type)

    healthy_stores = []
    try:
        tmp = tempfile.mkdtemp()
        # 3 healthy traces (real clip_grad → respects max_norm)
        for i in range(3):
            _run_clean_trace(seed=i, db_path=os.path.join(tmp, f"h{i}.duckdb"))
            trainaudit.disable()
        from trainaudit.store import TraceStore
        for i in range(3):
            healthy_stores.append(TraceStore(os.path.join(tmp, f"h{i}.duckdb")))

        schema = schema_introspect(healthy_stores[0])
        hyp = Hypothesis(relation_type=RelationType.PAYLOAD_FIELD_COMPARE)
        cands = enumerate_predicates(hyp, schema)
        clip_pred = next(
            p for p in cands
            if p.scope.hookpoint == "utils.clip_grad.post"
            and p.bound.value_is_field and p.bound.value == "max_norm")

        # Layer 3: this predicate must hold on all healthy
        v = validate_against_healthy(clip_pred, healthy_stores)
        assert v.accepted, f"healthy validation failed: {v}"

        # Now run a buggy trace (clip is bypassed) and re-apply the SAME
        # mined predicate. It must now fire.
        _nn_utils.clip_grad_norm_ = buggy_clip
        buggy_db = os.path.join(tmp, "buggy.duckdb")
        store = trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                                   db_path=buggy_db)
        try:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)
            opt.zero_grad()
            x = torch.randn(2, 8)
            (model(x).pow(2).sum() * 1e6).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            opt.step()
            store.flush()
        finally:
            trainaudit.disable()

        buggy_store = TraceStore(buggy_db)
        try:
            ids = violation_event_ids(buggy_store, clip_pred)
            assert ids, ("end-to-end mining loop failed: predicate accepted "
                         "on healthy but did NOT fire on buggy trace")
            print(f"[E2E mining] healthy-validated clip predicate fired on "
                  f"buggy trace: event_ids={ids}")
        finally:
            buggy_store.close()
    finally:
        _nn_utils.clip_grad_norm_ = real_clip
        for s in healthy_stores:
            try:
                s.close()
            except Exception:
                pass


def test_infer_tolerance_returns_pct_quantile():
    """Auto-learn tolerance: 99th-percentile of post_norm across healthy
    traces should be a positive finite number for our toy training."""
    healthy_stores = []
    try:
        tmp = tempfile.mkdtemp()
        for i in range(3):
            _run_clean_trace(seed=i, db_path=os.path.join(tmp, f"h{i}.duckdb"))
            trainaudit.disable()
        from trainaudit.store import TraceStore
        for i in range(3):
            healthy_stores.append(TraceStore(os.path.join(tmp, f"h{i}.duckdb")))

        v = infer_tolerance("post_norm", healthy_stores,
                             "utils.clip_grad.post", pct=99.0)
        assert v is not None and v > 0.0, (
            f"expected a positive 99th-percentile post_norm, got {v}")
    finally:
        for s in healthy_stores:
            try:
                s.close()
            except Exception:
                pass
