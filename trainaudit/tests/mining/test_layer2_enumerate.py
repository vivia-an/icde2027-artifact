"""B1 acceptance: schema introspection + per-relation-type enumeration."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

import trainaudit  # noqa: E402
from trainaudit.mining import (Hypothesis, RelationType,
                                enumerate_predicates, schema_introspect)


def _build_clean_trace_store():
    """Run a tiny trainaudit-instrumented training, return the store."""
    store = trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                               db_path=":memory:")
    torch.manual_seed(0)
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


def test_schema_introspect_finds_real_hookpoints():
    store = _build_clean_trace_store()
    try:
        schema = schema_introspect(store)
        # Must enumerate at least the major hookpoints
        for hp in ("module.fwd.post", "optim.step.post",
                    "utils.clip_grad.post", "build.snapshot"):
            assert hp in schema, f"missing {hp} in schema; got {list(schema)}"
        # The clip_grad post payload must expose post_norm + max_norm
        clip = schema["utils.clip_grad.post"]
        assert "$.post_norm" in clip and "$.max_norm" in clip
        # build.snapshot must expose model.n_parameters etc.
        bs = schema["build.snapshot"]
        assert "$.model.n_parameters" in bs
    finally:
        trainaudit.disable()


def test_enumerate_payload_field_compare_finds_clip_grad():
    """For PAYLOAD_FIELD_COMPARE on a clip_grad-bearing schema, the
    enumerator must output a (post_norm <= max_norm) predicate — the
    same shape as the existing T0-clip-grad-bounded YAML."""
    store = _build_clean_trace_store()
    try:
        schema = schema_introspect(store)
        hyp = Hypothesis(relation_type=RelationType.PAYLOAD_FIELD_COMPARE,
                          entities=["norm", "threshold"],
                          dimensions=[])
        cands = enumerate_predicates(hyp, schema)
        assert cands, "expected at least one candidate"
        clip_cand = [
            p for p in cands
            if p.scope.hookpoint == "utils.clip_grad.post"
            and (p.bound.field == "post_norm" if isinstance(p.bound.field, str)
                 else "post_norm" in p.bound.field)
        ]
        assert clip_cand, (
            f"expected a (post_norm vs *) predicate at clip_grad post, "
            f"got candidates: {[(p.scope.hookpoint, p.bound.field, p.bound.op, p.bound.value) for p in cands]}")
        # Must include the (post_norm, max_norm) variant
        canonical = [p for p in clip_cand
                      if p.bound.value_is_field and p.bound.value == "max_norm"]
        assert canonical, "expected post_norm vs max_norm canonical pair"
    finally:
        trainaudit.disable()


def test_enumerate_cross_rank_equal_yields_replica_predicate():
    store = _build_clean_trace_store()
    try:
        # Inject a build.snapshot with cross_rank_cksums to exercise the path
        store.emit("build.snapshot", {
            "cross_rank_cksums": [{"name": "w", "group_size": 2,
                                    "all_equal": True}],
        })
        store.flush()
        schema = schema_introspect(store)
        hyp = Hypothesis(relation_type=RelationType.CROSS_RANK_EQUAL,
                          entities=["param"], dimensions=["rank"])
        cands = enumerate_predicates(hyp, schema)
        assert len(cands) == 1
        p = cands[0]
        assert p.scope.payload_path == "$.cross_rank_cksums[*]"
        assert p.bound.field == "all_equal" and p.bound.value is True
    finally:
        trainaudit.disable()


def test_enumerate_cross_step_monotonic_yields_optim_step_predicate():
    store = _build_clean_trace_store()
    try:
        schema = schema_introspect(store)
        hyp = Hypothesis(relation_type=RelationType.CROSS_STEP_MONOTONIC,
                          entities=["state_step"], dimensions=["step"])
        cands = enumerate_predicates(hyp, schema)
        assert any(p.scope.hookpoint == "optim.step.post"
                    and p.bound.field == "state_step_max"
                    for p in cands), (
            f"expected optim.step.post / state_step_max monotonic; got "
            f"{[(p.scope.hookpoint, p.bound.field) for p in cands]}")
    finally:
        trainaudit.disable()


def test_enumerate_unknown_scope_returns_empty():
    """Schema with no matching hookpoint → enumerator returns empty list,
    not crash."""
    schema = {"some.weird.hookpoint": {"$.foo": "number"}}
    hyp = Hypothesis(relation_type=RelationType.CROSS_RANK_EQUAL)
    assert enumerate_predicates(hyp, schema) == []
