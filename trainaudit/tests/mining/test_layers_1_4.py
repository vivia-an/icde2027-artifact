"""B3/B4 acceptance: LLM Layer 1 (hypothesis generation) + Layer 4 (filter).

Tests use deterministic stubs so they're hermetic; production wiring is
a single constructor swap to a real LLMClient.

Layer 1 (hypothesis): given a code snippet, return a list of Hypotheses.
Layer 4 (filter): given a Predicate, return keep|drop decision.

Plus the full 4-layer end-to-end loop:
  source code → L1 propose → L2 enumerate → L3 validate → L4 filter
                          → set of verified, non-spurious predicates
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

import trainaudit  # noqa: E402
from trainaudit.dsl import (Bound, BoundKind, Predicate, Scope, Template,
                              violation_event_ids)
from trainaudit.mining import (CATALOG18, FilterDecision, Hypothesis,
                                RelationType, enumerate_predicates,
                                filter_predicates, mine_catalog,
                                propose_hypotheses, schema_introspect,
                                validate_against_healthy)


# ---- B3: Layer 1 propose --------------------------------------------------


def test_propose_hypotheses_for_clip_grad_source():
    src = """
    def clip_grad_norm_(parameters, max_norm):
        # Buggy: torch.max instead of torch.min
        clip_coef = torch.max(torch.tensor(1.0), max_norm / total_norm)
        for p in parameters:
            p.grad.mul_(clip_coef)
    """
    hyps = propose_hypotheses(src, framework="DeepSpeed")
    assert hyps, "expected at least one hypothesis"
    types = [h.relation_type for h in hyps]
    assert RelationType.PAYLOAD_FIELD_COMPARE in types, (
        f"expected payload_field_compare for clip_grad source, got {types}")


def test_propose_hypotheses_for_norm_source():
    src = "class RMSNorm(nn.Module): pass"
    hyps = propose_hypotheses(src, framework="OLMo")
    types = [h.relation_type for h in hyps]
    assert RelationType.TENSOR_STAT_BOUND in types


def test_propose_hypotheses_for_scheduler_source():
    src = ("class CosineLR:\n"
           "    def __init__(self, optim, last_epoch=-1):\n"
           "        ...")
    hyps = propose_hypotheses(src, framework="OLMo-core")
    types = [h.relation_type for h in hyps]
    assert RelationType.CONDITIONAL_CHECK in types


def test_mine_catalog_walks_18_directions():
    rows = mine_catalog("def train(): pass", framework="Megatron")
    assert [r[0] for r in rows] == [item[0] for item in CATALOG18]
    assert len(rows) == 18
    for rule_id, hyps in rows:
        assert hyps
        assert all(h.scope_hint.get("direction") == rule_id for h in hyps)


def test_propose_hypotheses_with_pluggable_llm():
    captured = {}

    def my_llm(system, user, *, max_tokens=1024):
        captured["src_seen"] = "router" in user.lower()
        return ('```json\n{"hypotheses": [{"relation_type": '
                '"tensor_stat_bound", "entities": ["x"], "dimensions": [], '
                '"rationale": "test"}]}\n```')

    hyps = propose_hypotheses("def router(): pass", llm_client=my_llm)
    assert captured["src_seen"]
    assert len(hyps) == 1
    assert hyps[0].relation_type == RelationType.TENSOR_STAT_BOUND
    assert hyps[0].rationale == "test"


# ---- B4: Layer 4 filter --------------------------------------------------


def test_filter_keeps_genuine_invariants():
    """Predicates over well-known semantic fields (all_equal, has_nan,
    has_initial_lr, expert_bias_dtype, etc.) must be kept."""
    preds = [
        Predicate(id="p1", template=Template.STRUCTURAL_PRESENCE,
                   scope=Scope(hookpoint="build.snapshot",
                               payload_path="$.cross_rank_cksums[*]"),
                   bound=Bound(kind=BoundKind.EQUALITY, field="all_equal",
                               value=True)),
        Predicate(id="p2", template=Template.PAYLOAD_FIELD_COMPARE,
                   scope=Scope(hookpoint="utils.clip_grad.post"),
                   bound=Bound(kind=BoundKind.BOUND, field="post_norm",
                               op="<=", value="max_norm",
                               value_is_field=True)),
    ]
    decisions = filter_predicates(preds)
    assert all(d.keep for d in decisions), (
        f"expected all kept, got {[(d.predicate_id, d.keep) for d in decisions]}")


def test_filter_rejects_workload_specific_threshold():
    """A predicate against an absolute numeric threshold (not value_is_field,
    not over a known-semantic field) is a workload artefact."""
    p = Predicate(id="bad/lr_eq_1e4",
                   template=Template.PAYLOAD_FIELD_COMPARE,
                   scope=Scope(hookpoint="optim.step.pre"),
                   bound=Bound(kind=BoundKind.BOUND, field="param_groups",
                               op="==", value=0.0001))
    [d] = filter_predicates([p])
    assert not d.keep, f"expected reject, got {d}"


def test_filter_with_pluggable_llm():
    def my_llm(system, user, *, max_tokens=1024):
        return ('```json\n{"keep": false, "reason": "custom"}\n```')

    p = Predicate(id="p", template=Template.STRUCTURAL_PRESENCE,
                   scope=Scope(hookpoint="x.y"),
                   bound=Bound(kind=BoundKind.EQUALITY, field="all_equal",
                               value=True))
    [d] = filter_predicates([p], llm_client=my_llm)
    assert not d.keep
    assert d.reason == "custom"


# ---- End-to-end 4-layer pipeline ----------------------------------------


def test_four_layer_pipeline_end_to_end():
    """Full §3.2 closure: code source → L1 hypotheses → L2 enumerate →
    L3 validate on healthy → L4 filter spurious. Final set must include
    the clip_grad invariant."""
    src = "def clip_grad_norm_(parameters, max_norm): ..."
    hyps = propose_hypotheses(src, framework="generic")
    assert hyps

    # Build a healthy trace store on disk so it survives trainaudit.disable()
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "healthy.duckdb")
    store = trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path=db_path)
    try:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16),
                               nn.GELU(), nn.Linear(16, 4))
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        trainaudit.snapshot_build(model, opt)
        for step in range(3):
            opt.zero_grad()
            x = torch.randn(2, 8)
            model(x).pow(2).sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        store.flush()
    finally:
        trainaudit.disable()

    from trainaudit.store import TraceStore
    healthy_store = TraceStore(db_path)
    try:
        schema = schema_introspect(healthy_store)

        # L2: enumerate from each hypothesis
        candidates: list = []
        for h in hyps:
            candidates.extend(enumerate_predicates(h, schema))
        assert candidates, "L2 produced no candidates"

        # L3: keep only those that hold on healthy
        accepted: list = []
        for p in candidates:
            r = validate_against_healthy(p, [healthy_store])
            if r.accepted:
                accepted.append(p)
        assert accepted, "no L3-accepted predicates"

        # L4: filter spurious
        decisions = filter_predicates(accepted)
        kept = [p for p, d in zip(accepted, decisions) if d.keep]
        assert kept, "L4 filtered everything"

        # Final set must contain the clip_grad invariant signature
        clip_kept = [p for p in kept
                       if p.scope.hookpoint == "utils.clip_grad.post"]
        assert clip_kept, (
            "expected clip_grad invariant to survive 4-layer pipeline; "
            f"final set: {[(p.scope.hookpoint, p.bound.field) for p in kept]}")
        print(f"[4-layer E2E] kept {len(kept)}/{len(candidates)} predicates")
        print(f"[4-layer E2E] clip kept: "
              f"{[(p.id, p.bound.field, p.bound.op, p.bound.value) for p in clip_kept]}")
    finally:
        healthy_store.close()
