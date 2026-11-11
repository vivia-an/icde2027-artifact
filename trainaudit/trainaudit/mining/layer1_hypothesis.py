"""Mining walks the 18 frozen rule files, one direction at a time.

Each file is the direction. Grounding may only emit that direction's
relation; a hypothesis for another direction is dropped. Expansion of a
kept hypothesis stays in layer 2.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

from ..diagnosis.rca_agent import LLMClient
from .hypothesis_schema import Hypothesis, RelationType

# The 18 directions registered in dsl/registry/MAPPING.md.
# Files live in catalog/frozen_20260712/rules.
CATALOG18: Sequence[Tuple[str, str, RelationType]] = (
    ("T0_no_nan_inf", "T0_no_nan_inf.py", RelationType.TENSOR_STAT_BOUND),
    ("T0_clip_grad_bounded", "T0_clip_grad_bounded.py", RelationType.PAYLOAD_FIELD_COMPARE),
    ("T0_softmax_degenerate", "T0_softmax_degenerate.py", RelationType.TENSOR_STAT_BOUND),
    ("T0_token_id_range", "T0_token_id_range.py", RelationType.TENSOR_STAT_BOUND),
    ("T0_initial_lr_present", "T0_initial_lr_present.py", RelationType.CONDITIONAL_CHECK),
    ("T0_norm_output_rms", "T0_norm_output_rms.py", RelationType.TENSOR_STAT_BOUND),
    ("T0_dtype_propagation", "T0_dtype_propagation.py", RelationType.PAYLOAD_FIELD_COMPARE),
    ("T0_optim_lr_positive", "T0_optim_lr_positive.py", RelationType.PAYLOAD_FIELD_COMPARE),
    ("T0_optim_step_counter", "T0_optim_step_counter.py", RelationType.CROSS_STEP_MONOTONIC),
    ("T0_layer_count", "T0_layer_count.py", RelationType.STRUCTURAL_PRESENCE),
    ("T0_checkpoint_preserve_rng", "T0_checkpoint_preserve_rng.py", RelationType.CONDITIONAL_CHECK),
    ("T1_replica_cksum_equal", "T1_replica_cksum_equal.py", RelationType.CROSS_RANK_EQUAL),
    ("T1_residual_stream", "T1_residual_stream.py", RelationType.PAYLOAD_FIELD_COMPARE),
    ("T1_router_attribute", "T1_router_attribute.py", RelationType.CONDITIONAL_CHECK),
    ("T1_expert_bias_dtype", "T1_expert_bias_dtype.py", RelationType.PAYLOAD_FIELD_COMPARE),
    ("T1_layer_count_strict", "T1_layer_count_strict.py", RelationType.STRUCTURAL_PRESENCE),
    ("T1_jitter_dtype", "T1_jitter_dtype.py", RelationType.PAYLOAD_FIELD_COMPARE),
    ("T1_sqrt_decay_direction", "T1_sqrt_decay_direction.py", RelationType.CROSS_STEP_MONOTONIC),
)


def frozen_rules_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(
        here, "..", "..", "..", "catalog", "frozen_20260712", "rules"))


_SYSTEM_PROMPT = (
    "You are an LLM training framework auditor. Given a snippet of "
    "framework source code, propose 2–4 invariant Hypotheses that, if "
    "violated at runtime, would indicate a silent error. Each Hypothesis "
    "must specify:\n"
    "  - relation_type ∈ {cross_rank_equal, tensor_stat_bound, "
    "payload_field_compare, cross_step_monotonic, structural_presence, "
    "conditional_check}\n"
    "  - entities: list of trace entities the invariant references "
    "(e.g. ['param', 'grad_norm'])\n"
    "  - dimensions: ['rank', 'step'] etc.\n"
    "  - rationale: one sentence on why this invariant matters\n\n"
    "Return JSON: {\"hypotheses\": [...]}\n")


def _build_user_prompt(source: str, framework: str = "",
                       direction: Optional[str] = None,
                       relation: Optional[str] = None,
                       guide: str = "") -> str:
    fw = f"Framework: {framework}\n" if framework else ""
    if direction:
        return (f"Direction: {direction} {relation}\n"
                f"{fw}Rule:\n{guide[:1500]}\n"
                f"Source code:\n```\n{source[:8000]}\n```\n\n"
                "Propose hypotheses only for this direction.\n")
    return (f"{fw}Source code:\n```\n{source[:8000]}\n```\n\n"
            f"Propose Hypotheses as JSON.")


def propose_hypotheses(source: str, *,
                        framework: str = "",
                        llm_client: Optional[LLMClient] = None,
                        direction: Optional[str] = None,
                        relation: Optional[str] = None,
                        guide: str = ""
                        ) -> List[Hypothesis]:
    """Ground `source` in one catalog direction, or propose freely when unset."""
    if llm_client is None:
        llm_client = _DefaultStubHypothesisLLM()
    system = _SYSTEM_PROMPT
    if direction:
        system += f"\nThis round is only {direction} ({relation}). Drop any other direction.\n"
    response = llm_client(
        system,
        _build_user_prompt(source, framework, direction, relation, guide))
    found = _parse_hypothesis_response(response)
    if direction is None:
        return found
    rel = RelationType(relation)
    kept = []
    for h in found:
        if h.relation_type != rel:
            continue
        h.scope_hint = dict(h.scope_hint or {})
        h.scope_hint["direction"] = direction
        kept.append(h)
    print(f"mining direction={direction} kept={len(kept)} dropped={len(found) - len(kept)}")
    return kept


def mine_catalog(source: str, *,
                 rules_dir: Optional[str] = None,
                 framework: str = "",
                 llm_client: Optional[LLMClient] = None
                 ) -> List[Tuple[str, List[Hypothesis]]]:
    """Walk the 18 frozen rules. Each file directs mining of that relation only."""
    rules_dir = rules_dir or frozen_rules_dir()
    out: List[Tuple[str, List[Hypothesis]]] = []
    for rule_id, filename, relation in CATALOG18:
        path = os.path.join(rules_dir, filename)
        print(f"mining direction={rule_id} file={filename}")
        guide = ""
        if os.path.isfile(path):
            with open(path) as f:
                guide = f.read(4000)
        else:
            print(f"mining direction={rule_id} missing={filename}")
        hyps = propose_hypotheses(
            source, framework=framework, llm_client=llm_client,
            direction=rule_id, relation=relation.value, guide=guide)
        out.append((rule_id, hyps))
    return out


def _parse_hypothesis_response(text: str) -> List[Hypothesis]:
    """Tolerantly extract a JSON {"hypotheses": [...]} block from LLM text."""
    # Try fenced ```json ... ``` first
    candidates = []
    fence = text.rfind("```json")
    if fence != -1:
        end = text.find("```", fence + 7)
        if end != -1:
            try:
                candidates.append(json.loads(text[fence + 7:end].strip()))
            except Exception:
                pass
    # Try last { ... } block
    last_open = text.rfind("{")
    if last_open != -1:
        try:
            candidates.append(json.loads(text[last_open:]))
        except Exception:
            pass
    for blob in candidates:
        if not isinstance(blob, dict):
            continue
        items = blob.get("hypotheses") or []
        out: List[Hypothesis] = []
        for h in items:
            try:
                out.append(Hypothesis(
                    relation_type=RelationType(h["relation_type"]),
                    entities=list(h.get("entities", [])),
                    dimensions=list(h.get("dimensions", [])),
                    scope_hint=h.get("scope_hint", {}),
                    rationale=h.get("rationale", ""),
                ))
            except Exception:
                continue
        if out:
            return out
    return []


# ---- stub that returns plausible hypotheses without LLM access ----------


class _DefaultStubHypothesisLLM:
    """Deterministic stub: pattern-match on source for known fingerprints
    (clip_grad / RMSNorm / softmax / scheduler) and emit a corresponding
    Hypothesis. Sufficient for harness self-test without real LLM."""

    def __call__(self, system: str, user: str, *,
                 max_tokens: int = 1024) -> str:
        for line in user.splitlines():
            if line.startswith("Direction:"):
                _, rule_id, relation = line.split()
                one = {
                    "relation_type": relation,
                    "entities": [rule_id],
                    "dimensions": [],
                    "scope_hint": {"direction": rule_id},
                    "rationale": f"ground {rule_id}",
                }
                return "```json\n" + json.dumps({"hypotheses": [one]}) + "\n```"
        src_lower = user.lower()
        hypotheses = []
        if "clip_grad" in src_lower or "max_norm" in src_lower:
            hypotheses.append({
                "relation_type": "payload_field_compare",
                "entities": ["norm", "threshold"],
                "dimensions": [],
                "rationale": ("clip_grad must enforce post_norm <= max_norm; "
                              "violation means clipping was bypassed"),
            })
        if "rmsnorm" in src_lower or "layernorm" in src_lower:
            hypotheses.append({
                "relation_type": "tensor_stat_bound",
                "entities": ["norm_output"],
                "dimensions": [],
                "rationale": ("normalizer output rms must stay near 1.0; "
                              "drift indicates eps placement / L2-vs-RMS bug"),
            })
        if "router" in src_lower or "softmax" in src_lower or "topk" in src_lower:
            hypotheses.append({
                "relation_type": "tensor_stat_bound",
                "entities": ["router_output"],
                "dimensions": [],
                "rationale": ("router output must NOT be a one-hot "
                              "distribution; topk-then-softmax over a size-1 "
                              "dim is a known degenerate case"),
            })
        if "scheduler" in src_lower or "last_epoch" in src_lower:
            hypotheses.append({
                "relation_type": "conditional_check",
                "entities": ["param_group", "scheduler"],
                "dimensions": [],
                "rationale": ("scheduler resume requires initial_lr present "
                              "on every param_group"),
            })
        if not hypotheses:
            hypotheses.append({
                "relation_type": "structural_presence",
                "entities": ["module"],
                "dimensions": [],
                "rationale": "fallback: build snapshot must report parameters",
            })
        return "```json\n" + json.dumps({"hypotheses": hypotheses}) + "\n```"
