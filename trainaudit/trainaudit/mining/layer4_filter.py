"""Layer 4: spurious predicate filter (LLM).

After Layer 3 accepts a predicate as healthy-validated, Layer 4 asks an
LLM whether the predicate is *meaningfully* invariant or just an
artefact of the workload (e.g. "step 0 has lr=1e-4" — true on every
healthy trace but config-specific, useless as a bug detector).

Same pluggable LLMClient interface. In CI we use a deterministic stub
that filters by simple heuristics (predicates referencing absolute
numeric thresholds learned from a single workload are likely spurious;
predicates over fields like `all_equal`, `has_nan`, `has_initial_lr`
are likely genuine).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from ..diagnosis.rca_agent import LLMClient
from ..dsl import Predicate


_SYSTEM_PROMPT = (
    "You are an LLM training auditor reviewing an automatically-mined "
    "invariant. Decide whether it is a genuine invariant of distributed "
    "training (catches a real bug class) or a spurious workload artefact "
    "(e.g. \"step 0 lr=1e-4\" — incidentally true on the trace but "
    "config-specific). Reply with strict JSON:\n"
    "  {\"keep\": true|false, \"reason\": \"<one sentence>\"}\n")


def _user_prompt(predicate: Predicate) -> str:
    pred_blob = {
        "id": predicate.id,
        "template": str(predicate.template.value if hasattr(predicate.template, "value")
                        else predicate.template),
        "description": predicate.description,
        "scope": {
            "hookpoint": predicate.scope.hookpoint,
            "payload_path": predicate.scope.payload_path,
        },
        "bound": {
            "kind": str(predicate.bound.kind.value if hasattr(predicate.bound.kind, "value")
                        else predicate.bound.kind),
            "field": predicate.bound.field,
            "op": predicate.bound.op,
            "value": predicate.bound.value,
            "value_is_field": predicate.bound.value_is_field,
        },
    }
    return ("Predicate to evaluate:\n```json\n"
            + json.dumps(pred_blob, indent=2, default=str)
            + "\n```\nIs this a genuine invariant?")


@dataclass
class FilterDecision:
    predicate_id: str
    keep: bool
    reason: str


def filter_predicates(predicates: List[Predicate], *,
                       llm_client: Optional[LLMClient] = None
                       ) -> List[FilterDecision]:
    """Run each predicate through the LLM filter; return the decision per
    predicate. Caller can then drop the rejected ones."""
    if llm_client is None:
        llm_client = _DefaultStubFilterLLM()
    decisions: List[FilterDecision] = []
    for p in predicates:
        try:
            response = llm_client(_SYSTEM_PROMPT, _user_prompt(p))
            keep, reason = _parse(response)
        except Exception as e:  # noqa: BLE001
            keep, reason = True, f"LLM failure → conservative keep: {e}"
        decisions.append(FilterDecision(predicate_id=p.id, keep=keep,
                                          reason=reason))
    return decisions


def _parse(text: str):
    """Tolerantly extract {"keep": ..., "reason": ...} from LLM output."""
    fence = text.rfind("```json")
    blobs = []
    if fence != -1:
        end = text.find("```", fence + 7)
        if end != -1:
            try:
                blobs.append(json.loads(text[fence + 7:end].strip()))
            except Exception:
                pass
    last_open = text.rfind("{")
    if last_open != -1:
        try:
            blobs.append(json.loads(text[last_open:]))
        except Exception:
            pass
    for blob in blobs:
        if isinstance(blob, dict) and "keep" in blob:
            return bool(blob["keep"]), str(blob.get("reason", ""))
    return True, "could not parse LLM response → conservative keep"


# ---- stub: heuristic filter without LLM ---------------------------------


_GENUINE_FIELDS = {
    "all_equal", "has_nan", "has_inf", "has_initial_lr",
    "preserve_rng_state", "expert_bias_dtype",
    "post_norm",  # B11 type
    "rms",
    "n_parameters", "n_modules",
    "state_step_max", "state_step_min",
}


class _DefaultStubFilterLLM:
    """Heuristic stub: keep predicates that reference well-known semantic
    fields (the kind that catch real bugs); reject predicates that
    compare against absolute numeric thresholds learned from one trace
    (those are workload artefacts unless the threshold is canonical)."""

    def __call__(self, system: str, user: str, *,
                 max_tokens: int = 1024) -> str:
        try:
            blob = json.loads(user[user.find("{"):user.rfind("}") + 1])
        except Exception:
            return ('```json\n{"keep": true, "reason": "could not parse"}\n```')
        bound = blob.get("bound", {})
        field = bound.get("field")
        fields = field if isinstance(field, list) else [field]
        if any(f in _GENUINE_FIELDS for f in fields):
            return ('```json\n{"keep": true, '
                     '"reason": "field is a known semantic invariant"}\n```')
        # Comparing against a literal absolute number (no value_is_field) →
        # likely workload-specific
        if not bound.get("value_is_field") and isinstance(bound.get("value"),
                                                            (int, float)):
            return ('```json\n{"keep": false, "reason": '
                     '"absolute numeric threshold is likely '
                     'config-specific"}\n```')
        return ('```json\n{"keep": true, "reason": '
                 '"default conservative keep"}\n```')
