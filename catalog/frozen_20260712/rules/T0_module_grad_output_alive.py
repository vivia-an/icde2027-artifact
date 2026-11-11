"""T0: every parameter-bearing module's grad_output during backward must be
non-zero in l2 (else the loss derivative didn't reach it — silent dead path).

Catches:
  - graph cut by accidental .detach()
  - module accidentally wrapped in torch.no_grad()
  - frozen-by-bug parameters (intended trainable but no_grad mode)
  - upstream block silently dropped its gradient

Distinct from T0-grad-norm-finite (which checks aggregate total_grad_l2 at
optim.step.pre) by being PER-MODULE — catches one block dying while other
blocks compensate, leaving total_grad_l2 normal.

Distinct from T0-no-nan-inf (which checks NaN/Inf) by checking literal zero.
"""
import json
import math

from ..tiers import Tier
from .base import RuleResult, rule


# Modules that own trainable parameters (so grad_output should propagate)
_PARAM_MODULES = {
    "Linear", "Embedding", "RMSNorm", "LayerNorm",
    "MoEMLP", "Conv2d", "Conv1d",
    "MoELinearRouter",
}


@rule(
    rule_id="T0-module-grad-output-alive",
    min_tier=Tier.T1_FW_METADATA,
    families=["F-NEW"],
    evidence_bugs=["mining-pipeline"],
    description="Each parameter-bearing module's grad_output[0].l2_norm "
                "during module.bwd must be > 0 (else dead path to module).",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'module.bwd'
    """).fetchall()
    violations = []
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        cls = p.get("module_class")
        if cls not in _PARAM_MODULES:
            continue
        # grad_output is a list of tensor-summary dicts; check first non-null
        grad_outputs = p.get("grad_output") or []
        if not grad_outputs:
            continue
        for go in grad_outputs:
            if go is None:
                continue
            l2 = go.get("l2_norm")
            if l2 is None or (isinstance(l2, float) and math.isnan(l2)):
                continue
            try:
                l2_f = float(l2)
            except (TypeError, ValueError):
                continue
            if l2_f == 0.0:
                violations.append({
                    "event_id": event_id,
                    "module_class": cls,
                    "module_name": p.get("module_name"),
                    "grad_output_l2": l2_f,
                    "issue": "grad_output l2_norm == 0 — backward graph cut or "
                             "module isolated from loss",
                })
                break  # one violation per event suffices

    if not violations:
        return RuleResult(
            rule_id="T0-module-grad-output-alive",
            violated=False,
            message="All parameter-bearing modules have non-zero grad_output.",
        )
    return RuleResult(
        rule_id="T0-module-grad-output-alive",
        violated=True,
        message=f"{len(violations)} module.bwd events have zero grad_output l2 "
                f"(backward graph cut for these modules)",
        evidence={
            "violation_event_ids": [v["event_id"] for v in violations][:50],
            "sample": violations[:5],
        },
    )
