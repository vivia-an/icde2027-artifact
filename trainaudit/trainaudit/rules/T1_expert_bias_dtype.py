"""T1: TopKRouter.expert_bias must be fp32.

Catches M-012: Float16Module silently demotes expert_bias to bf16/fp16,
breaking routing precision. Fix re-casts to fp32 every forward.

Detection: Megatron adapter's label_module attaches `expert_bias_dtype`
to module.fwd.post events for Router classes. Rule fires if any router
observation has expert_bias_dtype != float32.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T1-expert-bias-fp32",
    min_tier=Tier.T1_FW_METADATA,
    families=["F3"],
    evidence_bugs=["M-012"],
    description="TopKRouter.expert_bias must remain fp32 across forwards.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'module.fwd.post'
    """).fetchall()
    bad = []
    n_router = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        sem = p.get("semantic") or {}
        if not sem.get("is_router"):
            continue
        eb_dtype = sem.get("expert_bias_dtype")
        if eb_dtype is None:
            continue
        n_router += 1
        if eb_dtype != "float32":
            bad.append({
                "event_id": event_id,
                "module_class": p.get("module_class"),
                "expert_bias_dtype": eb_dtype,
            })
    if not n_router:
        return RuleResult(rule_id="T1-expert-bias-fp32", violated=False,
                          message="no router with expert_bias observed (rule passive)")
    if bad:
        return RuleResult(
            rule_id="T1-expert-bias-fp32",
            violated=True,
            message=f"{len(bad)} router(s) have non-fp32 expert_bias",
            evidence={
                "sample": bad[:3],
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T1-expert-bias-fp32",
        violated=False,
        message=f"all {n_router} router observations have fp32 expert_bias",
    )
