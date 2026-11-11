"""T1: TopKRouter.apply_input_jitter must preserve input dtype.

Catches M-024: torch.distributions.Uniform creates fp32 tensors regardless of
input dtype, so multiplying bf16 input by fp32 jitter silently promotes to fp32.

Detection: Megatron adapter active probe wraps apply_input_jitter and emits
jitter.probe events with (input_dtype, output_dtype). Rule fires on mismatch.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T1-jitter-preserves-dtype",
    min_tier=Tier.T1_FW_METADATA,
    families=["F3"],
    evidence_bugs=["M-024"],
    description="apply_input_jitter must produce output with the same dtype as input.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'jitter.probe'
    """).fetchall()
    if not rows:
        return RuleResult(rule_id="T1-jitter-preserves-dtype", violated=False,
                          message="no jitter.probe events (rule passive)")
    bad = []
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        if p.get("dtypes_match") is False:
            bad.append({
                "event_id": event_id,
                "module_class": p.get("module_class"),
                "input_dtype": p.get("input_dtype"),
                "output_dtype": p.get("output_dtype"),
            })
    if bad:
        return RuleResult(
            rule_id="T1-jitter-preserves-dtype",
            violated=True,
            message=f"{len(bad)} jitter calls promoted dtype",
            evidence={
                "sample": bad[:3],
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T1-jitter-preserves-dtype",
        violated=False,
        message=f"all {len(rows)} jitter calls preserved dtype",
    )
