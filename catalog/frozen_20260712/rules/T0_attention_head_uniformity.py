"""T0: per-head attention output l2 should not vary >10x across heads
within the same attention layer.

A healthy attention has heads contributing roughly equally — the
max/min l2 ratio is typically 2-5x. >10x indicates:
  - dead head (head's params not learning, output stuck near init)
  - head domination (one head saturates and others starved)
  - per-head bias bug (head bias init wrong on one head)

This rule reads `attention.fwd.post` events with per-head stats and flags
modules where max_to_min_ratio > 10.

Catches a class of attention-internal bugs invisible to layer-aggregate
output rules (T0-norm-output-rms / T1-fwd-output-uniformity).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


_HEAD_RATIO_THRESHOLD = 10.0


@rule(
    rule_id="T0-attention-head-uniformity",
    min_tier=Tier.T0_PYTORCH,
    families=["F-NEW"],
    evidence_bugs=["attention-correctness"],
    description="Per-head attention output l2 max/min ratio must be <10x "
                "(catches dead head, head domination).",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'attention.fwd.post'
    """).fetchall()
    violations = []
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        ratio = p.get("max_to_min_ratio")
        any_zero = p.get("any_zero_head", False)
        if ratio is None:
            continue
        # any_zero_head with non-zero max is extreme (one head completely dead)
        # otherwise check ratio threshold
        if any_zero or (isinstance(ratio, (int, float))
                         and ratio != float("inf")
                         and ratio > _HEAD_RATIO_THRESHOLD) \
                or ratio == float("inf"):
            violations.append({
                "event_id": event_id,
                "module_name": p.get("module_name"),
                "module_class": p.get("module_class"),
                "n_heads": p.get("n_heads"),
                "max_to_min_ratio": ratio,
                "min_head_l2": p.get("min_head_l2"),
                "max_head_l2": p.get("max_head_l2"),
                "issue": (f"per-head l2 ratio {ratio:.1f}x at "
                          f"{p.get('module_name')} (max={p.get('max_head_l2'):.2e}, "
                          f"min={p.get('min_head_l2'):.2e})"),
            })
    if not violations:
        return RuleResult(
            rule_id="T0-attention-head-uniformity",
            violated=False,
            message="All attention modules have uniform per-head output l2.",
        )
    return RuleResult(
        rule_id="T0-attention-head-uniformity",
        violated=True,
        message=f"{len(violations)} attention events with per-head l2 ratio >10x",
        evidence={
            "violation_event_ids": [v["event_id"] for v in violations][:50],
            "sample": violations[:5],
        },
    )
