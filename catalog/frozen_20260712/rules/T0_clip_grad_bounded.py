"""T0: post-clip grad norm must respect max_norm.

Catches B11 (DeepSpeed clip_grad_norm_ used max instead of min).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T0-clip-grad-bounded",
    min_tier=Tier.T0_PYTORCH,
    families=["F2"],
    evidence_bugs=["B11"],
    description="After clip_grad_norm_, total grad norm must be <= max_norm "
                "(within 1% tolerance).",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'utils.clip_grad.post'
    """).fetchall()
    violations = []
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        max_norm = p.get("max_norm")
        post = p.get("post_norm")
        if max_norm is None or post is None:
            continue
        if post > max_norm * 1.01:
            violations.append({"event_id": event_id, "max_norm": max_norm,
                               "post_norm": post,
                               "ratio": post / max_norm})
    if violations:
        return RuleResult(
            rule_id="T0-clip-grad-bounded",
            violated=True,
            message=f"{len(violations)} clip_grad calls left grad norm > max_norm",
            evidence={
                "sample": violations[:3],
                "violation_event_ids": [v["event_id"] for v in violations],
            },
        )
    return RuleResult(
        rule_id="T0-clip-grad-bounded",
        violated=False,
        message=f"all {len(rows)} clip_grad calls bounded by max_norm",
    )
