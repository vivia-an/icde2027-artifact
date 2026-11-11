"""T0: every optimizer param_group must have a positive 'lr'.

Catches a class of build-time misconfigurations where lr is missing or zero.
Family F4 (build structural).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T0-optim-lr-positive",
    min_tier=Tier.T0_PYTORCH,
    families=["F4"],
    evidence_bugs=["sanity"],
    description="Every optimizer param_group must contain a positive 'lr'.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'build.snapshot'
    """).fetchall()
    violations = []
    n_groups = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        opt = p.get("optimizer") or {}
        for g in opt.get("param_groups", []):
            n_groups += 1
            lr = g.get("lr")
            if lr is None or (isinstance(lr, (int, float)) and lr <= 0):
                violations.append({"event_id": event_id,
                                   "group_index": g.get("index"),
                                   "lr": lr})
    if violations:
        return RuleResult(
            rule_id="T0-optim-lr-positive",
            violated=True,
            message=f"{len(violations)} param_groups have non-positive lr",
            evidence={
                "sample": violations[:3],
                "violation_event_ids": [v["event_id"] for v in violations],
            },
        )
    return RuleResult(
        rule_id="T0-optim-lr-positive",
        violated=False,
        message=f"all {n_groups} param_groups have positive lr",
    )
