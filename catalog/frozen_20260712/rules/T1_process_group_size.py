"""T1: process-group sizes (TP/DP/EP/PP) actual must equal declared.

Catches B8 / similar: DeepSpeed initializes the EP communication group
with `num_experts` ranks instead of `ep_size`, so all-to-all routes
tokens to wrong ranks but does not crash.

Detection: build.snapshot.framework_invariants.<fw> carries both
`<role>_size_declared` (from config) and `<role>_size_actual` (from
torch.distributed group inspection at adapter init time). Rule fires
on any role where declared != actual and group_size > 1.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


_ROLES = ("ep", "tp", "dp", "pp", "cp", "sp")


@rule(
    rule_id="T1-process-group-size-correct",
    min_tier=Tier.T1_FW_METADATA,
    families=["F1"],
    evidence_bugs=["B8"],
    description="For every parallelism role (ep/tp/dp/pp/cp/sp) declared "
                "by the framework, the actual process group size must equal "
                "the declared size.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'build.snapshot'
    """).fetchall()
    if not rows:
        return RuleResult(rule_id="T1-process-group-size-correct",
                          violated=False,
                          message="no build snapshot (rule passive)")
    bad = []
    n_checked = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        fw_inv = p.get("framework_invariants") or {}
        for fw_name, inv in fw_inv.items():
            if not isinstance(inv, dict):
                continue
            for role in _ROLES:
                declared_key = f"{role}_size_declared"
                actual_key = f"{role}_size_actual"
                declared = inv.get(declared_key)
                actual = inv.get(actual_key)
                if declared is None or actual is None:
                    continue
                n_checked += 1
                if int(declared) != int(actual) and int(declared) > 1:
                    bad.append({
                        "event_id": event_id,
                        "framework": fw_name,
                        "role": role,
                        "declared": declared,
                        "actual": actual,
                    })
    if bad:
        return RuleResult(
            rule_id="T1-process-group-size-correct",
            violated=True,
            message=f"{len(bad)} parallelism group(s) have wrong size "
                    "(declared != actual)",
            evidence={
                "sample": bad[:3], "n_checked": n_checked,
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T1-process-group-size-correct",
        violated=False,
        message=f"all {n_checked} parallelism groups have correct size",
    )
