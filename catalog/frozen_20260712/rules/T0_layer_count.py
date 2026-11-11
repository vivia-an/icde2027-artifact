"""T0: model n_modules must be > 0 and consistent with declared depth.

This is a sanity-level T0 rule — catches gross layer-count bugs (M-020-style).
A T1/T2 version will know the configured depth from the framework.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T0-build-has-modules",
    min_tier=Tier.T0_PYTORCH,
    families=["F4"],
    evidence_bugs=["M-020"],
    description="Build snapshot must report at least one nn.Module and "
                "at least one parameter.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'build.snapshot'
    """).fetchall()
    if not rows:
        return RuleResult(rule_id="T0-build-has-modules", violated=False,
                          message="no build snapshot present (skip)")
    bad = []
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        m = p.get("model") or {}
        n_p = m.get("n_parameters", 0) or 0
        n_m = m.get("n_modules", 0) or 0
        if n_p == 0 or n_m == 0:
            bad.append({"event_id": event_id, "n_parameters": n_p, "n_modules": n_m})
    if bad:
        return RuleResult(
            rule_id="T0-build-has-modules",
            violated=True,
            message="build snapshot reports zero parameters or zero modules",
            evidence={
                "sample": bad[:3],
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T0-build-has-modules",
        violated=False,
        message=f"build snapshot has parameters and modules ({len(rows)} snapshots)",
    )
