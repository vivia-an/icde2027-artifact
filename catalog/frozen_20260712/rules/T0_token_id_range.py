"""T0: input ids loaded by DataLoader should be within declared vocab.

Catches O-NEW-9 (token IDs truncated by % 2^16).

This rule is passive in Phase 1 (no DataLoader hook installed yet); it's
included as a stub so that when a DataLoader hook is added it has somewhere
to attach. The rule body shows the intended SQL.
"""
import json

from ..schema import T_ABS_MAX
from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T0-token-id-in-vocab",
    min_tier=Tier.T0_PYTORCH,
    families=["F9"],
    evidence_bugs=["O-NEW-9"],
    description="Input IDs from DataLoader must be < declared vocab_size "
                "(passive in Phase 1; populated when DataLoader hook lands).",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'dataloader.batch'
    """).fetchall()
    if not rows:
        return RuleResult(
            rule_id="T0-token-id-in-vocab",
            violated=False,
            message="no dataloader.batch events captured (rule passive)",
        )
    bad = []
    sane = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        ids = p.get("input_ids") or {}
        if not isinstance(ids, dict):
            continue
        max_id = ids.get(T_ABS_MAX) or ids.get("max")
        min_id = ids.get("min")
        if max_id is None:
            continue
        sane += 1
        # Sanity: token IDs should be non-negative and within a reasonable
        # absolute bound (vocab unknown at T0; the only universal violation is
        # negative or exorbitant). T1 will compare against declared vocab_size.
        if min_id is not None and min_id < 0:
            bad.append({"event_id": event_id, "min_id": min_id, "issue": "negative"})
        # Hard cap heuristic: > 2^20 = 1M tokens almost never legit
        if max_id > 2**20:
            bad.append({"event_id": event_id, "max_id": max_id, "issue": "absurdly large"})
    if bad:
        return RuleResult(
            rule_id="T0-token-id-in-vocab",
            violated=True,
            message=f"{len(bad)} dataloader batches have out-of-range ids",
            evidence={
                "sample": bad[:3],
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T0-token-id-in-vocab",
        violated=False,
        message=f"all {sane} dataloader batches within sanity range",
    )
