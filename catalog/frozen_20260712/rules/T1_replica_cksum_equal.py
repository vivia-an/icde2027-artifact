"""T1: replica params must be cksum-equal across their replica group.

Catches B1 / M-005: SwitchMLP router weight differs across TP ranks because
init went through default CUDA RNG (per-rank diverged) instead of
get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()).

This rule reads the cross_rank_cksums field added at build_snapshot time.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T1-replica-cksum-equal",
    min_tier=Tier.T1_FW_METADATA,
    families=["F1"],
    evidence_bugs=["B1", "M-005"],
    description="For every parameter labelled as replica (not shard / not "
                "expert_local), its checksum must be equal across all ranks "
                "in the replica group at build time.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'build.snapshot'
    """).fetchall()
    if not rows:
        return RuleResult(rule_id="T1-replica-cksum-equal", violated=False,
                          message="no build snapshot (rule passive)")

    bad = []
    n_checked = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        for entry in p.get("cross_rank_cksums") or []:
            if entry.get("group_size", 1) <= 1:
                continue  # nothing to compare
            n_checked += 1
            if entry.get("all_equal") is False:
                bad.append({
                    "event_id": event_id,
                    "name": entry["name"],
                    "local": entry["local_cksum"],
                    "gathered": entry["gathered_cksums"],
                    "group_size": entry["group_size"],
                })
    if bad:
        return RuleResult(
            rule_id="T1-replica-cksum-equal",
            violated=True,
            message=f"{len(bad)} replica params diverged across replica group",
            evidence={
                "sample": bad[:3], "n_checked": n_checked,
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T1-replica-cksum-equal",
        violated=False,
        message=f"all {n_checked} replica params consistent across replica group",
    )
