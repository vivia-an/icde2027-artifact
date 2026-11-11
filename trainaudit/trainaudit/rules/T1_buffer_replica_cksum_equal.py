"""T1: replicated module buffers must be cksum-equal across replica group.

Catches Megatron CAND_MEGATRON_CUDAGRAPH_BUFFER_CORRUPTION (PR #4433): the
CUDA-graph warmup runs forward passes that mutate persistent module buffers
(e.g. MoE router `expert_bias`) but pre-fix code did NOT back the buffers up
across warmup. After warmup, ranks have differently-corrupted buffer state.
At the next build_snapshot's cross-rank gather, the buffer cksums diverge.

Same shape as T1-replica-cksum-equal but reads `cross_rank_buffer_cksums`
(populated at build_snapshot when at least one parallel state has > 1 rank).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T1-buffer-replica-cksum-equal",
    min_tier=Tier.T1_FW_METADATA,
    families=["F1"],
    evidence_bugs=["CAND_MEGATRON_CUDAGRAPH_BUFFER_CORRUPTION"],
    description="For every module buffer that is replicated across a "
                "parallel-state group (TP / DP), its checksum must be equal "
                "across all ranks in the replica group at build time.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'build.snapshot'
    """).fetchall()
    if not rows:
        return RuleResult(rule_id="T1-buffer-replica-cksum-equal",
                          violated=False,
                          message="no build snapshot (rule passive)")

    bad = []
    n_checked = 0
    n_buffers_total = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        for entry in p.get("cross_rank_buffer_cksums") or []:
            n_buffers_total += 1
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
            rule_id="T1-buffer-replica-cksum-equal",
            violated=True,
            message=f"{len(bad)} replica buffer(s) diverged across replica group "
                    f"(of {n_checked} checked)",
            evidence={
                "sample": bad[:3], "n_checked": n_checked,
                "n_buffers_total": n_buffers_total,
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T1-buffer-replica-cksum-equal",
        violated=False,
        message=(f"all {n_checked} replica buffers consistent across "
                 f"replica group (skipped {n_buffers_total - n_checked} "
                 f"buffers with group_size <= 1)" if n_checked
                 else f"no buffers in a >1-rank replica group "
                      f"(rule passive; {n_buffers_total} buffers in total)"),
    )
