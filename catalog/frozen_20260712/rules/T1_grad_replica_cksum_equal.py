"""T1: replica-kind params must have equal GRAD cksum across replica group
after backward (before optim.step apply).

Catches B2-class bugs: TP `LinearWithFrozenWeight` skips the input-gradient
all-reduce in backward → upstream replica params end up with different
grads across TP rank → after optim.step apply they diverge over time.

Detection: optim.step.pre payload carries `cross_rank_grad_cksums` (parallel
field to build.snapshot's `cross_rank_cksums`, but for `p.grad` rather
than `p`). Rule checks `all_equal == True` for every entry with
`group_size > 1` (size-1 groups are degenerate).

Difference from T1-replica-cksum-equal: that one fires once at build time
(catches init-time divergence, e.g. B1/M-005). This one fires every
optim step (catches backward-time divergence, e.g. B2).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T1-grad-replica-cksum-equal",
    min_tier=Tier.T1_FW_METADATA,
    families=["F2"],
    evidence_bugs=["B2"],
    description="For every replica-labelled parameter, its gradient must "
                "be bit-identical across the replica group at optim.step.pre.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'optim.step.pre'
    """).fetchall()
    if not rows:
        return RuleResult(rule_id="T1-grad-replica-cksum-equal",
                          violated=False,
                          message="no optim.step.pre events (rule passive)")
    bad = []
    n_checked = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        for entry in p.get("cross_rank_grad_cksums") or []:
            if entry.get("group_size", 1) <= 1:
                continue
            n_checked += 1
            if entry.get("all_equal") is False:
                bad.append({
                    "event_id": event_id,
                    "name": entry.get("name"),
                    "local": entry.get("local_cksum"),
                    "gathered": entry.get("gathered_cksums"),
                    "group_size": entry.get("group_size"),
                })
    if bad:
        return RuleResult(
            rule_id="T1-grad-replica-cksum-equal",
            violated=True,
            message=f"{len(bad)} replica params' gradients diverged "
                    "across replica group (B2-style backward all-reduce miss)",
            evidence={
                "sample": bad[:3], "n_checked": n_checked,
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T1-grad-replica-cksum-equal",
        violated=False,
        message=f"all {n_checked} replica grads consistent across replica group",
    )
