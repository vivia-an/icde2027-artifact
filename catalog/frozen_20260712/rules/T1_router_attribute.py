"""T1: Megatron MoE Router class must declare 'calculate_per_token_loss'.

Catches M-NEW-5: the missing attribute breaks downstream loss balancing.
This rule fires only when the Megatron adapter has labelled at least one
module as is_router (so it's a no-op on non-MoE training).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T1-router-has-calculate-per-token-loss",
    min_tier=Tier.T1_FW_METADATA,
    families=["F4"],
    evidence_bugs=["M-NEW-5"],
    description="Modules labelled is_router by the framework adapter must "
                "carry attribute calculate_per_token_loss.",
)
def check(conn) -> RuleResult:
    # Precondition: rule only applies when --calculate-per-token-loss is on.
    # Otherwise the attribute being absent is the framework's default state.
    bld = conn.execute("""
        SELECT payload FROM events WHERE hookpoint = 'build.snapshot'
    """).fetchall()
    feature_enabled = None
    for (p,) in bld:
        try:
            pl = json.loads(p)
        except Exception:
            continue
        meg = (pl.get("framework_invariants") or {}).get("megatron") or {}
        if "calculate_per_token_loss" in meg:
            feature_enabled = bool(meg["calculate_per_token_loss"])
    if not feature_enabled:
        return RuleResult(rule_id="T1-router-has-calculate-per-token-loss",
                          violated=False,
                          message="--calculate-per-token-loss not enabled (rule N/A)")

    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'module.fwd.post'
    """).fetchall()
    bad = []
    n_router = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        sem = p.get("semantic") or {}
        if not sem.get("is_router"):
            continue
        n_router += 1
        if sem.get("has_calculate_per_token_loss") is False:
            bad.append({
                "event_id": event_id,
                "module_class": p.get("module_class"),
            })
    if not n_router:
        return RuleResult(rule_id="T1-router-has-calculate-per-token-loss",
                          violated=False,
                          message="no router modules observed (rule passive)")
    if bad:
        return RuleResult(
            rule_id="T1-router-has-calculate-per-token-loss",
            violated=True,
            message=f"{len(bad)} router(s) missing calculate_per_token_loss attr",
            evidence={"sample": bad[:3]},
        )
    return RuleResult(
        rule_id="T1-router-has-calculate-per-token-loss",
        violated=False,
        message=f"all {n_router} router observations have the attribute",
    )
