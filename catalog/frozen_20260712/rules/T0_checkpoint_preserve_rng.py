"""T0: when activation checkpointing is in play AND any Dropout module exists,
torch.utils.checkpoint must be called with preserve_rng_state=True.

Catches O-005 (OLMo passes preserve_rng_state=False with dropout > 0, so
recomputed forward draws different dropout masks → silently wrong gradient).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T0-checkpoint-preserve-rng",
    min_tier=Tier.T0_PYTORCH,
    families=["F5"],
    evidence_bugs=["O-005"],
    description="If a Dropout module exists in the model and "
                "torch.utils.checkpoint is invoked, preserve_rng_state must "
                "be True (else False would silently desync rng on recompute).",
)
def check(conn) -> RuleResult:
    cp_rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'checkpoint.call'
    """).fetchall()
    if not cp_rows:
        return RuleResult(rule_id="T0-checkpoint-preserve-rng",
                          violated=False,
                          message="no checkpoint.call events (rule passive)")

    # See if any Dropout module exists in the build snapshot OR module hooks
    has_dropout = False
    for (p,) in conn.execute("""
        SELECT payload FROM events
         WHERE hookpoint IN ('build.snapshot', 'module.fwd.post')
    """).fetchall():
        try:
            pl = json.loads(p)
        except Exception:
            continue
        cls = pl.get("module_class", "") or ""
        if "Dropout" in cls:
            has_dropout = True
            break
        m = pl.get("model")
        if isinstance(m, dict):
            # walk parameter names; OLMo stores dropout layers as attribute on blocks
            for prm in m.get("parameters", []):
                if "dropout" in (prm.get("name") or "").lower():
                    has_dropout = True
                    break
    if not has_dropout:
        return RuleResult(rule_id="T0-checkpoint-preserve-rng",
                          violated=False,
                          message=f"{len(cp_rows)} checkpoint calls but no "
                                  "Dropout in model (rule N/A)")

    bad = []
    for event_id, payload_str in cp_rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        # preserve_rng_state default in PyTorch is True; explicit False is buggy
        prs = p.get("preserve_rng_state")
        if prs is False:
            bad.append({"event_id": event_id,
                        "function": p.get("function"),
                        "preserve_rng_state": prs})
    if bad:
        return RuleResult(
            rule_id="T0-checkpoint-preserve-rng",
            violated=True,
            message=f"{len(bad)} checkpoint calls used preserve_rng_state=False "
                    "while dropout is present",
            evidence={"sample": bad[:3]},
        )
    return RuleResult(
        rule_id="T0-checkpoint-preserve-rng",
        violated=False,
        message=f"all {len(cp_rows)} checkpoint calls preserve rng",
    )
