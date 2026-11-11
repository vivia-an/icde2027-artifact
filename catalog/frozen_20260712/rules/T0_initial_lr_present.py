"""T0: when an LRScheduler resumes (last_epoch != -1), every param_group
must carry 'initial_lr'.

Catches B12 / O-016 (OLMo-core AdamWConfig.build forgot to set initial_lr).
PyTorch's stock optimizers don't set initial_lr by default — it's the
scheduler's resume-time check that requires it. So we hook scheduler.__init__
and only flag missing initial_lr when last_epoch != -1.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T0-initial-lr-present",
    min_tier=Tier.T0_PYTORCH,
    families=["F4"],
    evidence_bugs=["B12", "O-016"],
    description="Whenever an LRScheduler is constructed with last_epoch != -1, "
                "every optimizer param_group must contain 'initial_lr'.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'scheduler.init'
    """).fetchall()
    if not rows:
        return RuleResult(rule_id="T0-initial-lr-present", violated=False,
                          message="no scheduler.init events (rule passive)")
    bad = []
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        last_epoch = p.get("last_epoch", -1)
        if last_epoch == -1:
            continue  # fresh start: PyTorch will set initial_lr itself
        for g in p.get("param_groups", []):
            if not g.get("has_initial_lr"):
                bad.append({
                    "event_id": event_id,
                    "scheduler_class": p.get("scheduler_class"),
                    "last_epoch": last_epoch,
                    "group_index": g.get("index"),
                })
    if bad:
        return RuleResult(
            rule_id="T0-initial-lr-present",
            violated=True,
            message=f"{len(bad)} scheduler-resume(s) found param_groups without initial_lr",
            evidence={
                "sample": bad[:3],
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T0-initial-lr-present",
        violated=False,
        message=f"all {len(rows)} scheduler.init events compatible",
    )
