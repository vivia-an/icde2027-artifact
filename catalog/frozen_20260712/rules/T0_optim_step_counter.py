"""T0: optimizer state['step'] must monotonically increment across optim.step calls.

Catches OC-NEW-2: SkipStepAdamW had `step.add_(step_factor)` commented out, so
state['step'] stays at initial value. Adam bias correction never updates.

Detection: optim_hook captures min/max of state[p]['step'] in each
optim.step.post event. Rule checks max(state_step_max) increases across calls.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T0-optim-step-counter-monotonic",
    min_tier=Tier.T0_PYTORCH,
    families=["F5"],
    evidence_bugs=["OC-NEW-2"],
    description="Optimizer state['step'] must increment with each optim.step() call.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events
         WHERE hookpoint = 'optim.step.post'
         ORDER BY event_id
    """).fetchall()
    if len(rows) < 2:
        return RuleResult(rule_id="T0-optim-step-counter-monotonic",
                          violated=False,
                          message=f"only {len(rows)} optim.step.post events (rule passive)")
    series = []
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        if "state_step_max" in p:
            series.append((event_id, p["state_step_max"], p.get("optimizer_class")))
    if len(series) < 2:
        return RuleResult(rule_id="T0-optim-step-counter-monotonic",
                          violated=False,
                          message="no optimizer state['step'] captured (rule passive)")
    # Check strict monotonic increase
    bad = []
    for i in range(1, len(series)):
        prev_max = series[i - 1][1]
        cur_max = series[i][1]
        if cur_max <= prev_max:
            bad.append({
                "event_id": series[i][0],
                "optimizer_class": series[i][2],
                "prev_step_max": prev_max,
                "cur_step_max": cur_max,
            })
    if bad:
        return RuleResult(
            rule_id="T0-optim-step-counter-monotonic",
            violated=True,
            message=f"optim state['step'] failed to increment in {len(bad)}/{len(series)-1} transitions",
            evidence={
                "sample": bad[:3],
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T0-optim-step-counter-monotonic",
        violated=False,
        message=f"state['step'] monotonic across {len(series)} optim.step events",
    )
