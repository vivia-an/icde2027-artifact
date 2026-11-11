"""T0: at every optim.step.post, total parameter delta should be > 0
(at least one param actually changed). And no param with non-zero grad
should have delta exactly 0 (frozen-by-bug: grad available but optimizer
skipped this param).

Catches:
  - frozen-by-bug params (custom hook that zeros grad after backward,
    or buggy parameter wrapping that ignores .grad)
  - completely failed optimizer step (everything unchanged → graph cut
    everywhere)
  - lr=0 cases (would fire on n_unchanged_with_grad but T0-optim-lr-positive
    catches the config side)

Distinct from T0-grad-norm-finite: that checks grads exist; this checks
they were APPLIED to params.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T0-param-update-applied",
    min_tier=Tier.T0_PYTORCH,
    families=["F-NEW"],
    evidence_bugs=["optimizer-state-drift"],
    description="optim.step.post must show non-zero param delta and no "
                "frozen-by-bug param (grad>0 but delta==0).",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'optim.step.post'
    """).fetchall()
    violations = []
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        delta = p.get("total_param_delta_l2")
        n_unchanged_with_grad = p.get("n_params_unchanged_with_grad", 0)
        n_unchanged = p.get("n_params_unchanged", 0)

        if delta is None:
            continue

        # Catastrophic: ALL params unchanged after step (delta==0 everywhere)
        if delta == 0.0:
            violations.append({
                "event_id": event_id,
                "issue": ("optim.step had zero total param delta — no params "
                          "updated (every param unchanged)"),
                "n_params_unchanged": n_unchanged,
            })
            continue

        # Frozen-by-bug: some params have non-zero grad but didn't update
        if n_unchanged_with_grad > 0:
            violations.append({
                "event_id": event_id,
                "issue": (f"{n_unchanged_with_grad} params had non-zero grad but "
                          f"delta==0 after step (frozen-by-bug suspect)"),
                "n_params_unchanged_with_grad": n_unchanged_with_grad,
                "total_param_delta_l2": delta,
            })

    if not violations:
        return RuleResult(
            rule_id="T0-param-update-applied",
            violated=False,
            message="All optim.step.post events show param updates were applied.",
        )
    return RuleResult(
        rule_id="T0-param-update-applied",
        violated=True,
        message=f"{len(violations)} optim.step events with frozen/un-updated params",
        evidence={
            "violation_event_ids": [v["event_id"] for v in violations][:50],
            "sample": violations[:5],
        },
    )
