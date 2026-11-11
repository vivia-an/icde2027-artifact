"""T0: every cross_entropy / nll_loss call must have reduction='mean'
during training. 'sum' or 'none' indicates an aggregation bug — loss
ends up scaled wrong, gradients are off by N, optimizer step too large.

This is one of the most common silent bugs in custom training loops:
  - someone copy-pastes inference loss code (reduction='sum') into training
  - someone wraps the loss for logging (reduction='none') and forgets to mean it

Catches a class production monitoring misses (loss is "correct" relative
to itself, but parameter updates are N× too large).

Tolerated: reduction='mean' (default and correct).
Flag-able:  reduction='sum' or 'none' or 'batchmean'.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


_OK_REDUCTIONS = {"mean"}


@rule(
    rule_id="T0-loss-reduction-mode-correct",
    min_tier=Tier.T0_PYTORCH,
    families=["F-NEW"],
    evidence_bugs=["loss-function-class"],
    description="loss.call reduction must be 'mean' during training; "
                "'sum' / 'none' indicate aggregation bug (gradient scale wrong).",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'loss.call'
    """).fetchall()
    violations = []
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        red = p.get("reduction")
        if red is None:
            continue
        if red not in _OK_REDUCTIONS:
            violations.append({
                "event_id": event_id,
                "fn": p.get("fn"),
                "reduction": red,
                "issue": (f"loss reduction='{red}' (expected 'mean'); gradient "
                          f"scale will be off by N (=batch*seq tokens)"),
            })
    if not violations:
        return RuleResult(
            rule_id="T0-loss-reduction-mode-correct",
            violated=False,
            message="All loss calls used reduction='mean'.",
        )
    return RuleResult(
        rule_id="T0-loss-reduction-mode-correct",
        violated=True,
        message=f"{len(violations)} loss calls with non-'mean' reduction",
        evidence={
            "violation_event_ids": [v["event_id"] for v in violations][:50],
            "sample": violations[:5],
        },
    )
