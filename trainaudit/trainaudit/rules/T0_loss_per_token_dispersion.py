"""T0: per-token loss distribution should not be wildly dispersed.

A healthy training step has per-token cross_entropy loss values within
~5x of the median. If max(per_token_loss) / median > 100, this signals:
  - one position always gives huge loss (model converged but for one token-class)
  - label / input alignment off-by-one (always wrong on first/last position)
  - degenerate prediction (one expert / head dominates, hurting one position)

Distinct from T0-no-nan-inf (which checks Inf/NaN at element level) and
from T0-grad-norm-finite (which checks aggregate). This is per-token
distribution shape — invisible to scalar loss monitoring.

Threshold 100x is loose; in healthy training max/median is typically
2-10x. 100x catches clear pathology while permitting natural variance.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


_DISPERSION_THRESHOLD = 100.0


@rule(
    rule_id="T0-loss-per-token-dispersion",
    min_tier=Tier.T0_PYTORCH,
    families=["F-NEW"],
    evidence_bugs=["loss-function-class"],
    description="per-token loss max/median ratio must be < 100x (catches "
                "degenerate prediction / label alignment off / dead token).",
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
        pt = p.get("per_token") or {}
        median = pt.get("median")
        max_ = pt.get("max")
        if median is None or max_ is None:
            continue
        if median <= 0 or pt.get("all_ignored"):
            continue
        ratio = max_ / median
        if ratio > _DISPERSION_THRESHOLD:
            violations.append({
                "event_id": event_id,
                "max_per_token_loss": max_,
                "median_per_token_loss": median,
                "max_to_median_ratio": ratio,
                "n_total": pt.get("n_total"),
                "n_ignored": pt.get("n_ignored"),
                "issue": (f"per-token loss max={max_:.2f} vs median={median:.2f} "
                          f"({ratio:.1f}x) — distribution highly skewed"),
            })
    if not violations:
        return RuleResult(
            rule_id="T0-loss-per-token-dispersion",
            violated=False,
            message="All loss.call events have bounded per-token dispersion.",
        )
    return RuleResult(
        rule_id="T0-loss-per-token-dispersion",
        violated=True,
        message=f"{len(violations)} loss events with >100x per-token dispersion",
        evidence={
            "violation_event_ids": [v["event_id"] for v in violations][:50],
            "sample": violations[:5],
        },
    )
