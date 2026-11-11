"""T0: total gradient L2 norm at optim.step.pre must be finite, non-zero,
and below an explosion threshold.

Catches three failure modes that complement T0-no-nan-inf:
  - **zero grad** (e.g. backward hook miswired, autograd graph cut) —
    training silently makes no progress
  - **gradient explosion** (loss instability, no clip) — training diverges
  - **inf grad** (overflow on fp16) — separate signal from per-tensor NaN

The rule is intentionally permissive on the upper bound (1e10) so that
healthy fp32 training with momentum spikes does not FP. The lower bound
is strict zero — a perfectly-zero total grad indicates a real wiring
problem, not a numerical accident.
"""
import json
import math

from ..tiers import Tier
from .base import RuleResult, rule


_EXPLODE_THRESHOLD = 1e10


@rule(
    rule_id="T0-grad-norm-finite",
    min_tier=Tier.T0_PYTORCH,
    families=["F2"],
    evidence_bugs=["sanity"],
    description="Total grad L2 norm at optim.step.pre must be finite, "
                "positive, and below 1e10 (explosion sentinel).",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'optim.step.pre'
    """).fetchall()
    if not rows:
        return RuleResult(rule_id="T0-grad-norm-finite", violated=False,
                          message="no optim.step.pre events (rule passive)")
    bad = []
    n_checked = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        gn = p.get("total_grad_l2")
        if gn is None:
            continue
        n_checked += 1
        try:
            gnf = float(gn)
        except (TypeError, ValueError):
            continue
        # NaN check first (NaN comparisons are always False)
        if math.isnan(gnf):
            bad.append({"event_id": event_id, "total_grad_l2": gn,
                        "issue": "NaN"})
            continue
        if math.isinf(gnf):
            bad.append({"event_id": event_id, "total_grad_l2": gn,
                        "issue": "Inf"})
            continue
        if gnf == 0.0:
            bad.append({"event_id": event_id, "total_grad_l2": 0.0,
                        "issue": "exactly zero — backward graph likely cut"})
            continue
        if gnf > _EXPLODE_THRESHOLD:
            bad.append({"event_id": event_id, "total_grad_l2": gn,
                        "issue": f"explosion ({gnf:.2e} > {_EXPLODE_THRESHOLD:.0e})"})
    if bad:
        return RuleResult(
            rule_id="T0-grad-norm-finite",
            violated=True,
            message=f"{len(bad)}/{n_checked} optim.step.pre events have "
                    "non-finite or pathological total_grad_l2",
            evidence={
                "sample": bad[:3], "n_checked": n_checked,
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T0-grad-norm-finite",
        violated=False,
        message=f"all {n_checked} optim.step.pre events have finite "
                f"non-zero grad norm",
    )
