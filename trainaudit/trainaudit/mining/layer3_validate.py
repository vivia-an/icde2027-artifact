"""Layer 3: validate candidate predicates against multiple healthy traces.

A candidate predicate is **accepted** as a verified invariant only if it
produces *zero* violations across every healthy trace. If it fires on
some, the predicate is either:
  - genuinely too strict — reject as spurious, or
  - missing a precondition — `precondition_inference.py` (P1) tries to
    derive a differentiating attribute (e.g. `zero_stage<3`)

Tolerance auto-learning: for `BOUND` predicates with no fixed threshold,
the validator records the field's percentile distribution across all
healthy events and reports `quantile_pct` so the predicate can be
re-emitted with a learned threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..dsl import Predicate, violation_event_ids
from ..store import TraceStore


@dataclass
class HealthyValidationResult:
    predicate_id: str
    accepted: bool
    n_violations_per_trace: List[int] = field(default_factory=list)
    """Length matches len(healthy_stores). Each int is the number of
    violation event_ids that predicate produced on that store."""

    rejection_reason: str = ""

    learned_tolerance: Optional[Dict[str, float]] = None
    """If set, a suggested {abs|rel|quantile_pct} that would have made
    the predicate hold across every healthy trace."""


def validate_against_healthy(predicate: Predicate,
                              healthy_stores: List[TraceStore],
                              *,
                              max_allowed_violations: int = 0
                              ) -> HealthyValidationResult:
    """Run `predicate` on each `healthy_store`. Accept only if every
    store yields ≤ `max_allowed_violations` violations.

    `max_allowed_violations > 0` lets very-noisy candidates through
    (rarely useful — paper §3.2 keeps strict zero-violations); kept as a
    knob for Layer 4 (LLM filter) to revisit borderline cases.
    """
    n_violations: List[int] = []
    for store in healthy_stores:
        try:
            ids = violation_event_ids(store, predicate)
        except Exception as e:  # noqa: BLE001
            return HealthyValidationResult(
                predicate_id=predicate.id, accepted=False,
                n_violations_per_trace=[],
                rejection_reason=f"compile/run error: "
                                  f"{type(e).__name__}: {e}")
        n_violations.append(len(ids))

    accepted = all(v <= max_allowed_violations for v in n_violations)
    reason = ""
    if not accepted:
        reason = (f"fired on {sum(1 for v in n_violations if v > 0)}/"
                  f"{len(n_violations)} healthy traces; counts={n_violations}")
    return HealthyValidationResult(
        predicate_id=predicate.id, accepted=accepted,
        n_violations_per_trace=n_violations, rejection_reason=reason)


def infer_tolerance(field_name: str,
                     healthy_stores: List[TraceStore],
                     hookpoint: str,
                     *,
                     pct: float = 99.0) -> Optional[float]:
    """Sample `field_name` across every event of `hookpoint` in every
    healthy store; return the `pct`-th percentile (e.g. 99th) so a
    bound predicate can be auto-tuned to clear all healthy values with
    a 1% safety margin.
    """
    import json
    import statistics
    samples: List[float] = []
    for store in healthy_stores:
        store.flush()
        rows = store.conn.execute(
            "SELECT payload FROM events WHERE hookpoint = ?",
            [hookpoint]).fetchall()
        for (payload_str,) in rows:
            try:
                p = json.loads(payload_str)
            except Exception:
                continue
            v = p.get(field_name)
            if isinstance(v, (int, float)):
                samples.append(float(v))
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0]
    samples.sort()
    idx = int((pct / 100.0) * (len(samples) - 1))
    return samples[idx]
