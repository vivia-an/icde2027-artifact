"""T1: sqrt decay LR schedule must decay fast initially, slow at end.

Catches OC-NEW-3: buggy `sqrt(progress)` curve has wrong shape (slow then fast).
Fixed `1 - sqrt(1 - progress)` produces the intended fast-then-slow curve.

Detection: OLMo-core adapter wraps _sqrt_decay; emits decay.probe events.
At rule time, fit slope at progress=0 vs progress=1; correct decay has
steeper |slope| at start than at end.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T1-sqrt-decay-front-loaded",
    min_tier=Tier.T1_FW_METADATA,
    families=["F8"],
    evidence_bugs=["OC-NEW-3"],
    description="sqrt decay should be steeper near progress=0 than near progress=1.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'decay.probe'
         ORDER BY event_id
    """).fetchall()
    if len(rows) < 5:
        return RuleResult(rule_id="T1-sqrt-decay-front-loaded", violated=False,
                          message=f"only {len(rows)} decay.probe events (rule passive)")

    samples = []
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        prog = p.get("progress")
        res = p.get("result")
        if prog is not None and res is not None:
            samples.append((prog, res))

    if len(samples) < 5:
        return RuleResult(rule_id="T1-sqrt-decay-front-loaded", violated=False,
                          message="insufficient probe samples (rule passive)")

    samples.sort(key=lambda x: x[0])
    # Estimate slopes near progress=0 (first quartile) vs progress=1 (last quartile)
    n = len(samples)
    early = samples[: max(2, n // 4)]
    late = samples[max(n - n // 4, n - 2):]

    def _slope(pts):
        if len(pts) < 2:
            return 0.0
        return (pts[-1][1] - pts[0][1]) / max(pts[-1][0] - pts[0][0], 1e-9)

    s_early = abs(_slope(early))
    s_late = abs(_slope(late))

    # Probe.progress = step_from_end/decay (high progress = near START of training).
    # Correct decay (fast-then-slow over real time):
    #   |dLR/dt| near start of training >> |dLR/dt| near end
    # → |slope at high progress| (s_late) >> |slope at low progress| (s_early)
    # Violation: opposite — s_early >> s_late means decay slow-then-fast.
    if s_early > s_late * 2:
        return RuleResult(
            rule_id="T1-sqrt-decay-front-loaded",
            violated=True,
            message=f"sqrt decay shape inverted: end-of-training |slope|={s_early:.4f} "
                    f">> start-of-training |slope|={s_late:.4f}",
            evidence={"slope_at_end_of_training": s_early,
                      "slope_at_start_of_training": s_late,
                      "n_samples": n},
        )
    return RuleResult(
        rule_id="T1-sqrt-decay-front-loaded",
        violated=False,
        message=f"sqrt decay shape correct: start-of-training |slope|={s_late:.4f} "
                f">= end-of-training |slope|={s_early:.4f}",
    )
