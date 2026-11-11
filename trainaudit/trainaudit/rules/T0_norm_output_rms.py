"""T0: LayerNorm / RMSNorm / GroupNorm output should have RMS roughly 1.

Catches O-NEW-1 (RMSNorm output way off) and similar normalization bugs.
Tolerance is intentionally wide (0.3..3.0) to avoid false positives on
training-time activations; real bugs (e.g. L2 instead of RMS) miss by 10x+.
"""
import json
import math

from ..tiers import Tier
from .base import RuleResult, rule


# Normalizers output should have unit RMS by construction; allow modest
# deviation for learned weight gain. Tolerance set after empirical evidence
# (O-NEW-1 buggy run produces rms~0.33; clean training produces rms~0.95-1.05).
_LOWER = 0.5
_UPPER = 2.0


@rule(
    rule_id="T0-norm-output-unit-rms",
    min_tier=Tier.T0_PYTORCH,
    families=["F8"],
    evidence_bugs=["O-NEW-1"],
    description="Normalizer layers (LayerNorm/RMSNorm/GroupNorm) should "
                "produce output with per-feature RMS close to 1.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'module.fwd.post'
    """).fetchall()
    violations = []
    n_norm = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        if not p.get("is_normalizer"):
            continue
        n_norm += 1
        out = p.get("output")
        if not isinstance(out, dict):
            continue
        l2 = out.get("l2_norm")
        shape = out.get("shape")
        if l2 is None or not shape:
            continue
        numel = 1
        for d in shape:
            numel *= int(d)
        if numel <= 0:
            continue
        rms = l2 / math.sqrt(numel)
        if not (_LOWER <= rms <= _UPPER):
            violations.append({"event_id": event_id, "rms": rms,
                               "module_class": p.get("module_class")})
    if violations:
        return RuleResult(
            rule_id="T0-norm-output-unit-rms",
            violated=True,
            message=f"{len(violations)} normalizer outputs have abnormal RMS",
            evidence={
                "sample": violations[:3], "n_normalizer_events": n_norm,
                "violation_event_ids": [v["event_id"] for v in violations],
            },
        )
    return RuleResult(
        rule_id="T0-norm-output-unit-rms",
        violated=False,
        message=f"all {n_norm} normalizer outputs within RMS [{_LOWER},{_UPPER}]",
    )
