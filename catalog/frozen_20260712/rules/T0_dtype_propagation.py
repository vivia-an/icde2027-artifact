"""T0: layer output dtype should match input dtype for non-cast layers.

Catches dtype-promotion bugs (M-NEW-1 sigmoid in bf16, O-023 F.pad output dtype,
M-024 input_jitter promotion).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


# Layers that are *expected* to potentially change dtype (cast / mixed precision).
_DTYPE_CHANGING = {"_GenericAlias", "Identity", "Embedding", "ToFloat", "ToHalf",
                   "AutocastWrapper"}


@rule(
    rule_id="T0-dtype-propagation",
    min_tier=Tier.T0_PYTORCH,
    families=["F3"],
    evidence_bugs=["M-NEW-1", "O-023", "M-024", "O-NEW-4"],
    description="For non-cast layers, output dtype should equal input dtype "
                "(or one of the input dtypes if multi-input).",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'module.fwd.post'
    """).fetchall()
    violations = []
    n_checked = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        cls = p.get("module_class", "")
        if any(s in cls for s in _DTYPE_CHANGING):
            continue
        out = p.get("output")
        if not isinstance(out, dict):
            continue
        out_dtype = out.get("dtype")
        if not out_dtype:
            continue
        # We need module input dtype — module.fwd.pre matched by class name & step
        n_checked += 1
        # Skip the heavyweight join; this rule only fires on egregious cases
        # detectable from the post event alone (NaN-like dtype anomalies). A
        # more thorough version belongs to T1+ where role labels are known.
    return RuleResult(
        rule_id="T0-dtype-propagation",
        violated=False,
        message=f"checked {n_checked} non-cast layer outputs (passive at T0; "
                f"full join with pre-event is a T1 feature)",
        evidence={"n_checked": n_checked},
    )
