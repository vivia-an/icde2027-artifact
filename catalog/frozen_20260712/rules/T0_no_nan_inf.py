"""T0: any captured tensor must be free of NaN / Inf."""
import json

from ..schema import T_HAS_INF, T_HAS_NAN
from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T0-no-nan-inf",
    min_tier=Tier.T0_PYTORCH,
    families=["F8"],
    evidence_bugs=["sanity"],
    description="No tensor in any traced event should contain NaN or Inf.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, hookpoint, payload FROM events
         WHERE hookpoint IN ('module.fwd.pre', 'module.fwd.post', 'module.bwd',
                             'optim.step.pre', 'optim.step.post',
                             'comm.pre', 'comm.post', 'build.snapshot')
    """).fetchall()
    bad = []
    for event_id, hp, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        for ts in _walk_tensor_summaries(p):
            if not isinstance(ts, dict):
                continue
            if ts.get(T_HAS_NAN) or ts.get(T_HAS_INF):
                bad.append({"event_id": event_id, "hookpoint": hp,
                            "has_nan": ts.get(T_HAS_NAN),
                            "has_inf": ts.get(T_HAS_INF)})
                break
    if bad:
        return RuleResult(
            rule_id="T0-no-nan-inf",
            violated=True,
            message=f"{len(bad)} events contain NaN/Inf tensors",
            evidence={
                "sample": bad[:3],
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(rule_id="T0-no-nan-inf", violated=False,
                      message="no NaN/Inf detected")


def _walk_tensor_summaries(obj):
    """Yield dicts that look like tensor summaries (have 'dtype' key)."""
    if isinstance(obj, dict):
        if "dtype" in obj and "shape" in obj:
            yield obj
        for v in obj.values():
            yield from _walk_tensor_summaries(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_tensor_summaries(v)
