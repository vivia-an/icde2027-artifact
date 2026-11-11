"""T0: softmax / router-style output should not be a degenerate one-hot
(every row max == 1 with all mass on a single position).

Catches M-014 (Megatron MoE Router topk=1 + post-softmax → softmax of single
value = 1.0 for every token, zero gradient).

We restrict to modules whose class name carries a routing/gating semantic
to avoid FP on the (legitimate) post-argmax onehot in some pipelines.
"""
import json
import math

from ..tiers import Tier
from .base import RuleResult, rule


_ROUTER_NAME_HINTS = ("Router", "Gate", "TopK", "MoEGate", "ExpertSelector")


@rule(
    rule_id="T0-softmax-degenerate",
    min_tier=Tier.T0_PYTORCH,
    families=["F8"],
    evidence_bugs=["M-014"],
    description="Router/Gate-class module forward output must not be a "
                "degenerate one-hot distribution (all mass on a single index).",
)
def check(conn) -> RuleResult:
    bad = []
    n_checked = 0

    # 1) Router-class module forward outputs
    mod_rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'module.fwd.post'
    """).fetchall()
    for event_id, payload_str in mod_rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        cls = p.get("module_class", "")
        if not any(h in cls for h in _ROUTER_NAME_HINTS):
            continue
        # Output may be tuple (scores, indices); check both single + list forms
        candidates = []
        if isinstance(p.get("output"), dict):
            candidates.append(p["output"])
        for x in (p.get("outputs") or []):
            if isinstance(x, dict):
                candidates.append(x)
        for ts in candidates:
            n_checked += 1
            v = _check_one(ts)
            if v:
                v.update(event_id=event_id, module_class=cls)
                bad.append(v)

    # 2) F.softmax outputs whose dim looks like the routing dim
    fn_rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'functional.softmax'
    """).fetchall()
    for event_id, payload_str in fn_rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        out = p.get("output")
        if not isinstance(out, dict):
            continue
        n_checked += 1
        v = _check_one(out)
        if v:
            v.update(event_id=event_id, source="F.softmax")
            bad.append(v)

    if bad:
        return RuleResult(
            rule_id="T0-softmax-degenerate",
            violated=True,
            message=f"{len(bad)} softmax/router outputs are degenerate one-hot",
            evidence={
                "sample": bad[:3],
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T0-softmax-degenerate",
        violated=False,
        message=f"checked {n_checked} softmax/router outputs, none degenerate",
    )


def _check_one(ts: dict):
    """Return a violation dict if the tensor summary looks one-hot, else None."""
    l2 = ts.get("l2_norm")
    amax = ts.get("abs_max") or ts.get("max")
    shape = ts.get("shape")
    if l2 is None or amax is None or not shape:
        return None
    # Two-shape cases:
    # (a) shape (..., K) with K>=2: row sums to 1, all mass on one index (one-hot)
    # (b) shape (..., 1):           softmax over a single value is trivially 1.0
    #     This IS the bug for M-014 (topk=1 + post-softmax → output [1.0]).
    n_rows = 1
    for d in shape[:-1]:
        n_rows *= int(d)
    if n_rows == 0:
        return None
    # One-hot signature: each row has one 1.0 → l2² = n_rows, abs_max = 1
    if abs(amax - 1.0) < 1e-3 and abs(l2 * l2 - n_rows) < n_rows * 0.05:
        return {
            "abs_max": amax,
            "l2_norm": l2,
            "shape": shape,
            "n_rows": n_rows,
            "degenerate_kind": "trivial_size_1" if shape[-1] == 1 else "one_hot",
        }
    return None
