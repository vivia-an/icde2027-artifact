"""T1: distributed comm tensors must use the framework's declared training dtype.

Catches B3-class bugs: DeepSpeed silently uses FP16 for the gradient
all-reduce path even when the user declared BFloat16 training. The
gradient values still flow but with the wrong precision profile;
training appears nominal but loss curves drift over many steps.

Detection: build.snapshot.framework_invariants.<fw>.training_dtype
declares the user's chosen compute dtype (e.g. "bfloat16"). All
comm.post events whose `op` is a gradient-class collective
(all_reduce / reduce_scatter / reduce / all_gather / all_gather_into_tensor /
all_to_all / all_to_all_single) must carry tensor_post.dtype matching
the declared dtype. If declared training_dtype is absent the rule is
passive (we cannot know what to expect).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


_GRADIENT_COMM_OPS = {
    "all_reduce", "reduce_scatter", "reduce",
    "all_gather", "all_gather_into_tensor",
    "all_to_all", "all_to_all_single",
}


@rule(
    rule_id="T1-comm-dtype-matches-training",
    min_tier=Tier.T1_FW_METADATA,
    families=["F3"],
    evidence_bugs=["B3"],
    description="Distributed-collective tensor dtype must match the "
                "framework's declared training_dtype.",
)
def check(conn) -> RuleResult:
    # Step 1: discover declared training_dtype from build snapshots.
    bld = conn.execute("""
        SELECT payload FROM events WHERE hookpoint = 'build.snapshot'
    """).fetchall()
    declared = None
    for (p,) in bld:
        try:
            pl = json.loads(p)
        except Exception:
            continue
        for fw_inv in (pl.get("framework_invariants") or {}).values():
            if isinstance(fw_inv, dict) and fw_inv.get("training_dtype"):
                declared = fw_inv["training_dtype"]
                break
        if declared is not None:
            break
    if declared is None:
        return RuleResult(rule_id="T1-comm-dtype-matches-training",
                          violated=False,
                          message="no training_dtype declared in any "
                                  "framework_invariants block (rule passive)")

    # Step 2: scan comm.post events for dtype mismatch.
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'comm.post'
    """).fetchall()
    bad = []
    n_checked = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        op = p.get("op") or ""
        if op not in _GRADIENT_COMM_OPS:
            continue
        ts = p.get("tensor_post")
        if not isinstance(ts, dict):
            continue
        dt = ts.get("dtype")
        if not dt:
            continue
        n_checked += 1
        # Normalise — adapters may emit "bfloat16" or "torch.bfloat16"
        dt_norm = dt.replace("torch.", "")
        decl_norm = str(declared).replace("torch.", "")
        if dt_norm != decl_norm:
            bad.append({
                "event_id": event_id, "op": op,
                "tensor_dtype": dt_norm,
                "declared_training_dtype": decl_norm,
            })
    if bad:
        return RuleResult(
            rule_id="T1-comm-dtype-matches-training",
            violated=True,
            message=f"{len(bad)} comm op(s) used dtype != declared "
                    f"training_dtype ({declared})",
            evidence={
                "sample": bad[:3], "n_checked": n_checked,
                "violation_event_ids": [v["event_id"] for v in bad],
            },
        )
    return RuleResult(
        rule_id="T1-comm-dtype-matches-training",
        violated=False,
        message=f"all {n_checked} grad-comm ops matched training_dtype "
                f"({declared})",
    )
