"""T1: actual transformer layers (across all ranks) must equal configured num_layers.

Catches M-020: PP layer count silently dropped by integer division when
num_layers % pp_size != 0.

Detection: framework_invariants[megatron] captures declared `num_layers` and
`pipeline_model_parallel_size`. Megatron adapter labels TransformerLayer
class. Rule counts is_transformer_layer events on rank 0 (one PP stage),
multiplies by pp_size, compares to declared.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T1-layer-count-strict",
    min_tier=Tier.T1_FW_METADATA,
    families=["F4"],
    evidence_bugs=["M-020"],
    description="Sum of TransformerLayer counts across PP ranks must equal "
                "declared num_layers. Fires when num_layers % pp_size != 0.",
)
def check(conn) -> RuleResult:
    bld = conn.execute("""
        SELECT payload FROM events WHERE hookpoint = 'build.snapshot'
    """).fetchall()
    declared_num_layers = None
    declared_pp_size = None
    actual_per_rank = None
    for (p,) in bld:
        try:
            pl = json.loads(p)
        except Exception:
            continue
        fw = pl.get("framework_invariants") or {}
        meg = fw.get("megatron") or {}
        if "num_layers" in meg:
            declared_num_layers = meg.get("num_layers")
        if "pipeline_model_parallel_size" in meg:
            declared_pp_size = meg.get("pipeline_model_parallel_size") or 1
        if "n_transformer_layers_in_local_module" in meg:
            actual_per_rank = meg.get("n_transformer_layers_in_local_module")
    if declared_num_layers is None:
        return RuleResult(rule_id="T1-layer-count-strict", violated=False,
                          message="no declared num_layers (rule passive)")
    if actual_per_rank is None:
        return RuleResult(rule_id="T1-layer-count-strict", violated=False,
                          message="no transformer layer count from adapter (rule passive)")

    pp_size = max(int(declared_pp_size or 1), 1)
    actual_total = actual_per_rank * pp_size
    if actual_total == int(declared_num_layers):
        return RuleResult(
            rule_id="T1-layer-count-strict",
            violated=False,
            message=f"layer count consistent: {actual_total} == declared {declared_num_layers}",
        )
    return RuleResult(
        rule_id="T1-layer-count-strict",
        violated=True,
        message=f"layer count mismatch: actual {actual_total} (per-rank "
                f"{actual_per_rank} × pp_size {pp_size}) != declared "
                f"{declared_num_layers}",
        evidence={
            "declared_num_layers": declared_num_layers,
            "declared_pp_size": pp_size,
            "actual_per_rank": actual_per_rank,
            "actual_total": actual_total,
        },
    )
