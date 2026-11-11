"""T1: across blocks of the same module class (e.g. all 12 MoE modules of a
12-layer transformer), the grad_input/grad_output L2 ratio should be roughly
uniform — outlier > 100x off the median signals a block-level dead path.

Rationale: even with depth-dependent residual scaling, the *attenuation* a
single block applies to its incoming gradient should be in similar ballpark
across blocks. A block that attenuates 100x more than its siblings is
silently absorbing gradient — its parameters get tiny updates while the
others train normally.

This rule is INDEPENDENT of T0-norm-output-unit-rms (which checks forward
output of normalizers). It catches dead-block patterns even when forward
output magnitudes are healthy, e.g., a block that has normal forward but
dead backward (detached graph segment).
"""
import json
import math
import statistics
import collections

from ..tiers import Tier
from .base import RuleResult, rule


_RATIO_OUTLIER_THRESHOLD = 100.0  # 100x departure from median fires

# Module classes whose grad-flow ratio is naturally depth-skewed (e.g. norms
# inherently have very different attenuation in early vs late blocks). Skip
# these — focus on parameter-compute modules where uniformity expectation
# is sound.
_SKIP_CLASSES = {"RMSNorm", "LayerNorm", "Identity", "RotaryEmbedding",
                  "Attention", "TorchAttentionBackend",
                  "FSDPMoELinearRouter", "MoELinearRouter"}


def _block_index(name: str):
    """Extract block index from module name like 'blocks.5.feed_forward_moe'."""
    if not name or "blocks." not in name:
        return None
    try:
        return int(name.split("blocks.")[1].split(".")[0])
    except (ValueError, IndexError):
        return None


@rule(
    rule_id="T1-grad-flow-block-uniformity",
    min_tier=Tier.T1_FW_METADATA,
    families=["F-NEW"],
    evidence_bugs=["mining-pipeline"],
    description="Per-block grad_input/grad_output ratio for same module class "
                "must stay within 100x of cross-block median (catches dead "
                "block backward path).",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'module.bwd'
    """).fetchall()
    # group by (module_class, block_idx) → list of ratios
    grouped = collections.defaultdict(list)
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        cls = p.get("module_class")
        if not cls or cls.strip('"') in _SKIP_CLASSES:
            continue
        name = p.get("module_name", "")
        bidx = _block_index(name)
        if bidx is None:
            continue
        gin_list = p.get("grad_input") or []
        gout_list = p.get("grad_output") or []
        if not gin_list or not gout_list:
            continue
        gin = gin_list[0]
        gout = gout_list[0]
        if not isinstance(gin, dict) or not isinstance(gout, dict):
            continue
        gin_l2 = gin.get("l2_norm")
        gout_l2 = gout.get("l2_norm")
        try:
            gin_l2 = float(gin_l2) if gin_l2 is not None else None
            gout_l2 = float(gout_l2) if gout_l2 is not None else None
        except (TypeError, ValueError):
            continue
        if gin_l2 is None or gout_l2 is None:
            continue
        if math.isnan(gin_l2) or math.isnan(gout_l2):
            continue
        if gout_l2 <= 0:
            continue  # avoid div0
        ratio = gin_l2 / gout_l2
        grouped[(cls, bidx)].append({
            "event_id": event_id,
            "ratio": ratio,
            "gin_l2": gin_l2,
            "gout_l2": gout_l2,
            "module_name": name,
        })

    # For each module_class, get median ratio across blocks; flag blocks 100x off
    by_cls = collections.defaultdict(dict)  # cls -> bidx -> median_ratio
    samples = collections.defaultdict(dict)
    for (cls, bidx), recs in grouped.items():
        if len(recs) < 3:
            continue  # need enough samples for stable median
        med = statistics.median([r["ratio"] for r in recs])
        by_cls[cls][bidx] = med
        samples[cls][bidx] = recs[0]

    violations = []
    for cls, block_meds in by_cls.items():
        if len(block_meds) < 4:
            continue  # need at least 4 blocks for cross-block uniformity test
        all_ratios = list(block_meds.values())
        cross_block_median = statistics.median(all_ratios)
        if cross_block_median <= 0:
            continue
        for bidx, ratio in block_meds.items():
            if ratio <= 0:
                continue
            deviation = max(ratio / cross_block_median,
                            cross_block_median / ratio)
            if deviation > _RATIO_OUTLIER_THRESHOLD:
                violations.append({
                    "module_class": cls,
                    "block_idx": bidx,
                    "block_ratio": ratio,
                    "cross_block_median": cross_block_median,
                    "deviation_x": deviation,
                    "module_name": samples[cls][bidx]["module_name"],
                    "sample_event_id": samples[cls][bidx]["event_id"],
                    "issue": (f"block {bidx} {cls} grad ratio {ratio:.2e} "
                              f"departs {deviation:.1f}x from cross-block "
                              f"median {cross_block_median:.2e}"),
                })

    if not violations:
        return RuleResult(
            rule_id="T1-grad-flow-block-uniformity",
            violated=False,
            message="All same-class modules have uniform grad ratio across blocks.",
        )
    return RuleResult(
        rule_id="T1-grad-flow-block-uniformity",
        violated=True,
        message=f"{len(violations)} block-level grad-flow outlier(s)",
        evidence={
            "violation_event_ids": [v["sample_event_id"] for v in violations][:50],
            "sample": violations[:5],
        },
    )
