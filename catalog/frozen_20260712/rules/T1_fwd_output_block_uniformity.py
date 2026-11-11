"""T1: forward output L2 magnitude per block, for the same module class,
should not deviate >100x from the cross-block median.

Forward-side companion to T1-grad-flow-block-uniformity. Catches dead-region
patterns where forward output of one block is dramatically smaller (or
larger) than its siblings — independent signal from grad uniformity.

Same SKIP_CLASSES as the grad rule (norms + routers + low-info wrappers).
"""
import json
import math
import statistics
import collections

from ..tiers import Tier
from .base import RuleResult, rule


_RATIO_OUTLIER_THRESHOLD = 100.0

_SKIP_CLASSES = {"RMSNorm", "LayerNorm", "Identity", "RotaryEmbedding",
                  "FSDPMoELinearRouter", "MoELinearRouter",
                  "FSDPEmbedding", "FSDPLMHead",
                  "TorchAttentionBackend"}


def _block_index(name: str):
    if not name or "blocks." not in name:
        return None
    try:
        return int(name.split("blocks.")[1].split(".")[0])
    except (ValueError, IndexError):
        return None


@rule(
    rule_id="T1-fwd-output-block-uniformity",
    min_tier=Tier.T1_FW_METADATA,
    families=["F-NEW"],
    evidence_bugs=["mining-pipeline"],
    description="Per-block forward output l2_norm for same module class "
                "must stay within 100x of cross-block median.",
)
def check(conn) -> RuleResult:
    rows = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'module.fwd.post'
    """).fetchall()
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
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
        out = p.get("output") or {}
        if not isinstance(out, dict):
            continue
        l2 = out.get("l2_norm")
        try:
            l2 = float(l2) if l2 is not None else None
        except (TypeError, ValueError):
            continue
        if l2 is None or math.isnan(l2) or l2 <= 0:
            continue
        grouped[cls][bidx].append({
            "event_id": event_id, "l2": l2, "module_name": name,
        })

    violations = []
    for cls, block_dict in grouped.items():
        if len(block_dict) < 4:
            continue
        block_meds = {}
        samples = {}
        for bidx, recs in block_dict.items():
            if len(recs) < 3:
                continue
            block_meds[bidx] = statistics.median([r["l2"] for r in recs])
            samples[bidx] = recs[0]
        if len(block_meds) < 4:
            continue
        all_meds = list(block_meds.values())
        cross_med = statistics.median(all_meds)
        if cross_med <= 0:
            continue
        for bidx, med in block_meds.items():
            if med <= 0:
                continue
            dev = max(med / cross_med, cross_med / med)
            if dev > _RATIO_OUTLIER_THRESHOLD:
                violations.append({
                    "module_class": cls,
                    "block_idx": bidx,
                    "block_l2_median": med,
                    "cross_block_l2_median": cross_med,
                    "deviation_x": dev,
                    "module_name": samples[bidx]["module_name"],
                    "sample_event_id": samples[bidx]["event_id"],
                    "issue": (f"block {bidx} {cls} fwd output l2 {med:.2e} "
                              f"departs {dev:.1f}x from cross-block median "
                              f"{cross_med:.2e}"),
                })

    if not violations:
        return RuleResult(
            rule_id="T1-fwd-output-block-uniformity",
            violated=False,
            message="All same-class modules have uniform fwd output l2 "
                    "across blocks.",
        )
    return RuleResult(
        rule_id="T1-fwd-output-block-uniformity",
        violated=True,
        message=f"{len(violations)} block-level fwd output outlier(s)",
        evidence={
            "violation_event_ids": [v["sample_event_id"] for v in violations][:50],
            "sample": violations[:5],
        },
    )
