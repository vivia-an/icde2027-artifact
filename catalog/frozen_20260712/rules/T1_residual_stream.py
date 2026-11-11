"""T1: in residual blocks, the post-block output must remain closer to
the block's *original* input than to its normalized version.

Catches B13 / O-002: OLMoSequentialBlock did `x = self.attn_norm(x)` which
clobbered the residual variable, so the identity path used normed x instead
of original x.

Detection has two paths:
  1) Primary (strong): read residual.probe events emitted by the OLMo
     adapter's active probe — directly compares (d_to_input, d_to_normed).
  2) Fallback (weak): l2-magnitude heuristic on module.fwd.pre/post events
     when no probe was installed.
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T1-residual-stream-preserved",
    min_tier=Tier.T1_FW_METADATA,
    families=["F7"],
    evidence_bugs=["B13", "O-002", "O-NEW-7", "O-NEW-8"],
    description="In an OLMo residual block, the block's output should remain "
                "closer to the block's input than to the attn_norm output.",
)
def check(conn) -> RuleResult:
    # === Primary signal: residual.probe events from OLMo adapter ===
    probes = conn.execute("""
        SELECT event_id, payload FROM events WHERE hookpoint = 'residual.probe'
    """).fetchall()
    if probes:
        bad = []
        for event_id, payload_str in probes:
            try:
                p = json.loads(payload_str)
            except Exception:
                continue
            d_in = p.get("d_to_original_input")
            d_norm = p.get("d_to_normed_input")
            if d_in is None or d_norm is None:
                continue
            if d_norm < d_in:
                bad.append({
                    "event_id": event_id,
                    "block_class": p.get("block_class"),
                    "d_to_original_input": d_in,
                    "d_to_normed_input": d_norm,
                })
        if bad:
            return RuleResult(
                rule_id="T1-residual-stream-preserved",
                violated=True,
                message=f"{len(bad)}/{len(probes)} residual blocks: output "
                        "closer to normed than to original input",
                evidence={
                    "sample": bad[:3],
                    "violation_event_ids": [v["event_id"] for v in bad],
                },
            )
        return RuleResult(
            rule_id="T1-residual-stream-preserved",
            violated=False,
            message=f"residual stream preserved across {len(probes)} probes",
        )

    # === Fallback: l2-magnitude heuristic (used when adapter probe missing) ===
    rows = conn.execute("""
        SELECT event_id, hookpoint, payload FROM events
         WHERE hookpoint IN ('module.fwd.pre', 'module.fwd.post')
         ORDER BY event_id
    """).fetchall()

    bad = []
    n_blocks = 0
    last_input_l2 = None
    last_normed_l2 = None
    for event_id, hp, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        cls = p.get("module_class", "")
        sem = p.get("semantic") or {}

        if hp == "module.fwd.pre" and sem.get("is_residual_block"):
            inputs = p.get("inputs") or []
            if inputs and isinstance(inputs[0], dict):
                last_input_l2 = inputs[0].get("l2_norm")

        if hp == "module.fwd.post" and "Norm" in cls and last_input_l2 is not None:
            out = p.get("output")
            if isinstance(out, dict):
                last_normed_l2 = out.get("l2_norm")

        if hp == "module.fwd.post" and sem.get("is_residual_block"):
            n_blocks += 1
            out = p.get("output")
            if not isinstance(out, dict) or last_input_l2 is None:
                continue
            block_out_l2 = out.get("l2_norm")
            if block_out_l2 is None:
                continue
            if last_normed_l2 is not None:
                d_in = abs(block_out_l2 - last_input_l2)
                d_norm = abs(block_out_l2 - last_normed_l2)
                if d_norm * 3 < d_in:
                    bad.append({
                        "event_id": event_id,
                        "block_class": cls,
                        "input_l2": last_input_l2,
                        "normed_l2": last_normed_l2,
                        "block_out_l2": block_out_l2,
                    })
            last_input_l2 = None
            last_normed_l2 = None

    if not n_blocks:
        return RuleResult(rule_id="T1-residual-stream-preserved",
                          violated=False,
                          message="no residual blocks observed (rule passive)")
    if bad:
        return RuleResult(
            rule_id="T1-residual-stream-preserved",
            violated=True,
            message=f"{len(bad)}/{n_blocks} residual blocks (l2-fallback): "
                    "output closer to normed than to input",
            evidence={"sample": bad[:3]},
        )
    return RuleResult(
        rule_id="T1-residual-stream-preserved",
        violated=False,
        message=f"residual stream preserved across {n_blocks} block forwards (l2-fallback)",
    )
