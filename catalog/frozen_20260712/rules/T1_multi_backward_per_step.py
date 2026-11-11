"""T1: detect multi-backward-per-optim-step (gradient accumulation pattern).

This is a STRUCTURAL detector, not by itself a violation. It flags steps in
which the same root-level module had multiple forward+backward pairs before
the next optim.step.pre. The user pattern is legitimate in many contexts
(e.g., RLHF, accumulation-boundary, multi-token prediction). The rule's
value is that — combined with framework-specific config (DeepSpeed
ZeRO-1/2 + offload_optimizer + gradient_accumulation_steps == 1) — this
pattern is the silent-error trigger of CAND_DEEPSPEED_ZERO_OFFLOAD_MULTI_BACKWARD
(upstream PR #7981): all backward passes' gradients except the last get
silently dropped.

Detection: count module.fwd.pre events with module_name IS NULL (root
module of the snapshot tree) per training step. > 1 → multi-forward;
emit advisory with the offending step IDs. The rule's `violated` flag is
True only when count > 1 AND any framework_invariant indicates the
fragile config combo (offload + ga=1).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


def _is_fragile_config(payloads_json) -> bool:
    """Return True iff a build.snapshot framework_invariants block declares
    DeepSpeed ZeRO-1/2 + offload_optimizer=True + gradient_accumulation_steps=1.

    Conservative: returns False unless ALL three are explicitly True/value.
    Since DeepSpeed adapter (as of iter 10) does not yet expose these fields,
    the rule will currently never fire. The detector code is in place so
    that future adapter extension lights it up automatically.
    """
    for p in payloads_json:
        try:
            pl = json.loads(p)
        except Exception:
            continue
        fw_inv = (pl.get("framework_invariants") or {}).get("deepspeed")
        if not isinstance(fw_inv, dict):
            continue
        if (fw_inv.get("offload_optimizer") is True
                and fw_inv.get("gradient_accumulation_steps") == 1
                and fw_inv.get("zero_stage") in (1, 2)):
            return True
    return False


@rule(
    rule_id="T1-multi-backward-per-step-fragile-config",
    min_tier=Tier.T1_FW_METADATA,
    families=["F11"],
    evidence_bugs=["CAND_DEEPSPEED_ZERO_OFFLOAD_MULTI_BACKWARD"],
    description="Multiple forwards (root-module) within the same training "
                "step combined with DeepSpeed ZeRO-1/2 + offload + "
                "gradient_accumulation_steps==1: all-but-last backward's "
                "grad is silently dropped (PR #7981 fix).",
)
def check(conn) -> RuleResult:
    # 1. count root-module forwards per step (module_name IS NULL or empty)
    # DuckDB's json_extract returns the literal string 'null' for JSON null,
    # not SQL NULL — so we match on that. Also accept empty-string module_name
    # for setups where root module is registered with name=''.
    fwd_rows = conn.execute("""
        SELECT step, COUNT(*) AS n FROM events
         WHERE hookpoint = 'module.fwd.pre'
           AND (json_extract(payload, '$.module_name') = 'null'
                OR json_extract(payload, '$.module_name') = '""')
         GROUP BY step
    """).fetchall()
    multi_fwd_steps = [(s, n) for (s, n) in fwd_rows if n and n > 1]

    if not multi_fwd_steps:
        return RuleResult(
            rule_id="T1-multi-backward-per-step-fragile-config",
            violated=False,
            message="all training steps have ≤ 1 root forward (single-pass)",
        )

    # 2. check framework config
    bld_payloads = [p for (p,) in conn.execute(
        "SELECT payload FROM events WHERE hookpoint = 'build.snapshot'"
    ).fetchall()]
    fragile = _is_fragile_config(bld_payloads)

    msg = (f"{len(multi_fwd_steps)} step(s) have multiple root-forward passes "
           f"(steps: {[s for s, _ in multi_fwd_steps[:5]]}...)")
    if not fragile:
        return RuleResult(
            rule_id="T1-multi-backward-per-step-fragile-config",
            violated=False,
            message=f"{msg}; no fragile DeepSpeed offload+ga=1 config "
                    "declared in build.snapshot — pattern is benign here.",
        )
    return RuleResult(
        rule_id="T1-multi-backward-per-step-fragile-config",
        violated=True,
        message=f"{msg}; DeepSpeed offload+ga=1 declared — silent grad loss "
                "expected (PR #7981 pattern)",
        evidence={
            "multi_fwd_steps": multi_fwd_steps[:10],
            "n_violation_steps": len(multi_fwd_steps),
        },
    )
