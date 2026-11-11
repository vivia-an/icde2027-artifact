"""Violation event_id → DiagnosisReport.

For every RuleResult.evidence['violation_event_ids'], pull the source
event from the trace and structure a DiagnosisReport with:
  - location (module_name/class/id, rank, step, callsite)
  - bug-specific evidence (pre/post norm for clip, gathered_cksums for
    replica, dtype + abs_max for softmax-degenerate, etc.)
  - surrounding trace slice on the same key (module_id or hookpoint)
  - a short deterministic hypothesis sentence
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..rules.base import RuleResult
from ..store import TraceStore
from .cross_rank_outlier import summarize_replica_violation
from .report import DiagnosisReport


# How many ±events around a violation to pull as context. Cheap; only fires
# at diagnosis time, not during training.
_CONTEXT_WINDOW = 6


def expand_results(store: TraceStore,
                   results: List[RuleResult]) -> List[DiagnosisReport]:
    """Expand every violation in `results` into a DiagnosisReport.
    Empty list returned for clean runs (no violations) — explicit
    "no violation, no diagnosis" rather than a crash."""
    store.flush()
    reports: List[DiagnosisReport] = []
    for r in results:
        if not r.violated:
            continue
        ids = (r.evidence or {}).get("violation_event_ids") or []
        if not ids:
            # Some rules carry only `sample` — fall back to sample event_ids
            for entry in (r.evidence or {}).get("sample", []) or []:
                if isinstance(entry, dict) and "event_id" in entry:
                    ids.append(entry["event_id"])
        for eid in ids:
            rep = expand_violation(store, r.rule_id, eid)
            if rep is not None:
                reports.append(rep)
    return reports


def expand_violation(store: TraceStore, rule_id: str,
                     event_id: int) -> Optional[DiagnosisReport]:
    row = store.conn.execute(
        "SELECT event_id, step, rank, hookpoint, payload FROM events "
        "WHERE event_id = ?", [event_id]).fetchone()
    if row is None:
        return None
    eid, step, rank, hookpoint, payload_str = row
    try:
        payload = json.loads(payload_str)
    except Exception:
        payload = {}

    rep = DiagnosisReport(
        rule_id=rule_id,
        violation_event_id=int(eid),
        hookpoint=str(hookpoint),
        suspect_step=step,
        suspect_rank=rank,
        suspect_module=payload.get("module_name"),
        suspect_module_class=payload.get("module_class"),
        suspect_module_id=payload.get("module_id"),
        callsite=payload.get("callsite"),
    )

    # Bug-specific evidence per rule
    rep.bug_specific = _bug_specific(rule_id, payload)

    # Pull context: ±N events on the same key (module_id if available,
    # else same hookpoint). Stays cheap — bounded N.
    rep.context_events = _context_events(store, rep, payload)

    rep.hypothesis = _hypothesis(rep)
    return rep


def _bug_specific(rule_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Per-rule extraction of the most diagnostic fields."""
    if rule_id == "T0-clip-grad-bounded":
        return {
            "fn": payload.get("fn", "torch.nn.utils.clip_grad_norm_"),
            "max_norm": payload.get("max_norm"),
            "pre_norm": payload.get("pre_norm"),
            "post_norm": payload.get("post_norm"),
            "ratio": payload.get("ratio"),
        }
    if rule_id == "T0-no-nan-inf":
        return {
            "n_inputs": len(payload.get("inputs", []) or []),
            "n_outputs": (1 if "output" in payload
                          else len(payload.get("outputs", []) or [])),
        }
    if rule_id == "T0-norm-output-unit-rms":
        out = payload.get("output") or {}
        return {
            "shape": out.get("shape"),
            "l2_norm": out.get("l2_norm"),
            "is_normalizer": payload.get("is_normalizer"),
        }
    if rule_id == "T0-softmax-degenerate":
        out = payload.get("output") or {}
        return {
            "shape": out.get("shape"),
            "abs_max": out.get("abs_max") or out.get("max"),
            "l2_norm": out.get("l2_norm"),
        }
    if rule_id == "T0-optim-step-counter-monotonic":
        return {
            "optimizer_class": payload.get("optimizer_class"),
            "state_step_max": payload.get("state_step_max"),
            "state_step_min": payload.get("state_step_min"),
        }
    if rule_id == "T0-initial-lr-present":
        return {
            "scheduler_class": payload.get("scheduler_class"),
            "last_epoch": payload.get("last_epoch"),
            "param_groups_missing_initial_lr": [
                g.get("index") for g in payload.get("param_groups", [])
                if not g.get("has_initial_lr")
            ],
        }
    if rule_id == "T1-replica-cksum-equal":
        # The violation event payload is build.snapshot which contains the
        # full cross_rank_cksums list — find the offending entry.
        for entry in payload.get("cross_rank_cksums", []) or []:
            if entry.get("group_size", 1) > 1 and entry.get("all_equal") is False:
                return summarize_replica_violation(entry)
        return {}
    if rule_id == "T1-expert-bias-fp32":
        sem = payload.get("semantic") or {}
        return {
            "expert_bias_dtype": sem.get("expert_bias_dtype"),
            "is_router": sem.get("is_router"),
        }
    return {}


def _context_events(store: TraceStore, rep: DiagnosisReport,
                    payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull ±N events keyed on module_id (preferred) or hookpoint."""
    cur_id = rep.violation_event_id
    if rep.suspect_module_id is not None:
        # Same-module slice
        rows = store.conn.execute(
            f"""
            (SELECT event_id, step, rank, hookpoint, payload FROM events
              WHERE event_id < ?
                AND CAST(json_extract(payload, '$.module_id') AS BIGINT) = ?
              ORDER BY event_id DESC LIMIT {_CONTEXT_WINDOW})
            UNION ALL
            (SELECT event_id, step, rank, hookpoint, payload FROM events
              WHERE event_id > ?
                AND CAST(json_extract(payload, '$.module_id') AS BIGINT) = ?
              ORDER BY event_id ASC LIMIT {_CONTEXT_WINDOW})
            """,
            [cur_id, rep.suspect_module_id,
             cur_id, rep.suspect_module_id]).fetchall()
    else:
        # Same-hookpoint slice
        rows = store.conn.execute(
            f"""
            (SELECT event_id, step, rank, hookpoint, payload FROM events
              WHERE event_id < ? AND hookpoint = ?
              ORDER BY event_id DESC LIMIT {_CONTEXT_WINDOW})
            UNION ALL
            (SELECT event_id, step, rank, hookpoint, payload FROM events
              WHERE event_id > ? AND hookpoint = ?
              ORDER BY event_id ASC LIMIT {_CONTEXT_WINDOW})
            """,
            [cur_id, rep.hookpoint, cur_id, rep.hookpoint]).fetchall()

    out: List[Dict[str, Any]] = []
    for eid, step, rank, hp, p in rows:
        try:
            pl = json.loads(p)
        except Exception:
            pl = {}
        out.append({
            "event_id": int(eid),
            "step": step, "rank": rank, "hookpoint": hp,
            # Just the salient bits, not the whole payload
            "module_class": pl.get("module_class"),
            "module_name": pl.get("module_name"),
        })
    return out


def _hypothesis(rep: DiagnosisReport) -> str:
    """One-sentence deterministic summary. C2 LLM agent can replace this
    with a richer prose explanation."""
    where = rep.suspect_module or rep.suspect_module_class or rep.hookpoint
    if rep.suspect_step is not None:
        where += f" at step {rep.suspect_step}"
    if rep.suspect_rank is not None:
        where += f" rank {rep.suspect_rank}"
    bs = rep.bug_specific or {}
    rule = rep.rule_id

    if rule == "T0-clip-grad-bounded":
        return (f"clip_grad_norm_ left grad norm at {bs.get('post_norm'):.3f} "
                f"with max_norm={bs.get('max_norm')} "
                f"(ratio={bs.get('ratio')}); the clip path likely returned "
                f"without scaling — check the clip_coef formula.")
    if rule == "T0-no-nan-inf":
        return f"NaN/Inf reached {where}; trace upstream events for the source."
    if rule == "T0-norm-output-unit-rms":
        return (f"normalizer output at {where} has abnormal magnitude "
                f"(l2_norm={bs.get('l2_norm')}, shape={bs.get('shape')}); "
                f"check eps placement and whether it's L2 vs RMS form.")
    if rule == "T0-softmax-degenerate":
        return (f"softmax / router output at {where} is one-hot "
                f"(abs_max={bs.get('abs_max')}, shape={bs.get('shape')}); "
                f"likely topk-then-softmax over a size-1 dim — softmax of a "
                f"single value is always 1.")
    if rule == "T0-optim-step-counter-monotonic":
        return (f"optimizer state['step'] is frozen at "
                f"{bs.get('state_step_max')} across calls — bias correction "
                f"will never update.")
    if rule == "T0-initial-lr-present":
        groups = bs.get("param_groups_missing_initial_lr") or []
        return (f"scheduler resume (last_epoch={bs.get('last_epoch')}) but "
                f"param_groups {groups} have no initial_lr — optimizer config "
                f"forgot to set it.")
    if rule == "T1-replica-cksum-equal":
        out_rank = bs.get("outlier_rank")
        rank_clause = (f" outlier rank={out_rank}" if out_rank is not None
                       else " (2-rank tie — both are suspect)")
        return (f"replica param '{bs.get('param_name')}' diverged across the "
                f"replica group{rank_clause}; init likely went through the "
                f"wrong RNG generator (default cuda RNG vs TP-aware RNG).")
    if rule == "T1-expert-bias-fp32":
        return (f"router expert_bias is {bs.get('expert_bias_dtype')}, expected "
                f"float32; Float16Module probably re-cast it on a path that "
                f"should have been excluded.")
    return f"violation of {rule} at {where}"
