"""Compile a `Predicate` into a `CompiledRule` (DuckDB SQL + optional postprocess).

Strategy per template:
- TENSOR_STAT_BOUND with walk_tensor_summaries → SQL fetch + Python walker
- TENSOR_STAT_BOUND with payload_path             → pure SQL
- PAYLOAD_FIELD_COMPARE                            → pure SQL
- CONDITIONAL_CHECK                                → pure SQL (precondition AS WHERE)
- STRUCTURAL_PRESENCE                              → pure SQL (multi-field uses OR-of-failures)

Multi-field (`bound.field` is a list) semantics: invariant requires *all* fields
satisfy the per-field condition; violation = ANY field fails.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .predicate import Bound, BoundCondition, BoundKind, Predicate, Scope, Template
from .sql_fragments import (cast_int, cast_num, extract, hookpoint_filter,
                             to_json_literal)


@dataclass
class Violation:
    event_id: int
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledRule:
    predicate_id: str
    sql: str
    """SQL that, when executed, yields one row per VIOLATION (unless
    `postprocess` is set, in which case it yields rows the postprocess
    function will reduce to violations)."""

    postprocess: Optional[Callable[[List[Tuple]], List[Violation]]] = None


# -- public ----------------------------------------------------------------


def compile_predicate(p: Predicate) -> CompiledRule:
    if p.bound.kind == BoundKind.MONOTONIC:
        return _compile_monotonic(p)
    if p.scope.walk_tensor_summaries:
        return _compile_walk(p)
    if p.scope.tensor_signature:
        return _compile_tensor_signature(p)
    if p.scope.payload_path is not None:
        return _compile_with_path(p)
    return _compile_flat(p)


# -- internals -------------------------------------------------------------

def _per_field_failure(row_src: str, b: Bound, tol_rel: float = 0.0) -> List[str]:
    """Return per-field SQL clause(s) that evaluate True iff THIS field FAILS.

    For multi-field bounds we OR these together at the call site to get a
    "any-field-fails" predicate, which matches the violation semantics.
    """
    out: List[str] = []
    for f in b.fields:
        ext = extract(row_src, f)
        if b.kind == BoundKind.BOUND:
            # `bound.op` describes the INVARIANT direction (what must hold);
            # failure clause is its negation.
            for cond in b.all_conditions:
                if cond.value_is_field:
                    rhs_raw = cast_num(extract(row_src, cond.value))
                else:
                    rhs_raw = (repr(cond.value) if isinstance(cond.value, str)
                               else str(cond.value))
                if tol_rel > 0:
                    if cond.op in ("<", "<="):
                        rhs = f"({rhs_raw} * {1.0 + tol_rel})"
                    elif cond.op in (">", ">="):
                        rhs = f"({rhs_raw} * {1.0 - tol_rel})"
                    else:
                        rhs = rhs_raw
                else:
                    rhs = rhs_raw
                out.append(f"NOT ({cast_num(ext)} {cond.op} {rhs})")
        elif b.kind == BoundKind.EQUALITY:
            # invariant: field == value; failure: field != value (and not null-as-pass)
            lit = to_json_literal(b.value)
            # Treat NULL/missing as "no info" → don't flag (matches Python rule)
            out.append(f"({ext} IS NOT NULL AND {ext} != {lit})")
        elif b.kind == BoundKind.PRESENT:
            out.append(f"({ext} IS NULL)")
        elif b.kind == BoundKind.MONOTONIC:
            # Window-based; emitted by caller, not per-field
            raise NotImplementedError("MONOTONIC compiled at outer level")
        else:
            raise ValueError(f"unsupported bound kind: {b.kind}")
    return out


def _compile_flat(p: Predicate) -> CompiledRule:
    """Predicate with no payload_path expansion: payload IS the row.

    Precondition.expr may reference `payload` directly.
    """
    hp = hookpoint_filter(p.scope.hookpoints)
    tol_rel = (p.tolerance.rel or 0.0) if p.tolerance else 0.0
    failures = _per_field_failure("payload", p.bound, tol_rel)
    failure = " OR ".join(failures)

    pre = ""
    if p.precondition is not None:
        pre = f"\n   AND ({p.precondition.expr})"

    sql = (
        f"SELECT event_id, payload\n"
        f"  FROM events\n"
        f" WHERE {hp}{pre}\n"
        f"   AND ({failure})"
    )
    return CompiledRule(predicate_id=p.id, sql=sql)


def _compile_with_path(p: Predicate) -> CompiledRule:
    """Predicate with payload_path: build a CTE so that `payload`
    (parent event JSON) and `row_payload` (per-list-element or projected
    object) are clean named columns. Precondition.expr can reference
    either name directly without alias hacks."""
    path = p.scope.payload_path or ""
    hp = hookpoint_filter(p.scope.hookpoints)
    tol_rel = (p.tolerance.rel or 0.0) if p.tolerance else 0.0

    if path.endswith("[*]"):
        list_path = path[:-3]  # strip [*]
        list_path_inner = list_path[2:] if list_path.startswith("$.") else list_path
        list_extract = extract("e.payload", list_path_inner)
        cte = (
            f"WITH expanded AS (\n"
            f"  SELECT e.event_id AS event_id, e.payload AS payload, "
            f"t.value AS row_payload\n"
            f"    FROM events e, json_each({list_extract}) t\n"
            f"   WHERE e.{hp}\n"
            f")"
        )
    else:
        obj_path = path[2:] if path.startswith("$.") else path
        row_src = extract("payload", obj_path)
        cte = (
            f"WITH expanded AS (\n"
            f"  SELECT event_id, payload, {row_src} AS row_payload\n"
            f"    FROM events\n"
            f"   WHERE {hp}\n"
            f")"
        )

    failures = _per_field_failure("row_payload", p.bound, tol_rel)
    failure = " OR ".join(failures)
    pre = ""
    if p.precondition is not None:
        pre = f"\n   AND ({p.precondition.expr})"
    sql = (
        f"{cte}\n"
        f"SELECT event_id, row_payload\n"
        f"  FROM expanded\n"
        f" WHERE row_payload IS NOT NULL{pre}\n"
        f"   AND ({failure})"
    )
    return CompiledRule(predicate_id=p.id, sql=sql)


def _compile_walk(p: Predicate) -> CompiledRule:
    """walk_tensor_summaries: fetch matching events, walk payload Python-side."""
    hp = hookpoint_filter(p.scope.hookpoints)
    sql = (
        f"SELECT event_id, hookpoint, payload\n"
        f"  FROM events\n"
        f" WHERE {hp}"
    )
    bound = p.bound

    def walker(rows: List[Tuple]) -> List[Violation]:
        out: List[Violation] = []
        for event_id, hp_name, payload_str in rows:
            try:
                payload = json.loads(payload_str)
            except Exception:  # noqa: BLE001
                continue
            for ts in _iter_tensor_summaries(payload):
                if not _check_summary(ts, bound):
                    continue
                out.append(Violation(event_id=event_id, detail={
                    "hookpoint": hp_name,
                    "field_values": {f: ts.get(f) for f in bound.fields},
                }))
                break  # one violation per event is enough
        return out

    return CompiledRule(predicate_id=p.id, sql=sql, postprocess=walker)


def _compile_monotonic(p: Predicate) -> CompiledRule:
    """MONOTONIC: field value must strictly increase across event_id-ordered
    events at this hookpoint. Used by T0-optim-step-counter-monotonic."""
    if isinstance(p.bound.field, list):
        raise ValueError(
            f"predicate {p.id}: MONOTONIC bound takes a single field, "
            f"not a list")
    field_name = p.bound.field
    hp = hookpoint_filter(p.scope.hookpoints)
    ext = extract("payload", field_name)
    sql = (
        f"WITH ordered AS (\n"
        f"  SELECT event_id, payload, {cast_num(ext)} AS v,\n"
        f"         ROW_NUMBER() OVER (ORDER BY event_id) AS rn\n"
        f"    FROM events\n"
        f"   WHERE {hp} AND {ext} IS NOT NULL\n"
        f")\n"
        f"SELECT a.event_id, a.payload\n"
        f"  FROM ordered a JOIN ordered b ON a.rn = b.rn + 1\n"
        f" WHERE NOT (a.v > b.v)"
    )
    return CompiledRule(predicate_id=p.id, sql=sql)


def _compile_tensor_signature(p: Predicate) -> CompiledRule:
    """tensor_signature: project payload_path (or whole payload) into a
    tensor-summary view; expose derived fields rms / n_rows / one_hot;
    apply bound on those derived names. Python postprocess so we don't
    fight DuckDB's JSON array math.
    """
    hp = hookpoint_filter(p.scope.hookpoints)
    pre = ""
    if p.precondition is not None:
        pre = f"\n   AND ({p.precondition.expr})"
    sql = (
        f"SELECT event_id, payload\n"
        f"  FROM events\n"
        f" WHERE {hp}{pre}"
    )
    bound = p.bound
    payload_path = p.scope.payload_path  # e.g. '$.output' or None

    def postprocess(rows: List[Tuple]) -> List[Violation]:
        out: List[Violation] = []
        for event_id, payload_str in rows:
            try:
                payload = json.loads(payload_str)
            except Exception:  # noqa: BLE001
                continue
            ts = _resolve_path(payload, payload_path) if payload_path else payload
            if not isinstance(ts, dict):
                continue
            sig = _tensor_signature(ts)
            if sig is None:
                continue
            for f in bound.fields:
                if f not in sig:
                    continue
                v = sig[f]
                if _check_signature_field(v, bound):
                    out.append(Violation(event_id=event_id, detail={
                        "field": f, "value": v,
                        "shape": ts.get("shape")}))
                    break
        return out

    return CompiledRule(predicate_id=p.id, sql=sql, postprocess=postprocess)


def _resolve_path(obj: Any, path: str) -> Any:
    """Walk a payload dict by '$.a.b.c' path. Returns None if any step misses."""
    if not path or not path.startswith("$."):
        return obj
    cur = obj
    for part in path[2:].split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _tensor_signature(ts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build derived fields from a tensor summary."""
    if "shape" not in ts:
        return None
    shape = ts["shape"]
    if not isinstance(shape, list) or not shape:
        return None
    try:
        numel = 1
        for d in shape:
            numel *= int(d)
        n_rows = 1
        for d in shape[:-1]:
            n_rows *= int(d)
    except (TypeError, ValueError):
        return None
    out: Dict[str, Any] = dict(ts)
    out["numel"] = numel
    out["n_rows"] = n_rows
    l2 = ts.get("l2_norm")
    if isinstance(l2, (int, float)) and numel > 0:
        out["rms"] = float(l2) / math.sqrt(numel)
    amax = ts.get("abs_max") or ts.get("max")
    if isinstance(l2, (int, float)) and isinstance(amax, (int, float)) \
            and n_rows > 0:
        out["one_hot"] = (
            abs(amax - 1.0) < 1e-3
            and abs(l2 * l2 - n_rows) < n_rows * 0.05
        )
    return out


def _check_signature_field(v: Any, bound: Bound) -> bool:
    """True iff the value FAILS the bound (= violation)."""
    if bound.kind == BoundKind.EQUALITY:
        return v != bound.value
    if bound.kind == BoundKind.BOUND:
        for c in bound.all_conditions:
            try:
                ok = eval(  # noqa: S307
                    f"a {c.op} b",
                    {"a": v, "b": c.value}, {})
            except Exception:  # noqa: BLE001
                return False
            if not ok:
                return True
        return False
    if bound.kind == BoundKind.PRESENT:
        return v is None
    return False


def _iter_tensor_summaries(obj: Any):
    """Yield every nested dict that has both `dtype` and `shape` keys
    (matches the Python `_walk_tensor_summaries` helper)."""
    if isinstance(obj, dict):
        if "dtype" in obj and "shape" in obj:
            yield obj
        for v in obj.values():
            yield from _iter_tensor_summaries(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensor_summaries(v)


def _check_summary(ts: Dict[str, Any], bound: Bound) -> bool:
    """Return True iff this summary FAILS the bound (= violation)."""
    for f in bound.fields:
        if f not in ts:
            continue  # missing → "no info"; matches Python rule
        v = ts[f]
        if bound.kind == BoundKind.EQUALITY:
            if v != bound.value:
                return True
        elif bound.kind == BoundKind.BOUND:
            assert bound.op is not None and bound.value is not None
            try:
                ok = eval(  # noqa: S307 — controlled op
                    f"a {bound.op} b", {"a": v, "b": bound.value}, {})
            except Exception:  # noqa: BLE001
                continue
            if not ok:
                return True
        elif bound.kind == BoundKind.PRESENT:
            if v is None:
                return True
    return False
