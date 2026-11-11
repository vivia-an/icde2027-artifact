"""Execute compiled DSL rules against a TraceStore. Produces RuleResult-shaped
output that's drop-in compatible with `verifier.run_rules` for diffing."""
from __future__ import annotations

import json
from typing import Any, List

from ..rules.base import RuleResult
from ..store import TraceStore
from .compiler import CompiledRule, Violation, compile_predicate
from .loader import load_predicates_dir
from .predicate import Predicate


def run_compiled_one(conn, compiled: CompiledRule) -> RuleResult:
    rows = conn.execute(compiled.sql).fetchall()
    if compiled.postprocess is None:
        violations: List[Violation] = []
        for r in rows:
            event_id = r[0]
            detail: dict = {}
            if len(r) >= 2 and r[1] is not None:
                # row_payload or full payload — store snippet for evidence
                try:
                    detail["row"] = json.loads(r[1]) if isinstance(r[1], str) else r[1]
                except Exception:  # noqa: BLE001
                    pass
            violations.append(Violation(event_id=event_id, detail=detail))
    else:
        violations = compiled.postprocess(rows)

    if violations:
        return RuleResult(
            rule_id=compiled.predicate_id,
            violated=True,
            message=f"{len(violations)} DSL violation(s)",
            evidence={
                "n_violations": len(violations),
                "sample": [{"event_id": v.event_id, **v.detail}
                           for v in violations[:3]],
            },
        )
    return RuleResult(
        rule_id=compiled.predicate_id,
        violated=False,
        message="DSL rule satisfied",
    )


def run_dsl_rules(store: TraceStore,
                  predicates: List[Predicate]) -> List[RuleResult]:
    """Compile + run each predicate, return list of RuleResult."""
    store.flush()
    results: List[RuleResult] = []
    for p in predicates:
        try:
            compiled = compile_predicate(p)
        except Exception as e:  # noqa: BLE001
            results.append(RuleResult(
                rule_id=p.id, violated=False,
                message=f"compile error: {type(e).__name__}: {e}"))
            continue
        try:
            results.append(run_compiled_one(store.conn, compiled))
        except Exception as e:  # noqa: BLE001
            results.append(RuleResult(
                rule_id=p.id, violated=False,
                message=f"runtime error: {type(e).__name__}: {e}"))
    return results


def violation_event_ids(store: TraceStore, p: Predicate) -> List[int]:
    """Helper used by A2 equivalence test — return raw violation event_id list."""
    store.flush()
    compiled = compile_predicate(p)
    rows = store.conn.execute(compiled.sql).fetchall()
    if compiled.postprocess is None:
        return [r[0] for r in rows]
    return [v.event_id for v in compiled.postprocess(rows)]
