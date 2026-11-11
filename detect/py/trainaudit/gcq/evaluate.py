"""Run compiled guarded constraints on a DuckDB connection that has the views installed."""
import copy
import time
from typing import Any, Dict, List, Tuple

from .compile import CompiledRule, compile_spec
from .spec import Spec


def compile_any(spec: Spec, disable: Tuple[str, ...] = ()) -> List[CompiledRule]:
    """A spec with "union": [branch, ...] is the union of its branches' violation sets
    (a disjunctive premise over different relations). Each branch is a full spec."""
    if spec.get("union"):
        out = []
        for i, b in enumerate(spec.get("union")):
            raw = copy.deepcopy(b)
            raw.setdefault("id", f"{spec.id}#{i}")
            raw.setdefault("profile", spec.profile)
            out.append(compile_spec(Spec(raw, path=spec.path), disable))
        return out
    return [compile_spec(spec, disable)]


def run_compiled(con, crs: List[CompiledRule], with_count: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    for cr in crs:
        for q in cr.applicable_sql:
            if not con.execute(q).fetchone()[0]:
                return dict(status="not_applicable", violated=None, witnesses=[], count=None,
                            seconds=time.time() - t0)
        for q in cr.specialize_sql:
            if not con.execute(q).fetchone()[0]:
                return dict(status="dropped_under_tau", violated=None, witnesses=[], count=None,
                            seconds=time.time() - t0)
    wit = []
    count = 0
    for cr in crs:
        rows = con.execute(cr.sql).fetchall()
        if cr.form == "aggregate":   # group key only (the aggregate values follow it)
            rows = [r[:len(cr.witness)] for r in rows]
        wit.extend(rows)
        if with_count:
            count += con.execute(cr.count_sql).fetchone()[0]
    wit = sorted(set(wit), key=lambda r: tuple((x is None, x) for x in r))
    return dict(status="ok", violated=bool(wit), witnesses=wit,
                count=count if with_count else None, seconds=time.time() - t0)


def fragment_status(crs: List[CompiledRule]) -> Dict[str, Any]:
    reasons = sorted({r for cr in crs for r in cr.fragment_reasons})
    windows = [cr.window for cr in crs]
    return dict(in_fragment=all(cr.in_fragment for cr in crs), reasons=reasons,
                window=None if any(w is None for w in windows) else max(windows),
                window_kind=sorted({cr.window_kind for cr in crs}),
                k=max(cr.k for cr in crs), k_meta=max(cr.k_meta for cr in crs),
                forms=sorted({cr.form for cr in crs}),
                extensions=sorted({e for cr in crs for e in cr.extensions}),
                hooks=sorted({h for cr in crs for h in cr.hooks}),
                class_rewrite=any(cr.class_sql for cr in crs))
