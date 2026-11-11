"""Rule form for E12 (BENCH.md A12): step-aligned aggregate (extension E6). New module; compile.py,
spec.py and forms_ext.py are unchanged and reused.

E6  aggregate over rows of DIFFERENT steps aligned to one step. An aggregate rule groups the rows
    of one State variable; with w=1 the group key contains t.step. Gather consistency needs, per
    object, the gathered copy at step s and the at-rest shards written at the end of step s-1 in
    the same group. E6 adds a derived attribute astep = step + align[hook] (align: {hook: offset},
    default 0) and groups by t.astep:

        SELECT g, α FROM (SELECT s.*, s.step + CASE s.hook WHEN h THEN c ... ELSE 0 END AS astep
                          FROM State s WHERE s.hook IN H) t
        GROUP BY g HAVING NOT θ                       (g contains t.astep)

    The window is w = 1 + max(align) - min(0, min(align)) (w = 2 for a one-step offset): a
    witness at astep s depends only on rows of steps s-1 and s. Spec (JSON):
      {"id", "profile", "form": "aggregate_aligned", "hooks": [...], "align": {hook: int},
       "group_by": ["t.astep", ...], "aggregates": {...as aggregate form...}, "theta": <expr>,
       "witness": [...], "evidence": {...}, "notes": "..."}
    Outside the base fragment (aggregate keyed by a derived step): reported as "E6-aligned-step".

`load_any_v3(path_or_dict)` returns an AlignedSpec, an ExtSpec (completeness) or a Spec;
`compile_rule_v3(spec)` dispatches.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import compile as C
from . import forms_ext as FX
from .spec import Spec, SpecError

ALIGNED_KEYS = {"id", "profile", "form", "hooks", "align", "group_by", "aggregates", "theta",
                "witness", "evidence", "notes", "extensions"}


@dataclass
class AlignedSpec:
    raw: Dict[str, Any]
    path: str = ""
    id: str = field(init=False)

    def __post_init__(self):
        r = self.raw
        unknown = set(r) - ALIGNED_KEYS
        if unknown:
            raise SpecError(f"{self.path}: unknown keys {sorted(unknown)}")
        for k in ("id", "hooks", "group_by", "aggregates", "theta", "witness"):
            if k not in r:
                raise SpecError(f"{self.path}: missing key {k}")
        if "t.astep" not in r["group_by"]:
            raise SpecError(f"{self.path}: group_by must contain t.astep")
        self.id = r["id"]

    def get(self, k, d=None):
        return self.raw.get(k, d)

    @property
    def profile(self):
        return self.raw.get("profile", "layout_v3")


def _from_raw(raw: Dict[str, Any], path: str = ""):
    if raw.get("form") == "aggregate_aligned":
        return AlignedSpec(raw, path=path)
    if raw.get("form") in FX.EXT_FORMS:
        return FX.ExtSpec(raw, path=path)
    return Spec(raw, path=path)


def load_any_v3(src):
    if isinstance(src, dict):
        return _from_raw(src)
    with open(src) as f:
        return _from_raw(json.load(f), path=src)


class _AlignedProfile(C.Profile):
    def __init__(self, base):
        self.base = base
        self.name = base.name
        self.meta = base.meta
        self.relations = dict(base.relations)
        self.relations["StateA"] = dict(base.relations["State"], astep="bigint")

    def rel_sql(self, rel):
        return self.base.rel_sql(rel)


def compile_aligned(spec: AlignedSpec) -> C.CompiledRule:
    base = C.PROFILES[spec.profile]
    prof = _AlignedProfile(base)
    ctx = C.Ctx(Spec({"id": spec.id, "form": "tuple", "vars": [{"name": "t", "rel": "StateA"}],
                      "schema": True}), prof)
    hooks = list(spec.get("hooks"))
    align = {h: int(c) for h, c in (spec.get("align") or {}).items()}
    case = "CASE " + " ".join(f"WHEN s.hook = {C._sqlstr(h)} THEN {c}" for h, c in sorted(align.items())) \
        + " ELSE 0 END" if align else "0"
    hl = ", ".join(C._sqlstr(h) for h in hooks)
    sub = (f"(SELECT s.*, s.step + {case} AS astep FROM {base.rel_sql('State')} s "
           f"WHERE s.hook IN ({hl}))")
    for name, a in spec.get("aggregates").items():
        fn = a["fn"]
        arg = C.expr(a["arg"], ctx)[0] if a.get("arg") is not None else "*"
        sqlfn = {"count": "COUNT", "count_distinct": "COUNT", "sum": "SUM", "min": "MIN", "max": "MAX"}[fn]
        inner = f"DISTINCT {arg}" if fn == "count_distinct" else arg
        flt = f" FILTER (WHERE {C.expr(a['filter'], ctx)[0]})" if a.get("filter") is not None else ""
        ctx.aggs[name] = f"{sqlfn}({inner}){flt}"
        ctx.agg_types[name] = "bigint" if fn in ("count", "count_distinct") else "double"
    theta = C.expr(spec.get("theta"), ctx)[0]
    gcols = [C.expr({"attr": g}, ctx)[0] for g in spec.get("group_by")]
    agg_sel = ", ".join(f"{s} AS {n}" for n, s in ctx.aggs.items())
    body = f"FROM {sub} t GROUP BY {', '.join(gcols)} HAVING NOT ({theta})"
    sql = f"SELECT {', '.join(g + f' AS g{i}' for i, g in enumerate(gcols))}, {agg_sel} {body}"
    count_sql = f"SELECT COUNT(*) FROM (SELECT 1 {body})"
    offs = list(align.values()) + [0]
    window = 1 + max(offs) - min(offs)
    return C.CompiledRule(id=spec.id, form="aggregate", profile=spec.profile, k=1, k_meta=0,
                          window=window, window_kind="step-aligned", hooks=sorted(hooks),
                          monotone=False, extensions=["E4-multi-aggregate", "E6-aligned-step"],
                          in_fragment=False,
                          fragment_reasons=["needs extension(s) E4-multi-aggregate, E6-aligned-step"],
                          sql=sql, count_sql=count_sql, witness=list(spec.get("witness")))


def compile_rule_v3(spec) -> List[C.CompiledRule]:
    if isinstance(spec, AlignedSpec):
        return [compile_aligned(spec)]
    return FX.compile_rule(spec)
