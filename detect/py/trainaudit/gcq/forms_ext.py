"""Extended rule forms for GCQ (E9 / BENCH.md A10). New module; compile.py and spec.py unchanged.

E5  completeness (domain anti-join). The base fragment has no way to detect a MISSING tuple: a
    tuple-level rule quantifies over tuples that exist, and an aggregate rule over groups that
    exist. A completeness constraint states that every element of a domain has a matching tuple:

        forall d in Dom:  exists t in R:  match(d, t)          V = Dom  ANTI-JOIN  R

    Dom is a conjunctive query over Meta relations (and the step domain), R a selection over one
    State/Event relation. Spec (JSON):
      {"id", "profile", "form": "completeness",
       "domain":  {"vars": [{"name", "rel"}, ...], "where": <expr over domain vars> | null},
       "target":  {"var": "t", "rel": "State", "hooks": [hookpoint, ...], "where": <expr> | null},
       "match":   [["d.rank", "t.rank"], ["d.obj", "t.obj"], ["s.step", "t.step"]],
       "witness": ["d.rank", "d.obj", "s.step"],
       "evidence": {...}, "notes": "..."}
    Compiled SQL:
      SELECT DISTINCT <witness> FROM <domain vars> WHERE <domain where>
        AND NOT EXISTS (SELECT 1 FROM R t WHERE t.hook IN (...) AND <target where> AND <match>)
    The violation set is not monotone under insertion of R tuples (a late tuple removes a
    violation), but it is monotone in the domain and, for a step domain closed at step s, final
    once step s is complete; the compiler reports it outside the base fragment ("E5-domain-antijoin").

The averaging-consistency rule of E9 (post-sync value vs. the mean of the pre-sync values over a
group) does NOT need a new form: it is an aggregate rule with several FILTERed aggregates, which
compile.py already supports as extension E4-multi-aggregate (sum/count of pre-sync rows, min/max
of post-sync rows in one group; theta compares them).

`load_any(path)` returns a Spec (tuple/aggregate, compiled by compile.py) or an ExtSpec
(completeness, compiled here); `compile_rule(spec, disable)` dispatches.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from . import compile as C
from .spec import Spec, SpecError

EXT_FORMS = {"completeness"}
EXT_KEYS = {"id", "profile", "form", "domain", "target", "match", "witness", "evidence", "notes",
            "rule_file", "fragment", "extensions"}


@dataclass
class ExtSpec:
    raw: Dict[str, Any]
    path: str = ""
    id: str = field(init=False)

    def __post_init__(self):
        r = self.raw
        unknown = set(r) - EXT_KEYS
        if unknown:
            raise SpecError(f"{self.path}: unknown keys {sorted(unknown)}")
        if r.get("form") not in EXT_FORMS:
            raise SpecError(f"{self.path}: ExtSpec form must be one of {sorted(EXT_FORMS)}")
        for k in ("id", "domain", "target", "match", "witness"):
            if k not in r:
                raise SpecError(f"{self.path}: missing key {k}")
        self.id = r["id"]

    def get(self, k, d=None):
        return self.raw.get(k, d)

    @property
    def profile(self):
        return self.raw.get("profile", "trainaudit")


def load_any(path: str):
    with open(path) as f:
        raw = json.load(f)
    if raw.get("form") in EXT_FORMS:
        return ExtSpec(raw, path=path)
    return Spec(raw, path=path)


def _ctx(spec: ExtSpec, prof, vars_):
    return C.Ctx(Spec({"id": spec.id, "form": "tuple", "vars": vars_, "schema": True}), prof)


def compile_completeness(spec: ExtSpec) -> C.CompiledRule:
    prof = C.PROFILES[spec.profile]
    dom = spec.get("domain")
    tgt = spec.get("target")
    tv = tgt["var"]
    vars_ = list(dom["vars"]) + [{"name": tv, "rel": tgt["rel"]}]
    for v in dom["vars"]:
        if v["rel"] not in prof.meta:
            raise SpecError(f"{spec.id}: domain variable {v['name']} must range over Meta, "
                            f"not {v['rel']}")
    ctx = _ctx(spec, prof, vars_)
    dfrom = ", ".join(f"{prof.rel_sql(v['rel'])} {v['name']}" for v in dom["vars"])
    dwhere = [C.expr(dom["where"], ctx)[0]] if dom.get("where") is not None else []
    inner = []
    if tgt.get("hooks"):
        inner.append(f"{tv}.{prof.hook_col} IN (" + ", ".join(C._sqlstr(h) for h in tgt["hooks"]) + ")")
    if tgt.get("where") is not None:
        inner.append(C.expr(tgt["where"], ctx)[0])
    for a, b in spec.get("match"):
        inner.append(f"{C.expr({'attr': a}, ctx)[0]} = {C.expr({'attr': b}, ctx)[0]}")
    nexists = (f"NOT EXISTS (SELECT 1 FROM {prof.rel_sql(tgt['rel'])} {tv} WHERE "
               + " AND ".join(inner or ["TRUE"]) + ")")
    where = " AND ".join(dwhere + [nexists])
    witness = spec.get("witness")
    wsql = ", ".join(C.expr({"attr": w}, ctx)[0] + f" AS w{i}" for i, w in enumerate(witness))
    body = f"FROM {dfrom} WHERE {where}"
    sql = f"SELECT DISTINCT {wsql} {body}"
    count_sql = f"SELECT COUNT(*) FROM (SELECT DISTINCT {wsql} {body})"
    hooks = sorted(tgt.get("hooks") or [])
    return C.CompiledRule(id=spec.id, form="completeness", profile=spec.profile, k=1,
                          k_meta=len(dom["vars"]), window=1, window_kind="step", hooks=hooks,
                          monotone=False, extensions=["E5-domain-antijoin"], in_fragment=False,
                          fragment_reasons=["needs extension(s) E5-domain-antijoin"],
                          sql=sql, count_sql=count_sql, witness=list(witness))


def compile_rule(spec, disable: Tuple[str, ...] = ()) -> List[C.CompiledRule]:
    if isinstance(spec, ExtSpec):
        return [compile_completeness(spec)]
    from .evaluate import compile_any
    return compile_any(spec, disable)
