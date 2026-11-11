"""Declarative guarded-constraint spec (JSON).

A spec is one guarded constraint r = (x̄, π_phase, π_topo, π_schema) of Definition 1, or an
aggregate guarded constraint r = (x, π_phase, π_topo, g, α, θ).

Top-level keys
  id            rule id (frozen rule id for translations)
  rule_file     source of the translated rule (documentation only)
  profile       "trainaudit" (views.py) or "legacy_megatron" (legacy_views.py)
  form          "tuple" | "aggregate"
  vars          [{"name": "t1", "rel": "State"|"Event"|"Obj"|"Config"|"OptGroup"|"Topo",
                  "unnest": "<json path of a list in args>"   (optional, extension E1)}]
                1 or 2 variables over State/Event (k <= 2); Meta variables do not count towards k
                for the window but are reported.
  phase         π_phase:
                  hooks   {var: [hookpoint, ...]}         selection atoms on hook
                  select  <expr>                           further selection atoms on step/micro/hook
                  steps   [{"var": v, "base": u, "offset": c}]   atoms v.step = u.step + c
                  micro   [{"var": v, "base": u}]          atoms v.micro = u.micro
                  eq      [["t1.obj", "t2.obj"], ...]      object identity between variables
                  succ    {"var": v, "base": u, "series": <expr on one var>}  extension E2
                          (v is the next row after u in seq order among rows satisfying series)
  topo          π_topo: {"dims": [...], "atoms": [
                  {"grp": d, "vars": [u, v], "ordered": bool},   grp_d(u, v)
                  {"repl": d, "var": u, "specialize": bool}]}    repl_d(u)
                d = "gathered" in the trainaudit profile (the replica group that the collector
                resolved and gathered in-hook); d in {DP, TP, PP, CP} in the legacy profile.
  pre           <expr> applicability premise (selection on non-phase attributes). In the
                paper's terms it is part of π_schema: ps' = (NOT pre) OR ps.
  schema        <expr> π_schema (tuple form)
  group_by      [attr, ...]                      (aggregate form)
  aggregates    {"name": {"fn": count|count_distinct|sum|min|max, "arg": <expr>|null,
                          "filter": <expr>|null}}  (one aggregate = paper's form; more = ext. E4)
  theta         <expr> over {"agg": name}        aggregate form: violation iff NOT theta
  weight        {"agg": name}                    optional: what a violating group counts as
  meta_guard    <expr> over Meta variables (aggregate form; compiled as EXISTS)
  witness       [attr, ...]                      columns returned per violation
  applicable    legacy only: conditions under which the rule is evaluated at all (else None)
  extensions    list of extension tags the spec needs (checked against the compiler's view)
  fragment      "in" | "outside": the author's claim, re-checked by the compiler
  notes         free text: line-by-line correspondence with the Python rule

Expressions (JSON):
  literals       numbers, true/false/null, {"str": "..."}
  attributes     {"attr": "t.col"} or {"attr": "t.jsoncol.path.to.key", "type": "double"|"bigint"|
                 "varchar"|"bool"|"json", "missing": <literal>}
  operators      {"op": eq|ne|lt|le|gt|ge|is_distinct|not_distinct|and|or|not|add|sub|mul|div|
                  in|not_in|like|contains|is_null|is_not_null|case, "args": [...]}
                 numeric comparisons follow IEEE-754 (Python): NaN compares false, NaN != x true
  functions      {"fn": abs|sqrt|isnan|isinf|coalesce|greatest|least|trunc|replace|lower|
                  truthy|is_true|is_false|jtype|has_key|numel|prod_prefix|json_len, "args": [...]}
  aggregate ref  {"agg": "name"}
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

KNOWN_KEYS = {"id", "rule_file", "profile", "form", "vars", "phase", "topo", "pre", "schema",
              "group_by", "aggregates", "theta", "weight", "meta_guard", "witness", "applicable",
              "extensions", "fragment", "notes", "python_semantics", "evidence", "union",
              "variants", "count_mode"}


class SpecError(ValueError):
    pass


@dataclass
class Spec:
    raw: Dict[str, Any]
    path: str = ""
    id: str = field(init=False)

    def __post_init__(self):
        r = self.raw
        unknown = set(r) - KNOWN_KEYS
        if unknown:
            raise SpecError(f"{self.path}: unknown keys {sorted(unknown)}")
        for k in ("id", "form", "vars"):
            if k not in r:
                raise SpecError(f"{self.path}: missing key {k}")
        if r["form"] not in ("tuple", "aggregate"):
            raise SpecError(f"{self.path}: form must be tuple|aggregate")
        names = [v["name"] for v in r["vars"]]
        if len(set(names)) != len(names):
            raise SpecError(f"{self.path}: duplicate variable names")
        if r["form"] == "tuple" and "schema" not in r and not r.get("union"):
            raise SpecError(f"{self.path}: tuple form needs schema")
        if r["form"] == "aggregate":
            for k in ("group_by", "aggregates", "theta"):
                if k not in r:
                    raise SpecError(f"{self.path}: aggregate form needs {k}")
        self.id = r["id"]

    def get(self, k, d=None):
        return self.raw.get(k, d)

    @property
    def profile(self) -> str:
        return self.raw.get("profile", "trainaudit")


def load_spec(path: str) -> Spec:
    with open(path) as f:
        return Spec(json.load(f), path=path)


def load_specs(directory: str) -> List[Spec]:
    out = []
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(".json"):
            out.append(load_spec(os.path.join(directory, fn)))
    return out
