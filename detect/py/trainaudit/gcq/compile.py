"""Compiler: guarded-constraint spec -> SQL over the logical views.

  tuple-level rule:   SELECT DISTINCT <witness> FROM x̄ WHERE π_phase AND π_topo AND pre AND NOT π_schema
  aggregate rule:     SELECT g, α FROM x WHERE π_phase AND π_topo AND pre [AND EXISTS meta_guard]
                      GROUP BY g HAVING NOT θ
  replica-class rewrite (Proposition 1): a binary tuple rule whose π_schema is one equality
                      t1.a = t2.a, whose π_topo consists of grp_d/repl_d atoms and whose other
                      binary atoms are equalities is also compiled to
                      SELECT key, cls FROM t1 WHERE unary(π_phase) AND repl(t1)
                      GROUP BY (equality attributes, cls) HAVING COUNT(DISTINCT a) > 1

The compiler also computes the window w(r) (largest step offset related by π_phase plus one;
None = unbounded), the hookpoint index of the rule, monotonicity, the extensions the spec needs,
and whether the rule is inside the GCQ fragment (k <= 2, w <= 2, monotone tuple rule or
aggregate grouped by step, no extension).
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import views as V
from .spec import Spec, SpecError

# ---------------------------------------------------------------------------------------------
# profiles


class Profile:
    name = "base"
    relations: Dict[str, Dict[str, str]] = {}
    meta: set = set()
    hook_col = "hook"

    def rel_sql(self, rel: str) -> str:
        return rel

    def grp_sql(self, d, u, v, ordered, ctx) -> str:
        raise SpecError(f"grp_{d} not available in profile {self.name}")

    def repl_sql(self, d, u, ctx) -> str:
        raise SpecError(f"repl_{d} not available in profile {self.name}")

    def class_key(self, d, u, ctx) -> List[str]:
        raise SpecError(f"no class identifier for grp_{d} in profile {self.name}")


class TrainAuditProfile(Profile):
    name = "trainaudit"
    relations = V.RELATIONS
    meta = V.META_RELATIONS

    # Topo.group/coord are never recorded, so grp_DP/TP/PP/EP/CP cannot be evaluated. The only
    # group the trace identifies is the one the collector resolved and gathered in-hook
    # (build_snapshot._resolve_group: the TP group when TP>1 and the parameter is neither
    # TP-sharded nor expert-parallel); its members are the State rows with slot='replica'
    # of one (event, obj, obj_class).
    def grp_sql(self, d, u, v, ordered, ctx):
        if d != "gathered":
            raise SpecError(f"grp_{d}: Topo(rank, dim, group, coord) has no group/coord in the "
                            "trace (list_groups() is never called); only grp_gathered exists")
        cmp = "<" if ordered else "<>"
        return (f"{u}.event_id = {v}.event_id AND {u}.obj = {v}.obj AND {u}.obj_class = {v}.obj_class "
                f"AND {u}.slot = 'replica' AND {v}.slot = 'replica' AND {u}.idx {cmp} {v}.idx")

    def repl_sql(self, d, u, ctx):
        if d != "gathered":
            raise SpecError(f"repl_{d}: not decidable from the trace; use repl_gathered")
        return f"{u}.xr_group_size > 1"

    def class_key(self, d, u, ctx):
        if d != "gathered":
            raise SpecError("only grp_gathered has a class identifier")
        return [f"{u}.event_id", f"{u}.obj_class"]


class LegacyMegatronProfile(Profile):
    name = "legacy_megatron"
    hook_col = "hook"

    @property
    def relations(self):
        from . import legacy_views as L
        return L.RELATIONS

    @property
    def meta(self):
        from . import legacy_views as L
        return L.META_RELATIONS

    def rel_sql(self, rel):
        from . import legacy_views as L
        return L.NAMES[rel]

    def grp_sql(self, d, u, v, ordered, ctx):
        gu, gv = ctx.alias(f"g_{u}_{d}"), ctx.alias(f"g_{v}_{d}")
        ctx.extra_from.append(f"{self.rel_sql('Topo')} {gu}")
        ctx.extra_from.append(f"{self.rel_sql('Topo')} {gv}")
        cmp = "<" if ordered else "<>"
        return (f"{gu}.rank = {u}.rank AND {gu}.dim = '{d}' AND {gv}.rank = {v}.rank AND {gv}.dim = '{d}' "
                f"AND {gu}.grp = {gv}.grp AND {gu}.coord {cmp} {gv}.coord")

    def repl_sql(self, d, u, ctx):
        return (f"EXISTS (SELECT 1 FROM {self.rel_sql('Obj')} o WHERE o.obj = {u}.obj AND o.kind = {u}.kind "
                f"AND NOT list_contains(o.shard_dims, '{d}'))")

    def class_key(self, d, u, ctx):
        # replica class along d = the Topo group id of d (equal coordinates in all other dims)
        return [f"(SELECT gk.grp FROM {self.rel_sql('Topo')} gk WHERE gk.rank = {u}.rank AND gk.dim = '{d}')"]


# E1 (a tuple variable bound to the elements of one list-valued argument) is a view-level
# extension: the same rule over a view that unnests the list has k=1 and the same window.
VIEW_LEVEL_EXT = {"E1-unnest"}

PROFILES = {"trainaudit": TrainAuditProfile(), "legacy_megatron": LegacyMegatronProfile()}

# ---------------------------------------------------------------------------------------------
# expressions

_CMP = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_ARITH = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_JSON_FN = {"truthy": "gcq_truthy", "is_true": "gcq_is_true", "is_false": "gcq_is_false"}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _jpath(keys: List[str]) -> str:
    if not keys:
        return "$"
    return "$" + "".join("." + k if _IDENT.match(k) else '."' + k + '"' for k in keys)


def _sqlstr(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


class Ctx:
    def __init__(self, spec: Spec, profile: Profile):
        self.spec = spec
        self.profile = profile
        self.vars: Dict[str, Dict] = {v["name"]: v for v in spec.get("vars")}
        self.extra_from: List[str] = []
        self.class_from: List[str] = []
        self.used_vars: set = set()
        self._n = 0
        self.aggs: Dict[str, str] = {}
        self.agg_types: Dict[str, str] = {}

    def alias(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    def rel_of(self, var):
        if var not in self.vars:
            raise SpecError(f"{self.spec.id}: unknown variable {var}")
        return self.vars[var]["rel"]


def _attr(node, ctx: Ctx) -> Tuple[str, str, Optional[Tuple[str, str]]]:
    """-> (sql, type, (json_col_sql, path) or None)."""
    ref = node["attr"]
    parts = ref.split(".")
    var, col, rest = parts[0], parts[1], parts[2:]
    ctx.used_vars.add(var)
    rel = ctx.rel_of(var)
    vdef = ctx.vars[var]
    if col == "item" and vdef.get("unnest"):
        col_sql, ctype = f"{var}__u.value", "json"
    else:
        cols = ctx.profile.relations[rel]
        if col not in cols:
            raise SpecError(f"{ctx.spec.id}: {rel} has no column {col}")
        col_sql, ctype = f"{var}.{col}", cols[col]
    if not rest:
        return col_sql, ctype, ((col_sql, "$") if ctype == "json" else None)
    if ctype != "json":
        raise SpecError(f"{ctx.spec.id}: {ref}: path into non-JSON column")
    path = _jpath(rest)
    typ = node.get("type", "json")
    jp = _sqlstr(path)
    if typ == "double":
        s = f"gcq_dbl({col_sql}, {jp})"
    elif typ == "bigint":
        s = f"CAST(gcq_dbl({col_sql}, {jp}) AS BIGINT)"
    elif typ == "varchar":
        s = f"gcq_str({col_sql}, {jp})"
    elif typ == "bool":
        s = f"(CASE WHEN json_type({col_sql}, {jp}) = 'BOOLEAN' THEN json_extract({col_sql}, {jp})::BOOLEAN END)"
    elif typ == "json":
        s = f"json_extract({col_sql}, {jp})"
    else:
        raise SpecError(f"{ctx.spec.id}: bad type {typ}")
    if "missing" in node:
        d, _ = expr(node["missing"], ctx)
        s = f"(CASE WHEN json_type({col_sql}, {jp}) IS NULL THEN {d} ELSE {s} END)"
    return s, typ, (col_sql, path)


def _is_lit(node):
    return not isinstance(node, dict) or "str" in node or "list" in node or "num" in node


def expr(node, ctx: Ctx) -> Tuple[str, str]:
    if node is None:
        return "NULL", "null"
    if isinstance(node, bool):
        return ("TRUE" if node else "FALSE"), "bool"
    if isinstance(node, (int, float)):
        return repr(float(node)) if isinstance(node, float) else str(node), "double"
    if not isinstance(node, dict):
        raise SpecError(f"{ctx.spec.id}: bad expression {node!r}")
    if "str" in node:
        return _sqlstr(node["str"]), "varchar"
    if "num" in node:   # non-JSON float literals: "inf", "-inf", "nan"
        return f"'{node['num']}'::DOUBLE", "double"
    if "attr" in node:
        s, t, _ = _attr(node, ctx)
        return s, t
    if "agg" in node:
        if node["agg"] not in ctx.aggs:
            raise SpecError(f"{ctx.spec.id}: unknown aggregate {node['agg']}")
        return ctx.aggs[node["agg"]], ctx.agg_types.get(node["agg"], "double")
    if "list" in node:
        return "(" + ", ".join(expr(x, ctx)[0] for x in node["list"]) + ")", "list"
    if "op" in node:
        return _op(node, ctx)
    if "fn" in node:
        return _fn(node, ctx)
    raise SpecError(f"{ctx.spec.id}: bad expression {node!r}")


def _op(node, ctx):
    op = node["op"]
    args = node.get("args", [])
    if op in _CMP:
        (a, ta), (b, tb) = expr(args[0], ctx), expr(args[1], ctx)
        base = f"({a} {_CMP[op]} {b})"
        nan = [f"isnan({s})" for s, t, n in ((a, ta, args[0]), (b, tb, args[1]))
               if t == "double" and not _is_lit(n)]
        if not nan:
            return base, "bool"
        if op == "ne":   # IEEE: NaN != x is true
            return f"({base} OR {' OR '.join(nan)})", "bool"
        return f"({base} AND NOT ({' OR '.join(nan)}))", "bool"   # IEEE: NaN compares false
    if op in ("is_distinct", "not_distinct"):
        (a, _), (b, _) = expr(args[0], ctx), expr(args[1], ctx)
        kw = "IS DISTINCT FROM" if op == "is_distinct" else "IS NOT DISTINCT FROM"
        return f"({a} {kw} {b})", "bool"
    if op in ("and", "or"):
        if not args:
            return ("TRUE" if op == "and" else "FALSE"), "bool"
        return "(" + f" {op.upper()} ".join(expr(x, ctx)[0] for x in args) + ")", "bool"
    if op == "not":
        return f"(NOT {expr(args[0], ctx)[0]})", "bool"
    if op in _ARITH:
        (a, _), (b, _) = expr(args[0], ctx), expr(args[1], ctx)
        return f"({a} {_ARITH[op]} {b})", "double"
    if op == "neg":
        return f"(-{expr(args[0], ctx)[0]})", "double"
    if op in ("in", "not_in"):
        a, _ = expr(args[0], ctx)
        lst, _ = expr(args[1], ctx)
        return f"({a} {'IN' if op == 'in' else 'NOT IN'} {lst})", "bool"
    if op == "like":
        return f"({expr(args[0], ctx)[0]} LIKE {expr(args[1], ctx)[0]})", "bool"
    if op == "contains":   # Python `needle in haystack` for strings
        return f"contains({expr(args[0], ctx)[0]}, {expr(args[1], ctx)[0]})", "bool"
    if op == "is_null":
        return f"({expr(args[0], ctx)[0]} IS NULL)", "bool"
    if op == "is_not_null":
        return f"({expr(args[0], ctx)[0]} IS NOT NULL)", "bool"
    if op == "case":
        parts = ["CASE"]
        t = "null"
        for cond, val in node["when"]:
            vs, t = expr(val, ctx)
            parts.append(f"WHEN {expr(cond, ctx)[0]} THEN {vs}")
        if "else" in node:
            parts.append(f"ELSE {expr(node['else'], ctx)[0]}")
        parts.append("END")
        return "(" + " ".join(parts) + ")", t
    raise SpecError(f"{ctx.spec.id}: unknown op {op}")


def _json_arg(node, ctx):
    if not (isinstance(node, dict) and "attr" in node):
        raise SpecError(f"{ctx.spec.id}: JSON function needs an attribute")
    s, t, jp = _attr(dict(node, type="json"), ctx)
    if jp is None:
        return None, s, t
    return jp, s, t


def _fn(node, ctx):
    fn = node["fn"]
    args = node.get("args", [])
    if fn in _JSON_FN:
        jp, s, t = _json_arg(args[0], ctx)
        if jp is None:   # typed column: Python truthiness on the typed value
            if fn == "truthy":
                if t == "bool":
                    return f"COALESCE({s}, FALSE)", "bool"
                if t == "double":
                    return f"COALESCE({s} <> 0, FALSE)", "bool"
                if t == "varchar":
                    return f"COALESCE(length({s}) > 0, FALSE)", "bool"
            if fn in ("is_true", "is_false") and t == "bool":
                return f"COALESCE({s} = {'TRUE' if fn == 'is_true' else 'FALSE'}, FALSE)", "bool"
            raise SpecError(f"{ctx.spec.id}: {fn} on column of type {t}")
        return f"{_JSON_FN[fn]}({jp[0]}, {_sqlstr(jp[1])})", "bool"
    if fn == "jtype":
        jp, s, t = _json_arg(args[0], ctx)
        return f"json_type({jp[0]}, {_sqlstr(jp[1])})", "varchar"
    if fn == "is_none":   # Python: d.get(k) is None  (missing key or JSON null)
        jp, s, t = _json_arg(args[0], ctx)
        q = _sqlstr(jp[1])
        return f"(json_type({jp[0]}, {q}) IS NULL OR json_type({jp[0]}, {q}) = 'NULL')", "bool"
    if fn == "has_key":
        jp, s, t = _json_arg(args[0], ctx)
        return f"(json_type({jp[0]}, {_sqlstr(jp[1])}) IS NOT NULL)", "bool"
    if fn == "json_len":
        jp, s, t = _json_arg(args[0], ctx)
        return (f"(CASE WHEN json_type({jp[0]}, {_sqlstr(jp[1])}) = 'ARRAY' "
                f"THEN json_array_length({jp[0]}, {_sqlstr(jp[1])}) END)"), "double"
    if fn == "numel":
        jp, s, t = _json_arg(args[0], ctx)
        return f"gcq_numel({jp[0]}, {_sqlstr(jp[1])})", "double"
    if fn == "prod_prefix":   # Python: prod(int(d) for d in shape[:-1]) (1 for a 1-d shape)
        jp, s, t = _json_arg(args[0], ctx)
        lst = f"TRY_CAST(json_extract({jp[0]}, {_sqlstr(jp[1])}) AS DOUBLE[])"
        return (f"(CASE WHEN json_type({jp[0]}, {_sqlstr(jp[1])}) = 'ARRAY' THEN "
                f"COALESCE(list_aggregate(list_slice({lst}, 1, len({lst}) - 1), 'product'), 1.0) END)"), "double"
    simple = {"abs": "abs", "sqrt": "sqrt", "isnan": "isnan", "isinf": "isinf", "coalesce": "coalesce",
              "greatest": "greatest", "least": "least", "trunc": "trunc", "replace": "replace",
              "lower": "lower", "length": "length"}
    if fn in simple:
        sargs = [expr(a, ctx) for a in args]
        t = sargs[0][1] if sargs else "double"
        if fn in ("isnan", "isinf"):
            t = "bool"
        if fn in ("replace", "lower"):
            t = "varchar"
        if fn in ("trunc", "abs", "sqrt", "length"):
            t = "double"
        return f"{simple[fn]}(" + ", ".join(s for s, _ in sargs) + ")", t
    raise SpecError(f"{ctx.spec.id}: unknown function {fn}")


# ---------------------------------------------------------------------------------------------
# compiled rule


@dataclass
class CompiledRule:
    id: str
    form: str
    profile: str
    k: int                      # tuple variables over State/Event
    k_meta: int                 # variables over Meta relations
    window: Optional[int]       # w(r); None = unbounded (no step relation between variables)
    window_kind: str            # "step" | "unbounded" | "seq-successor"
    hooks: List[str]
    monotone: bool
    extensions: List[str]
    in_fragment: bool
    fragment_reasons: List[str]
    sql: str                    # violation query: one row per violation witness
    count_sql: str              # |V_r| (tuples / groups)
    class_sql: Optional[str] = None          # Prop. 1 rewrite (violating replica classes)
    pairwise_class_sql: Optional[str] = None  # classes projected from the pairwise query
    pairs_from_classes_sql: Optional[str] = None
    specialize_sql: List[str] = field(default_factory=list)
    applicable_sql: List[str] = field(default_factory=list)
    witness: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return dict(id=self.id, form=self.form, profile=self.profile, k=self.k, k_meta=self.k_meta,
                    window=self.window, window_kind=self.window_kind, hooks=self.hooks,
                    monotone=self.monotone, extensions=self.extensions,
                    in_fragment=self.in_fragment, fragment_reasons=self.fragment_reasons,
                    has_class_rewrite=self.class_sql is not None)


def _window(spec: Spec, ctx: Ctx) -> Tuple[Optional[int], str]:
    data_vars = [v["name"] for v in spec.get("vars") if v["rel"] not in ctx.profile.meta]
    ph = spec.get("phase") or {}
    if spec.get("form") == "aggregate":
        keys = [g if isinstance(g, str) else g.get("attr", "") for g in spec.get("group_by")]
        return (1, "step") if any(k.endswith(".step") for k in keys) else (None, "unbounded")
    if len(data_vars) <= 1:
        return 1, "step"
    if ph.get("succ"):
        return None, "seq-successor"
    # union-find over step atoms with offsets
    adj = {v: [] for v in data_vars}
    for a in ph.get("steps", []):
        adj.setdefault(a["var"], []).append((a["base"], a["offset"]))
        adj.setdefault(a["base"], []).append((a["var"], -a["offset"]))
    root = data_vars[0]
    off = {root: 0}
    stack = [root]
    while stack:
        u = stack.pop()
        for w, c in adj.get(u, []):
            if w not in off:
                off[w] = off[u] - c
                stack.append(w)
    if any(v not in off for v in data_vars):
        return None, "unbounded"
    return max(off.values()) - min(off.values()) + 1, "step"


def _schema_equality(spec: Spec):
    """Return attribute name a if π_schema is exactly t1.a = t2.a, else None."""
    s = spec.get("schema")
    if not (isinstance(s, dict) and s.get("op") == "eq"):
        return None
    a, b = s["args"]
    if not (isinstance(a, dict) and isinstance(b, dict) and "attr" in a and "attr" in b):
        return None
    va, ca = a["attr"].split(".", 1)
    vb, cb = b["attr"].split(".", 1)
    if ca != cb or va == vb or "." in ca:
        return None
    return va, vb, ca


def _refs(node) -> set:
    out = set()
    if isinstance(node, dict):
        if "attr" in node:
            out.add(node["attr"].split(".")[0])
        for v in node.values():
            out |= _refs(v)
    elif isinstance(node, list):
        for v in node:
            out |= _refs(v)
    return out


def compile_spec(spec: Spec, disable: Tuple[str, ...] = ()) -> CompiledRule:
    """`disable` may contain "phase" and/or "topo": atoms tagged {"guard": "phase"|"topo"}
    are dropped (guard ablation, used for the Megatron 2x2)."""
    prof = PROFILES[spec.profile]
    ctx = Ctx(spec, prof)
    vars_ = spec.get("vars")
    data_vars = [v["name"] for v in vars_ if v["rel"] not in prof.meta]
    meta_vars = [v["name"] for v in vars_ if v["rel"] in prof.meta]
    ph = spec.get("phase") or {}
    topo = spec.get("topo") or {}
    ext = set(spec.get("extensions") or [])
    reasons: List[str] = []

    def keep(atom):
        return not (isinstance(atom, dict) and atom.get("guard") in disable)

    # ---- FROM
    from_items = []
    for v in vars_:
        if spec.get("form") == "aggregate" and v["rel"] in prof.meta:
            continue   # Meta variables of aggregate rules are existential guards
        from_items.append(f"{prof.rel_sql(v['rel'])} {v['name']}")
        if v.get("unnest"):
            ext.add("E1-unnest")
            p = _jpath(v["unnest"].split("."))
            from_items.append(f"json_each({v['name']}.args, {_sqlstr(p)}) {v['name']}__u")

    # ---- π_phase
    wh: List[str] = []
    for var, hooks in (ph.get("hooks") or {}).items():
        ctx.used_vars.add(var)
        wh.append(f"{var}.{prof.hook_col} IN (" + ", ".join(_sqlstr(h) for h in hooks) + ")")
    sel_atoms = ph.get("select")
    sel_atoms = [] if sel_atoms is None else (sel_atoms if isinstance(sel_atoms, list) else [sel_atoms])
    for atom in sel_atoms:
        if keep(atom):
            wh.append(expr(atom, ctx)[0])
    for a in ph.get("steps", []):
        wh.append(f"{a['var']}.step = {a['base']}.step + {int(a['offset'])}")
    for a in ph.get("micro", []):
        wh.append(f"{a['var']}.micro = {a['base']}.micro")
    for a, b in ph.get("eq", []):
        wh.append(f"{expr({'attr': a}, ctx)[0]} = {expr({'attr': b}, ctx)[0]}")
    succ = ph.get("succ")
    if succ:
        ext.add("E2-seq-successor")
        u, v = succ["base"], succ["var"]
        ser_u = expr(succ["series"], ctx)[0]
        ser_v = ser_u.replace(f"{u}.", f"{v}.")
        rel = ctx.rel_of(u)
        ser_w = ser_u.replace(f"{u}.", "w.")
        wh.append(f"({ser_u}) AND ({ser_v}) AND {v}.seq = (SELECT min(w.seq) FROM {rel} w "
                  f"WHERE w.seq > {u}.seq AND ({ser_w}))")
    # ---- π_topo
    specialize = []
    for atom in topo.get("atoms", []):
        if not keep(atom):
            continue
        if "grp" in atom:
            wh.append(prof.grp_sql(atom["grp"], atom["vars"][0], atom["vars"][1],
                                   atom.get("ordered", False), ctx))
        elif "repl" in atom:
            wh.append(prof.repl_sql(atom["repl"], atom["var"], ctx))
            if atom.get("specialize"):
                # repl_d over Meta for the object kind the rule addresses: the rule is dropped
                # under τ if no object of that kind is replicated along d
                specialize.append(
                    f"SELECT EXISTS (SELECT 1 FROM {prof.rel_sql('Obj')} o WHERE o.kind = {_sqlstr(atom['kind'])} "
                    f"AND NOT list_contains(o.shard_dims, '{atom['repl']}'))")
        else:
            raise SpecError(f"{spec.id}: bad topo atom {atom}")
    # ---- premise
    if spec.get("pre") is not None:
        wh.append(expr(spec.get("pre"), ctx)[0])

    applicable = []
    for cond in spec.get("applicable") or []:
        if "exists" in cond:
            sub = Ctx(Spec({"id": spec.id, "form": "tuple", "vars": [{"name": "a", "rel": cond.get("rel", "State")}],
                            "schema": True}), prof)
            applicable.append(f"SELECT EXISTS (SELECT 1 FROM {prof.rel_sql(cond.get('rel', 'State'))} a "
                              f"WHERE {expr(cond['exists'], sub)[0]})")
        elif "topo_size_gt" in cond:
            d, n = cond["topo_size_gt"]
            applicable.append(f"SELECT COALESCE((SELECT max(size) FROM {prof.rel_sql('Topo')} WHERE dim = '{d}'), 0) > {n}")
        else:
            raise SpecError(f"{spec.id}: bad applicable {cond}")

    window, wkind = _window(spec, ctx)
    hooks = sorted({h for hs in (ph.get("hooks") or {}).values() for h in hs})
    k = len(data_vars)
    if k > 2:
        reasons.append(f"k={k} > 2")

    if spec.get("form") == "tuple":
        if spec.get("schema") is None:
            raise SpecError(f"{spec.id}: no schema")
        ps = expr(spec.get("schema"), ctx)[0]
        where = " AND ".join(wh) or "TRUE"
        all_from = ", ".join(from_items + ctx.extra_from)
        witness = spec.get("witness") or [f"{v}.event_id" for v in data_vars
                                          if "event_id" in prof.relations[ctx.rel_of(v)]]
        wsql = ", ".join(expr({"attr": w}, ctx)[0] + f" AS w{i}" for i, w in enumerate(witness))
        body = f"FROM {all_from} WHERE {where} AND NOT ({ps})"
        sql = f"SELECT DISTINCT {wsql} {body}"
        keycols = []
        for v in vars_:
            rel = v["rel"]
            for c in ("event_id", "rid", "idx", "obj", "obj_class", "slot", "rank", "framework"):
                if c in prof.relations[rel]:
                    keycols.append(f"{v['name']}.{c}")
            if v.get("unnest"):
                keycols.append(f"{v['name']}__u.key")
        count_sql = f"SELECT COUNT(*) FROM (SELECT {', '.join(keycols)} {body})"
        monotone = True
        cr = CompiledRule(id=spec.id, form="tuple", profile=spec.profile, k=k, k_meta=len(meta_vars),
                          window=window, window_kind=wkind, hooks=hooks, monotone=monotone,
                          extensions=sorted(ext), in_fragment=False, fragment_reasons=reasons,
                          sql=sql, count_sql=count_sql, specialize_sql=specialize,
                          applicable_sql=applicable, witness=witness)
        _replica_rewrite(spec, cr, prof, disable, body)
    else:
        aggs = spec.get("aggregates")
        if len(aggs) > 1:
            ext.add("E4-multi-aggregate")
        for name, a in aggs.items():
            fn = a["fn"]
            arg = expr(a["arg"], ctx)[0] if a.get("arg") is not None else "*"
            sqlfn = {"count": "COUNT", "count_distinct": "COUNT", "sum": "SUM", "min": "MIN",
                     "max": "MAX"}[fn]
            inner = f"DISTINCT {arg}" if fn == "count_distinct" else arg
            f = f" FILTER (WHERE {expr(a['filter'], ctx)[0]})" if a.get("filter") is not None else ""
            ctx.aggs[name] = f"{sqlfn}({inner}){f}"
            ctx.agg_types[name] = "bigint" if fn in ("count", "count_distinct") else "double"
        theta = expr(spec.get("theta"), ctx)[0]
        if spec.get("meta_guard") is not None:
            mfrom = ", ".join(f"{prof.rel_sql(ctx.rel_of(m))} {m}" for m in meta_vars)
            wh.append(f"EXISTS (SELECT 1 FROM {mfrom} WHERE {expr(spec.get('meta_guard'), ctx)[0]})")
        where = " AND ".join(wh) or "TRUE"
        gcols = [expr(g if isinstance(g, dict) else {"attr": g}, ctx)[0] for g in spec.get("group_by")]
        agg_sel = ", ".join(f"{s} AS {n}" for n, s in ctx.aggs.items())
        all_from = ", ".join(from_items + ctx.extra_from)
        body = (f"FROM {all_from} WHERE {where} GROUP BY {', '.join(gcols)} HAVING NOT ({theta})")
        sql = f"SELECT {', '.join(g + f' AS g{i}' for i, g in enumerate(gcols))}, {agg_sel} {body}"
        if spec.get("weight"):
            wexp = ctx.aggs[spec.get("weight")["agg"]]
            count_sql = f"SELECT COALESCE(SUM(w), 0) FROM (SELECT {wexp} AS w {body})"
        else:
            count_sql = f"SELECT COUNT(*) FROM (SELECT 1 {body})"
        monotone = False
        cr = CompiledRule(id=spec.id, form="aggregate", profile=spec.profile, k=k,
                          k_meta=len(meta_vars), window=window, window_kind=wkind, hooks=hooks,
                          monotone=monotone, extensions=sorted(ext), in_fragment=False,
                          fragment_reasons=reasons, sql=sql, count_sql=count_sql,
                          specialize_sql=specialize, applicable_sql=applicable,
                          witness=list(spec.get("group_by")))
    # ---- fragment check
    if window is None:
        reasons.append(f"window unbounded ({wkind})")
    elif window > 2:
        reasons.append(f"w={window} > 2")
    if cr.form == "aggregate" and wkind != "step":
        reasons.append("aggregate group not keyed by step")
    semantic_ext = sorted(e for e in ext if e not in VIEW_LEVEL_EXT)
    if semantic_ext:
        reasons.append("needs extension(s) " + ", ".join(semantic_ext))
    cr.extensions = sorted(ext)
    cr.fragment_reasons = reasons
    cr.in_fragment = not reasons
    return cr


def _replica_rewrite(spec: Spec, cr: CompiledRule, prof: Profile, disable, body: str) -> None:
    """Proposition 1: binary equality under an equivalence guard -> grouped scan."""
    eqa = _schema_equality(spec)
    vars_ = [v["name"] for v in spec.get("vars")]
    if eqa is None or len(vars_) != 2 or cr.k != 2:
        return
    u, v, a = eqa
    ph = spec.get("phase") or {}
    if ph.get("succ") or any(s["offset"] != 0 for s in ph.get("steps", [])):
        return
    if spec.get("pre") is not None:
        return
    hooks = ph.get("hooks") or {}
    if hooks and sorted(hooks.get(u, [])) != sorted(hooks.get(v, [])):
        return
    sel = ph.get("select")
    sel = [] if sel is None else (sel if isinstance(sel, list) else [sel])
    sel = [x for x in sel if not (isinstance(x, dict) and x.get("guard") in disable)]
    if any(_refs(x) - {u} for x in sel):
        return
    atoms = [x for x in (spec.get("topo") or {}).get("atoms", [])
             if not (isinstance(x, dict) and x.get("guard") in disable)]
    grps = [x for x in atoms if "grp" in x]
    repls = [x for x in atoms if "repl" in x]
    if not grps:
        return
    ctx = Ctx(spec, prof)
    wh = []
    if hooks.get(u):
        wh.append(f"{u}.{prof.hook_col} IN (" + ", ".join(_sqlstr(h) for h in hooks[u]) + ")")
    for x in sel:
        wh.append(expr(x, ctx)[0])
    key = []
    for s in ph.get("steps", []):
        key.append(f"{u}.step")
    for m in ph.get("micro", []):
        key.append(f"{u}.micro")
    for x, y in ph.get("eq", []):
        xv, xc = x.split(".", 1)
        key.append(expr({"attr": x if xv == u else y}, ctx)[0])
    for g in grps:
        key += prof.class_key(g["grp"], u, ctx)
        if g["grp"] == "gathered":
            wh.append(f"{u}.slot = 'replica'")
    for r in repls:
        if r["var"] != u:
            return
        wh.append(prof.repl_sql(r["repl"], u, ctx))
    key = list(dict.fromkeys(key))
    frm = f"{prof.rel_sql(spec.get('vars')[0]['rel'])} {u}"
    where = " AND ".join(wh) or "TRUE"
    kal = ", ".join(f"{k} AS k{i}" for i, k in enumerate(key))
    kn = ", ".join(f"k{i}" for i in range(len(key)))
    cr.class_sql = (f"SELECT {kal}, COUNT(DISTINCT {u}.{a}) AS n_distinct FROM {frm} "
                    f"WHERE {where} GROUP BY ALL HAVING COUNT(DISTINCT {u}.{a}) > 1")
    # unordered violating pairs derived from the class histogram (O(n) grouped scan):
    # sum over classes of C(n,2) - sum over values C(n_v,2), over non-NULL values
    cr.pairs_from_classes_sql = (
        f"WITH h AS (SELECT {kal}, {u}.{a} AS val, COUNT(*)::HUGEINT AS n FROM {frm} "
        f"WHERE {where} AND {u}.{a} IS NOT NULL GROUP BY ALL), "
        f"c AS (SELECT SUM(n) AS tot, SUM(n * (n - 1) // 2) AS same FROM h GROUP BY {kn}) "
        f"SELECT COALESCE(SUM(tot * (tot - 1) // 2 - same), 0) FROM c")
    # the violating classes projected from the pairwise query (checks Prop. 1 on data)
    cr.pairwise_class_sql = f"SELECT DISTINCT {kal} {body}"
