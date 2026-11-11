"""Fifth view definition (E12 / BENCH.md A12): relations over the fw.* hookpoints written by
core_trace/framework_hooks_v3.py (Obj v3), compiler profile "layout_v3". New module; it registers
the profile at import time; compile.py, spec.py, forms_ext.py, views_layout*.py are unchanged.

Differences from layout_v2 (E10):
  * Obj v3 per (object, phase): shard_dims, pending collective (op, dims), order inside the step,
    dtype; per object: kind (param | hparam), owner (single-owner objects, ZeRO-1), rest hook.
  * Replica-class columns, all computed ONCE per installed job from Obj x Topo:
      cls        coordinates of the phase's shard dims plus its pending dims (the guarded class)
      cls_np     coordinates of the phase's shard dims only (ignores the pending collective; used
                 by the "phase guard one hook too early" ablation)
      rest_cls   coordinates of the at-rest shard dims (ablation "ignore phase-dependent sharding")
      cls_dponly coordinates of every dim except DP (ablation "replica class with DP only")
      gkey       coordinates of the dims sharded in EVERY phase of the object (key shared by an
                 object's gathered copies and its at-rest shards; gather consistency)
  * owner, kind, dtype, rest_hook, rest_nslices (number of distinct at-rest slices of the object
    within its gkey class) as State columns.
  * Step domain from punctuation exactly as layout_v2: Steps = {0..wm}, wm = min over Topo ranks of
    the largest recorded marker step (the recorded loop counter is trusted; see the E12 report).

Relations (prefix L3_): Topo, Obj, State, Steps, Ranks, TraceRanks, Marker, AnyRec, Scalar.
"""
from typing import List, Optional

from . import compile as C
from .spec import SpecError

HP_PRE, HP_POST = "fw.grad_pre_sync", "fw.grad_post_sync"
HP_PLOCAL, HP_PARAM, HP_GATHER = "fw.param_local_step", "fw.param_after_step", "fw.param_gathered"
HP_HPARAM = "fw.opt_hparam"
STATE_HOOKS = (HP_GATHER, HP_PRE, HP_POST, HP_PLOCAL, HP_HPARAM, HP_PARAM)
HOOK_ORDER = {HP_GATHER: 0, HP_PRE: 1, HP_POST: 2, HP_PLOCAL: 3, HP_HPARAM: 4, HP_PARAM: 5,
              "fw.marker": 6, "fw.scalar": 7}

RELATIONS = {
    "State": dict(event_id="bigint", rank="bigint", step="bigint", hook="varchar", htype="varchar",
                  obj="varchar", rkey="varchar", slice="bigint", cksum="varchar", sum="double",
                  l2="double", numel="double", nonfinite="bool", cls_c="varchar",
                  shard_dims="varchar[]", cls="varchar", cls_np="varchar", rest_cls="varchar",
                  cls_dponly="varchar", gkey="varchar", dp_grp="varchar", tp_grp="varchar",
                  pp_grp="varchar", red_grp="varchar", red_scale="double", owner="bigint",
                  kind="varchar", dtype="varchar", rest_hook="varchar", rest_nslices="bigint",
                  requires_grad="bool"),
    "Topo": dict(rank="bigint", dim="varchar", size="bigint", grp="varchar", coord="bigint"),
    "Obj": dict(rank="bigint", obj="varchar", hook="varchar", slice="bigint",
                shard_dims="varchar[]", pending_op="varchar", pending_dims="varchar[]",
                ord="bigint", dtype="varchar", rkey="varchar", tied="varchar", kind="varchar",
                requires_grad="bool", opt_group="bigint", red_op="varchar", red_dim="varchar",
                red_from="varchar", red_to="varchar", owner="bigint", rest_hook="varchar"),
    "Steps": dict(step="bigint"),
    "Ranks": dict(rank="bigint"),
    "TraceRanks": dict(rank="bigint"),
    "Marker": dict(rank="bigint", step="bigint", n="bigint"),
    "AnyRec": dict(rank="bigint", step="bigint", hook="varchar"),
    "Scalar": dict(rank="bigint", step="bigint", loss="double"),
}
META_RELATIONS = {"Topo", "Obj", "Steps", "Ranks", "TraceRanks"}
NAMES = {k: "L3_" + k for k in RELATIONS}


class LayoutV3Profile(C.Profile):
    name = "layout_v3"
    relations = RELATIONS
    meta = META_RELATIONS

    def rel_sql(self, rel):
        return NAMES[rel]

    _COL = {"REPLICA": "cls", "NOPEND": "cls_np", "REST": "rest_cls", "DPONLY": "cls_dponly",
            "GKEY": "gkey", "DP": "dp_grp", "TP": "tp_grp", "PP": "pp_grp"}

    def grp_sql(self, d, u, v, ordered, ctx):
        col = self._COL.get(d)
        if col is None:
            raise SpecError(f"grp_{d} not available in profile layout_v3")
        cmp = "<" if ordered else "<>"
        return f"{u}.{col} = {v}.{col} AND {u}.rank {cmp} {v}.rank"

    def repl_sql(self, d, u, ctx):
        return f"NOT list_contains({u}.shard_dims, '{d}')"

    def class_key(self, d, u, ctx) -> List[str]:
        col = self._COL.get(d)
        if col is None:
            raise SpecError(f"no class identifier for grp_{d}")
        return [f"{u}.{col}"]


C.PROFILES["layout_v3"] = LayoutV3Profile()


def _coords(rank_expr: str, dims_expr: str) -> str:
    return (f"(SELECT COALESCE(string_agg(t.dim || '=' || t.coord, ',' ORDER BY t.dim), '') "
            f"FROM L3_Topo t WHERE t.rank = {rank_expr} AND list_contains({dims_expr}, t.dim))")


def install_meta(con, src: str = "L3_events") -> None:
    """Meta relations and the per-(rank, obj, hook) class table, from the build-time rows."""
    x = con.execute
    x(f"""CREATE OR REPLACE TEMP TABLE L3_Topo AS
      SELECT e.rank::BIGINT AS rank, d.key AS dim, json_extract(d.value, '$.size')::BIGINT AS size,
             json_extract_string(d.value, '$.group') AS grp, json_extract(d.value, '$.coord')::BIGINT AS coord
      FROM {src} e, json_each(e.payload, '$.dims') d WHERE e.hookpoint = 'fw.topo'""")
    x(f"""CREATE OR REPLACE TEMP TABLE L3_ObjRaw AS
      SELECT e.rank::BIGINT AS rank, o.value AS v FROM {src} e, json_each(e.payload, '$.objects') o
      WHERE e.hookpoint = 'fw.obj'""")
    common = """json_extract_string(o.v, '$.obj') AS obj,
             COALESCE(json_extract_string(o.v, '$.tied'), json_extract_string(o.v, '$.obj')) AS rkey,
             json_extract_string(o.v, '$.tied') AS tied,
             COALESCE(json_extract_string(o.v, '$.kind'), 'param') AS kind,
             json_extract(o.v, '$.requires_grad')::BOOLEAN AS requires_grad,
             json_extract(o.v, '$.opt_group')::BIGINT AS opt_group,
             json_extract_string(o.v, '$.red.op') AS red_op, json_extract_string(o.v, '$.red.dim') AS red_dim,
             json_extract_string(o.v, '$.red.from') AS red_from, json_extract_string(o.v, '$.red.to') AS red_to,
             TRY_CAST(json_extract(o.v, '$.owner') AS BIGINT) AS owner,
             COALESCE(json_extract_string(o.v, '$.rest.hook'), 'fw.param_after_step') AS rest_hook"""
    x(f"""CREATE OR REPLACE TEMP TABLE L3_Obj AS
      SELECT o.rank, p.key AS hook, sl.value::BIGINT AS slice,
             CAST(json_extract(p.value, '$.shard_dims') AS VARCHAR[]) AS shard_dims,
             json_extract_string(p.value, '$.pending.op') AS pending_op,
             COALESCE(TRY_CAST(json_extract(p.value, '$.pending.dims') AS VARCHAR[]), []::VARCHAR[]) AS pending_dims,
             TRY_CAST(json_extract(p.value, '$.order') AS BIGINT) AS ord,
             json_extract_string(p.value, '$.dtype') AS dtype, {common}
      FROM L3_ObjRaw o, json_each(o.v, '$.phases') p, json_each(p.value, '$.slices') sl
      UNION ALL
      SELECT o.rank, 'rest' AS hook, json_extract(o.v, '$.rest.slice')::BIGINT AS slice,
             CAST(json_extract(o.v, '$.rest.shard_dims') AS VARCHAR[]) AS shard_dims,
             NULL AS pending_op, []::VARCHAR[] AS pending_dims, NULL AS ord, NULL AS dtype, {common}
      FROM L3_ObjRaw o""")
    # dims sharded in every phase of the object (on this rank)
    x("""CREATE OR REPLACE TEMP TABLE L3_Persist AS
      SELECT rank, obj, list_reduce(list(shard_dims), (a, b) -> list_intersect(a, b)) AS pdims
      FROM (SELECT DISTINCT rank, obj, hook, shard_dims FROM L3_Obj) GROUP BY rank, obj""")
    x(f"""CREATE OR REPLACE TEMP TABLE L3_Ph AS
      WITH ph AS (SELECT DISTINCT rank, obj, hook, shard_dims, pending_dims, rkey, tied, red_op, red_dim,
                         owner, kind, dtype, rest_hook, requires_grad FROM L3_Obj),
           rest AS (SELECT DISTINCT rank, obj, shard_dims AS rest_dims, slice AS rest_slice FROM L3_Obj WHERE hook = 'rest')
      SELECT ph.*, rest.rest_slice,
             {_coords('ph.rank', 'list_concat(ph.shard_dims, ph.pending_dims)')} AS cls,
             {_coords('ph.rank', 'ph.shard_dims')} AS cls_np,
             {_coords('ph.rank', 'rest.rest_dims')} AS rest_cls,
             (SELECT COALESCE(string_agg(t.dim || '=' || t.coord, ',' ORDER BY t.dim), '')
              FROM L3_Topo t WHERE t.rank = ph.rank AND t.dim <> 'DP') AS cls_dponly,
             {_coords('ph.rank', 'pe.pdims')} AS gkey,
             CASE ph.red_dim WHEN 'TIED' THEN 'TIED:' || ph.tied
                  ELSE (SELECT t.grp FROM L3_Topo t WHERE t.rank = ph.rank AND t.dim = ph.red_dim) END AS red_grp,
             CASE WHEN ph.red_op IS NULL THEN NULL WHEN ph.red_op = 'sum' THEN 1.0
                  ELSE 1.0 / (SELECT t.size FROM L3_Topo t WHERE t.rank = ph.rank AND t.dim = ph.red_dim) END AS red_scale
      FROM ph JOIN rest ON rest.rank = ph.rank AND rest.obj = ph.obj
      LEFT JOIN L3_Persist pe ON pe.rank = ph.rank AND pe.obj = ph.obj""")
    x("""CREATE OR REPLACE TEMP TABLE L3_NSl AS
      SELECT rkey, gkey, count(DISTINCT rest_slice) AS rest_nslices FROM L3_Ph WHERE hook = 'rest'
      GROUP BY rkey, gkey""")
    x("""CREATE OR REPLACE TEMP TABLE L3_PhX AS
      SELECT p.*, n.rest_nslices, td.grp AS dp_grp, tt.grp AS tp_grp, tq.grp AS pp_grp
      FROM L3_Ph p LEFT JOIN L3_NSl n ON n.rkey = p.rkey AND n.gkey = p.gkey
      LEFT JOIN L3_Topo td ON td.rank = p.rank AND td.dim = 'DP'
      LEFT JOIN L3_Topo tt ON tt.rank = p.rank AND tt.dim = 'TP'
      LEFT JOIN L3_Topo tq ON tq.rank = p.rank AND tq.dim = 'PP'""")
    x("CREATE OR REPLACE TEMP TABLE L3_Ranks AS SELECT DISTINCT rank FROM L3_Topo")


def install_state(con, src: str, steps_override: Optional[List[int]] = None) -> None:
    hooks = ", ".join(f"'{h}'" for h in STATE_HOOKS)
    x = con.execute
    x(f"""CREATE OR REPLACE TEMP TABLE L3_State AS
      WITH s AS (
        SELECT e.rank::BIGINT AS rank, e.step::BIGINT AS step, e.hookpoint AS hook,
               json_extract_string(r.value, '$.obj') AS obj, json_extract(r.value, '$.slice')::BIGINT AS slice,
               json_extract_string(r.value, '$.cls') AS cls_c, json_extract_string(r.value, '$.cksum') AS cksum,
               json_extract(r.value, '$.sum')::DOUBLE AS sum, json_extract(r.value, '$.l2')::DOUBLE AS l2,
               json_extract(r.value, '$.numel')::DOUBLE AS numel,
               json_extract(r.value, '$.nonfinite')::BOOLEAN AS nonfinite, CAST(r.key AS BIGINT) AS pos
        FROM {src} e, json_each(e.payload, '$.tensors') r WHERE e.hookpoint IN ({hooks}))
      SELECT row_number() OVER (ORDER BY s.rank, s.step, s.hook, s.pos) AS event_id,
             s.rank, s.step, s.hook,
             CASE WHEN s.hook IN ('{HP_PRE}', '{HP_POST}') THEN 'grad'
                  WHEN s.hook = '{HP_HPARAM}' THEN 'hparam' ELSE 'param' END AS htype,
             s.obj, COALESCE(p.rkey, s.obj) AS rkey, s.slice, s.cksum, s.sum, s.l2, s.numel,
             s.nonfinite, s.cls_c, p.shard_dims, p.cls, p.cls_np, p.rest_cls, p.cls_dponly, p.gkey,
             p.dp_grp, p.tp_grp, p.pp_grp, p.red_grp, p.red_scale, p.owner, p.kind, p.dtype,
             p.rest_hook, p.rest_nslices, p.requires_grad
      FROM s LEFT JOIN L3_PhX p ON p.rank = s.rank AND p.obj = s.obj AND p.hook = s.hook""")
    x(f"""CREATE OR REPLACE TEMP TABLE L3_Marker AS
      SELECT e.rank::BIGINT AS rank, json_extract(e.payload, '$.step')::BIGINT AS step,
             json_extract(e.payload, '$.n')::BIGINT AS n FROM {src} e WHERE e.hookpoint = 'fw.marker'""")
    x("CREATE OR REPLACE TEMP TABLE L3_TraceRanks AS SELECT DISTINCT rank FROM L3_State")
    if steps_override is not None:
        vals = ", ".join(f"({int(s)})" for s in steps_override) or "(NULL)"
        x(f"CREATE OR REPLACE TEMP TABLE L3_Steps AS SELECT col0::BIGINT AS step FROM (VALUES {vals}) "
          "WHERE col0 IS NOT NULL")
    else:
        x("""CREATE OR REPLACE TEMP TABLE L3_Steps AS
          WITH m AS (SELECT r.rank, max(k.step) AS mx FROM L3_Ranks r
                     LEFT JOIN L3_Marker k ON k.rank = r.rank GROUP BY r.rank),
               w AS (SELECT CASE WHEN count(mx) < count(*) OR count(*) = 0 THEN -1 ELSE min(mx) END AS wm FROM m)
          SELECT range::BIGINT AS step FROM w, range(0, w.wm + 1)""")
    x("""CREATE OR REPLACE TEMP TABLE L3_AnyRec AS
      SELECT DISTINCT rank, step, hook FROM L3_State
      UNION SELECT rank, step, 'fw.marker' FROM L3_Marker""")
    x(f"""CREATE OR REPLACE TEMP TABLE L3_Scalar AS
      SELECT e.rank::BIGINT AS rank, e.step::BIGINT AS step,
             TRY_CAST(json_extract(e.payload, '$.loss') AS DOUBLE) AS loss
      FROM {src} e WHERE e.hookpoint = 'fw.scalar'""")


def install(con, src: str = "L3_events", steps_override: Optional[List[int]] = None,
            meta: bool = True) -> None:
    """Install every relation over `src`. meta=False keeps the Meta tables installed earlier
    (the class columns are then computed once per job, not per window)."""
    if meta:
        install_meta(con, src)
    install_state(con, src, steps_override)


def watermark(con) -> int:
    r = con.execute("SELECT COALESCE(max(step), -1) FROM L3_Steps").fetchone()[0]
    return int(r)


def open_job(rank_files: List[str], threads: int = 1, memory_limit: str = "2GB",
             install_views: bool = True):
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    parts = []
    for i, f in enumerate(sorted(rank_files)):
        con.execute(f"ATTACH '{f}' AS r{i} (READ_ONLY)")
        parts.append(f"SELECT * FROM r{i}.events")
    con.execute("CREATE OR REPLACE TEMP TABLE L3_events AS " + " UNION ALL ".join(parts))
    for i in range(len(parts)):
        con.execute(f"DETACH r{i}")
    if install_views:
        install(con, "L3_events")
    return con


def read_meta(con, src: str = "L3_events"):
    """(topo, objs) from the build-time rows: topo = {rank: {dim: {...}}}, objs = {rank: [obj, ...]}."""
    import json
    topo, objs = {}, {}
    for rank, hp, payload in con.execute(
            f"SELECT rank, hookpoint, payload FROM {src} WHERE hookpoint IN ('fw.topo', 'fw.obj') "
            f"ORDER BY rank, hookpoint").fetchall():
        p = json.loads(payload) if isinstance(payload, str) else payload
        if hp == "fw.topo":
            topo[int(rank)] = p["dims"]
        else:
            objs[int(rank)] = p["objects"]
    return topo, objs
