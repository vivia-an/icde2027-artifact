"""Fourth view definition (E10 / BENCH.md A11): relations over the fw.* hookpoints written by
core_trace/framework_hooks.py, and the compiler profile "layout_v2". New module: it registers the
profile in compile.PROFILES at import time; compile.py, spec.py, forms_ext.py, views_layout.py (E9,
frozen) are unchanged.

Differences from layout_v1 (E9):
  * Obj depends on (object, hook phase) and carries slices (dim-0 chunk index; -1 = whole local
    tensor): FSDP2 shards at rest vs. all-gathered in forward, ZeRO-1 owner-only before the
    broadcast. cls (replica class) is computed per (rank, obj, hook) from the phase's shard dims;
    rest_cls from the at-rest shard dims (used by the phase-unguarded variants).
  * Tied replicas: rkey = COALESCE(tied group, obj) (pp2: tok_emb.weight on stage 0 and
    lm_head.weight on stage 1 share rkey 'emb').
  * Reduction convention as Meta: red_grp (group the gradient reduction runs over: the DP group, or
    'TIED:<group>' for tied-embedding sync) and red_scale (1/size(DP) for mean, 1 for sum).
  * Step domain from punctuation, not from State: Steps = {0..wm} where the watermark wm = min over
    the ranks of Topo of the largest step for which the rank emitted a step marker (-1 if some rank
    emitted none). A rank that skips the marker of step s but continues still advances wm past s,
    so s is in the domain and its missing marker is a witness of G5. A rank that stops emitting
    markers stalls wm (liveness, not detected by these rules). `steps_override` replaces the
    domain (window evaluation passes the step being closed).

Relations (prefix L2_):
  Topo(rank, dim, size, grp, coord)
  Obj(rank, obj, hook, slice, shard_dims, rkey, tied, requires_grad, opt_group, red_op, red_dim)
      one row per (rank, object, phase hook, slice); hook = 'rest' rows describe the object at rest
  State(event_id, rank, step, hook, htype, obj, rkey, slice, cksum, sum, l2, numel, nonfinite,
        cls_c, shard_dims, cls, rest_cls, dp_grp, tp_grp, pp_grp, red_grp, red_scale)
      cls_c = class id computed by the collector at collection time (shared-scan optimisation)
  Steps(step)          punctuated step domain (Meta)
  Ranks(rank)          ranks of Topo (Meta)
  TraceRanks(rank)     ranks that wrote any State row (used only by the topology-unguarded G5)
  Marker(rank, step, n)
  AnyRec(rank, step, hook)   every per-step record (State rows and markers)
  Scalar(rank, step, loss)
"""
from typing import List, Optional

from . import compile as C
from .spec import SpecError

HP_PRE, HP_POST = "fw.grad_pre_sync", "fw.grad_post_sync"
HP_PLOCAL, HP_PARAM, HP_GATHER = "fw.param_local_step", "fw.param_after_step", "fw.param_gathered"
STATE_HOOKS = (HP_GATHER, HP_PRE, HP_POST, HP_PLOCAL, HP_PARAM)
PARAM_HOOKS = (HP_PLOCAL, HP_PARAM, HP_GATHER)
GRAD_HOOKS = (HP_PRE, HP_POST)
HOOK_ORDER = {HP_GATHER: 0, HP_PRE: 1, HP_POST: 2, HP_PLOCAL: 3, HP_PARAM: 4, "fw.marker": 5,
              "fw.scalar": 6}

RELATIONS = {
    "State": dict(event_id="bigint", rank="bigint", step="bigint", hook="varchar", htype="varchar",
                  obj="varchar", rkey="varchar", slice="bigint", cksum="varchar", sum="double",
                  l2="double", numel="double", nonfinite="bool", cls_c="varchar",
                  shard_dims="varchar[]", cls="varchar", rest_cls="varchar", dp_grp="varchar",
                  tp_grp="varchar", pp_grp="varchar", red_grp="varchar", red_scale="double"),
    "Topo": dict(rank="bigint", dim="varchar", size="bigint", grp="varchar", coord="bigint"),
    "Obj": dict(rank="bigint", obj="varchar", hook="varchar", slice="bigint",
                shard_dims="varchar[]", rkey="varchar", tied="varchar", requires_grad="bool",
                opt_group="bigint", red_op="varchar", red_dim="varchar"),
    "Steps": dict(step="bigint"),
    "Ranks": dict(rank="bigint"),
    "TraceRanks": dict(rank="bigint"),
    "Marker": dict(rank="bigint", step="bigint", n="bigint"),
    "AnyRec": dict(rank="bigint", step="bigint", hook="varchar"),
    "Scalar": dict(rank="bigint", step="bigint", loss="double"),
}
META_RELATIONS = {"Topo", "Obj", "Steps", "Ranks", "TraceRanks"}
NAMES = {k: "L2_" + k for k in RELATIONS}


class LayoutV2Profile(C.Profile):
    name = "layout_v2"
    relations = RELATIONS
    meta = META_RELATIONS

    def rel_sql(self, rel):
        return NAMES[rel]

    _COL = {"REPLICA": "cls", "REST": "rest_cls", "DP": "dp_grp", "TP": "tp_grp", "PP": "pp_grp"}

    def grp_sql(self, d, u, v, ordered, ctx):
        col = self._COL.get(d)
        if col is None:
            raise SpecError(f"grp_{d} not available in profile layout_v2")
        cmp = "<" if ordered else "<>"
        return f"{u}.{col} = {v}.{col} AND {u}.rank {cmp} {v}.rank"

    def repl_sql(self, d, u, ctx):
        return f"NOT list_contains({u}.shard_dims, '{d}')"

    def class_key(self, d, u, ctx) -> List[str]:
        col = self._COL.get(d)
        if col is None:
            raise SpecError(f"no class identifier for grp_{d}")
        return [f"{u}.{col}"]


C.PROFILES["layout_v2"] = LayoutV2Profile()


def install(con, src: str = "L2_events", steps_override: Optional[List[int]] = None) -> None:
    hooks = ", ".join(f"'{h}'" for h in STATE_HOOKS)
    x = con.execute
    x(f"""CREATE OR REPLACE TEMP TABLE L2_Topo AS
      SELECT e.rank::BIGINT AS rank, d.key AS dim, json_extract(d.value, '$.size')::BIGINT AS size,
             json_extract_string(d.value, '$.group') AS grp, json_extract(d.value, '$.coord')::BIGINT AS coord
      FROM {src} e, json_each(e.payload, '$.dims') d WHERE e.hookpoint = 'fw.topo'""")
    x(f"""CREATE OR REPLACE TEMP TABLE L2_ObjRaw AS
      SELECT e.rank::BIGINT AS rank, o.value AS v FROM {src} e, json_each(e.payload, '$.objects') o
      WHERE e.hookpoint = 'fw.obj'""")
    common = """json_extract_string(v, '$.obj') AS obj,
             COALESCE(json_extract_string(v, '$.tied'), json_extract_string(v, '$.obj')) AS rkey,
             json_extract_string(v, '$.tied') AS tied,
             json_extract(v, '$.requires_grad')::BOOLEAN AS requires_grad,
             json_extract(v, '$.opt_group')::BIGINT AS opt_group,
             json_extract_string(v, '$.red.op') AS red_op, json_extract_string(v, '$.red.dim') AS red_dim"""
    x(f"""CREATE OR REPLACE TEMP TABLE L2_Obj AS
      SELECT rank, json_extract_string(v, '$.obj') AS obj, p.key AS hook, sl.value::BIGINT AS slice,
             CAST(json_extract(p.value, '$.shard_dims') AS VARCHAR[]) AS shard_dims, {common.replace('json_extract_string(v', 'json_extract_string(o.v').replace('json_extract(v', 'json_extract(o.v')}
      FROM L2_ObjRaw o, json_each(o.v, '$.phases') p, json_each(p.value, '$.slices') sl
      UNION ALL
      SELECT rank, json_extract_string(v, '$.obj') AS obj, 'rest' AS hook,
             json_extract(v, '$.rest.slice')::BIGINT AS slice,
             CAST(json_extract(v, '$.rest.shard_dims') AS VARCHAR[]) AS shard_dims, {common}
      FROM L2_ObjRaw""")
    # per (rank, obj, hook): shard dims, class, at-rest class, reduction group and scale
    x("""CREATE OR REPLACE TEMP TABLE L2_Ph AS
      WITH ph AS (SELECT DISTINCT rank, obj, hook, shard_dims, rkey, tied, red_op, red_dim FROM L2_Obj),
           rest AS (SELECT DISTINCT rank, obj, shard_dims AS rest_dims FROM L2_Obj WHERE hook = 'rest')
      SELECT ph.rank, ph.obj, ph.hook, ph.shard_dims, ph.rkey,
             (SELECT COALESCE(string_agg(t.dim || '=' || t.coord, ',' ORDER BY t.dim), '')
              FROM L2_Topo t WHERE t.rank = ph.rank AND list_contains(ph.shard_dims, t.dim)) AS cls,
             (SELECT COALESCE(string_agg(t.dim || '=' || t.coord, ',' ORDER BY t.dim), '')
              FROM L2_Topo t WHERE t.rank = ph.rank AND list_contains(rest.rest_dims, t.dim)) AS rest_cls,
             CASE ph.red_dim WHEN 'TIED' THEN 'TIED:' || ph.tied
                  ELSE (SELECT t.grp FROM L2_Topo t WHERE t.rank = ph.rank AND t.dim = ph.red_dim) END AS red_grp,
             CASE ph.red_op WHEN 'sum' THEN 1.0
                  ELSE 1.0 / (SELECT t.size FROM L2_Topo t WHERE t.rank = ph.rank AND t.dim = ph.red_dim) END AS red_scale
      FROM ph JOIN rest ON rest.rank = ph.rank AND rest.obj = ph.obj""")
    x(f"""CREATE OR REPLACE TEMP TABLE L2_State AS
      WITH s AS (
        SELECT e.event_id AS eid, e.rank::BIGINT AS rank, e.step::BIGINT AS step, e.hookpoint AS hook,
               json_extract_string(r.value, '$.obj') AS obj, json_extract(r.value, '$.slice')::BIGINT AS slice,
               json_extract_string(r.value, '$.cls') AS cls_c, json_extract_string(r.value, '$.cksum') AS cksum,
               json_extract(r.value, '$.sum')::DOUBLE AS sum, json_extract(r.value, '$.l2')::DOUBLE AS l2,
               json_extract(r.value, '$.numel')::DOUBLE AS numel,
               json_extract(r.value, '$.nonfinite')::BOOLEAN AS nonfinite, CAST(r.key AS BIGINT) AS pos
        FROM {src} e, json_each(e.payload, '$.tensors') r WHERE e.hookpoint IN ({hooks}))
      SELECT row_number() OVER (ORDER BY s.rank, s.step, s.hook, s.pos) AS event_id,
             s.rank, s.step, s.hook,
             CASE WHEN s.hook IN ('{HP_PRE}', '{HP_POST}') THEN 'grad' ELSE 'param' END AS htype,
             s.obj, COALESCE(p.rkey, rp.rkey, s.obj) AS rkey, s.slice, s.cksum, s.sum, s.l2, s.numel,
             s.nonfinite, s.cls_c,
             COALESCE(p.shard_dims, rp.shard_dims) AS shard_dims, COALESCE(p.cls, rp.cls) AS cls,
             COALESCE(p.rest_cls, rp.rest_cls) AS rest_cls,
             td.grp AS dp_grp, tt.grp AS tp_grp, tq.grp AS pp_grp,
             COALESCE(p.red_grp, rp.red_grp) AS red_grp, COALESCE(p.red_scale, rp.red_scale) AS red_scale
      FROM s
      LEFT JOIN L2_Ph p ON p.rank = s.rank AND p.obj = s.obj AND p.hook = s.hook
      LEFT JOIN L2_Ph rp ON rp.rank = s.rank AND rp.obj = s.obj AND rp.hook = 'rest'
      LEFT JOIN L2_Topo td ON td.rank = s.rank AND td.dim = 'DP'
      LEFT JOIN L2_Topo tt ON tt.rank = s.rank AND tt.dim = 'TP'
      LEFT JOIN L2_Topo tq ON tq.rank = s.rank AND tq.dim = 'PP'""")
    x(f"""CREATE OR REPLACE TEMP TABLE L2_Marker AS
      SELECT e.rank::BIGINT AS rank, json_extract(e.payload, '$.step')::BIGINT AS step,
             json_extract(e.payload, '$.n')::BIGINT AS n FROM {src} e WHERE e.hookpoint = 'fw.marker'""")
    x("CREATE OR REPLACE TEMP TABLE L2_Ranks AS SELECT DISTINCT rank FROM L2_Topo")
    x("CREATE OR REPLACE TEMP TABLE L2_TraceRanks AS SELECT DISTINCT rank FROM L2_State")
    if steps_override is not None:
        vals = ", ".join(f"({int(s)})" for s in steps_override) or "(NULL)"
        x(f"CREATE OR REPLACE TEMP TABLE L2_Steps AS SELECT col0::BIGINT AS step FROM (VALUES {vals}) "
          "WHERE col0 IS NOT NULL")
    else:
        x("""CREATE OR REPLACE TEMP TABLE L2_Steps AS
          WITH m AS (SELECT r.rank, max(k.step) AS mx FROM L2_Ranks r
                     LEFT JOIN L2_Marker k ON k.rank = r.rank GROUP BY r.rank),
               w AS (SELECT CASE WHEN count(mx) < count(*) OR count(*) = 0 THEN -1 ELSE min(mx) END AS wm FROM m)
          SELECT range::BIGINT AS step FROM w, range(0, w.wm + 1)""")
    x("""CREATE OR REPLACE TEMP TABLE L2_AnyRec AS
      SELECT DISTINCT rank, step, hook FROM L2_State
      UNION SELECT rank, step, 'fw.marker' FROM L2_Marker""")
    x(f"""CREATE OR REPLACE TEMP TABLE L2_Scalar AS
      SELECT e.rank::BIGINT AS rank, e.step::BIGINT AS step,
             TRY_CAST(json_extract(e.payload, '$.loss') AS DOUBLE) AS loss
      FROM {src} e WHERE e.hookpoint = 'fw.scalar'""")


def watermark(con) -> int:
    """Current punctuation watermark of the installed relations (max step of L2_Steps, -1 if empty)."""
    r = con.execute("SELECT COALESCE(max(step), -1) FROM L2_Steps").fetchone()[0]
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
    con.execute("CREATE OR REPLACE TEMP TABLE L2_events AS " + " UNION ALL ".join(parts))
    for i in range(len(parts)):
        con.execute(f"DETACH r{i}")
    if install_views:
        install(con, "L2_events")
    return con
