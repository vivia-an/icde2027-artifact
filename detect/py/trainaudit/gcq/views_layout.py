"""Third view definition (E9 / BENCH.md A10): the paper's relations over the layout.* hookpoints
written by core_trace/layout_state_hook.py, plus the compiler profile "layout_v1".

New module. It registers the profile in compile.PROFILES at import time; compile.py, spec.py,
views.py and every existing spec are unchanged.

Physical input: the events table of every rank of one job (one DuckDB file per rank, combined by
`open_job` into one relation `LAY_events` with UNION ALL).

Views (prefixed LAY_ so they never collide with views.py):
  LAY_Topo(rank, dim, size, grp, coord)          from layout.topo (one row per rank and dimension)
  LAY_Obj(rank, obj, base, kind, shard_dims, opt_group, requires_grad)   from layout.obj
  LAY_State(event_id, seq, rank, step, hook, obj, base, kind, cksum, sum, l2, numel, nonfinite,
            shard_dims, cls, dp_grp, tp_grp)
      one row per tensor of layout.grad_pre_sync / grad_post_sync / param_after_step.
      kind = 'grad_local' (pre-sync), 'grad' (post-sync), 'param' (after step).
      shard_dims = LAY_Obj.shard_dims of (rank, obj) (NULL when the rank does not hold obj, e.g.
      the pre-sync chunks of other ranks' shards under full sharding).
      cls = cls(rank, obj; tau): the replica class. For an object replicated along the set of
      dimensions D(obj) = all dims minus shard_dims(obj), two records of the same (step, hook, obj)
      are replicas iff their ranks agree on the coordinate of every dimension NOT in D(obj), i.e.
      on every dimension in shard_dims(obj) (round-1 correction of Sec. III-C). cls is the tuple of
      those coordinates, rendered 'TP=1' / 'DP=0' / '' (replicated everywhere), computed from
      LAY_Topo and LAY_Obj only (Meta).
      dp_grp / tp_grp = LAY_Topo.grp of the rank for DP / TP (same DP group <=> same coordinates
      in every dimension except DP).
  LAY_Steps(step)                                the step domain: every step any rank recorded
  LAY_Scalar(rank, step, loss, payload)          layout.scalar

Profile "layout_v1" topology atoms:
  {"grp": "REPLICA", "vars": [u, v], "ordered": b}   u.cls = v.cls and u.rank <> v.rank (< if ordered)
  {"grp": "DP"|"TP", ...}                           same group of that dimension
  {"repl": d, "var": u}                             d not in u.shard_dims
  class key (Prop. 1 rewrite): REPLICA -> u.cls; DP -> u.dp_grp; TP -> u.tp_grp
"""
from typing import List

from . import compile as C
from .spec import SpecError

RELATIONS = {
    "State": dict(event_id="bigint", seq="bigint", rank="bigint", step="bigint", hook="varchar",
                  obj="varchar", base="varchar", kind="varchar", cksum="varchar", sum="double",
                  l2="double", numel="double", nonfinite="bool", shard_dims="varchar[]",
                  cls="varchar", dp_grp="varchar", tp_grp="varchar"),
    "Topo": dict(rank="bigint", dim="varchar", size="bigint", grp="varchar", coord="bigint"),
    "Obj": dict(rank="bigint", obj="varchar", base="varchar", kind="varchar",
                shard_dims="varchar[]", opt_group="bigint", requires_grad="bool"),
    "Steps": dict(step="bigint"),
    "Scalar": dict(rank="bigint", step="bigint", loss="double", payload="json"),
}
META_RELATIONS = {"Topo", "Obj", "Steps"}
NAMES = {"State": "LAY_State", "Topo": "LAY_Topo", "Obj": "LAY_Obj", "Steps": "LAY_Steps",
         "Scalar": "LAY_Scalar"}
STATE_HOOKS = ("layout.grad_pre_sync", "layout.grad_post_sync", "layout.param_after_step")
HOOK_ORDER = {"layout.grad_pre_sync": 0, "layout.grad_post_sync": 1, "layout.param_after_step": 2}


class LayoutProfile(C.Profile):
    name = "layout_v1"
    relations = RELATIONS
    meta = META_RELATIONS

    def rel_sql(self, rel):
        return NAMES[rel]

    def grp_sql(self, d, u, v, ordered, ctx):
        cmp = "<" if ordered else "<>"
        col = {"REPLICA": "cls", "DP": "dp_grp", "TP": "tp_grp"}.get(d)
        if col is None:
            raise SpecError(f"grp_{d} not available in profile layout_v1")
        return f"{u}.{col} = {v}.{col} AND {u}.rank {cmp} {v}.rank"

    def repl_sql(self, d, u, ctx):
        return f"NOT list_contains({u}.shard_dims, '{d}')"

    def class_key(self, d, u, ctx) -> List[str]:
        col = {"REPLICA": "cls", "DP": "dp_grp", "TP": "tp_grp"}.get(d)
        if col is None:
            raise SpecError(f"no class identifier for grp_{d}")
        return [f"{u}.{col}"]


C.PROFILES["layout_v1"] = LayoutProfile()


def install(con, src: str = "LAY_events") -> None:
    hooks = ", ".join(f"'{h}'" for h in STATE_HOOKS)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE LAY_Topo AS
      SELECT e.rank::BIGINT AS rank, d.key AS dim,
             json_extract(d.value, '$.size')::BIGINT AS size,
             json_extract_string(d.value, '$.group') AS grp,
             json_extract(d.value, '$.coord')::BIGINT AS coord
      FROM {src} e, json_each(e.payload, '$.dims') d WHERE e.hookpoint = 'layout.topo'""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE LAY_Obj AS
      SELECT e.rank::BIGINT AS rank, json_extract_string(o.value, '$.obj') AS obj,
             json_extract_string(o.value, '$.base') AS base,
             json_extract_string(o.value, '$.kind') AS kind,
             CAST(json_extract(o.value, '$.shard_dims') AS VARCHAR[]) AS shard_dims,
             json_extract(o.value, '$.opt_group')::BIGINT AS opt_group,
             json_extract(o.value, '$.requires_grad')::BOOLEAN AS requires_grad
      FROM {src} e, json_each(e.payload, '$.objects') o WHERE e.hookpoint = 'layout.obj'""")
    con.execute("""CREATE OR REPLACE TEMP TABLE LAY_Cls AS
      SELECT o.rank, o.obj,
             COALESCE(string_agg(t.dim || '=' || t.coord, ',' ORDER BY t.dim)
                      FILTER (WHERE list_contains(o.shard_dims, t.dim)), '') AS cls
      FROM LAY_Obj o JOIN LAY_Topo t ON t.rank = o.rank GROUP BY o.rank, o.obj""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE LAY_State AS
      WITH s AS (
        SELECT e.event_id, e.rank::BIGINT AS rank, e.step::BIGINT AS step, e.hookpoint AS hook,
               json_extract_string(x.value, '$.obj') AS obj,
               json_extract_string(x.value, '$.cksum') AS cksum,
               json_extract(x.value, '$.sum')::DOUBLE AS sum,
               json_extract(x.value, '$.l2')::DOUBLE AS l2,
               json_extract(x.value, '$.numel')::DOUBLE AS numel,
               json_extract(x.value, '$.nonfinite')::BOOLEAN AS nonfinite,
               CAST(x.key AS BIGINT) AS pos
        FROM {src} e, json_each(e.payload, '$.tensors') x WHERE e.hookpoint IN ({hooks}))
      SELECT row_number() OVER (ORDER BY s.rank, s.step, s.hook, s.pos) AS event_id,
             s.event_id AS seq, s.rank, s.step, s.hook, s.obj,
             regexp_replace(s.obj, '#s[0-9]+$', '') AS base,
             CASE s.hook WHEN 'layout.grad_pre_sync' THEN 'grad_local'
                         WHEN 'layout.grad_post_sync' THEN 'grad' ELSE 'param' END AS kind,
             s.cksum, s.sum, s.l2, s.numel, s.nonfinite,
             o.shard_dims, c.cls,
             td.grp AS dp_grp, tt.grp AS tp_grp
      FROM s
      LEFT JOIN LAY_Obj o ON o.rank = s.rank AND o.obj = s.obj
      LEFT JOIN LAY_Cls c ON c.rank = s.rank AND c.obj = s.obj
      LEFT JOIN LAY_Topo td ON td.rank = s.rank AND td.dim = 'DP'
      LEFT JOIN LAY_Topo tt ON tt.rank = s.rank AND tt.dim = 'TP'""")
    con.execute("CREATE OR REPLACE TEMP TABLE LAY_Steps AS SELECT DISTINCT step FROM LAY_State")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE LAY_Scalar AS
      SELECT e.rank::BIGINT AS rank, e.step::BIGINT AS step,
             TRY_CAST(json_extract(e.payload, '$.loss') AS DOUBLE) AS loss, e.payload::JSON AS payload
      FROM {src} e WHERE e.hookpoint = 'layout.scalar'""")


def open_job(rank_files: List[str], threads: int = 1, memory_limit: str = "2GB"):
    """In-memory DuckDB with every rank file attached read-only, LAY_events = their union, and the
    LAY_ views installed."""
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    parts = []
    for i, f in enumerate(sorted(rank_files)):
        con.execute(f"ATTACH '{f}' AS r{i} (READ_ONLY)")
        parts.append(f"SELECT * FROM r{i}.events")
    con.execute("CREATE OR REPLACE TEMP TABLE LAY_events AS " + " UNION ALL ".join(parts))
    install(con, "LAY_events")
    return con
