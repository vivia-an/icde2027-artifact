"""Logical relations of the paper's data model (Sec. 3.1) as DuckDB views over the physical trace.

Physical table (store.py):  events(event_id, step, rank, hookpoint, ts_ns, schema_version, payload VARCHAR/JSON)
One DuckDB file per rank; the frozen rules see one file at a time, so the views are defined per file.

Logical relations (all views, nothing is copied unless `materialize=True`):

  State(event_id, seq, rank, step, hook, ts_ns, obj, obj_class, obj_id, slot, in_list, idx,
        dtype, shape, ndim, numel, l2, abs_max, max, min, mean, std, zero_rate, has_nan, has_inf,
        cksum, xr_group_size, summary, args)
      one row per tensor summary that a hook recorded (see HOOK_MAP below), plus one row per
      member checksum of a cross-rank replica gather (slot='replica').
  Event(event_id, seq, rank, step, hook, ts_ns, kind, fn, module_class, module_name, args)
      one row per recorded call event; args is the whole JSON payload.
  Meta:
    Topo(rank, dim, size, grp, coord, source)    parallel-dimension sizes that the trace records;
                                                 grp and coord are never recorded (always NULL)
    Obj(event_id, rank, obj, kind, dtype, shape, requires_grad, is_leaf, replica_group_kind,
        shard_dims, shard_dim, sequence_parallel, tp_size, dp_size, ep_size, source_adapter,
        attr_tensor_model_parallel, attr_expert_parallel, has_attr_ds_id, is_fsdp_flat,
        xr_group_size, xr_all_equal, local_cksum, entry, semantic)
    Config(event_id, rank, step, framework, inv_type, inv)   build.snapshot.framework_invariants
    OptGroup(event_id, rank, step, idx, item)                build.snapshot.optimizer.param_groups

Fields of the paper's model that the trace does not contain (left out, never synthesized):
  State.micro (no micro-batch index on any hook; grad_accum.backward has microbatch_idx_within_step
  but that hook is not installed by enable()), Topo.group and Topo.coord (list_groups() is never
  called; replica_group_id is id() of a process-local Python object), Obj.tied_group, Obj.opt_group
  (param_groups record only n_params, not membership), the CP dimension, Event.ret (post-call values
  live in the *.post payloads), per-step parameter/gradient/optimizer-state rows (optimizer hooks
  record only totals; per-parameter checksums exist only once, inside build.snapshot).
"""
from typing import Dict, List

# hookpoint -> (relation, slots) mapping; documented in REPORT.md
HOOK_MAP: Dict[str, Dict] = {
    "module.fwd.pre":     {"Event": True, "State": [("inputs", "list")]},
    "module.fwd.post":    {"Event": True, "State": [("output", "dict"), ("outputs", "list")]},
    "module.bwd":         {"Event": True, "State": [("grad_input", "list"), ("grad_output", "list")]},
    "comm.pre":           {"Event": True, "State": [("tensor_pre", "dict"), ("tensor_pre", "list")]},
    "comm.post":          {"Event": True, "State": [("tensor_post", "dict"), ("tensor_post", "list")]},
    "functional.softmax": {"Event": True, "State": [("output", "dict")]},
    "dataloader.batch":   {"Event": True, "State": [("input_ids", "dict"), ("fields", "dict-of-dict")]},
    "build.snapshot":     {"Event": True, "State": [("param", "model.parameters[*].tensor_summary"),
                                                    ("replica", "cross_rank_cksums[*].gathered_cksums[*]"),
                                                    ("replica", "cross_rank_buffer_cksums[*].gathered_cksums[*]")],
                           "Meta": ["Obj", "Topo", "Config", "OptGroup"]},
    "optim.step.pre":     {"Event": True, "State": [("replica", "cross_rank_grad_cksums[*].gathered_cksums[*]")]},
    "optim.step.post":    {"Event": True, "State": []},
    "utils.clip_grad.pre":  {"Event": True, "State": []},
    "utils.clip_grad.post": {"Event": True, "State": []},
    "scheduler.init":     {"Event": True, "State": []},
    "checkpoint.call":    {"Event": True, "State": []},
    "loss.call":          {"Event": True, "State": []},
    "attention.fwd.post": {"Event": True, "State": []},
    "grad_accum.backward": {"Event": True, "State": []},
    "jitter.probe":       {"Event": True, "State": []},
    "residual.probe":     {"Event": True, "State": []},
    "decay.probe":        {"Event": True, "State": []},
}

# Column catalogue used by the compiler for typing. 'json' columns accept a JSON path suffix.
RELATIONS: Dict[str, Dict[str, str]] = {
    "State": dict(event_id="bigint", seq="bigint", rank="bigint", step="bigint", hook="varchar",
                  ts_ns="bigint", obj="varchar", obj_class="varchar", obj_id="varchar",
                  slot="varchar", in_list="bool", idx="bigint", dtype="varchar", shape="json",
                  ndim="bigint", numel="double", l2="double", abs_max="double", max="double",
                  min="double", mean="double", std="double", zero_rate="double",
                  has_nan="bool", has_inf="bool", cksum="varchar", xr_group_size="double",
                  summary="json", args="json"),
    "Event": dict(event_id="bigint", seq="bigint", rank="bigint", step="bigint", hook="varchar",
                  ts_ns="bigint", kind="varchar", fn="varchar", module_class="varchar",
                  module_name="varchar", args="json"),
    "Topo": dict(rank="bigint", dim="varchar", size="double", grp="varchar", coord="bigint",
                 source="varchar"),
    "Obj": dict(event_id="bigint", rank="bigint", obj="varchar", kind="varchar", dtype="varchar",
                shape="json", requires_grad="json", is_leaf="json", replica_group_kind="varchar",
                shard_dims="varchar[]", shard_dim="json", sequence_parallel="json",
                tp_size="double", dp_size="double", ep_size="double", source_adapter="varchar",
                attr_tensor_model_parallel="json", attr_expert_parallel="json",
                has_attr_ds_id="json", is_fsdp_flat="json", xr_group_size="double",
                xr_all_equal="json", local_cksum="varchar", entry="json", semantic="json"),
    "Config": dict(event_id="bigint", rank="bigint", step="bigint", framework="varchar",
                   inv_type="varchar", inv="json"),
    "OptGroup": dict(event_id="bigint", rank="bigint", step="bigint", idx="bigint", item="json"),
}
META_RELATIONS = {"Topo", "Obj", "Config", "OptGroup"}   # no step: never widen the window

# ---------------------------------------------------------------------------------------------
# Typed JSON access. Python semantics of the rules: json.loads -> dict.get; numbers compare as
# floats (bool is an int), None is "missing". These macros mirror that.
MACROS = [
    # double: JSON numbers (incl. NaN/Infinity), booleans as 1/0 (Python bool is int); else NULL
    """CREATE OR REPLACE TEMP MACRO gcq_dbl(j, p) AS CASE json_type(j, p)
         WHEN 'DOUBLE' THEN json_extract(j, p)::DOUBLE WHEN 'BIGINT' THEN json_extract(j, p)::DOUBLE
         WHEN 'UBIGINT' THEN json_extract(j, p)::DOUBLE
         WHEN 'BOOLEAN' THEN CASE WHEN json_extract(j, p)::BOOLEAN THEN 1.0 ELSE 0.0 END
         ELSE NULL END""",
    """CREATE OR REPLACE TEMP MACRO gcq_str(j, p) AS json_extract_string(j, p)""",
    # strict JSON boolean (Python `is True` / `is False`)
    """CREATE OR REPLACE TEMP MACRO gcq_is_true(j, p) AS
         COALESCE(json_type(j, p) = 'BOOLEAN' AND json_extract(j, p)::BOOLEAN, FALSE)""",
    """CREATE OR REPLACE TEMP MACRO gcq_is_false(j, p) AS
         COALESCE(json_type(j, p) = 'BOOLEAN' AND NOT json_extract(j, p)::BOOLEAN, FALSE)""",
    # Python truthiness of the decoded value (missing/None -> False)
    """CREATE OR REPLACE TEMP MACRO gcq_truthy(j, p) AS COALESCE(CASE json_type(j, p)
         WHEN 'BOOLEAN' THEN json_extract(j, p)::BOOLEAN
         WHEN 'DOUBLE' THEN json_extract(j, p)::DOUBLE <> 0
         WHEN 'BIGINT' THEN json_extract(j, p)::DOUBLE <> 0
         WHEN 'UBIGINT' THEN json_extract(j, p)::DOUBLE <> 0
         WHEN 'VARCHAR' THEN length(json_extract_string(j, p)) > 0
         WHEN 'ARRAY' THEN json_array_length(j, p) > 0
         WHEN 'OBJECT' THEN json_extract(j, p)::VARCHAR <> '{}'
         ELSE FALSE END, FALSE)""",
    # numel of a JSON shape list (product of dims; 1 for []); NULL if not a list
    """CREATE OR REPLACE TEMP MACRO gcq_numel(j, p) AS CASE WHEN json_type(j, p) = 'ARRAY'
         THEN COALESCE(list_aggregate(TRY_CAST(json_extract(j, p) AS DOUBLE[]), 'product'), 1.0)
         ELSE NULL END""",
]


def _stats_select(base: str) -> str:
    """Typed tensor-summary columns on top of a UNION of (…, summary) rows."""
    return f"""
SELECT b.*,
  gcq_str(b.summary, '$.dtype')            AS dtype,
  json_extract(b.summary, '$.shape')       AS shape,
  CASE WHEN json_type(b.summary, '$.shape') = 'ARRAY'
       THEN json_array_length(b.summary, '$.shape') END AS ndim,
  gcq_numel(b.summary, '$.shape')          AS numel,
  gcq_dbl(b.summary, '$.l2_norm')          AS l2,
  gcq_dbl(b.summary, '$.abs_max')          AS abs_max,
  gcq_dbl(b.summary, '$.max')              AS max,
  gcq_dbl(b.summary, '$.min')              AS min,
  gcq_dbl(b.summary, '$.mean')             AS mean,
  gcq_dbl(b.summary, '$.std')              AS std,
  gcq_dbl(b.summary, '$.zero_rate')        AS zero_rate,
  gcq_truthy(b.summary, '$.has_nan')       AS has_nan,
  gcq_truthy(b.summary, '$.has_inf')       AS has_inf
FROM ({base}) b"""


def _ev_cols(e: str = "e") -> str:
    return (f"{e}.event_id AS event_id, {e}.event_id AS seq, {e}.rank AS rank, {e}.step AS step, "
            f"{e}.hookpoint AS hook, {e}.ts_ns AS ts_ns")


_MODULE_OBJ = ("gcq_str(e.payload, '$.module_name') AS obj, gcq_str(e.payload, '$.module_class') AS obj_class, "
               "gcq_str(e.payload, '$.module_id') AS obj_id")


def state_sql(src: str) -> str:
    parts: List[str] = []
    tail = "NULL::VARCHAR AS cksum, NULL::DOUBLE AS xr_group_size"

    def list_slot(hook, key, obj_sql):
        parts.append(f"""SELECT {_ev_cols()}, {obj_sql}, '{key}' AS slot, TRUE AS in_list,
  CAST(j.key AS BIGINT) AS idx, j.value AS summary, e.payload AS args, {tail}
FROM {src} e, json_each(e.payload, '$.{key}') j
WHERE e.hookpoint = '{hook}' AND json_type(e.payload, '$.{key}') = 'ARRAY' AND j.type = 'OBJECT'""")

    def dict_slot(hook, key, obj_sql):
        parts.append(f"""SELECT {_ev_cols()}, {obj_sql}, '{key}' AS slot, FALSE AS in_list,
  0::BIGINT AS idx, json_extract(e.payload, '$.{key}') AS summary, e.payload AS args, {tail}
FROM {src} e
WHERE e.hookpoint = '{hook}' AND json_type(e.payload, '$.{key}') = 'OBJECT'""")

    list_slot("module.fwd.pre", "inputs", _MODULE_OBJ)
    dict_slot("module.fwd.post", "output", _MODULE_OBJ)
    list_slot("module.fwd.post", "outputs", _MODULE_OBJ)
    list_slot("module.bwd", "grad_input", _MODULE_OBJ)
    list_slot("module.bwd", "grad_output", _MODULE_OBJ)
    comm_obj = "gcq_str(e.payload, '$.op') AS obj, 'comm' AS obj_class, NULL::VARCHAR AS obj_id"
    for hook, key in (("comm.pre", "tensor_pre"), ("comm.post", "tensor_post")):
        dict_slot(hook, key, comm_obj)
        list_slot(hook, key, comm_obj)
    dict_slot("functional.softmax", "output",
              "gcq_str(e.payload, '$.fn') AS obj, 'functional' AS obj_class, NULL::VARCHAR AS obj_id")
    dict_slot("dataloader.batch", "input_ids",
              "'input_ids' AS obj, 'batch' AS obj_class, NULL::VARCHAR AS obj_id")
    parts.append(f"""SELECT {_ev_cols()}, j.key AS obj, 'batch' AS obj_class, NULL::VARCHAR AS obj_id,
  'fields' AS slot, FALSE AS in_list, 0::BIGINT AS idx, j.value AS summary, e.payload AS args, {tail}
FROM {src} e, json_each(e.payload, '$.fields') j
WHERE e.hookpoint = 'dataloader.batch' AND json_type(e.payload, '$.fields') = 'OBJECT' AND j.type = 'OBJECT'""")
    # build.snapshot: per-parameter tensor summary
    parts.append(f"""SELECT {_ev_cols()}, gcq_str(j.value, '$.name') AS obj, 'parameter' AS obj_class,
  NULL::VARCHAR AS obj_id, 'param' AS slot, TRUE AS in_list, CAST(j.key AS BIGINT) AS idx,
  json_extract(j.value, '$.tensor_summary') AS summary, e.payload AS args, {tail}
FROM {src} e, json_each(e.payload, '$.model.parameters') j
WHERE e.hookpoint = 'build.snapshot' AND json_type(e.payload, '$.model.parameters') = 'ARRAY'
  AND j.type = 'OBJECT' AND json_type(j.value, '$.tensor_summary') = 'OBJECT'""")
    # cross-rank replica gathers: one row per member of the gathered group
    for hook, key, cls in (("build.snapshot", "cross_rank_cksums", "param"),
                           ("build.snapshot", "cross_rank_buffer_cksums", "buffer"),
                           ("optim.step.pre", "cross_rank_grad_cksums", "grad")):
        parts.append(f"""SELECT {_ev_cols()}, gcq_str(x.value, '$.name') AS obj, '{cls}' AS obj_class,
  NULL::VARCHAR AS obj_id, 'replica' AS slot, TRUE AS in_list, CAST(g.key AS BIGINT) AS idx,
  NULL::JSON AS summary, e.payload AS args, g.value::VARCHAR AS cksum,
  COALESCE(gcq_dbl(x.value, '$.group_size'), 1.0) AS xr_group_size
FROM {src} e, json_each(e.payload, '$.{key}') x, json_each(x.value, '$.gathered_cksums') g
WHERE e.hookpoint = '{hook}' AND json_type(e.payload, '$.{key}') = 'ARRAY' AND x.type = 'OBJECT'
  AND json_type(x.value, '$.gathered_cksums') = 'ARRAY'""")
    return _stats_select("\nUNION ALL\n".join(parts))


def event_sql(src: str) -> str:
    return f"""SELECT {_ev_cols()},
  gcq_str(e.payload, '$.kind') AS kind,
  COALESCE(gcq_str(e.payload, '$.fn'),
           CASE WHEN e.hookpoint LIKE 'comm.%' THEN 'torch.distributed.' || gcq_str(e.payload, '$.op')
                WHEN e.hookpoint LIKE 'optim.step.%' THEN gcq_str(e.payload, '$.optimizer_class') || '.step'
                WHEN e.hookpoint = 'scheduler.init' THEN gcq_str(e.payload, '$.scheduler_class') || '.__init__'
                WHEN e.hookpoint = 'checkpoint.call' THEN 'torch.utils.checkpoint.checkpoint'
                WHEN e.hookpoint LIKE 'module.%' OR e.hookpoint = 'attention.fwd.post'
                     THEN gcq_str(e.payload, '$.module_class') || '.forward'
                ELSE e.hookpoint END) AS fn,
  gcq_str(e.payload, '$.module_class') AS module_class,
  gcq_str(e.payload, '$.module_name') AS module_name,
  e.payload::JSON AS args
FROM {src} e"""


def obj_sql(src: str) -> str:
    return f"""WITH p AS (
  SELECT e.event_id, e.rank, j.value AS entry, gcq_str(j.value, '$.name') AS obj
  FROM {src} e, json_each(e.payload, '$.model.parameters') j
  WHERE e.hookpoint = 'build.snapshot' AND json_type(e.payload, '$.model.parameters') = 'ARRAY' AND j.type = 'OBJECT'
), x AS (
  SELECT e.event_id, e.rank, gcq_str(j.value, '$.name') AS obj, j.value AS xe, 'param' AS kind
  FROM {src} e, json_each(e.payload, '$.cross_rank_cksums') j
  WHERE e.hookpoint = 'build.snapshot' AND json_type(e.payload, '$.cross_rank_cksums') = 'ARRAY' AND j.type = 'OBJECT'
), xb AS (
  SELECT e.event_id, e.rank, gcq_str(j.value, '$.name') AS obj, j.value AS xe, 'buffer' AS kind
  FROM {src} e, json_each(e.payload, '$.cross_rank_buffer_cksums') j
  WHERE e.hookpoint = 'build.snapshot' AND json_type(e.payload, '$.cross_rank_buffer_cksums') = 'ARRAY' AND j.type = 'OBJECT'
)
SELECT p.event_id, p.rank, p.obj, 'param' AS kind,
  gcq_str(p.entry, '$.dtype') AS dtype, json_extract(p.entry, '$.shape') AS shape,
  json_extract(p.entry, '$.requires_grad') AS requires_grad, json_extract(p.entry, '$.is_leaf') AS is_leaf,
  gcq_str(p.entry, '$.semantic.replica_group_kind') AS replica_group_kind,
  list_filter([CASE WHEN gcq_truthy(p.entry, '$.attr_tensor_model_parallel') THEN 'TP' END,
               CASE WHEN gcq_truthy(p.entry, '$.attr_expert_parallel') THEN 'EP' END,
               CASE WHEN gcq_truthy(p.entry, '$.has_attr_ds_id') OR gcq_truthy(p.entry, '$.semantic.is_fsdp_flat')
                    THEN 'DP' END], s -> s IS NOT NULL) AS shard_dims,
  json_extract(p.entry, '$.semantic.shard_dim') AS shard_dim,
  json_extract(p.entry, '$.semantic.sequence_parallel') AS sequence_parallel,
  gcq_dbl(p.entry, '$.semantic.tp_size') AS tp_size, gcq_dbl(p.entry, '$.semantic.dp_size') AS dp_size,
  gcq_dbl(p.entry, '$.semantic.ep_size') AS ep_size,
  gcq_str(p.entry, '$.semantic.source_adapter') AS source_adapter,
  json_extract(p.entry, '$.attr_tensor_model_parallel') AS attr_tensor_model_parallel,
  json_extract(p.entry, '$.attr_expert_parallel') AS attr_expert_parallel,
  json_extract(p.entry, '$.has_attr_ds_id') AS has_attr_ds_id,
  json_extract(p.entry, '$.semantic.is_fsdp_flat') AS is_fsdp_flat,
  COALESCE(gcq_dbl(x.xe, '$.group_size'), CASE WHEN x.xe IS NOT NULL THEN 1.0 END) AS xr_group_size,
  json_extract(x.xe, '$.all_equal') AS xr_all_equal,
  json_extract(x.xe, '$.local_cksum')::VARCHAR AS local_cksum,
  p.entry::JSON AS entry, json_extract(p.entry, '$.semantic') AS semantic
FROM p LEFT JOIN x ON x.event_id = p.event_id AND x.obj = p.obj
UNION ALL
SELECT xb.event_id, xb.rank, xb.obj, 'buffer' AS kind, NULL, NULL, NULL, NULL, NULL, []::VARCHAR[],
  NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
  COALESCE(gcq_dbl(xb.xe, '$.group_size'), 1.0), json_extract(xb.xe, '$.all_equal'),
  json_extract(xb.xe, '$.local_cksum')::VARCHAR, xb.xe::JSON, NULL
FROM xb"""


def topo_sql(src: str) -> str:
    """Only dimension sizes are recorded. grp/coord stay NULL (never recorded)."""
    return f"""WITH b AS (SELECT * FROM {src} WHERE hookpoint = 'build.snapshot'),
inv AS (
  SELECT b.rank, 'TP' AS dim, gcq_dbl(b.payload, '$.framework_invariants.megatron.tensor_model_parallel_size') AS size,
         'framework_invariants.megatron.tensor_model_parallel_size' AS source FROM b
  UNION ALL
  SELECT b.rank, 'PP', gcq_dbl(b.payload, '$.framework_invariants.megatron.pipeline_model_parallel_size'),
         'framework_invariants.megatron.pipeline_model_parallel_size' FROM b
  UNION ALL
  SELECT b.rank, 'EP', gcq_dbl(b.payload, '$.framework_invariants.megatron.expert_model_parallel_size'),
         'framework_invariants.megatron.expert_model_parallel_size' FROM b
  UNION ALL
  SELECT b.rank, 'DP', gcq_dbl(j.value, '$.semantic.dp_size'), 'model.parameters[*].semantic.dp_size'
  FROM b, json_each(b.payload, '$.model.parameters') j
  UNION ALL
  SELECT b.rank, 'TP', gcq_dbl(j.value, '$.semantic.tp_size'), 'model.parameters[*].semantic.tp_size'
  FROM b, json_each(b.payload, '$.model.parameters') j
  UNION ALL
  SELECT b.rank, 'EP', gcq_dbl(j.value, '$.semantic.ep_size'), 'model.parameters[*].semantic.ep_size'
  FROM b, json_each(b.payload, '$.model.parameters') j
)
SELECT DISTINCT rank, dim, size, NULL::VARCHAR AS grp, NULL::BIGINT AS coord, source
FROM inv WHERE size IS NOT NULL"""


def config_sql(src: str) -> str:
    return f"""SELECT e.event_id, e.rank, e.step, j.key AS framework, j.type AS inv_type, j.value::JSON AS inv
FROM {src} e, json_each(e.payload, '$.framework_invariants') j
WHERE e.hookpoint = 'build.snapshot' AND json_type(e.payload, '$.framework_invariants') = 'OBJECT'"""


def optgroup_sql(src: str) -> str:
    return f"""SELECT e.event_id, e.rank, e.step, CAST(j.key AS BIGINT) AS idx, j.value::JSON AS item
FROM {src} e, json_each(e.payload, '$.optimizer.param_groups') j
WHERE e.hookpoint = 'build.snapshot' AND json_type(e.payload, '$.optimizer.param_groups') = 'ARRAY'"""


VIEW_BUILDERS = {"State": state_sql, "Event": event_sql, "Obj": obj_sql, "Topo": topo_sql,
                 "Config": config_sql, "OptGroup": optgroup_sql}


def install_macros(conn) -> None:
    for m in MACROS:
        conn.execute(m)


def install_views(conn, src: str = "events", materialize: bool = False, only=None) -> None:
    """Create TEMP views State, Event, Topo, Obj, Config, OptGroup over relation `src`.

    `src` is the physical events relation, e.g. 'events' or 'tr.events' after
    ATTACH '<trace>.duckdb' AS tr (READ_ONLY)."""
    install_macros(conn)
    for name, fn in VIEW_BUILDERS.items():
        if only is not None and name not in only:
            continue
        kind = "TABLE" if materialize else "VIEW"
        conn.execute(f"DROP VIEW IF EXISTS {name}")
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.execute(f"CREATE TEMP {kind} {name} AS {fn(src)}")


def open_trace(path: str, materialize: bool = False, threads: int = 1, memory_limit: str = "4GB"):
    """In-memory DuckDB connection with the trace attached read-only and the views installed."""
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"ATTACH '{path}' AS tr (READ_ONLY)")
    install_views(con, "tr.events", materialize=materialize)
    return con
