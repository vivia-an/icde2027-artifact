"""Second view definition: the paper's relations over the legacy Megatron `coredump(step, stage, data)`.

The State table itself is built by the pre-registered A3 evaluator
(benchmark/eval/icde_v1/07_bench/t2_megatron/gcq_guarded.load), unchanged:
  State(dp, tp, pp, cp, rid, step, stage, name, cksum, shape, dtype, requires_grad, grad_cksum, rn)
  (dp,tp,pp,cp) come from the file names coredump_dp*_tp*_pp*_cp*.db; rn = occurrence index of
  (rank, step, stage, name) in emission order (pairs duplicate micro-batch rows).

Views created here (names prefixed GCQ_ so they do not collide with gcq_guarded's own `State`):
  GCQ_State(rank, dp, tp, pp, cp, rid, step, hook, stage, obj, name, micro, cksum, shape, dtype,
            requires_grad, grad_cksum, kind)
      rank = 'dp{dp}_tp{tp}_pp{pp}_cp{cp}', hook = stage, obj = name, micro = rn,
      kind = param (model-*, main-param-*) | grad (main-grad-*) | optstate (optimizer-state-*) | other
  GCQ_Topo(rank, dim, grp, coord, size)  dims DP, TP, PP, CP; grp = coordinates of the other three
      dimensions (equal grp along d <=> same group of dimension d); from the file names.
  GCQ_Obj(obj, kind, shard_dims)
      param:    'TP' in shard_dims unless gcq_guarded.TP_REPL matches the name (Megatron naming
                convention fixed in A3: norms, row-parallel biases, position embeddings)
      grad, optstate: 'DP' in shard_dims iff the run uses the distributed optimizer
                (dist_opt True); [] if False; NULL if unknown (None) -> repl_DP undecidable
  GCQ_Run(dist_opt)
EP is not recorded in the legacy relation; it is not in GCQ_Topo.
"""
import os
import sys

A3_DIR = os.environ.get("LEGACY_A3_DIR", "")

RELATIONS = {
    "State": dict(rank="varchar", dp="bigint", tp="bigint", pp="bigint", cp="bigint", rid="bigint",
                  step="bigint", hook="varchar", stage="varchar", obj="varchar", name="varchar",
                  micro="bigint", cksum="varchar", shape="varchar", dtype="varchar",
                  requires_grad="varchar", grad_cksum="varchar", kind="varchar"),
    "Topo": dict(rank="varchar", dim="varchar", grp="varchar", coord="bigint", size="bigint"),
    "Obj": dict(obj="varchar", kind="varchar", shard_dims="varchar[]"),
    "Run": dict(dist_opt="bool"),
}
META_RELATIONS = {"Topo", "Obj", "Run"}
NAMES = {"State": "GCQ_State", "Topo": "GCQ_Topo", "Obj": "GCQ_Obj", "Run": "GCQ_Run"}


def _gcq_guarded():
    if not A3_DIR:
        raise RuntimeError("set LEGACY_A3_DIR to the legacy evaluator checkout")
    if A3_DIR not in sys.path:
        sys.path.insert(0, A3_DIR)
    import gcq_guarded  # noqa: E402  (pre-registered A3 evaluator, imported, not modified)
    return gcq_guarded


def install(con, dist_opt):
    G = _gcq_guarded()
    con.execute("""CREATE OR REPLACE TEMP VIEW GCQ_State AS
      SELECT 'dp' || dp || '_tp' || tp || '_pp' || pp || '_cp' || cp AS rank, dp, tp, pp, cp, rid,
             step, stage AS hook, stage, name AS obj, name, rn AS micro, cksum, shape, dtype,
             requires_grad, grad_cksum,
             CASE WHEN stage LIKE 'model-%' OR stage LIKE 'main-param-%' THEN 'param'
                  WHEN stage LIKE 'main-grad-%' THEN 'grad'
                  WHEN stage LIKE 'optimizer-state-%' THEN 'optstate' ELSE 'other' END AS kind
      FROM State""")
    con.execute("""CREATE OR REPLACE TEMP TABLE GCQ_Topo AS
      WITH r AS (SELECT DISTINCT dp, tp, pp, cp FROM State),
      n AS (SELECT count(DISTINCT dp) ndp, count(DISTINCT tp) ntp, count(DISTINCT pp) npp,
                   count(DISTINCT cp) ncp FROM State)
      SELECT 'dp' || dp || '_tp' || tp || '_pp' || pp || '_cp' || cp AS rank, d.dim,
             CASE d.dim WHEN 'DP' THEN 'tp' || tp || '_pp' || pp || '_cp' || cp
                        WHEN 'TP' THEN 'dp' || dp || '_pp' || pp || '_cp' || cp
                        WHEN 'PP' THEN 'dp' || dp || '_tp' || tp || '_cp' || cp
                        ELSE 'dp' || dp || '_tp' || tp || '_pp' || pp END AS grp,
             CASE d.dim WHEN 'DP' THEN dp WHEN 'TP' THEN tp WHEN 'PP' THEN pp ELSE cp END AS coord,
             CASE d.dim WHEN 'DP' THEN ndp WHEN 'TP' THEN ntp WHEN 'PP' THEN npp ELSE ncp END AS size
      FROM r, n, (VALUES ('DP'), ('TP'), ('PP'), ('CP')) d(dim)""")
    names = con.execute("SELECT DISTINCT name, kind FROM GCQ_State WHERE name IS NOT NULL").fetchall()
    rows = []
    for name, kind in names:
        if kind == "param":
            sd = [] if G.TP_REPL.search(name) else ["TP"]
        elif kind in ("grad", "optstate"):
            sd = None if dist_opt is None else (["DP"] if dist_opt else [])
        else:
            sd = []
        rows.append((name, kind, sd))
    con.execute("CREATE OR REPLACE TEMP TABLE GCQ_Obj (obj VARCHAR, kind VARCHAR, shard_dims VARCHAR[])")
    if rows:
        con.executemany("INSERT INTO GCQ_Obj VALUES (?, ?, ?)", rows)
    con.execute("CREATE OR REPLACE TEMP TABLE GCQ_Run (dist_opt BOOLEAN)")
    con.execute("INSERT INTO GCQ_Run VALUES (?)", [dist_opt])


def open_collector(coll_dir, dist_opt):
    """gcq_guarded.load(coll_dir) (in-memory DuckDB with its State table) + the GCQ views."""
    G = _gcq_guarded()
    con = G.load(coll_dir)
    install(con, dist_opt)
    return con
