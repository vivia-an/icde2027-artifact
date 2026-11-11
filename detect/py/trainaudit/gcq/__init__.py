"""Guarded constraint queries (GCQ) over a TrainAudit trace.

  views.py         logical relations State / Event / Meta(Topo, Obj, Config, OptGroup) as DuckDB
                   views over the physical table events(event_id, step, rank, hookpoint, ts_ns,
                   schema_version, payload JSON)
  legacy_views.py  the same logical relations over the legacy Megatron coredump(step, stage, data)
                   State table built by 07_bench/t2_megatron/gcq_guarded.load
  spec.py          declarative rule spec (JSON) and its validation
  compile.py       compiler spec -> SQL (tuple-level, aggregate, replica-class rewrite of Prop. 1),
                   window w(r), hookpoint index, fragment check
  evaluate.py      run compiled rules on a connection

Nothing in this package changes the trace, the frozen rules, or the online runner.
"""
from .compile import CompiledRule, compile_spec  # noqa: F401
from .spec import Spec, load_spec, load_specs  # noqa: F401
