"""TrainAudit invariant DSL.

A predicate is the structured form of an invariant rule. Each predicate
fits one of 4 templates and compiles to a DuckDB SQL query (Workstream
A2). The DSL co-exists with hand-written Python rules — see
`registry/MAPPING.md` for the classification of every existing rule.
"""
from .predicate import (Bound, BoundCondition, BoundKind, Precondition,
                        Predicate, Scope, Template, Tolerance)
from .loader import LoaderError, load_predicate, load_predicates_dir
from .compiler import CompiledRule, Violation, compile_predicate
from .runner import run_compiled_one, run_dsl_rules, violation_event_ids

__all__ = [
    "Bound", "BoundCondition", "BoundKind", "Precondition", "Predicate",
    "Scope", "Template", "Tolerance",
    "LoaderError", "load_predicate", "load_predicates_dir",
    "CompiledRule", "Violation", "compile_predicate",
    "run_compiled_one", "run_dsl_rules", "violation_event_ids",
]
