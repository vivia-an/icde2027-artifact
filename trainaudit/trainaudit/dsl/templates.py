"""Template metadata — describes which templates allow which Bound kinds.

Used by `loader.py` to validate that a YAML predicate's (template, bound.kind)
combination is well-formed before it reaches the SQL compiler.
"""
from __future__ import annotations

from typing import Dict, FrozenSet

from .predicate import BoundKind, Template

# Which BoundKind(s) each template can carry.
TEMPLATE_BOUND_KINDS: Dict[Template, FrozenSet[BoundKind]] = {
    Template.TENSOR_STAT_BOUND: frozenset({
        BoundKind.BOUND, BoundKind.EQUALITY, BoundKind.PRESENT,
        BoundKind.MONOTONIC,  # MONOTONIC over a tensor stat across steps
    }),
    Template.PAYLOAD_FIELD_COMPARE: frozenset({
        BoundKind.BOUND, BoundKind.EQUALITY, BoundKind.MONOTONIC,
    }),
    Template.CONDITIONAL_CHECK: frozenset({
        BoundKind.BOUND, BoundKind.EQUALITY, BoundKind.PRESENT,
    }),
    Template.STRUCTURAL_PRESENCE: frozenset({
        BoundKind.PRESENT, BoundKind.EQUALITY, BoundKind.BOUND,
    }),
}


def is_compatible(template: Template, kind: BoundKind) -> bool:
    return kind in TEMPLATE_BOUND_KINDS[template]
