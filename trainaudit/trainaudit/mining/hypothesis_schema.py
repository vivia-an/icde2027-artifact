"""Hypothesis schema — the structured input to the Layer 2 enumerator.

A Hypothesis names a *pattern* that an invariant could take, without
binding it to any specific trace field. The Layer 2 enumerator combines
the hypothesis with trace schema introspection (which fields actually
exist in events) to produce a list of concrete `Predicate` candidates.

The 6 paper relation types map 1-to-1 to the 4 DSL templates plus
2 cross-event derivations:

  cross_rank_equal     → STRUCTURAL_PRESENCE on build.snapshot.cross_rank_cksums
  tensor_stat_bound    → TENSOR_STAT_BOUND   (numeric stats on tensor outputs)
  payload_field_compare→ PAYLOAD_FIELD_COMPARE (one or two scalar fields)
  cross_step_monotonic → PAYLOAD_FIELD_COMPARE with bound.kind=monotonic
  structural_presence  → STRUCTURAL_PRESENCE (keys / counts must be non-zero)
  conditional_check    → CONDITIONAL_CHECK (precondition gates the bound)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RelationType(str, Enum):
    CROSS_RANK_EQUAL = "cross_rank_equal"
    TENSOR_STAT_BOUND = "tensor_stat_bound"
    PAYLOAD_FIELD_COMPARE = "payload_field_compare"
    CROSS_STEP_MONOTONIC = "cross_step_monotonic"
    STRUCTURAL_PRESENCE = "structural_presence"
    CONDITIONAL_CHECK = "conditional_check"


@dataclass
class Hypothesis:
    """One pattern statement.

    Example — "for every replica param, cross-rank cksums must agree":
        Hypothesis(
            relation_type=RelationType.CROSS_RANK_EQUAL,
            entities=["param"],
            dimensions=["rank"],
            scope_hint={"hookpoint": "build.snapshot",
                         "payload_path": "$.cross_rank_cksums[*]"},
        )

    Example — "in any module forward output, l2_norm must be finite":
        Hypothesis(
            relation_type=RelationType.TENSOR_STAT_BOUND,
            entities=["tensor"],
            dimensions=[],
            scope_hint={"hookpoints": ["module.fwd.post"]},
        )
    """

    relation_type: RelationType
    entities: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    scope_hint: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.relation_type, RelationType):
            self.relation_type = RelationType(self.relation_type)
