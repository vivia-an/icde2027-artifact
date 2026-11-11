"""TrainAudit invariant mining.

The outer loop is mine_catalog: the 18 frozen rule files, one direction
at a time. Layer 2 enumerates predicates inside a kept direction.
Layer 3: healthy-trace validation + tolerance auto-learning (layer3_validate.py)
Layer 4: spurious filter (LLM, layer4_filter.py — P1-gated)

A kept direction plus a trace schema yields up to 50 predicates.
Healthy traces then keep the ones that hold. A hypothesis whose relation
is not the current direction is dropped before that expansion.
"""
from .hypothesis_schema import Hypothesis, RelationType
from .layer1_hypothesis import CATALOG18, mine_catalog, propose_hypotheses
from .layer2_enumerate import enumerate_predicates, schema_introspect
from .layer3_validate import (HealthyValidationResult, infer_tolerance,
                               validate_against_healthy)
from .layer4_filter import FilterDecision, filter_predicates

__all__ = [
    "Hypothesis", "RelationType",
    "CATALOG18", "mine_catalog", "propose_hypotheses",
    "enumerate_predicates", "schema_introspect",
    "HealthyValidationResult", "infer_tolerance",
    "validate_against_healthy",
    "FilterDecision", "filter_predicates",
]
