"""Predicate dataclass — structured form of an invariant rule.

Compiles to DuckDB SQL via `dsl/compiler.py` (Workstream A2). The schema
matches doc 22 §A1 minimal field set:
    id, family, template, scope.{hookpoint, module_class_regex, payload_path},
    predicate.{kind: bound|equality|present|monotonic},
    precondition.expr, tolerance.{abs, rel, quantile_pct}, evidence_bugs[]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Union


class Template(str, Enum):
    """The 4 invariant templates (paper §3.2)."""

    TENSOR_STAT_BOUND = "TENSOR_STAT_BOUND"
    PAYLOAD_FIELD_COMPARE = "PAYLOAD_FIELD_COMPARE"
    CONDITIONAL_CHECK = "CONDITIONAL_CHECK"
    STRUCTURAL_PRESENCE = "STRUCTURAL_PRESENCE"


class BoundKind(str, Enum):
    """How `predicate.bound` reads its operand."""

    BOUND = "bound"           # `field <op> threshold` — op in {<,<=,>,>=,==,!=}
    EQUALITY = "equality"     # `field == constant`
    PRESENT = "present"       # `field IS NOT NULL` / list non-empty
    MONOTONIC = "monotonic"   # field strictly increases across event_id sequence


@dataclass
class Scope:
    """Where to look in the trace."""

    hookpoint: Union[str, List[str]]
    """Required. e.g. 'utils.clip_grad.post', 'build.snapshot'.
    May be a list to match any of several hookpoints."""

    module_class_regex: Optional[str] = None
    """Optional regex on payload.module_class (for module-scoped rules)."""

    payload_path: Optional[str] = None
    """Optional JSON path inside payload. Two forms supported:
      - `$.field.sub`  — extract a single object/value
      - `$.field[*]`   — extract each list element as its own row
    """

    walk_tensor_summaries: bool = False
    """If True, the rule applies to every nested dict containing both
    `dtype` and `shape` keys (anywhere in payload). Used by tensor-stat
    rules (T0-no-nan-inf, etc.)."""

    tensor_signature: bool = False
    """If True, the row-level payload is treated as a tensor summary and
    the compiler exposes derived fields:
      - rms       = l2_norm / sqrt(numel(shape))
      - n_rows    = product(shape[:-1])
      - one_hot   = (abs(amax-1) < 1e-3 AND |l2² - n_rows| < 0.05*n_rows)
    Bound.field can reference any of these names. Triggers a Python
    postprocess (no SQL JSONPath gymnastics)."""

    @property
    def hookpoints(self) -> List[str]:
        return [self.hookpoint] if isinstance(self.hookpoint, str) \
            else list(self.hookpoint)


@dataclass
class Tolerance:
    """Numeric tolerance applied to BOUND/EQUALITY predicates."""

    abs_: Optional[float] = None
    rel: Optional[float] = None
    quantile_pct: Optional[float] = None  # for B2 auto-learning


@dataclass
class BoundCondition:
    """One (op, value) pair inside a multi-condition bound."""

    op: str            # comparison operator
    value: Any         # threshold or field-ref
    value_is_field: bool = False


@dataclass
class Bound:
    """Numeric / boolean condition the field is checked against.

    Multi-field semantics: when `field` is a list, the invariant requires
    *all* fields satisfy the condition; violation fires when *any* one
    fails. (matches `T0-no-nan-inf` `field=[has_nan, has_inf]` value=false)

    Multi-condition semantics: when `conditions` is set (list of (op,value)
    pairs), the invariant requires *all* conditions to hold; violation
    fires if *any* fails. Used for range checks (`0.5 <= rms <= 2.0`).
    """

    kind: BoundKind
    field: Union[str, List[str]]
    """JSON field name or list of names inside the row context.
    Compiler resolves names to json_extract calls."""

    op: Optional[str] = None       # for BOUND: one of <, <=, >, >=, ==, !=
    value: Optional[Any] = None    # threshold (BOUND/EQUALITY)
    value_is_field: bool = False   # if True, `value` is interpreted as
    # another field name (e.g. `post_norm > max_norm`)

    conditions: Optional[List[BoundCondition]] = None
    """If set, replaces (op, value) — list of conditions, all must hold.
    Used for range bounds. Mutually exclusive with single op/value."""

    @property
    def fields(self) -> List[str]:
        return [self.field] if isinstance(self.field, str) else list(self.field)

    @property
    def all_conditions(self) -> List[BoundCondition]:
        """Normalised view: a single op/value comes back as a 1-element
        list. Empty for non-bound predicates (EQUALITY/PRESENT/MONOTONIC)."""
        if self.conditions is not None:
            return list(self.conditions)
        if self.kind == BoundKind.BOUND and self.op is not None:
            return [BoundCondition(op=self.op, value=self.value,
                                   value_is_field=self.value_is_field)]
        return []


@dataclass
class Precondition:
    """SQL/expression that must hold for predicate to fire (else passive)."""

    expr: str
    """Arbitrary DuckDB expression evaluated against trace store. May reference
    `events` and JSON paths via json_extract. Examples:
      - "json_extract(payload, '$.framework_invariants.megatron.calculate_per_token_loss') = 'true'"
      - "EXISTS (SELECT 1 FROM events WHERE hookpoint = 'scheduler.init')"
    """


@dataclass
class Predicate:
    """A complete invariant rule in structured form."""

    id: str
    template: Template
    scope: Scope
    bound: Bound
    family: str = ""
    description: str = ""
    evidence_bugs: List[str] = field(default_factory=list)
    precondition: Optional[Precondition] = None
    tolerance: Optional[Tolerance] = None
    min_tier: str = "T0_PYTORCH"   # Tier name; resolved by runner

    def __post_init__(self) -> None:
        if not isinstance(self.template, Template):
            self.template = Template(self.template)
        if not isinstance(self.bound.kind, BoundKind):
            self.bound.kind = BoundKind(self.bound.kind)
        # cross-field validity
        if self.bound.kind == BoundKind.BOUND and self.bound.conditions is None:
            if self.bound.op not in ("<", "<=", ">", ">=", "==", "!="):
                raise ValueError(
                    f"predicate {self.id}: bound.op must be a comparison "
                    f"operator, got {self.bound.op!r}")
        if self.bound.kind in (BoundKind.BOUND, BoundKind.EQUALITY) \
                and self.bound.conditions is None \
                and self.bound.value is None and not self.bound.value_is_field:
            raise ValueError(
                f"predicate {self.id}: bound.value required for kind "
                f"{self.bound.kind}")
        if self.bound.conditions is not None:
            for c in self.bound.conditions:
                if c.op not in ("<", "<=", ">", ">=", "==", "!="):
                    raise ValueError(
                        f"predicate {self.id}: bound.conditions[].op invalid "
                        f"({c.op!r})")
