"""Layer 2: template enumerator.

Given a `Hypothesis` and a trace store (used for schema introspection),
produce up to ~50 concrete `Predicate` candidates. The enumerator is
deterministic — no LLM. Output predicates feed Layer 3 (healthy-trace
validator) which keeps the ones that hold across multiple healthy runs.

Schema introspection: walk `events.payload` JSON for each hookpoint,
collect the set of leaf keys and infer their primitive types. The
resulting `{hookpoint: {field_path: type}}` map is the substrate the
enumerator pulls from.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from ..dsl import (Bound, BoundCondition, BoundKind, Precondition, Predicate,
                    Scope, Template)
from ..store import TraceStore
from .hypothesis_schema import Hypothesis, RelationType


# How many candidates per hypothesis (doc 22 §B1: 5–50).
_MAX_CANDIDATES_PER_HYP = 50


# ---- schema introspection -----------------------------------------------


def _walk(obj: Any, prefix: str = "$"):
    """Yield (path, primitive_type) for every leaf in a JSON structure."""
    if obj is None:
        yield prefix, "null"
        return
    if isinstance(obj, bool):
        yield prefix, "bool"
        return
    if isinstance(obj, (int, float)):
        yield prefix, "number"
        return
    if isinstance(obj, str):
        yield prefix, "string"
        return
    if isinstance(obj, list):
        # Don't enumerate every list index; record list with element type
        for elem in obj[:1]:  # only inspect first element type
            yield prefix + "[*]", "list"
            yield from _walk(elem, prefix + "[*]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{prefix}.{k}")


def schema_introspect(store: TraceStore) -> Dict[str, Dict[str, str]]:
    """Return {hookpoint: {json_path: type}} for every distinct hookpoint."""
    store.flush()
    rows = store.conn.execute(
        "SELECT DISTINCT hookpoint, payload FROM events").fetchall()
    schema: Dict[str, Dict[str, str]] = defaultdict(dict)
    for hp, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:  # noqa: BLE001
            continue
        for path, typ in _walk(p):
            schema[hp][path] = typ
    return dict(schema)


# ---- enumerators per relation type --------------------------------------


def enumerate_predicates(hyp: Hypothesis,
                          schema: Dict[str, Dict[str, str]]
                          ) -> List[Predicate]:
    """Return up to _MAX_CANDIDATES_PER_HYP predicate candidates.

    Two-pass: (1) hardcoded shape-specific enumerator (existing 6 funcs);
    (2) parametric generator that uses hyp.entities + hyp.dimensions to
    auto-construct predicates against the schema. Emits both — L3 filters
    spurious ones via healthy validation.
    """
    fn = _ENUMERATORS.get(hyp.relation_type)
    out: List[Predicate] = []
    if fn is not None:
        out.extend(fn(hyp, schema))
    out.extend(_parametric_enum(hyp, schema))
    seen_ids = set()
    deduped = []
    for p in out:
        if p.id in seen_ids:
            continue
        seen_ids.add(p.id)
        deduped.append(p)
    return deduped[:_MAX_CANDIDATES_PER_HYP]


# ---- parametric enumerator (uses hyp.entities + hyp.dimensions) ----------


def _resolve_hookpoints(hyp: Hypothesis,
                         schema: Dict[str, Dict[str, str]]) -> List[str]:
    """Map L1's `entities` (often free-form) to actual hookpoints in schema.

    LLM tends to emit entities like 'comm.post', 'optim.step.pre',
    'module.fwd.post' (which match hookpoints) — but also things like
    'comm.post:all_reduce' or 'average_tensor:rank_and_offsets' (which
    don't). We do a coarse match: an entity matches a hookpoint if the
    hookpoint string is a prefix of the entity (or vice versa).
    """
    out = []
    for ent in hyp.entities:
        ent_norm = ent.split(":")[0].strip()
        for hp in schema.keys():
            if hp == ent_norm or hp.startswith(ent_norm) or ent_norm.startswith(hp):
                if hp not in out:
                    out.append(hp)
    if not out:
        out = list(schema.keys())  # fall back to all hookpoints if no match
    return out


def _resolve_field_paths(dim: str, hp_schema: Dict[str, str]) -> List[str]:
    """Find candidate JSON field paths in the hookpoint schema matching the
    dimension name. Be tolerant — LLM names rarely match schema exactly.

    Matching tiers (most specific → least):
      1. exact `$.{dim}`
      2. suffix `.{dim}` or `.{leaf-of-dim}`
      3. leaf-of-dim equals last segment of any path
      4. partial substring (last resort)
    """
    candidates = []
    dim_norm = dim.lstrip("$.").strip()
    target = f"$.{dim_norm}"
    leaf = dim_norm.split(".")[-1]

    # Tier 1: exact match
    if target in hp_schema:
        candidates.append(target)

    # Tier 2: suffix match on the full dim or leaf
    for path in hp_schema:
        if path == target:
            continue
        if path.endswith(f".{dim_norm}") and path not in candidates:
            candidates.append(path)
        if path.endswith(f".{leaf}") and path not in candidates:
            candidates.append(path)

    # Tier 3: leaf == last segment of any path
    if not candidates:
        for path in hp_schema:
            if path.split(".")[-1] == leaf and path not in candidates:
                candidates.append(path)

    return candidates


def _enumerate_numeric_fields(hp_schema: Dict[str, str]) -> List[str]:
    """All `$.path.to.numeric_field` paths in the hookpoint schema."""
    return [p for p, t in hp_schema.items() if t == "number"]


def _is_likely_param_groups_innerlist(path: str) -> bool:
    """Filter out per-element fields inside list payloads (param_groups[*]
    inner fields would emit redundant predicates per element)."""
    return "[*]" in path


def _parametric_enum(hyp: Hypothesis,
                      schema: Dict[str, Dict[str, str]]) -> Iterable[Predicate]:
    """Schema-aware predicate generator. Produces predicates parameterised
    on (hookpoint, field) pairs derived from `hyp.entities` x
    `hyp.dimensions` x `schema`. Does NOT need a hardcoded template list.
    """
    if not hyp.entities:
        return  # at least one entity hint required

    hookpoints = _resolve_hookpoints(hyp, schema)
    rt = hyp.relation_type

    for hp in hookpoints:
        hp_schema = schema.get(hp, {})

        # Build the set of (field_path, source) pairs we'll emit predicates for.
        # If L1 specified dimensions, prefer those (after fuzzy resolution);
        # otherwise, fall back to all numeric fields at this hookpoint
        # (lets parametric mode work even when L1's dims are vague).
        field_paths_set: List[str] = []
        if hyp.dimensions:
            for dim in hyp.dimensions:
                for fp in _resolve_field_paths(dim, hp_schema):
                    if fp not in field_paths_set and \
                            not _is_likely_param_groups_innerlist(fp):
                        field_paths_set.append(fp)
        if not field_paths_set:
            field_paths_set = [p for p in _enumerate_numeric_fields(hp_schema)
                               if not _is_likely_param_groups_innerlist(p)]

        for fp in field_paths_set:
                field_leaf = fp.lstrip("$.").rsplit(".", 1)[-1]
                json_path = fp  # e.g. $.output.l2_norm or $.total_grad_l2

                # Pick predicate shape based on relation_type
                if rt == RelationType.CROSS_STEP_MONOTONIC:
                    yield Predicate(
                        id=f"hyp/param/csm/{hp}/{json_path[2:]}",
                        template=Template.PAYLOAD_FIELD_COMPARE,
                        family="F-PARAM",
                        scope=Scope(hookpoint=hp),
                        bound=Bound(kind=BoundKind.MONOTONIC,
                                    field=json_path[2:]),
                        description=(f"{json_path} should be monotonic "
                                     f"across {hp} (parametric)."),
                    )

                elif rt == RelationType.PAYLOAD_FIELD_COMPARE:
                    # Numeric fields → positivity check (most useful baseline)
                    typ = hp_schema.get(fp, "")
                    if typ == "number":
                        yield Predicate(
                            id=f"hyp/param/pfc/{hp}/{json_path[2:]}_positive",
                            template=Template.PAYLOAD_FIELD_COMPARE,
                            family="F-PARAM",
                            scope=Scope(hookpoint=hp),
                            bound=Bound(kind=BoundKind.BOUND,
                                        field=json_path[2:],
                                        op=">", value=0),
                            description=(f"{json_path} > 0 at {hp} "
                                         f"(parametric)."),
                        )

                elif rt == RelationType.TENSOR_STAT_BOUND:
                    # For tensor stat fields like l2_norm, max, etc.
                    # emit "is finite" + "> 0" baselines
                    if any(x in fp for x in ["l2_norm", ".max", ".min", "_norm"]):
                        yield Predicate(
                            id=f"hyp/param/tsb/{hp}/{json_path[2:]}_positive",
                            template=Template.PAYLOAD_FIELD_COMPARE,
                            family="F-PARAM",
                            scope=Scope(hookpoint=hp),
                            bound=Bound(kind=BoundKind.BOUND,
                                        field=json_path[2:],
                                        op=">", value=0),
                            description=(f"{json_path} > 0 at {hp} "
                                         f"(parametric tensor-stat)."),
                        )

                elif rt == RelationType.STRUCTURAL_PRESENCE:
                    # Emit "field present and non-zero" if numeric; skip non-numeric
                    typ = hp_schema.get(fp, "")
                    if typ == "number":
                        yield Predicate(
                            id=f"hyp/param/sp/{hp}/{json_path[2:]}_nonzero",
                            template=Template.STRUCTURAL_PRESENCE,
                            family="F-PARAM",
                            scope=Scope(hookpoint=hp),
                            bound=Bound(kind=BoundKind.BOUND,
                                        field=json_path[2:],
                                        op=">", value=0),
                            description=(f"{json_path} > 0 at {hp} "
                                         f"(parametric structural)."),
                        )

                elif rt == RelationType.CROSS_RANK_EQUAL:
                    # Emit cross_rank equality if the field has a cksums
                    # neighbor. For now: if path ends with .all_equal, emit
                    if fp.endswith(".all_equal"):
                        yield Predicate(
                            id=f"hyp/param/cre/{hp}/{json_path[2:]}",
                            template=Template.STRUCTURAL_PRESENCE,
                            family="F-PARAM",
                            scope=Scope(hookpoint=hp),
                            bound=Bound(kind=BoundKind.EQUALITY,
                                        field=json_path[2:], value=True),
                            description=(f"{json_path} must be true at "
                                         f"{hp} (parametric)."),
                        )


def _enum_cross_rank_equal(hyp: Hypothesis,
                            schema: Dict[str, Dict[str, str]]
                            ) -> Iterable[Predicate]:
    """Look for a `cross_rank_cksums[*]` list with `all_equal` field —
    exactly the shape produced by trainaudit's build_snapshot adapter
    when at least one framework adapter is active. Yields one predicate
    matching `T1-replica-cksum-equal`.
    """
    bs_schema = schema.get("build.snapshot", {})
    if "$.cross_rank_cksums[*].all_equal" not in bs_schema:
        return
    yield Predicate(
        id="hyp/cross_rank_equal/replica_cksum",
        template=Template.STRUCTURAL_PRESENCE,
        family="F1",
        description="Replica params must have equal cksum across replica group.",
        scope=Scope(
            hookpoint="build.snapshot",
            payload_path="$.cross_rank_cksums[*]",
        ),
        precondition=Precondition(
            expr="CAST(json_extract(row_payload, '$.group_size') AS INTEGER) > 1"),
        bound=Bound(kind=BoundKind.EQUALITY, field="all_equal", value=True),
        evidence_bugs=["B1", "M-005"],
        min_tier="T1_FW_METADATA",
    )


def _enum_payload_field_compare(hyp: Hypothesis,
                                 schema: Dict[str, Dict[str, str]]
                                 ) -> Iterable[Predicate]:
    """For every numeric field that has a 'companion' threshold field at
    the same hookpoint (`x` and `max_x`, `pre_norm` and `max_norm`, etc.),
    propose `x <= max_x`. This is the pattern that catches B11
    (post_norm <= max_norm)."""
    seen = 0
    pairs = [
        ("post_norm", "max_norm", "<="),
        ("post_norm", "pre_norm", "<="),  # weak check
        ("lr", 0, ">"),
        ("learning_rate", 0, ">"),
    ]
    for hp, fields in schema.items():
        for left, right, op in pairs:
            left_path = f"$.{left}"
            if left_path not in fields:
                continue
            value: Any = right
            value_is_field = False
            if isinstance(right, str):
                if f"$.{right}" not in fields:
                    continue
                value_is_field = True
            yield Predicate(
                id=f"hyp/pfc/{hp}/{left}_{op}_{right}".replace(" ", ""),
                template=Template.PAYLOAD_FIELD_COMPARE,
                family="F2",
                scope=Scope(hookpoint=hp),
                bound=Bound(kind=BoundKind.BOUND, field=left, op=op,
                            value=value, value_is_field=value_is_field),
                tolerance=None,
                description=f"{left} {op} {right} at {hp}",
            )
            seen += 1
            if seen >= _MAX_CANDIDATES_PER_HYP:
                return


def _enum_tensor_stat_bound(hyp: Hypothesis,
                             schema: Dict[str, Dict[str, str]]
                             ) -> Iterable[Predicate]:
    """For every hookpoint that emits a tensor summary (has dtype + shape),
    propose: no NaN/Inf. (l2/rms bounds need numeric tolerance learned by
    Layer 3.)"""
    targets: List[str] = []
    for hp, fields in schema.items():
        for path in fields:
            if path.endswith(".has_nan") or path.endswith("[*].has_nan"):
                targets.append(hp)
                break
    targets = sorted(set(targets))
    if not targets:
        return
    yield Predicate(
        id="hyp/tsb/no_nan_inf",
        template=Template.TENSOR_STAT_BOUND,
        family="F8",
        scope=Scope(hookpoint=targets, walk_tensor_summaries=True),
        bound=Bound(kind=BoundKind.EQUALITY,
                    field=["has_nan", "has_inf"], value=False),
        description="No traced tensor may contain NaN or Inf.",
    )


def _enum_cross_step_monotonic(hyp: Hypothesis,
                                schema: Dict[str, Dict[str, str]]
                                ) -> Iterable[Predicate]:
    """For every numeric field at a per-step hookpoint that ought to
    increase monotonically (state_step_max, global_step), emit a
    monotonic predicate."""
    candidates = [
        ("optim.step.post", "state_step_max"),
        ("optim.step.post", "state_step_min"),
    ]
    seen = 0
    for hp, field in candidates:
        if hp not in schema or f"$.{field}" not in schema[hp]:
            continue
        yield Predicate(
            id=f"hyp/csm/{hp}/{field}",
            template=Template.PAYLOAD_FIELD_COMPARE,
            family="F5",
            scope=Scope(hookpoint=hp),
            bound=Bound(kind=BoundKind.MONOTONIC, field=field),
            description=f"{field} must increase monotonically across {hp} events.",
        )
        seen += 1
        if seen >= _MAX_CANDIDATES_PER_HYP:
            return


def _enum_structural_presence(hyp: Hypothesis,
                               schema: Dict[str, Dict[str, str]]
                               ) -> Iterable[Predicate]:
    """Build snapshot must report at least one module + one parameter.
    Generic catch for empty-model regressions (M-020-style)."""
    bs_schema = schema.get("build.snapshot", {})
    if ("$.model.n_parameters" in bs_schema
            and "$.model.n_modules" in bs_schema):
        yield Predicate(
            id="hyp/sp/build_has_modules",
            template=Template.STRUCTURAL_PRESENCE,
            family="F4",
            scope=Scope(hookpoint="build.snapshot",
                        payload_path="$.model"),
            bound=Bound(kind=BoundKind.BOUND,
                        field=["n_parameters", "n_modules"],
                        op=">", value=0),
            description="Build snapshot must have parameters and modules.",
        )


def _enum_conditional_check(hyp: Hypothesis,
                             schema: Dict[str, Dict[str, str]]
                             ) -> Iterable[Predicate]:
    """For scheduler.init: when last_epoch != -1 (resume), every
    param_group must carry initial_lr. Catches B12."""
    sched_schema = schema.get("scheduler.init", {})
    if "$.last_epoch" not in sched_schema:
        return
    if "$.param_groups[*].has_initial_lr" not in sched_schema:
        return
    yield Predicate(
        id="hyp/cond/scheduler_resume_initial_lr",
        template=Template.CONDITIONAL_CHECK,
        family="F4",
        scope=Scope(hookpoint="scheduler.init",
                    payload_path="$.param_groups[*]"),
        precondition=Precondition(
            expr="json_extract(payload, '$.last_epoch') != to_json(-1)"),
        bound=Bound(kind=BoundKind.EQUALITY,
                    field="has_initial_lr", value=True),
        description="Scheduler resume requires initial_lr on every param_group.",
        evidence_bugs=["B12", "O-016"],
    )


_ENUMERATORS = {
    RelationType.CROSS_RANK_EQUAL: _enum_cross_rank_equal,
    RelationType.PAYLOAD_FIELD_COMPARE: _enum_payload_field_compare,
    RelationType.TENSOR_STAT_BOUND: _enum_tensor_stat_bound,
    RelationType.CROSS_STEP_MONOTONIC: _enum_cross_step_monotonic,
    RelationType.STRUCTURAL_PRESENCE: _enum_structural_presence,
    RelationType.CONDITIONAL_CHECK: _enum_conditional_check,
}
