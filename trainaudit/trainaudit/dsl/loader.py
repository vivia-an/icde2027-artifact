"""YAML → Predicate loader with schema validation.

Recognises the minimal field set from doc 22 §A1; raises `LoaderError` with
clear context on every malformed input. Test fixtures must drive both the
positive cases (7 dsl_native rules parse cleanly) and 5+ negative cases
(missing hookpoint, negative tolerance, wrong template, bad payload_path,
unknown family) — see `tests/dsl/test_loader.py`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

from .predicate import (Bound, BoundCondition, BoundKind, Precondition,
                        Predicate, Scope, Template, Tolerance)
from .templates import is_compatible

# -- public ----------------------------------------------------------------


class LoaderError(ValueError):
    """Raised on any malformed YAML predicate. Message must include rule id
    if known, the offending field, and what the loader expected."""


def load_predicate(src: Union[str, Path, Dict[str, Any]]) -> Predicate:
    """Load one predicate from a YAML path, raw YAML string, or already-parsed dict."""
    if isinstance(src, Path) or (isinstance(src, str)
                                 and not src.lstrip().startswith(("id:", "{", "-"))):
        path = Path(src)
        if not path.exists():
            raise LoaderError(f"predicate file not found: {path}")
        if yaml is None:
            raise LoaderError("PyYAML required to load YAML predicates "
                              "(`pip install pyyaml`)")
        with path.open() as f:
            data = yaml.safe_load(f)
    elif isinstance(src, dict):
        data = src
    else:
        if yaml is None:
            raise LoaderError("PyYAML required to load YAML predicates "
                              "(`pip install pyyaml`)")
        data = yaml.safe_load(src)
    if not isinstance(data, dict):
        raise LoaderError(f"predicate root must be a mapping, got "
                          f"{type(data).__name__}")
    return _from_dict(data)


def load_predicates_dir(dir_path: Union[str, Path]) -> List[Predicate]:
    p = Path(dir_path)
    if not p.exists():
        raise LoaderError(f"predicates dir not found: {p}")
    out: List[Predicate] = []
    for f in sorted(p.rglob("*.yaml")):
        out.append(load_predicate(f))
    return out


# -- internals -------------------------------------------------------------

_PAYLOAD_PATH_RE = re.compile(r"^\$\.[A-Za-z_][\w.\[\]\*]*$")
_HOOKPOINT_RE = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)+$")


def _validate_hookpoint(hp: Any, ctx: str) -> Union[str, List[str]]:
    if isinstance(hp, str):
        if not _HOOKPOINT_RE.match(hp):
            raise LoaderError(f"{ctx} scope.hookpoint: '{hp!r}' is not a "
                              f"dotted lowercase identifier")
        return hp
    if isinstance(hp, list) and hp and all(isinstance(x, str) for x in hp):
        for x in hp:
            if not _HOOKPOINT_RE.match(x):
                raise LoaderError(f"{ctx} scope.hookpoint[]: '{x!r}' is not a "
                                  f"dotted lowercase identifier")
        return list(hp)
    raise LoaderError(f"{ctx} scope.hookpoint: must be string or non-empty "
                      f"list of strings, got {type(hp).__name__}")


def _req(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise LoaderError(f"{ctx}: missing required field '{key}'")
    return d[key]


def _from_dict(d: Dict[str, Any]) -> Predicate:
    pid = d.get("id") or "<unknown>"
    ctx = f"predicate '{pid}'"
    rid = _req(d, "id", "predicate")
    template_raw = _req(d, "template", ctx)
    try:
        template = Template(template_raw)
    except ValueError as e:
        raise LoaderError(f"{ctx}: unknown template '{template_raw}'. "
                          f"Valid: {[t.value for t in Template]}") from e

    scope_raw = _req(d, "scope", ctx)
    if not isinstance(scope_raw, dict):
        raise LoaderError(f"{ctx}: scope must be a mapping")
    hookpoint = _validate_hookpoint(_req(scope_raw, "hookpoint", f"{ctx} scope"), ctx)
    payload_path = scope_raw.get("payload_path")
    if payload_path is not None and not _PAYLOAD_PATH_RE.match(payload_path):
        raise LoaderError(f"{ctx} scope.payload_path: '{payload_path}' must "
                          f"start with '$.', use only word/dot/[]/* chars")
    walk = bool(scope_raw.get("walk_tensor_summaries", False))
    tensor_sig = bool(scope_raw.get("tensor_signature", False))
    if walk and payload_path is not None:
        raise LoaderError(f"{ctx} scope: walk_tensor_summaries and payload_path "
                          f"are mutually exclusive")
    if walk and tensor_sig:
        raise LoaderError(f"{ctx} scope: walk_tensor_summaries and "
                          f"tensor_signature are mutually exclusive")
    scope = Scope(hookpoint=hookpoint,
                  module_class_regex=scope_raw.get("module_class_regex"),
                  payload_path=payload_path,
                  walk_tensor_summaries=walk,
                  tensor_signature=tensor_sig)

    bound_raw = _req(d, "bound", ctx)
    if not isinstance(bound_raw, dict):
        raise LoaderError(f"{ctx}: bound must be a mapping")
    bk_raw = _req(bound_raw, "kind", f"{ctx} bound")
    try:
        bk = BoundKind(bk_raw)
    except ValueError as e:
        raise LoaderError(f"{ctx} bound.kind: '{bk_raw}' invalid. "
                          f"Valid: {[k.value for k in BoundKind]}") from e
    if not is_compatible(template, bk):
        raise LoaderError(
            f"{ctx}: template {template.value} is incompatible with "
            f"bound.kind {bk.value}. See dsl/templates.py")
    field_raw = _req(bound_raw, "field", f"{ctx} bound")
    if not (isinstance(field_raw, str) or
            (isinstance(field_raw, list) and field_raw
             and all(isinstance(x, str) for x in field_raw))):
        raise LoaderError(f"{ctx} bound.field: must be string or non-empty "
                          f"list of strings")
    value = bound_raw.get("value")
    value_is_field = bool(bound_raw.get("value_is_field", False))
    if value_is_field and not isinstance(value, str):
        raise LoaderError(f"{ctx} bound: value_is_field=True requires "
                          f"value to be a field name string, got {type(value).__name__}")
    conditions_raw = bound_raw.get("conditions")
    conditions: Optional[list] = None
    if conditions_raw is not None:
        if not isinstance(conditions_raw, list) or not conditions_raw:
            raise LoaderError(f"{ctx} bound.conditions: must be non-empty list")
        if value is not None or bound_raw.get("op") is not None:
            raise LoaderError(f"{ctx} bound: conditions and op/value are "
                              f"mutually exclusive")
        conditions = []
        for i, c in enumerate(conditions_raw):
            if not isinstance(c, dict) or "op" not in c:
                raise LoaderError(
                    f"{ctx} bound.conditions[{i}]: must be mapping with 'op'")
            cv = c.get("value")
            cvf = bool(c.get("value_is_field", False))
            if cv is None and not cvf:
                raise LoaderError(
                    f"{ctx} bound.conditions[{i}]: value required")
            conditions.append(BoundCondition(op=c["op"], value=cv,
                                             value_is_field=cvf))
    bound = Bound(kind=bk, field=field_raw, op=bound_raw.get("op"),
                  value=value, value_is_field=value_is_field,
                  conditions=conditions)

    tolerance = None
    if "tolerance" in d:
        t = d["tolerance"]
        if not isinstance(t, dict):
            raise LoaderError(f"{ctx}: tolerance must be a mapping")
        for k in ("abs", "rel", "quantile_pct"):
            v = t.get(k)
            if v is not None and (not isinstance(v, (int, float)) or v < 0):
                raise LoaderError(
                    f"{ctx} tolerance.{k}: must be non-negative number, got {v!r}")
        tolerance = Tolerance(abs_=t.get("abs"), rel=t.get("rel"),
                              quantile_pct=t.get("quantile_pct"))

    precondition = None
    if "precondition" in d:
        pc = d["precondition"]
        if not isinstance(pc, dict) or "expr" not in pc:
            raise LoaderError(f"{ctx}: precondition must be a mapping with 'expr'")
        precondition = Precondition(expr=pc["expr"])

    family = d.get("family", "")
    evidence = d.get("evidence_bugs") or []
    if not isinstance(evidence, list):
        raise LoaderError(f"{ctx}: evidence_bugs must be a list")
    description = d.get("description", "")
    min_tier = d.get("min_tier", "T0_PYTORCH")

    return Predicate(
        id=rid, template=template, scope=scope, bound=bound,
        family=family, description=description,
        evidence_bugs=list(evidence), precondition=precondition,
        tolerance=tolerance, min_tier=min_tier,
    )
