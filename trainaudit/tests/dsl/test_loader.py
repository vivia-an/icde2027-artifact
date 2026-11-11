"""A1 loader unit tests: positive 1 + negative 5 (PoC scope).

Full A1 acceptance is 7 positive + 5 negative — those land alongside the
remaining 6 dsl_native YAMLs (T0_no_nan_inf, T0_optim_lr_positive, ...).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from trainaudit.dsl import (BoundKind, LoaderError, Template, load_predicate,
                            load_predicates_dir)

REGISTRY = Path(__file__).resolve().parents[2] / "trainaudit" / "dsl" / "registry"


# --- positive ---


def test_load_clip_grad_yaml():
    p = load_predicate(REGISTRY / "T0" / "T0_clip_grad_bounded.yaml")
    assert p.id == "T0-clip-grad-bounded"
    assert p.template == Template.PAYLOAD_FIELD_COMPARE
    assert p.scope.hookpoint == "utils.clip_grad.post"
    assert p.bound.kind == BoundKind.BOUND
    # `op` describes the INVARIANT (post must be <= max), not the violation
    assert p.bound.op == "<="
    assert p.bound.field == "post_norm"
    assert p.bound.value == "max_norm"
    assert p.bound.value_is_field is True
    assert p.evidence_bugs == ["B11"]
    assert p.tolerance is not None and p.tolerance.rel == 0.01


def test_load_predicates_dir_finds_all():
    preds = load_predicates_dir(REGISTRY)
    ids = {p.id for p in preds}
    expected = {
        "T0-clip-grad-bounded", "T0-no-nan-inf", "T0-optim-lr-positive",
        "T0-build-has-modules", "T0-initial-lr-present",
        "T0-token-id-in-vocab", "T1-replica-cksum-equal",
    }
    missing = expected - ids
    assert not missing, f"missing predicates from registry: {missing}"


def test_load_no_nan_inf_multi_hookpoint_and_walk():
    p = next(p for p in load_predicates_dir(REGISTRY)
             if p.id == "T0-no-nan-inf")
    assert isinstance(p.scope.hookpoint, list)
    assert "module.fwd.post" in p.scope.hookpoint
    assert p.scope.walk_tensor_summaries is True
    assert p.bound.fields == ["has_nan", "has_inf"]
    assert p.bound.value is False


def test_load_clip_grad_value_is_field():
    p = next(p for p in load_predicates_dir(REGISTRY)
             if p.id == "T0-clip-grad-bounded")
    assert p.bound.value_is_field is True
    assert p.bound.value == "max_norm"


def test_load_initial_lr_has_precondition():
    p = next(p for p in load_predicates_dir(REGISTRY)
             if p.id == "T0-initial-lr-present")
    assert p.precondition is not None
    assert "last_epoch" in p.precondition.expr


# --- negative ---


def _badload(yaml_text: str):
    """Helper: raise LoaderError on a YAML string."""
    return load_predicate(yaml_text)


def test_missing_hookpoint():
    with pytest.raises(LoaderError, match=r"hookpoint"):
        _badload("""
id: bad-no-hookpoint
template: PAYLOAD_FIELD_COMPARE
scope:
  payload_path: $.foo
bound:
  kind: bound
  field: x
  op: ">"
  value: 1
""")


def test_negative_tolerance():
    with pytest.raises(LoaderError, match=r"tolerance"):
        _badload("""
id: bad-neg-tol
template: PAYLOAD_FIELD_COMPARE
scope:
  hookpoint: foo.bar
bound:
  kind: bound
  field: x
  op: ">"
  value: 1
tolerance:
  rel: -0.01
""")


def test_unknown_template():
    with pytest.raises(LoaderError, match=r"unknown template"):
        _badload("""
id: bad-template
template: NOT_A_TEMPLATE
scope:
  hookpoint: foo.bar
bound:
  kind: bound
  field: x
  op: ">"
  value: 1
""")


def test_bad_payload_path():
    with pytest.raises(LoaderError, match=r"payload_path"):
        _badload("""
id: bad-payload-path
template: PAYLOAD_FIELD_COMPARE
scope:
  hookpoint: foo.bar
  payload_path: foo.bar.baz
bound:
  kind: bound
  field: x
  op: ">"
  value: 1
""")


def test_template_kind_mismatch():
    # MONOTONIC is incompatible with STRUCTURAL_PRESENCE per templates.py
    with pytest.raises(LoaderError, match=r"incompatible"):
        _badload("""
id: bad-template-kind
template: STRUCTURAL_PRESENCE
scope:
  hookpoint: foo.bar
bound:
  kind: monotonic
  field: x
""")


def test_walk_and_payload_path_mutually_exclusive():
    with pytest.raises(LoaderError, match=r"mutually exclusive"):
        _badload("""
id: bad-walk-and-path
template: TENSOR_STAT_BOUND
scope:
  hookpoint: foo.bar
  payload_path: $.foo
  walk_tensor_summaries: true
bound:
  kind: equality
  field: has_nan
  value: false
""")


def test_value_is_field_requires_string():
    with pytest.raises(LoaderError, match=r"value_is_field"):
        _badload("""
id: bad-vif
template: PAYLOAD_FIELD_COMPARE
scope:
  hookpoint: foo.bar
bound:
  kind: bound
  field: x
  op: ">"
  value: 42
  value_is_field: true
""")


def test_empty_field_list_rejected():
    with pytest.raises(LoaderError, match=r"field"):
        _badload("""
id: bad-empty-field
template: PAYLOAD_FIELD_COMPARE
scope:
  hookpoint: foo.bar
bound:
  kind: bound
  field: []
  op: ">"
  value: 0
""")
