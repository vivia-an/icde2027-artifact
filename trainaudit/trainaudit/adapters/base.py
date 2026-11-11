"""Framework adapter base interface.

Each framework adapter answers semantic questions about runtime objects:
  - Is this tensor a replica or a shard? Which group?
  - What semantic role does this module play (residual block? router?)?
  - Does this object satisfy framework-declared structural contracts?

Adapters are auto-detected at enable() time. If a framework's import is
present, its adapter activates. Multiple adapters can co-exist (e.g. OLMo
running on PyTorch's FSDP).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class FrameworkAdapter:
    """Base class. Subclass and register via `register_adapter()`."""

    name: str = "base"

    def detect(self) -> bool:
        """Return True if this framework is currently active."""
        return False

    # ---- Tensor labels ----
    def label_param(self, name: str, p: Any) -> Dict[str, Any]:
        """Label one named parameter with semantic attributes:
        {replica_group_kind, replica_group_id, role, is_expert_local, ...}.
        Return {} if no labels apply."""
        return {}

    # ---- Module labels ----
    def label_module(self, mod: Any) -> Dict[str, Any]:
        """Label one module instance with semantic attributes:
        {is_residual_block, is_router, is_attention, ...}.
        Return {} if no labels apply."""
        return {}

    # ---- Process groups ----
    def list_groups(self) -> List[Dict[str, Any]]:
        """Return list of active process groups with semantic kinds:
        [{kind: 'tp'|'dp'|'pp'|'ep', size, ranks, ...}, ...]."""
        return []

    # ---- Free-form checks at build time ----
    def build_invariants(self, model: Any, optimizer: Any) -> Dict[str, Any]:
        """Return framework-specific invariants discovered at build time
        (e.g., expected layer count, declared num_experts, declared precision)."""
        return {}


_REGISTRY: List[FrameworkAdapter] = []


def register_adapter(adapter: FrameworkAdapter) -> None:
    _REGISTRY.append(adapter)


def all_adapters() -> List[FrameworkAdapter]:
    return list(_REGISTRY)


def active_adapters() -> List[FrameworkAdapter]:
    return [a for a in _REGISTRY if a.detect()]


def label_param(name: str, p: Any) -> Dict[str, Any]:
    """Aggregate label_param across all active adapters."""
    out: Dict[str, Any] = {}
    for a in active_adapters():
        try:
            d = a.label_param(name, p)
            if d:
                out.update(d)
                out.setdefault("source_adapter", a.name)
        except Exception:
            pass
    return out


def label_module(mod: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for a in active_adapters():
        try:
            d = a.label_module(mod)
            if d:
                out.update(d)
                out.setdefault("source_adapter", a.name)
        except Exception:
            pass
    return out


def list_all_groups() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in active_adapters():
        try:
            out.extend(a.list_groups())
        except Exception:
            pass
    return out


def collect_build_invariants(model, optimizer) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for a in active_adapters():
        try:
            d = a.build_invariants(model, optimizer)
            if d:
                out[a.name] = d
        except Exception:
            pass
    return out
