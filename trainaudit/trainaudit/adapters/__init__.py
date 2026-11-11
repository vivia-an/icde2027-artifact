"""Adapter registry. Importing this side-loads all built-in adapters."""
from .base import (FrameworkAdapter, active_adapters, all_adapters,
                   collect_build_invariants, label_module, label_param,
                   list_all_groups, register_adapter)
from .deepspeed import DeepSpeedAdapter
from .fsdp import FSDPAdapter
from .megatron import MegatronAdapter
from .olmo import OLMoAdapter
from .olmo_core import OLMoCoreAdapter

# Auto-register all built-in adapters. detect() decides whether each is active.
_BUILT_INS = [MegatronAdapter, DeepSpeedAdapter, OLMoAdapter, OLMoCoreAdapter, FSDPAdapter]
for _cls in _BUILT_INS:
    try:
        register_adapter(_cls())
    except Exception:
        pass

__all__ = [
    "FrameworkAdapter", "register_adapter", "active_adapters", "all_adapters",
    "label_param", "label_module", "list_all_groups", "collect_build_invariants",
    "MegatronAdapter", "DeepSpeedAdapter", "OLMoAdapter", "FSDPAdapter",
]
