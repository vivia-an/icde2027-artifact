"""FSDP adapter (PyTorch built-in)."""
from typing import Any, Dict, List

from .base import FrameworkAdapter


class FSDPAdapter(FrameworkAdapter):
    name = "fsdp"

    def detect(self) -> bool:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel  # noqa: F401
            return True
        except ImportError:
            return False

    def label_param(self, name: str, p: Any) -> Dict[str, Any]:
        try:
            from torch.distributed.fsdp.flat_param import FlatParameter
            if isinstance(p, FlatParameter):
                return {"replica_group_kind": "shard", "is_fsdp_flat": True}
        except ImportError:
            pass
        return {}

    def label_module(self, mod: Any) -> Dict[str, Any]:
        cls = type(mod).__qualname__
        if "FullyShardedDataParallel" in cls or "FSDP" in cls:
            return {"is_fsdp_wrapped": True}
        return {}

    def list_groups(self) -> List[Dict[str, Any]]:
        # Walking FSDP module tree to enumerate process groups requires a
        # model handle. Defer to build_invariants where we have model.
        return []
