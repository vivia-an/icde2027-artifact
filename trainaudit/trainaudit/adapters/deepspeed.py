"""DeepSpeed adapter (minimal).

In ZeRO, every param is sharded → 'shard' is the right replica_group_kind.
Without ZeRO, params are 'replica' under DP.
"""
from typing import Any, Dict, List

from .base import FrameworkAdapter


class DeepSpeedAdapter(FrameworkAdapter):
    name = "deepspeed"

    def detect(self) -> bool:
        try:
            import deepspeed  # noqa: F401
            return True
        except ImportError:
            return False

    def label_param(self, name: str, p: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # ds_id appears on params under ZeRO partition
        if hasattr(p, "ds_id"):
            out["replica_group_kind"] = "shard"
            out["ds_id"] = int(getattr(p, "ds_id", -1))
        else:
            out["replica_group_kind"] = "replica"
        if hasattr(p, "ds_status"):
            out["ds_status"] = str(getattr(p, "ds_status"))
        return out

    def label_module(self, mod: Any) -> Dict[str, Any]:
        cls = type(mod).__qualname__
        if cls == "MoE" or "MOELayer" in cls:
            return {"is_moe_layer": True}
        return {}

    def list_groups(self) -> List[Dict[str, Any]]:
        # DeepSpeed groups are accessible from engine; we don't have a
        # handle to the engine here. T2 will expose this once engine hook lands.
        return []
