"""OLMo-core adapter.

Currently provides one active probe: wrap olmo_core.optim.scheduler._sqrt_decay
to emit decay.probe events for OC-NEW-3 detection (sqrt decay direction wrong).
"""
from typing import Any, Dict, List

from .base import FrameworkAdapter


class OLMoCoreAdapter(FrameworkAdapter):
    name = "olmo_core"

    def detect(self) -> bool:
        try:
            import olmo_core  # noqa: F401
            return True
        except ImportError:
            return False

    def label_param(self, name: str, p: Any) -> Dict[str, Any]:
        return {"replica_group_kind": "replica"}

    def label_module(self, mod: Any) -> Dict[str, Any]:
        return {}

    def list_groups(self) -> List[Dict[str, Any]]:
        return []

    def build_invariants(self, model, optimizer) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            self._install_sqrt_decay_probe()
        except Exception as e:
            out["sqrt_decay_probe_error"] = str(e)
        return out

    def _install_sqrt_decay_probe(self) -> None:
        try:
            from .. import get_store
            store = get_store()
        except Exception:
            store = None
        if store is None:
            return
        try:
            from olmo_core.optim import scheduler as _sched_mod
        except ImportError:
            return
        if not hasattr(_sched_mod, "_sqrt_decay"):
            return
        if getattr(_sched_mod._sqrt_decay, "_trainaudit_wrapped", False):
            return
        orig = _sched_mod._sqrt_decay

        def wrapped(initial_lr, step_from_end, decay, decay_min_lr=0.0):
            result = orig(initial_lr, step_from_end, decay, decay_min_lr)
            try:
                progress = (min(step_from_end, decay) / decay) if decay else 0
                store.emit("decay.probe", {
                    "kind": "sqrt_decay",
                    "progress": float(progress),
                    "result": float(result.item()) if hasattr(result, "item") else float(result),
                    "initial_lr": float(initial_lr.item()) if hasattr(initial_lr, "item") else float(initial_lr),
                    "decay_min_lr": float(decay_min_lr),
                })
            except Exception:
                pass
            return result

        wrapped._trainaudit_wrapped = True  # type: ignore[attr-defined]
        wrapped.__wrapped__ = orig             # type: ignore[attr-defined]
        _sched_mod._sqrt_decay = wrapped
