"""Hook torch.nn.functional ops that are widely used and stable.

Currently:
  - F.softmax: emits softmax.call event with output tensor summary.
    Catches inline softmax calls that bypass the nn.Module hook (Megatron's
    TopKRouter uses F.softmax inside .routing()).

Lifecycle: store is read at call time via `_utils.active_store()`, not
captured by closure, so trainaudit.disable() → enable() rebinds correctly.
"""
import torch
import torch.nn.functional as F

from ..store import TraceStore
from ._utils import (active_store, callsite, set_active_store,
                     summarize_tensor)


def install_functional_hooks(store: TraceStore) -> bool:
    set_active_store(store)
    orig_softmax = F.softmax
    if getattr(orig_softmax, "_trainaudit_wrapped", False):
        return False

    def wrapped_softmax(input, dim=None, _stacklevel=3, dtype=None):
        out = orig_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
        s = active_store()
        if s is None:
            return out
        try:
            s.emit("functional.softmax", {
                "kind": "functional",
                "fn": "F.softmax",
                "dim": dim,
                "output": summarize_tensor(out),
                "callsite": callsite(skip=2),
            })
        except Exception:
            pass
        return out

    wrapped_softmax._trainaudit_wrapped = True  # type: ignore[attr-defined]
    wrapped_softmax.__wrapped__ = orig_softmax  # type: ignore[attr-defined]
    F.softmax = wrapped_softmax
    import torch.nn.functional as _F
    _F.softmax = wrapped_softmax
    return True
