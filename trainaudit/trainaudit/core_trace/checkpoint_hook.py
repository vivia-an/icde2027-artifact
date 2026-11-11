"""Hook torch.utils.checkpoint to emit checkpoint.call events.

Capture (function name, preserve_rng_state, use_reentrant) — enables rules
that check rng-state preservation when dropout is in play (catches O-005).

Lifecycle: store is read at call time via `_utils.active_store()`, not
captured by closure, so trainaudit.disable() → enable() rebinds correctly.
"""
from ..store import TraceStore
from ._utils import active_store, callsite, set_active_store


def install_checkpoint_hook(store: TraceStore) -> bool:
    set_active_store(store)
    try:
        from torch.utils import checkpoint as _cp
    except ImportError:
        return False
    orig = getattr(_cp, "checkpoint", None)
    if orig is None or getattr(orig, "_trainaudit_wrapped", False):
        return False

    def wrapped(function, *args, use_reentrant=None, **kwargs):
        s = active_store()
        if s is not None:
            preserve = kwargs.get("preserve_rng_state")
            s.emit("checkpoint.call", {
                "kind": "checkpoint",
                "function": getattr(function, "__qualname__", str(function)),
                "preserve_rng_state": preserve,
                "use_reentrant": use_reentrant,
                "kwargs_keys": sorted(kwargs.keys()),
                "callsite": callsite(skip=2),
            })
        if use_reentrant is not None:
            return orig(function, *args, use_reentrant=use_reentrant, **kwargs)
        return orig(function, *args, **kwargs)

    wrapped._trainaudit_wrapped = True  # type: ignore[attr-defined]
    wrapped.__wrapped__ = orig            # type: ignore[attr-defined]
    _cp.checkpoint = wrapped

    # Some PyTorch / framework code paths import `torch.utils.checkpoint.checkpoint`
    # under an alias at *module load time* (e.g. `torch_utils_checkpoint`) and then
    # bind it via functools.partial — so they capture the un-wrapped reference and
    # the global patch above would not fire on those call sites.
    # Reach into the known offenders and replace their alias too. Discovered when
    # validating CAND_OLMOCORE_RNGCKPT end-to-end on real OLMo-core code:
    # `apply_activation_checkpointing` routes through
    # `torch.distributed.algorithms._checkpoint.checkpoint_wrapper.CheckpointWrapper`
    # whose `__init__` does `partial(torch_utils_checkpoint, ...)` against the
    # un-wrapped reference imported at module load time.
    try:
        import sys as _sys
        for _modname in ("torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
                          "torch.distributed._composable.checkpoint_activation"):
            _m = _sys.modules.get(_modname)
            if _m is None:
                continue
            for _alias in ("torch_utils_checkpoint", "checkpoint"):
                _val = getattr(_m, _alias, None)
                if _val is orig:
                    setattr(_m, _alias, wrapped)
    except Exception:
        pass
    return True
