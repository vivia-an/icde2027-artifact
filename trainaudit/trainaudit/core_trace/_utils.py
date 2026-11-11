"""Tensor summarization helpers + active TraceStore cell + trace context.

Lifecycle-safe wiring:
  `active_store()` / `set_active_store()` — function-level wrappers
  (installed once per process) read this at emit time so re-enabling
  trainaudit transparently rebinds. Per-Module hooks register/remove
  on enable()/disable() and so are not affected.

Trace context (C0):
  `register_module_names(model)` populates id(mod) → dotted name at
  snapshot time so `module.fwd.post` events can carry a stable
  `module_name` and downstream diagnosis can resolve violations to a
  source-tree location. `callsite()` produces best-effort
  (file, line, function) for function-patching hooks (clip_grad,
  checkpoint, F.softmax) so violations can point at user code rather
  than just the framework op.
"""
from typing import Any, Dict, List, Optional

import torch

from ..schema import (T_ABS_MAX, T_DEVICE, T_DTYPE, T_HAS_INF, T_HAS_NAN,
                      T_L2_NORM, T_SHAPE)
from ..store import TraceStore

_active: Optional[TraceStore] = None


def active_store() -> Optional[TraceStore]:
    s = _active
    if s is None or getattr(s, "_closed", False):
        return None
    return s


def set_active_store(store: TraceStore) -> None:
    global _active
    _active = store


def clear_active_store() -> None:
    global _active
    _active = None


# ---- Per-hookpoint sampling (P0 #2) -------------------------------------
# When `_sample_rates[hookpoint] < 1.0`, only every K-th event for that
# hookpoint actually fires (K = round(1/rate)). Hooks check this BEFORE
# doing summarize_tensor / any expensive work, so sampled-out events
# pay zero cost. Deterministic counter rather than random for
# reproducibility — across runs, the same step + hookpoint sequence
# produces the same sampled subset.

import threading as _threading
from collections import defaultdict as _defaultdict

_sample_rates: Dict[str, float] = {}
_sample_counters: Dict[str, int] = _defaultdict(int)
_sample_lock = _threading.Lock()


def set_sample_rates(rates: Optional[Dict[str, float]]) -> None:
    """Configure per-hookpoint sample rates. Pass None or empty dict to
    disable sampling. Rates outside [0, 1] are clamped."""
    global _sample_rates
    with _sample_lock:
        if not rates:
            _sample_rates = {}
        else:
            _sample_rates = {hp: max(0.0, min(1.0, float(r)))
                              for hp, r in rates.items()}
        _sample_counters.clear()


def get_sample_rate(hookpoint: str) -> float:
    return _sample_rates.get(hookpoint, 1.0)


def should_emit(hookpoint: str) -> bool:
    """True iff this event should fire given configured sample rate.
    Hooks call this first; if False, return immediately without computing
    tensor stats — that's where the production overhead win comes from.
    """
    rate = _sample_rates.get(hookpoint)
    if rate is None or rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    with _sample_lock:
        _sample_counters[hookpoint] += 1
        n = _sample_counters[hookpoint]
    k = max(1, round(1.0 / rate))
    return (n - 1) % k == 0


def reset_sampling() -> None:
    """Clear all sampling state. Used by tests + on disable()."""
    global _sample_rates
    with _sample_lock:
        _sample_rates = {}
        _sample_counters.clear()


# ---- C0: trace context (module_name, callsite) ---------------------------

# id(mod) → dotted module name from model.named_modules(). Populated at
# snapshot_build time so the global module hook can resolve a module to a
# stable name without holding a strong reference to it.
_module_name_map: Dict[int, str] = {}


def register_module_names(model: Any) -> int:
    """Walk model.named_modules() and stash id(mod) → dotted name.
    Called by snapshot_build. Returns count of registered names."""
    if model is None:
        return 0
    n = 0
    try:
        for name, mod in model.named_modules():
            if name:  # skip the root entry which has empty name
                _module_name_map[id(mod)] = name
                n += 1
    except Exception:
        pass
    return n


def lookup_module_name(mod: Any) -> Optional[str]:
    return _module_name_map.get(id(mod))


def clear_module_names() -> None:
    _module_name_map.clear()


def callsite(skip: int = 2) -> Optional[Dict[str, Any]]:
    """Return {file, line, function} for the caller `skip` frames up the stack.
    Best-effort; returns None on any failure to keep hot paths cheap.
    """
    import sys
    try:
        frame = sys._getframe(skip)
        # walk past trainaudit's own wrappers
        while frame is not None:
            f_name = frame.f_globals.get("__name__", "")
            if not f_name.startswith("trainaudit."):
                break
            frame = frame.f_back
        if frame is None:
            return None
        return {
            "file": frame.f_code.co_filename,
            "line": frame.f_lineno,
            "function": frame.f_code.co_name,
        }
    except Exception:
        return None


def summarize_tensor(t: Optional[torch.Tensor], *, with_stats: bool = True) -> Optional[Dict[str, Any]]:
    """Summarize one tensor to a plain-Python dict.

    `with_stats=False` skips the reduction kernels (faster path for hot loops).
    """
    if t is None or not torch.is_tensor(t):
        return None
    out: Dict[str, Any] = {
        T_DTYPE:  str(t.dtype).replace("torch.", ""),
        T_SHAPE:  list(t.shape),
        T_DEVICE: str(t.device),
    }
    if with_stats and t.numel() > 0:
        with torch.no_grad():
            try:
                tf = t.detach().float()
                if t.is_floating_point():
                    out[T_L2_NORM] = float(tf.norm(2).item())
                    out[T_HAS_NAN] = bool(torch.isnan(t).any().item())
                    out[T_HAS_INF] = bool(torch.isinf(t).any().item())
                # max/min/mean work on int too — useful for token-id range checks
                out[T_ABS_MAX] = float(tf.abs().max().item())
                out["max"] = float(tf.max().item())
                out["min"] = float(tf.min().item())
                out["mean"] = float(tf.mean().item())
                # A3: distribution shape signals
                if t.is_floating_point() and tf.numel() > 1:
                    out["std"] = float(tf.std().item())
                    # zero_rate: fraction of elements that are exactly 0.
                    # Catches dead-activation patterns (ReLU pruned neurons,
                    # all-zero MoE expert slots, masked-out positions).
                    out["zero_rate"] = float((tf == 0).float().mean().item())
            except Exception:
                pass
    return out


def summarize_tensor_list(items: Any) -> List[Optional[Dict[str, Any]]]:
    """Recursive summarization for tuples/lists of tensors."""
    out = []
    if items is None:
        return out
    if torch.is_tensor(items):
        return [summarize_tensor(items)]
    if isinstance(items, dict):
        items = list(items.values())
    if isinstance(items, (list, tuple)):
        for x in items:
            if torch.is_tensor(x):
                out.append(summarize_tensor(x))
            elif isinstance(x, (list, tuple, dict)):
                out.extend(summarize_tensor_list(x))
    return out
