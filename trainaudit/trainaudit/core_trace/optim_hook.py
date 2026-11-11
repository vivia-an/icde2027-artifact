"""Wrap torch.optim.Optimizer.step and torch.nn.utils.clip_grad_norm_.

Hooks resolve their TraceStore at *call time* via the module-level
`store_ref` mutable cell rather than capturing the store by closure. This
makes `enable() → disable() → enable()` safe across the lifetime of a
process: the second `enable()` swaps `store_ref` to the new store; the
already-installed wrappers transparently pick it up. Without this, the
old wrappers continue to write to the closed store and emits silently
no-op (returning -1).
"""
import math
from typing import Any, Optional

import torch
import torch.nn.utils as nn_utils
import torch.optim as optim

from ..schema import (C_KIND, HP_CLIP_GRAD_POST, HP_CLIP_GRAD_PRE,
                      HP_LR_SCHED_INIT, HP_OPTIM_STEP_POST, HP_OPTIM_STEP_PRE)
from ..store import TraceStore
from ._utils import active_store as _store
from ._utils import callsite as _callsite
from ._utils import set_active_store as _set_active_store


def _instance_step_wrapper(opt_instance, store: TraceStore):
    """Replace `opt_instance.step` with a wrapper that emits pre/post events.

    Subclassing-safe: wraps the bound method on the instance, so it works
    even when AdamW / SGD / etc. override `step` (which they do).

    The wrapper uses `_store()` (mutable cell) at call time, NOT the
    closed-over `store` parameter — so re-enabling trainaudit with a new
    store correctly redirects emits.
    """
    real_step = opt_instance.step
    if getattr(real_step, "_trainaudit_wrapped", False):
        return False

    def step_wrapper(*args, **kwargs):
        store_ref = _store()
        if store_ref is None:
            # trainaudit disabled → pass-through
            return real_step(*args, **kwargs)
        param_groups_summary = []
        total_grad_l2_pre = 0.0
        total_param_l2_pre = 0.0
        n_params = 0
        # A4: capture pre-step param data refs for delta computation
        pre_params = []  # list of (param_ref, cloned_data) pairs
        for g in opt_instance.param_groups:
            g_payload = {k: _safe_repr(v) for k, v in g.items() if k != "params"}
            g_payload["n_params"] = len(g.get("params", []))
            g_payload["param_keys"] = sorted(g.keys())
            param_groups_summary.append(g_payload)
            for p in g.get("params", []):
                if p is None:
                    continue
                n_params += 1
                with torch.no_grad():
                    total_param_l2_pre += float(p.detach().float().norm(2).item()) ** 2
                    if p.grad is not None:
                        total_grad_l2_pre += float(p.grad.detach().float().norm(2).item()) ** 2
                    # A4: capture pre-step data for per-param delta
                    pre_params.append((p, p.detach().clone()))

        store_ref.emit(HP_OPTIM_STEP_PRE, {
            C_KIND: "optim_step",
            "optimizer_class": type(opt_instance).__qualname__,
            "param_groups": param_groups_summary,
            "n_params": n_params,
            "total_grad_l2": math.sqrt(total_grad_l2_pre),
            "total_param_l2": math.sqrt(total_param_l2_pre),
        })

        result = real_step(*args, **kwargs)

        total_param_l2_post = 0.0
        # Capture state['step'] for OC-NEW-2-style "counter not incrementing" detection
        step_values = []
        for g in opt_instance.param_groups:
            for p in g.get("params", []):
                if p is None:
                    continue
                with torch.no_grad():
                    total_param_l2_post += float(p.detach().float().norm(2).item()) ** 2
                state = opt_instance.state.get(p, {})
                if "step" in state:
                    sv = state["step"]
                    try:
                        step_values.append(float(sv.item()) if hasattr(sv, "item") else float(sv))
                    except Exception:
                        pass

        # A4: compute per-param delta and aggregate
        total_delta_sq = 0.0
        n_unchanged = 0
        n_unchanged_with_grad = 0  # subset that ALSO had non-zero grad
        max_delta_per_param = 0.0
        for p, pre_data in pre_params:
            try:
                with torch.no_grad():
                    delta = (p.detach() - pre_data).float()
                    d_l2 = float(delta.norm(2).item())
                    total_delta_sq += d_l2 ** 2
                    if d_l2 > max_delta_per_param:
                        max_delta_per_param = d_l2
                    if d_l2 == 0.0:
                        n_unchanged += 1
                        # Did this param have a non-trivial gradient?
                        if p.grad is not None and float(p.grad.detach().float().norm(2).item()) > 0:
                            n_unchanged_with_grad += 1
            except Exception:
                pass

        post_payload = {
            C_KIND: "optim_step",
            "optimizer_class": type(opt_instance).__qualname__,
            "total_param_l2": math.sqrt(total_param_l2_post),
            "total_param_delta_l2": math.sqrt(total_delta_sq),
            "max_param_delta_l2": max_delta_per_param,
            "n_params_unchanged": n_unchanged,
            "n_params_unchanged_with_grad": n_unchanged_with_grad,
        }
        if step_values:
            post_payload["state_step_min"] = min(step_values)
            post_payload["state_step_max"] = max(step_values)
            post_payload["state_step_n_params"] = len(step_values)
        store_ref.emit(HP_OPTIM_STEP_POST, post_payload)
        return result

    step_wrapper._trainaudit_wrapped = True  # type: ignore[attr-defined]
    step_wrapper.__wrapped__ = real_step     # type: ignore[attr-defined]
    # PyTorch's LRScheduler probes `optimizer.step.__func__` and
    # `optimizer.step.__self__` to wrap step with a counter; preserve those
    # attributes so the scheduler init doesn't AttributeError.
    step_wrapper.__func__ = getattr(real_step, "__func__", real_step)  # type: ignore[attr-defined]
    step_wrapper.__self__ = opt_instance      # type: ignore[attr-defined]
    opt_instance.step = step_wrapper          # type: ignore[assignment]
    return True


def install_optim_hooks(store: TraceStore) -> bool:
    """Patch Optimizer.__init__ so that every constructed optimizer has its
    `step` wrapped at instance level. Subclass-safe and re-enable-safe."""
    _set_active_store(store)
    orig_init = optim.Optimizer.__init__
    if getattr(orig_init, "_trainaudit_wrapped", False):
        # Already wrapped — just rebind store and expose late-bind helper
        install_optim_hooks._wrap_instance = (  # type: ignore[attr-defined]
            lambda opt: _instance_step_wrapper(opt, _store()))
        return False

    def wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        s = _store()
        if s is not None:
            _instance_step_wrapper(self, s)

    wrapped_init._trainaudit_wrapped = True  # type: ignore[attr-defined]
    wrapped_init.__wrapped__ = orig_init      # type: ignore[attr-defined]
    optim.Optimizer.__init__ = wrapped_init
    install_optim_hooks._wrap_instance = (  # type: ignore[attr-defined]
        lambda opt: _instance_step_wrapper(opt, _store()))
    return True


def install_clip_grad_hook(store: TraceStore) -> bool:
    """Wrap torch.nn.utils.clip_grad_norm_ to record (max_norm, pre_norm, post_norm)."""
    _set_active_store(store)
    orig = getattr(nn_utils, "clip_grad_norm_", None)
    if orig is None or getattr(orig, "_trainaudit_wrapped", False):
        return False

    def wrapped(parameters, max_norm, norm_type=2.0, *args, **kwargs):
        s = _store()
        if s is None:
            return orig(parameters, max_norm, norm_type=norm_type,
                        *args, **kwargs)
        params = list(parameters) if not isinstance(parameters, list) else parameters
        params = [p for p in params if p is not None and p.grad is not None]
        with torch.no_grad():
            pre_sq = 0.0
            for p in params:
                pre_sq += float(p.grad.detach().float().norm(2).item()) ** 2
            pre_norm = math.sqrt(pre_sq)
        cs = _callsite(skip=2)
        s.emit(HP_CLIP_GRAD_PRE, {
            C_KIND: "clip_grad",
            "fn": "torch.nn.utils.clip_grad_norm_",
            "max_norm": float(max_norm),
            "pre_norm": pre_norm,
            "n_params": len(params),
            "callsite": cs,
        })
        result = orig(params, max_norm, norm_type=norm_type, *args, **kwargs)
        with torch.no_grad():
            post_sq = 0.0
            for p in params:
                post_sq += float(p.grad.detach().float().norm(2).item()) ** 2
            post_norm = math.sqrt(post_sq)
        s.emit(HP_CLIP_GRAD_POST, {
            C_KIND: "clip_grad",
            "fn": "torch.nn.utils.clip_grad_norm_",
            "max_norm": float(max_norm),
            "pre_norm": pre_norm,
            "post_norm": post_norm,
            "ratio": post_norm / pre_norm if pre_norm > 0 else 1.0,
            "callsite": cs,
        })
        return result

    wrapped._trainaudit_wrapped = True  # type: ignore[attr-defined]
    wrapped.__wrapped__ = orig          # type: ignore[attr-defined]
    nn_utils.clip_grad_norm_ = wrapped
    # Also expose under torch.nn.utils.clip_grad_norm_ in case caller imported the module reference
    try:
        import torch.nn.utils.clip_grad as _cg
        _cg.clip_grad_norm_ = wrapped
    except Exception:
        pass
    return True


def install_lr_scheduler_hook(store: TraceStore) -> bool:
    """Wrap torch.optim.lr_scheduler.LRScheduler.__init__ so we can capture
    (last_epoch, param_groups_keys) at scheduler construction. Used by
    T0-initial-lr-present to detect resume-time initial_lr absence (B12)."""
    _set_active_store(store)
    try:
        from torch.optim.lr_scheduler import LRScheduler as _Sched
    except ImportError:
        try:
            from torch.optim.lr_scheduler import _LRScheduler as _Sched  # older torch
        except ImportError:
            return False

    orig_init = _Sched.__init__
    if getattr(orig_init, "_trainaudit_wrapped", False):
        return False

    def wrapped_init(self, optimizer, *args, **kwargs):
        s = _store()
        last_epoch = kwargs.get("last_epoch", -1)
        if len(args) >= 1 and isinstance(args[0], int):
            last_epoch = args[0]
        if s is not None:
            groups = []
            if hasattr(optimizer, "param_groups"):
                for i, g in enumerate(optimizer.param_groups):
                    groups.append({
                        "index": i,
                        "keys": sorted(k for k in g.keys() if k != "params"),
                        "has_initial_lr": "initial_lr" in g,
                        "lr": g.get("lr"),
                    })
            s.emit(HP_LR_SCHED_INIT, {
                C_KIND: "scheduler_init",
                "scheduler_class": type(self).__qualname__,
                "optimizer_class": type(optimizer).__qualname__,
                "last_epoch": last_epoch,
                "param_groups": groups,
            })
        return orig_init(self, optimizer, *args, **kwargs)

    wrapped_init._trainaudit_wrapped = True  # type: ignore[attr-defined]
    wrapped_init.__wrapped__ = orig_init      # type: ignore[attr-defined]
    _Sched.__init__ = wrapped_init
    return True


def _safe_repr(v: Any):
    try:
        if isinstance(v, (int, float, bool, str)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [_safe_repr(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _safe_repr(x) for k, x in v.items()}
        if torch.is_tensor(v):
            return {"_tensor": True, "shape": list(v.shape), "dtype": str(v.dtype)}
        return str(v)
    except Exception:
        return None
