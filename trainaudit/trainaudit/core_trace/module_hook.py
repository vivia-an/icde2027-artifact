"""Global nn.Module forward / backward hooks.

C0 contract: every emitted module event carries `module_class` (Python
type name), `module_id` (stable identity = id(mod)), and `module_name`
(dotted name from `model.named_modules()`, or null if the module wasn't
registered at snapshot_build time). Diagnosis (C1) keys on these to
resolve a violation event_id back to a specific submodule rather than
just a class.
"""
from typing import List

import torch
import torch.nn as nn

from ..schema import (C_KIND, HP_MODULE_BWD, HP_MODULE_FWD_POST,
                      HP_MODULE_FWD_PRE)
from ..store import TraceStore
from ._utils import (lookup_module_name, should_emit, summarize_tensor,
                     summarize_tensor_list)


# Module classes whose output stats are particularly informative for T0 rules.
_NORMALIZER_CLASSES = (nn.LayerNorm, nn.GroupNorm)
try:
    _NORMALIZER_CLASSES = _NORMALIZER_CLASSES + (nn.RMSNorm,)  # PyTorch 2.4+
except AttributeError:
    pass


def _is_normalizer(mod) -> bool:
    """isinstance check + class-name fallback for framework-custom norms."""
    if isinstance(mod, _NORMALIZER_CLASSES):
        return True
    name = type(mod).__name__
    if name in {"RMSNorm", "LayerNorm", "GroupNorm", "FusedLayerNorm",
                "MixedFusedLayerNorm", "MixedFusedRMSNorm"}:
        return True
    if name.endswith("Norm") and "Linear" not in name and "Embed" not in name:
        return True
    return False


def _ctx(mod) -> dict:
    """C0 identity fields present on every module event."""
    return {
        "module_class": type(mod).__qualname__,
        "module_id": id(mod),
        "module_name": lookup_module_name(mod),
    }


def install_module_hooks(store: TraceStore,
                         capture_inputs: bool = True,
                         capture_outputs: bool = True,
                         capture_grads: bool = True) -> List:
    """Install global forward/backward hooks.

    Returns the list of removable handles (call .remove() to detach).
    """
    handles = []

    if capture_inputs:
        def _fwd_pre(mod, args):
            # P0 #2: skip summarize_tensor entirely if sampled out
            if not should_emit(HP_MODULE_FWD_PRE):
                return
            payload = {
                C_KIND: "forward",
                **_ctx(mod),
                "is_normalizer": _is_normalizer(mod),
                "training": mod.training,
                "grad_enabled": torch.is_grad_enabled(),
                "inputs": summarize_tensor_list(args),
            }
            store.emit(HP_MODULE_FWD_PRE, payload)
        handles.append(nn.modules.module.register_module_forward_pre_hook(_fwd_pre))

    if capture_outputs:
        try:
            from ..adapters import label_module as _adapter_label
        except Exception:
            _adapter_label = lambda mod: {}  # type: ignore[assignment]

        def _fwd_post(mod, args, output):
            if not should_emit(HP_MODULE_FWD_POST):
                return
            payload = {
                C_KIND: "forward",
                **_ctx(mod),
                "is_normalizer": _is_normalizer(mod),
                "training": mod.training,
            }
            sem = _adapter_label(mod)
            if sem:
                payload["semantic"] = sem
                if sem.get("is_normalizer"):
                    payload["is_normalizer"] = True
            if torch.is_tensor(output):
                payload["output"] = summarize_tensor(output)
            else:
                payload["outputs"] = summarize_tensor_list(output)
            store.emit(HP_MODULE_FWD_POST, payload)
        handles.append(nn.modules.module.register_module_forward_hook(_fwd_post))

    if capture_grads:
        def _bwd(mod, grad_input, grad_output):
            if not should_emit(HP_MODULE_BWD):
                return
            store.emit(HP_MODULE_BWD, {
                C_KIND: "backward",
                **_ctx(mod),
                "grad_input": summarize_tensor_list(grad_input),
                "grad_output": summarize_tensor_list(grad_output),
            })
        try:
            handles.append(nn.modules.module.register_module_full_backward_hook(_bwd))
        except AttributeError:
            handles.append(nn.modules.module.register_module_backward_hook(_bwd))

    return handles
