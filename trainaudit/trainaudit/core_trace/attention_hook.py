"""Attention-specific forward post hook.

Splits attention output [B, T, d_model] into [B, T, n_heads, head_dim] and
records per-head l2/abs_max stats. Catches:
  - dead attention head (one head's output l2 << others)
  - dead head class (block-level: all heads in a block dead)
  - per-head magnitude divergence (one head dominating)

Module discovery: walks model.named_modules() at snapshot time, picks any
module whose class name contains 'Attention' AND has either `n_heads` or
`num_heads` attribute. Registers per-module forward post-hook.

Distinct from module.fwd.post (which only records aggregate output l2).
This hookpoint emits 'attention.fwd.post' with per-head breakdown.

Lifecycle: handles list returned for clean removal on disable().
"""
from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..store import TraceStore
from ._utils import lookup_module_name, should_emit


_HOOKPOINT = "attention.fwd.post"


def _attention_n_heads(mod: nn.Module):
    """Best-effort lookup of n_heads on the module."""
    for attr in ("n_heads", "num_heads", "n_head", "num_attention_heads",
                  "n_kv_heads"):
        v = getattr(mod, attr, None)
        if isinstance(v, int) and v > 0:
            return attr, v
    return None, None


def _is_attention_module(mod: nn.Module) -> bool:
    name = type(mod).__qualname__
    if "Attention" not in name:
        return False
    if "Norm" in name:  # AttentionNorm not really attention
        return False
    attr, n = _attention_n_heads(mod)
    if n is not None:
        return True
    # Backend modules (TorchAttentionBackend, FlashAttentionBackend) often
    # don't store n_heads but emit 4D [B, T, H, hd] output directly. We
    # detect head structure from output shape at runtime instead.
    if "Backend" in name:
        return True
    return False


def _per_head_stats(out: torch.Tensor, n_heads) -> Dict[str, Any]:
    """Compute per-head l2 / abs_max stats.

    If `out` is 4D [..., n_heads, head_dim] (pre-projection backend output),
    use directly. If 3D [..., d_model], reshape using n_heads (post-proj
    wrapper output — note head structure is fictional after w_out, but rule
    is still useful for catching gross magnitude anomalies).
    """
    if not torch.is_tensor(out) or out.numel() == 0:
        return {}
    if out.dim() == 4:
        # backend-style output, already per-head
        n_heads = out.shape[-2]
        head_dim = out.shape[-1]
    elif out.dim() == 3 and n_heads is not None:
        last = out.shape[-1]
        if last % n_heads != 0:
            return {"reshape_failed": True, "last_dim": last, "n_heads": n_heads}
        head_dim = last // n_heads
    else:
        return {}
    try:
        with torch.no_grad():
            if out.dim() == 4:
                tf = out.detach().float()
            else:
                *batch_dims, _ = out.shape
                tf = out.detach().float().reshape(*batch_dims, n_heads, head_dim)
            # Reduce over all dims except the n_heads axis (second-to-last)
            head_axis = tf.dim() - 2
            reduce_dims = [d for d in range(tf.dim()) if d != head_axis]
            l2_per_head = tf.pow(2).sum(dim=reduce_dims).sqrt()
            abs_max_per_head = tf.abs().amax(dim=reduce_dims)
            l2_list = [float(x.item()) for x in l2_per_head]
            am_list = [float(x.item()) for x in abs_max_per_head]
            mean_l2 = sum(l2_list) / max(len(l2_list), 1)
            min_l2 = min(l2_list) if l2_list else 0.0
            max_l2 = max(l2_list) if l2_list else 0.0
            return {
                "n_heads": n_heads,
                "head_dim": head_dim,
                "per_head_l2": l2_list,
                "per_head_abs_max": am_list,
                "min_head_l2": min_l2,
                "max_head_l2": max_l2,
                "mean_head_l2": mean_l2,
                "max_to_min_ratio": (max_l2 / min_l2 if min_l2 > 0
                                     else (float("inf") if max_l2 > 0 else 1.0)),
                "any_zero_head": min_l2 == 0.0 and max_l2 > 0,
            }
    except Exception as e:
        return {"per_head_error": f"{type(e).__name__}: {e}"}


def install_attention_hooks(store: TraceStore, model: nn.Module) -> List:
    """Walk model, install per-attention-module forward post hooks.

    Call after model.build() (and after FSDP wrap, since we want post-FSDP
    output shape). Returns list of handles for cleanup.
    """
    handles = []
    n_installed = 0
    if model is None:
        return handles

    for name, mod in model.named_modules():
        if not _is_attention_module(mod):
            continue
        attr_name, n_heads = _attention_n_heads(mod)
        # n_heads may be None for Backend modules — they emit 4D output and
        # the hook will infer head count from output shape.

        def make_hook(_mod, _name, _n_heads):
            def hook(self_mod, inputs, output):
                from ._utils import active_store
                s = active_store()
                if s is None or not should_emit(_HOOKPOINT):
                    return
                # Output may be a tuple (out, attn_weights)
                t = output
                if isinstance(t, (tuple, list)):
                    t = t[0]
                if not torch.is_tensor(t):
                    return
                try:
                    payload = {
                        "kind": "attention",
                        "module_class": type(_mod).__qualname__,
                        "module_id": id(_mod),
                        "module_name": _name,
                        "output_shape": list(t.shape),
                    }
                    payload.update(_per_head_stats(t, _n_heads))
                    s.emit(_HOOKPOINT, payload)
                except Exception:
                    pass
            return hook

        h = mod.register_forward_hook(make_hook(mod, name, n_heads))
        handles.append(h)
        n_installed += 1

    if n_installed > 0:
        print(f"[trainaudit] installed {n_installed} attention-module hooks",
              flush=True)
    return handles
