"""OLMo adapter.

OLMo doesn't shard params (no TP), so all params are 'replica' under DP/FSDP.
Main contribution: identify residual-block class so a residual-stream
invariant can attach + install an active residual probe.
"""
from typing import Any, Dict, List

import torch

from .base import FrameworkAdapter


_RESIDUAL_BLOCK_CLASSES = {
    "OLMoSequentialBlock", "OLMoParallelBlock", "OLMoLlamaBlock",
}
_NORMALIZER_HINT_CLASSES = {"RMSLayerNorm", "RMSNorm", "LPLayerNorm"}


class OLMoAdapter(FrameworkAdapter):
    name = "olmo"

    def detect(self) -> bool:
        try:
            import olmo  # noqa: F401
            return True
        except ImportError:
            return False

    def label_param(self, name: str, p: Any) -> Dict[str, Any]:
        # OLMo doesn't tag params with TP/EP attrs by default; everything is replica
        return {"replica_group_kind": "replica"}

    def label_module(self, mod: Any) -> Dict[str, Any]:
        cls = type(mod).__qualname__
        out: Dict[str, Any] = {}
        if cls in _RESIDUAL_BLOCK_CLASSES or any(c in cls for c in _RESIDUAL_BLOCK_CLASSES):
            out["is_residual_block"] = True
            # In OLMo, block has attr `attn_norm` and `ff_norm` if pre-LN
            if hasattr(mod, "attn_norm"):
                out["has_attn_norm"] = True
            if hasattr(mod, "ff_norm"):
                out["has_ff_norm"] = True
            cfg = getattr(mod, "config", None)
            if cfg is not None and hasattr(cfg, "norm_after"):
                out["norm_after"] = bool(cfg.norm_after)
        if cls in _NORMALIZER_HINT_CLASSES:
            out["is_normalizer"] = True
        return out

    def list_groups(self) -> List[Dict[str, Any]]:
        # OLMo uses PyTorch's distributed primitives directly; let
        # FSDPAdapter / DDP detection handle group enumeration.
        return []

    def build_invariants(self, model, optimizer) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for k in ("d_model", "n_heads", "n_layers", "vocab_size",
                      "embedding_size", "block_type", "norm_after",
                      "attention_dropout", "residual_dropout"):
                if hasattr(cfg, k):
                    out[k] = getattr(cfg, k)

        # Install residual probe on all OLMo blocks (T1 active probe — emits
        # residual.probe event with directly computed (d_to_input, d_to_normed)).
        try:
            self._install_residual_probe(model)
        except Exception as e:
            out["residual_probe_error"] = str(e)
        return out

    def _install_residual_probe(self, model) -> None:
        from ..core_trace.module_hook import _is_normalizer  # type: ignore[attr-defined]

        store = _get_store()
        if store is None:
            return

        for mod in model.modules():
            cls = type(mod).__qualname__
            if cls not in _RESIDUAL_BLOCK_CLASSES and not any(c in cls for c in _RESIDUAL_BLOCK_CLASSES):
                continue
            attn_norm = getattr(mod, "attn_norm", None)
            if attn_norm is None:
                continue
            # Use a closure to capture mod and attn_norm
            self._wrap_block(mod, attn_norm, store)

    def _wrap_block(self, block, attn_norm, store) -> None:
        if getattr(block, "_trainaudit_residual_probe", False):
            return
        orig_forward = block.forward

        def wrapped(x, *args, **kwargs):
            # Snapshot the original input before any mutation
            try:
                x_orig = x.detach().clone() if torch.is_tensor(x) else None
            except Exception:
                x_orig = None
            out = orig_forward(x, *args, **kwargs)
            try:
                out_t = out[0] if isinstance(out, tuple) else out
                if x_orig is not None and torch.is_tensor(out_t):
                    with torch.no_grad():
                        # Re-compute normed version from the unmutated input
                        normed = attn_norm(x_orig)
                        d_in = float((out_t - x_orig).abs().mean().item())
                        d_normed = float((out_t - normed).abs().mean().item())
                    store.emit("residual.probe", {
                        "kind": "residual_probe",
                        "block_class": type(block).__qualname__,
                        "norm_class": type(attn_norm).__qualname__,
                        "d_to_original_input": d_in,
                        "d_to_normed_input": d_normed,
                        "ratio_normed_over_input": (d_normed / d_in) if d_in > 0 else None,
                    })
            except Exception:
                pass
            return out

        block.forward = wrapped
        block._trainaudit_residual_probe = True


def _get_store():
    """Best-effort fetch of the global trainaudit store."""
    try:
        from .. import get_store
        return get_store()
    except Exception:
        return None
