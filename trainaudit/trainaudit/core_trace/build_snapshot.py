"""One-shot scan of model + optimizer right after construction."""
from typing import Any, Optional

import torch
import torch.nn as nn

from ..schema import HP_BUILD_SNAPSHOT
from ..store import TraceStore
from ._utils import register_module_names, summarize_tensor


def take_build_snapshot(store: TraceStore,
                        model: Optional[nn.Module] = None,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        *,
                        rank: int = 0) -> int:
    """Emit a single build.snapshot event with full structural info.

    Side effect (C0): registers `id(mod) → dotted name` for every module
    in `model.named_modules()` so subsequent `module.fwd.*` events carry
    a stable `module_name`."""
    payload: dict[str, Any] = {}

    if model is not None:
        # C0: populate module_name lookup for hooks
        register_module_names(model)

        # T1: ask all active framework adapters to label each param
        try:
            from ..adapters import (active_adapters, collect_build_invariants,
                                    label_param)
            adapter_names = [a.name for a in active_adapters()]
        except Exception:
            adapter_names = []
            label_param = lambda n, p: {}  # type: ignore[assignment]
            collect_build_invariants = lambda m, o: {}  # type: ignore[assignment]

        params_payload = []
        n_layers = 0
        for name, p in model.named_parameters():
            entry = {
                "name": name,
                "dtype": str(p.dtype).replace("torch.", ""),
                "shape": list(p.shape),
                "requires_grad": p.requires_grad,
                "is_leaf": p.is_leaf,
                "tensor_summary": summarize_tensor(p, with_stats=True),
                # Raw framework attrs (kept for backward-compat)
                "attr_tensor_model_parallel": getattr(p, "tensor_model_parallel", None),
                "attr_expert_parallel": getattr(p, "expert_parallel", None),
                "has_attr_ds_id": hasattr(p, "ds_id"),
            }
            # T1: semantic labels from adapters (replica_group_kind, etc.)
            entry["semantic"] = label_param(name, p)
            params_payload.append(entry)
        for _ in model.modules():
            n_layers += 1

        payload["model"] = {
            "class": type(model).__qualname__,
            "n_parameters": len(params_payload),
            "n_modules": n_layers,
            "parameters": params_payload,
            "active_adapters": adapter_names,
        }
        payload["framework_invariants"] = collect_build_invariants(model, optimizer)

        # T1 cross-rank cksum: only meaningful if a Megatron-style parallel
        # state is active. We resolve the replica group per param via the
        # adapter (which knows tensor_model_parallel etc.).
        try:
            from .cross_rank import gather_param_cksums

            def _resolve_group(name, p):
                try:
                    from megatron.core import parallel_state as ps
                    # for non-TP-shard params on a TP>1 setup, the replica
                    # group is the TP group (params should be equal across TP)
                    if ps.get_tensor_model_parallel_world_size() > 1:
                        if not getattr(p, "tensor_model_parallel", False) and \
                           not getattr(p, "expert_parallel", False):
                            return ps.get_tensor_model_parallel_group()
                except Exception:
                    pass
                return None

            xrank = gather_param_cksums(model, replica_group_id_fn=_resolve_group)
            payload["cross_rank_cksums"] = xrank
        except Exception:
            pass

        # Cross-rank cksum gather for BUFFERS too — catches CUDA-graph
        # warmup-corruption case (Megatron PR #4433) and any other path that
        # silently leaves replicated buffers diverged across the replica group.
        try:
            from .cross_rank import gather_buffer_cksums

            def _resolve_buffer_group(name, b):
                # Buffers don't carry tensor_model_parallel attr typically;
                # default to TP-replica group when TP > 1.
                try:
                    from megatron.core import parallel_state as ps
                    if ps.get_tensor_model_parallel_world_size() > 1:
                        return ps.get_tensor_model_parallel_group()
                except Exception:
                    pass
                return None

            payload["cross_rank_buffer_cksums"] = gather_buffer_cksums(
                model, replica_group_id_fn=_resolve_buffer_group)
        except Exception:
            pass

    if optimizer is not None:
        # Late-bind: optimizer might have been constructed before enable().
        # Wrap its .step here so subsequent step() calls emit events.
        try:
            from .optim_hook import install_optim_hooks
            wrap = getattr(install_optim_hooks, "_wrap_instance", None)
            if wrap is not None:
                wrap(optimizer)
        except Exception:
            pass

        groups_payload = []
        for i, g in enumerate(optimizer.param_groups):
            keys = sorted(k for k in g.keys() if k != "params")
            groups_payload.append({
                "index": i,
                "keys": keys,
                "lr": g.get("lr"),
                "has_initial_lr": "initial_lr" in g,
                "n_params": len(g.get("params", [])),
            })
        payload["optimizer"] = {
            "class": type(optimizer).__qualname__,
            "n_param_groups": len(groups_payload),
            "param_groups": groups_payload,
        }

    return store.emit(HP_BUILD_SNAPSHOT, payload, rank=rank)
