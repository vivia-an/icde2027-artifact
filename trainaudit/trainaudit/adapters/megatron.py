"""Megatron-LM adapter.

Reads framework-specific tagging:
  - param.tensor_model_parallel  → 'shard' (TP-sharded weight)
  - param.expert_parallel        → 'expert_local' (MoE expert weight)
  - param.sequence_parallel      → marker
Reads parallel_state for groups (TP / DP / PP / EP).
"""
from typing import Any, Dict, List

from .base import FrameworkAdapter


class MegatronAdapter(FrameworkAdapter):
    name = "megatron"

    def detect(self) -> bool:
        try:
            import megatron  # noqa: F401
            return True
        except ImportError:
            return False

    def label_param(self, name: str, p: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # Megatron tags TP-sharded params with attribute tensor_model_parallel
        is_tp_shard = bool(getattr(p, "tensor_model_parallel", False))
        is_ep_local = bool(getattr(p, "expert_parallel", False)) or "expert" in name.lower()
        is_seq_par = bool(getattr(p, "sequence_parallel", False))

        if is_ep_local:
            out["replica_group_kind"] = "expert_local"
        elif is_tp_shard:
            out["replica_group_kind"] = "shard"
            out["shard_dim"] = getattr(p, "partition_dim", None)
        else:
            # default: DP-replicated
            out["replica_group_kind"] = "replica"

        if is_seq_par:
            out["sequence_parallel"] = True

        # Attach group ids best-effort
        try:
            from megatron.core import parallel_state as ps
            if is_tp_shard:
                out["replica_group_id"] = id(ps.get_tensor_model_parallel_group())
                out["tp_size"] = ps.get_tensor_model_parallel_world_size()
            elif is_ep_local:
                if hasattr(ps, "get_expert_model_parallel_group"):
                    out["replica_group_id"] = id(ps.get_expert_model_parallel_group())
                    out["ep_size"] = ps.get_expert_model_parallel_world_size()
            else:
                out["replica_group_id"] = id(ps.get_data_parallel_group())
                out["dp_size"] = ps.get_data_parallel_world_size()
        except Exception:
            pass
        return out

    def label_module(self, mod: Any) -> Dict[str, Any]:
        cls = type(mod).__qualname__
        out: Dict[str, Any] = {}
        if "Router" in cls or "TopK" in cls:
            out["is_router"] = True
            # Router-specific structural expectations
            out["has_calculate_per_token_loss"] = hasattr(mod, "calculate_per_token_loss")
            # M-012 evidence: expert_bias dtype
            eb = getattr(mod, "expert_bias", None)
            if eb is not None and hasattr(eb, "dtype"):
                out["expert_bias_dtype"] = str(eb.dtype).replace("torch.", "")
        if "SwitchMLP" in cls or "MoE" in cls:
            out["is_moe_layer"] = True
        if "ParallelLinear" in cls or "ColumnParallelLinear" in cls or "RowParallelLinear" in cls:
            out["is_tp_linear"] = True
        # Layer-count source for M-020
        if "TransformerLayer" in cls or cls == "ParallelTransformerLayer":
            out["is_transformer_layer"] = True
        return out

    def list_groups(self) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        try:
            from megatron.core import parallel_state as ps
            try:
                groups.append({
                    "kind": "tp",
                    "size": ps.get_tensor_model_parallel_world_size(),
                    "rank": ps.get_tensor_model_parallel_rank(),
                    "group_id": id(ps.get_tensor_model_parallel_group()),
                })
            except Exception:
                pass
            try:
                groups.append({
                    "kind": "dp",
                    "size": ps.get_data_parallel_world_size(),
                    "rank": ps.get_data_parallel_rank(),
                    "group_id": id(ps.get_data_parallel_group()),
                })
            except Exception:
                pass
            try:
                groups.append({
                    "kind": "pp",
                    "size": ps.get_pipeline_model_parallel_world_size(),
                    "rank": ps.get_pipeline_model_parallel_rank(),
                })
            except Exception:
                pass
            if hasattr(ps, "get_expert_model_parallel_world_size"):
                try:
                    groups.append({
                        "kind": "ep",
                        "size": ps.get_expert_model_parallel_world_size(),
                        "rank": ps.get_expert_model_parallel_rank(),
                    })
                except Exception:
                    pass
        except Exception:
            pass
        return groups

    def build_invariants(self, model, optimizer) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            from megatron.training import get_args
            args = get_args()
            # Capture structural expectations for downstream rules
            for k in ("num_layers", "num_experts", "moe_router_topk",
                      "tensor_model_parallel_size", "pipeline_model_parallel_size",
                      "expert_model_parallel_size", "fp16", "bf16",
                      "calculate_per_token_loss"):
                if hasattr(args, k):
                    out[k] = getattr(args, k)
        except Exception:
            pass

        # Count actual transformer layers by walking model.modules()
        # so the layer-count rule has authoritative evidence.
        n_layers = 0
        for m in model.modules():
            cls = type(m).__qualname__
            if "TransformerLayer" in cls or cls == "ParallelTransformerLayer":
                n_layers += 1
        out["n_transformer_layers_in_local_module"] = n_layers

        # M-024 active probe: wrap apply_input_jitter on every TopKRouter
        try:
            self._install_jitter_probe(model)
        except Exception as e:
            out["jitter_probe_error"] = str(e)
        return out

    def _install_jitter_probe(self, model) -> None:
        try:
            from .. import get_store
            store = get_store()
        except Exception:
            store = None
        if store is None:
            return
        for mod in model.modules():
            cls = type(mod).__qualname__
            if "Router" not in cls and "TopK" not in cls:
                continue
            if not hasattr(mod, "apply_input_jitter"):
                continue
            if getattr(mod, "_trainaudit_jitter_probe", False):
                continue
            orig = mod.apply_input_jitter

            def _make(orig_fn, mod_ref, store_ref):
                def wrapped(input):
                    in_dt = str(input.dtype).replace("torch.", "") if hasattr(input, "dtype") else None
                    out_t = orig_fn(input)
                    out_dt = str(out_t.dtype).replace("torch.", "") if hasattr(out_t, "dtype") else None
                    try:
                        store_ref.emit("jitter.probe", {
                            "kind": "jitter_probe",
                            "module_class": type(mod_ref).__qualname__,
                            "input_dtype": in_dt,
                            "output_dtype": out_dt,
                            "dtypes_match": in_dt == out_dt,
                        })
                    except Exception:
                        pass
                    return out_t
                return wrapped
            mod.apply_input_jitter = _make(orig, mod, store)
            mod._trainaudit_jitter_probe = True
