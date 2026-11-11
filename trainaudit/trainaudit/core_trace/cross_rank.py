"""Cross-rank cksum gather for replica params.

At build snapshot time, for each replica-kind param, hash its bytes locally
and all_gather across the replica group. Rank 0 stores the gathered list;
rule then checks whether all hashes are equal.
"""
import hashlib
from typing import Any, Dict, List, Optional

import torch


def _cksum(t: torch.Tensor) -> int:
    """SHA-256-based 60-bit signed-int hash of the tensor's float32 bytes."""
    try:
        data = t.detach().contiguous().cpu().float().numpy().tobytes()
        return int(hashlib.sha256(data).hexdigest()[:15], 16)
    except Exception:
        return -1


def gather_param_cksums(model, replica_group_id_fn=None) -> List[Dict[str, Any]]:
    """For each named parameter, hash locally and (if dist initialised)
    all_gather the hash across the replica group.

    `replica_group_id_fn(name, param) -> torch.distributed.ProcessGroup or None`:
    if returned group is non-trivial, do all_gather; else only emit local hash.
    Returns list of {name, local_cksum, gathered_cksums, all_equal, group_size}.
    """
    return _gather_tensor_cksums_for_params(
        model.named_parameters(),
        replica_group_id_fn=replica_group_id_fn,
        tensor_getter=lambda p: p)


def gather_buffer_cksums(model, replica_group_id_fn=None) -> List[Dict[str, Any]]:
    """Cross-rank cksum gather for module BUFFERS (not weights).

    Catches Megatron CUDA-graph buffer-corruption silent error: graph warmup
    forwards mutate persistent buffers (e.g. MoE router expert_bias), and the
    pre-fix code did not back them up across warmup. After warmup, ranks have
    different buffer state, which then biases routing for the rest of
    training.

    Same protocol as gather_param_cksums, but iterates `model.named_buffers()`.
    Persistent and non-persistent buffers are both gathered (the rule then
    filters by group_size > 1).
    """
    # named_buffers() also yields shared buffers; dedupe by id to avoid
    # double-counting when a buffer is registered under multiple parent
    # module dotted names.
    seen = set()
    def _iter_named_buffers():
        for name, b in model.named_buffers():
            if id(b) in seen:
                continue
            seen.add(id(b))
            yield name, b
    return _gather_tensor_cksums_for_params(
        _iter_named_buffers(),
        replica_group_id_fn=replica_group_id_fn,
        tensor_getter=lambda b: b)


def gather_grad_cksums(model, replica_group_id_fn=None) -> List[Dict[str, Any]]:
    """Cross-rank cksum gather for backward GRADIENTS (not weights).

    Same protocol as gather_param_cksums, but hashes p.grad instead of p.
    Used to detect B2-style bugs where TP frozen-weight backward skips
    the input-gradient all-reduce — replica params then end up with
    different grads across TP rank.

    Called from optim.step.pre (after backward, before apply) so grads
    are populated. Skips params with no grad (frozen forward but no
    backward path, etc.).
    """
    return _gather_tensor_cksums_for_params(
        model.named_parameters(),
        replica_group_id_fn=replica_group_id_fn,
        tensor_getter=lambda p: p.grad,
        skip_if_none=True)


def _gather_tensor_cksums_for_params(named_params, *,
                                       replica_group_id_fn=None,
                                       tensor_getter=lambda p: p,
                                       skip_if_none: bool = False
                                       ) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        import torch.distributed as dist
        dist_ok = dist.is_available() and dist.is_initialized()
    except Exception:
        dist_ok = False

    for name, p in named_params:
        t = tensor_getter(p)
        if t is None:
            if skip_if_none:
                continue
        local = _cksum(t) if t is not None else -1
        entry: Dict[str, Any] = {
            "name": name,
            "local_cksum": local,
            "gathered_cksums": None,
            "all_equal": None,
            "group_size": 1,
        }
        if dist_ok and replica_group_id_fn is not None and t is not None:
            try:
                grp = replica_group_id_fn(name, p)
                if grp is not None:
                    world = dist.get_world_size(group=grp)
                    if world > 1:
                        local_t = torch.tensor(
                            [local], dtype=torch.long,
                            device=t.device if t.is_cuda else "cpu")
                        gathered = [torch.zeros(1, dtype=torch.long,
                                                 device=local_t.device)
                                    for _ in range(world)]
                        dist.all_gather(gathered, local_t, group=grp)
                        gh = [int(g.item()) for g in gathered]
                        entry["gathered_cksums"] = gh
                        entry["all_equal"] = len(set(gh)) == 1
                        entry["group_size"] = world
            except Exception:
                pass
        out.append(entry)
    return out
