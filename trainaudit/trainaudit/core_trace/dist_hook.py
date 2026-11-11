"""Wrap torch.distributed comm ops to emit pre/post events."""
from typing import Any

import torch
import torch.distributed as dist

from ..schema import C_GROUP_SIZE, C_KIND, C_OP, HP_COMM_POST, HP_COMM_PRE
from ..store import TraceStore
from ._utils import (active_store, set_active_store, summarize_tensor,
                     summarize_tensor_list)


_TARGETS = ("all_reduce", "broadcast", "reduce_scatter",
            "all_gather", "all_gather_into_tensor", "all_to_all",
            "all_to_all_single", "reduce")


def _rank() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _world_size(group=None) -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(group=group)
    except Exception:
        pass
    return 1


def _summary_for(args, kwargs) -> Any:
    """Best-effort summary of the comm tensor argument."""
    candidate = None
    if args:
        candidate = args[0]
    if candidate is None:
        candidate = kwargs.get("tensor")
    if candidate is None:
        candidate = kwargs.get("output_tensor")
    if torch.is_tensor(candidate):
        return summarize_tensor(candidate)
    if isinstance(candidate, (list, tuple)):
        return summarize_tensor_list(candidate)
    return None


def install_dist_hooks(store: TraceStore) -> int:
    """Monkey-patch a stable set of torch.distributed ops.

    Returns the number of ops actually wrapped (some may be missing on old PyTorch).
    """
    set_active_store(store)
    n = 0
    for name in _TARGETS:
        orig = getattr(dist, name, None)
        if orig is None or getattr(orig, "_trainaudit_wrapped", False):
            continue
        wrapped = _make_wrapper(orig, name)
        setattr(dist, name, wrapped)
        n += 1
    return n


def _make_wrapper(orig_fn, op_name: str):
    def wrapped(*args, **kwargs):
        s = active_store()
        if s is None:
            return orig_fn(*args, **kwargs)
        group = kwargs.get("group")
        pre_summary = _summary_for(args, kwargs)
        s.emit(HP_COMM_PRE, {
            C_KIND: "comm",
            C_OP: op_name,
            C_GROUP_SIZE: _world_size(group),
            "tensor_pre": pre_summary,
        }, rank=_rank())
        try:
            result = orig_fn(*args, **kwargs)
        finally:
            post_summary = _summary_for(args, kwargs)
            s.emit(HP_COMM_POST, {
                C_KIND: "comm",
                C_OP: op_name,
                C_GROUP_SIZE: _world_size(group),
                "tensor_post": post_summary,
            }, rank=_rank())
        return result

    wrapped._trainaudit_wrapped = True  # type: ignore[attr-defined]
    wrapped.__wrapped__ = orig_fn       # type: ignore[attr-defined]
    return wrapped
