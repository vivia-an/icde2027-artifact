"""Hook PyTorch DataLoader iteration to emit dataloader.batch events.

Stable since PyTorch 1.0: DataLoader's `_BaseDataLoaderIter._next_data` is
the choke-point for any iter usage (single-process or worker-based).
"""
from typing import Any

import torch

from ..schema import HP_DATALOADER_BATCH
from ..store import TraceStore
from ._utils import active_store, set_active_store, summarize_tensor


def install_dataloader_hook(store: TraceStore) -> bool:
    set_active_store(store)
    try:
        from torch.utils.data import dataloader as _dl
    except ImportError:
        return False

    base_iter = getattr(_dl, "_BaseDataLoaderIter", None)
    if base_iter is None:
        return False

    # PyTorch's _SingleProcessDataLoaderIter and
    # _MultiProcessingDataLoaderIter both OVERRIDE `_next_data`, so
    # patching the base class alone is a no-op. Patch every concrete
    # subclass that defines its own `_next_data`.
    targets = [base_iter]
    for name in ("_SingleProcessDataLoaderIter",
                  "_MultiProcessingDataLoaderIter"):
        cls = getattr(_dl, name, None)
        if cls is not None:
            targets.append(cls)

    def _make_wrapper(orig_method):
        def wrapped(self):
            batch = orig_method(self)
            s = active_store()
            if s is None:
                return batch
            try:
                payload: dict[str, Any] = {"kind": "data_load"}
                if torch.is_tensor(batch):
                    payload["input_ids"] = summarize_tensor(batch)
                elif isinstance(batch, dict):
                    fields: dict[str, Any] = {}
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            fields[k] = summarize_tensor(v)
                    if fields:
                        payload["fields"] = fields
                        if "input_ids" in fields:
                            payload["input_ids"] = fields["input_ids"]
                elif isinstance(batch, (list, tuple)):
                    if len(batch) > 0 and torch.is_tensor(batch[0]):
                        payload["input_ids"] = summarize_tensor(batch[0])
                s.emit(HP_DATALOADER_BATCH, payload)
            except Exception:
                pass
            return batch
        wrapped._trainaudit_wrapped = True  # type: ignore[attr-defined]
        wrapped.__wrapped__ = orig_method     # type: ignore[attr-defined]
        return wrapped

    n_wrapped = 0
    for cls in targets:
        # Only wrap a class that defines _next_data on itself (not just inherited)
        if "_next_data" not in cls.__dict__:
            continue
        orig = cls.__dict__["_next_data"]
        if getattr(orig, "_trainaudit_wrapped", False):
            continue
        cls._next_data = _make_wrapper(orig)
        n_wrapped += 1
    return n_wrapped > 0
