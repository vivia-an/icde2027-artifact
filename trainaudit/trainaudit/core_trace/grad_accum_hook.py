"""Hook torch.Tensor.backward to track gradient accumulation pattern.

We hook `torch.autograd.backward` (and `Tensor.backward`) to count
backward passes between optim.step calls. Combined with the existing
optim.step hook, this gives us:

  - n_microbatches_in_step: how many .backward() between consecutive .step()
  - per-microbatch grad delta: change in total_grad_l2 across each backward

Why this matters: production silent errors in grad accumulation include
  - BF16 boundary leak (DS PR #7985 — last microbatch grad lost when
    crossing accumulation boundary)
  - Wrong scaling at boundary (each microbatch grad scaled by 1/N but
    accumulator already does mean-reduction internally → double-scaled)
  - Stale grad reuse (grad not zeroed between accumulation cycles)

These manifest as:
  - n_microbatches_in_step varies unexpectedly across steps
  - total_grad_l2 NOT proportional to microbatch count
  - sudden drop in accumulated grad at last microbatch

The hook emits 'grad_accum.backward' per backward call.
"""
import math
from typing import Optional

import torch

from ..store import TraceStore
from ._utils import (active_store, callsite, set_active_store, should_emit)


_HOOKPOINT = "grad_accum.backward"

# Process-level state: count of backward calls since last optim.step
_state = {
    "backward_count": 0,        # consecutive backward calls
    "step_count": 0,             # total optim.steps fired
    "last_grad_l2": None,        # snapshot of total_grad_l2 after last backward
}


def reset_grad_accum_state() -> None:
    _state["backward_count"] = 0
    _state["step_count"] = 0
    _state["last_grad_l2"] = None


def on_optim_step_fired() -> None:
    """Called from optim_hook just before step. Resets backward counter,
    bumps step count. The optim_hook can use this to know how many
    microbatches accumulated."""
    _state["backward_count"] = 0
    _state["step_count"] += 1
    _state["last_grad_l2"] = None


def install_grad_accum_hook(store: TraceStore) -> bool:
    """Wrap torch.autograd.backward to emit accumulation-tracking events."""
    set_active_store(store)
    orig_backward = torch.autograd.backward
    if getattr(orig_backward, "_trainaudit_wrapped", False):
        return False

    def wrapped_backward(tensors, grad_tensors=None, retain_graph=None,
                          create_graph=False, grad_variables=None,
                          inputs=None):
        # Run real backward first
        result = orig_backward(tensors, grad_tensors=grad_tensors,
                                retain_graph=retain_graph,
                                create_graph=create_graph,
                                grad_variables=grad_variables, inputs=inputs)
        s = active_store()
        if s is None or not should_emit(_HOOKPOINT):
            return result
        try:
            _state["backward_count"] += 1
            mb_idx = _state["backward_count"]
            payload = {
                "kind": "grad_accum",
                "microbatch_idx_within_step": mb_idx,
                "outer_step": _state["step_count"],
                "callsite": callsite(skip=2),
            }
            s.emit(_HOOKPOINT, payload)
        except Exception:
            pass
        return result

    wrapped_backward._trainaudit_wrapped = True  # type: ignore[attr-defined]
    wrapped_backward.__wrapped__ = orig_backward  # type: ignore[attr-defined]
    torch.autograd.backward = wrapped_backward
    return True
