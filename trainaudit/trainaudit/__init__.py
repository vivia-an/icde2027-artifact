"""TrainAudit public API — Phase 1 (T0 PyTorch core).

Typical usage:

    import trainaudit
    trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path="./trace.duckdb")
    trainaudit.snapshot_build(model, optimizer)
    # ... training ...
    results = trainaudit.run_rules()
    print(trainaudit.summarize(results))
    trainaudit.disable()
"""
from typing import Dict, List, Optional

from . import core_trace, rules, schema
from .core_trace import (install_checkpoint_hook, install_clip_grad_hook,
                         install_dataloader_hook, install_dist_hooks,
                         install_functional_hooks, install_loss_hooks,
                         install_lr_scheduler_hook,
                         install_module_hooks, install_optim_hooks,
                         take_build_snapshot)
from .diagnosis import DiagnosisReport, expand_results, expand_violation
from .rules.base import RuleResult
from .store import TraceStore
from .tiers import Tier
from .verifier import run_rules as _run_rules
from .verifier import summarize

__all__ = [
    "Tier", "TraceStore", "RuleResult", "DiagnosisReport",
    "enable", "disable", "snapshot_build", "set_step", "set_rank",
    "run_rules", "summarize", "get_store", "is_enabled",
    "diagnose", "expand_violation",
]


def diagnose(results=None):
    """Expand every violation in `results` (default: from current trace)
    into a list of DiagnosisReport. Returns [] on a clean run."""
    if not _state["enabled"]:
        raise RuntimeError("call trainaudit.enable() first")
    if results is None:
        results = _run_rules(_state["store"])  # type: ignore[arg-type]
    return expand_results(_state["store"], results)  # type: ignore[arg-type]


_state = {
    "enabled": False,
    "tier": None,
    "store": None,
    "module_handles": [],
}


def enable(tier: Tier = Tier.T0_PYTORCH,
           db_path: str = ":memory:",
           *,
           async_mode: bool = False,
           async_queue_size: int = 8192,
           sample_rates: Optional[Dict[str, float]] = None) -> TraceStore:
    """Install hooks and start writing trace to `db_path`.

    Idempotent: subsequent calls return the existing store.

    Production knobs (P0 #1):
      - `async_mode=True` runs json.dumps + DuckDB INSERT off the training
        step's critical path via a daemon thread. emit() returns after a
        single counter++ + queue.put_nowait. Required for paper §4.3 <5%
        overhead. Adds a small queueing window before events become
        visible to rule reads — `flush()` waits for drain so callers
        don't see this. Use sync mode (default) for tests + tight
        debugging loops.
      - `async_queue_size` caps in-memory queue. When full, emit() blocks
        50ms then drops the event (counter still advances; counter
        recorded in `store.health()['dropped']`). Default 8192 holds ~10s
        of events at typical hookpoint rates.
    """
    if _state["enabled"]:
        return _state["store"]  # type: ignore[return-value]

    store = TraceStore(db_path, async_mode=async_mode,
                        async_queue_size=async_queue_size)
    # P0 #2: per-hookpoint sampling. Keys are hookpoint strings (e.g.
    # 'module.fwd.post'); values in [0, 1]. Hooks check should_emit()
    # before computing tensor stats, so sampled-out events pay zero cost.
    if sample_rates:
        from .core_trace._utils import set_sample_rates
        set_sample_rates(sample_rates)
    if tier >= Tier.T0_PYTORCH:
        install_dist_hooks(store)
        _state["module_handles"] = install_module_hooks(store)
        install_optim_hooks(store)
        install_clip_grad_hook(store)
        install_lr_scheduler_hook(store)
        install_checkpoint_hook(store)
        install_dataloader_hook(store)
        install_functional_hooks(store)
        install_loss_hooks(store)

    if tier >= Tier.T1_FW_METADATA:
        # Phase 2 — adapter metadata reader install hooks here
        pass
    if tier >= Tier.T2_FW_PRIMITIVE:
        pass
    if tier >= Tier.T3_FW_SPECIFIC:
        pass

    _state.update(enabled=True, tier=tier, store=store)
    return store


def disable() -> None:
    if not _state["enabled"]:
        return
    for h in _state.get("module_handles", []):
        try:
            h.remove()
        except Exception:
            pass
    for h in _state.get("attention_handles", []):
        try:
            h.remove()
        except Exception:
            pass
    if _state["store"] is not None:
        _state["store"].close()  # type: ignore[union-attr]
    _state.update(enabled=False, tier=None, store=None, module_handles=[],
                   attention_handles=[])
    # Clear any sampling config so the next enable() starts fresh
    try:
        from .core_trace._utils import reset_sampling
        reset_sampling()
    except Exception:
        pass


def is_enabled() -> bool:
    return bool(_state["enabled"])


def get_store() -> Optional[TraceStore]:
    return _state["store"]  # type: ignore[return-value]


def snapshot_build(model=None, optimizer=None, *, rank: int = 0) -> int:
    if not _state["enabled"]:
        raise RuntimeError("call trainaudit.enable() first")
    n = take_build_snapshot(_state["store"], model=model,    # type: ignore[arg-type]
                              optimizer=optimizer, rank=rank)
    # Install per-attention-module hooks now that we have the model
    if model is not None:
        from .core_trace.attention_hook import install_attention_hooks
        try:
            handles = install_attention_hooks(_state["store"], model)
            _state.setdefault("attention_handles", []).extend(handles)
        except Exception:
            pass
    return n


def set_step(step: int) -> None:
    if _state["store"] is not None:
        _state["store"].set_step(step)  # type: ignore[union-attr]


def set_rank(rank: int) -> None:
    if _state["store"] is not None:
        _state["store"].set_rank(rank)  # type: ignore[union-attr]


def run_rules(tier: Optional[Tier] = None, *,
              use_dsl: bool = False) -> List[RuleResult]:
    """Run rules. `use_dsl=True` switches from the Python rule registry to
    the YAML DSL registry under `trainaudit/dsl/registry/` (M1 gate path)."""
    if not _state["enabled"]:
        raise RuntimeError("call trainaudit.enable() first")
    if tier is None:
        tier = _state["tier"] or Tier.T0_PYTORCH  # type: ignore[assignment]
    return _run_rules(_state["store"], tier, use_dsl=use_dsl)  # type: ignore[arg-type]
