"""Hook torch.nn.functional loss ops to trace per-token loss distribution.

Currently:
  - F.cross_entropy: emits loss.call event with per-token + aggregate stats.
  - F.nll_loss: same.

Why per-token: aggregate loss alone hides bugs like:
  - reduction=='sum' instead of 'mean' (loss is N× too large but trains)
  - ignore_index wrong (some real labels masked → some positions never train)
  - degenerate prediction (one position always gives huge loss)
  - label leak (positions always give 0 loss because of label/input alignment)

The per-token recompute uses reduction='none' on the same logits/target — adds
~1 forward over the loss kernel but no gradient cost.

Lifecycle: store is read at call time via _utils.active_store().
"""
import torch
import torch.nn.functional as F

from ..store import TraceStore
from ._utils import (active_store, callsite, set_active_store, should_emit,
                     summarize_tensor)


_HOOKPOINT = "loss.call"


def _per_token_loss_stats(input_logits: torch.Tensor,
                            target: torch.Tensor,
                            ignore_index: int = -100,
                            ce_fn=None) -> dict:
    """Recompute loss with reduction='none' to get per-token distribution.

    Pass `ce_fn` = the UNWRAPPED F.cross_entropy so recompute doesn't
    re-trigger the wrapper hook (infinite event spam).

    Returns dict with median, p99, max, min, std, mean, n_ignored, n_total.
    Skipped silently if recompute fails (e.g. weird shapes).
    """
    out = {}
    if ce_fn is None:
        ce_fn = F.cross_entropy
    try:
        with torch.no_grad():
            per_tok = ce_fn(
                input_logits.detach().float(),
                target.detach(),
                ignore_index=ignore_index,
                reduction="none",
            )
            valid_mask = target != ignore_index
            out["n_total"] = int(target.numel())
            out["n_ignored"] = int((~valid_mask).sum().item())
            valid = per_tok[valid_mask] if valid_mask.dim() == per_tok.dim() else per_tok
            if valid.numel() == 0:
                out["all_ignored"] = True
                return out
            v = valid.float()
            out["mean"] = float(v.mean().item())
            out["std"] = float(v.std().item()) if valid.numel() > 1 else 0.0
            out["min"] = float(v.min().item())
            out["max"] = float(v.max().item())
            out["median"] = float(v.median().item())
            # p99 via topk for efficiency
            k = max(1, min(int(0.01 * valid.numel()), valid.numel()))
            out["p99"] = float(torch.topk(v.flatten(), k).values.min().item())
            out["all_ignored"] = False
    except Exception as e:
        out["recompute_error"] = type(e).__name__
    return out


def install_loss_hooks(store: TraceStore) -> bool:
    """Wrap F.cross_entropy and F.nll_loss to emit loss.call events."""
    set_active_store(store)
    n_wrapped = 0

    orig_ce = F.cross_entropy
    if not getattr(orig_ce, "_trainaudit_wrapped", False):

        def wrapped_cross_entropy(input, target, weight=None, size_average=None,
                                    ignore_index=-100, reduce=None,
                                    reduction="mean", label_smoothing=0.0):
            out = orig_ce(input, target, weight=weight,
                          size_average=size_average,
                          ignore_index=ignore_index, reduce=reduce,
                          reduction=reduction, label_smoothing=label_smoothing)
            s = active_store()
            if s is None or not should_emit(_HOOKPOINT):
                return out
            try:
                payload = {
                    "kind": "loss",
                    "fn": "F.cross_entropy",
                    "reduction": reduction,
                    "ignore_index": ignore_index,
                    "label_smoothing": float(label_smoothing),
                    "input_shape": list(input.shape) if torch.is_tensor(input) else None,
                    "target_shape": list(target.shape) if torch.is_tensor(target) else None,
                    "input_dtype": (str(input.dtype).replace("torch.", "")
                                     if torch.is_tensor(input) else None),
                    "aggregate_loss": (float(out.detach().float().item())
                                        if torch.is_tensor(out) and out.numel() == 1
                                        else None),
                    "callsite": callsite(skip=2),
                }
                # Per-token recompute (only when input is a logit tensor we can
                # re-run cross_entropy on). Skip if reduction is already 'none'.
                if (torch.is_tensor(input) and torch.is_tensor(target)
                        and input.dim() >= 2 and reduction != "none"):
                    payload["per_token"] = _per_token_loss_stats(
                        input, target, ignore_index=ignore_index,
                        ce_fn=orig_ce)
                s.emit(_HOOKPOINT, payload)
            except Exception:
                pass
            return out

        wrapped_cross_entropy._trainaudit_wrapped = True  # type: ignore[attr-defined]
        wrapped_cross_entropy.__wrapped__ = orig_ce  # type: ignore[attr-defined]
        F.cross_entropy = wrapped_cross_entropy
        import torch.nn.functional as _F
        _F.cross_entropy = wrapped_cross_entropy
        n_wrapped += 1

    orig_nll = F.nll_loss
    if not getattr(orig_nll, "_trainaudit_wrapped", False):

        def wrapped_nll_loss(input, target, weight=None, size_average=None,
                              ignore_index=-100, reduce=None, reduction="mean"):
            out = orig_nll(input, target, weight=weight,
                            size_average=size_average,
                            ignore_index=ignore_index, reduce=reduce,
                            reduction=reduction)
            s = active_store()
            if s is None or not should_emit(_HOOKPOINT):
                return out
            try:
                s.emit(_HOOKPOINT, {
                    "kind": "loss",
                    "fn": "F.nll_loss",
                    "reduction": reduction,
                    "ignore_index": ignore_index,
                    "input_shape": list(input.shape) if torch.is_tensor(input) else None,
                    "target_shape": list(target.shape) if torch.is_tensor(target) else None,
                    "aggregate_loss": (float(out.detach().float().item())
                                        if torch.is_tensor(out) and out.numel() == 1
                                        else None),
                    "callsite": callsite(skip=2),
                })
            except Exception:
                pass
            return out

        wrapped_nll_loss._trainaudit_wrapped = True  # type: ignore[attr-defined]
        wrapped_nll_loss.__wrapped__ = orig_nll  # type: ignore[attr-defined]
        F.nll_loss = wrapped_nll_loss
        import torch.nn.functional as _F
        _F.nll_loss = wrapped_nll_loss
        n_wrapped += 1

    return n_wrapped > 0
