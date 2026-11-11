"""Framework-hook State collector v3 (E12 / BENCH.md A12). New module: framework_hooks.py (E10,
frozen) is imported, never modified.

What changes against E10 (round-3 review R2 W1): no probe sits inside a communication path.

  torch DDP (ddp2, ddp4, zero1, tp2dp2, ddp2bf16)
      grad_pre_sync    per-parameter post-accumulate-grad hooks (Tensor.register_post_accumulate_
                       grad_hook) on every trainable parameter, armed only during the backward of
                       the last micro-batch (the one DDP synchronizes). They fire in AccumulateGrad,
                       after the local gradient has been accumulated into param.grad and before
                       DDP's reducer hook (a post hook of the same AccumulateGrad node) copies it
                       into a bucket. The value is the rank-local accumulated gradient.
      grad_post_sync   param.grad read by the training loop after backward() returned (DDP's
                       reducer has finalized: reduced buckets copied back into param.grad), for
                       the parameters DDP synchronizes. That set is read from the DDP object
                       after wrapping (DDP._module_parameters, i.e. the module's parameters minus
                       DDP.parameters_to_ignore); a parameter that is in no DDP bucket has no
                       synchronized gradient and therefore no grad_post_sync row.
      DDP keeps its built-in C++ reducer (no TrainAudit communication hook is registered); a
      user communication hook, if any, belongs to the training script, not to the collector.
  ZeroRedundancyOptimizer    as E10: step post-hook of the rank-local inner optimizer
                       (param_local_step: before the parameter broadcast) and of the outer one.
  torch FSDP2 (fsdp2, fsdp2bf16)   as E10 (foreach_reduce wrapper = unsharded gradients entering
                       the reduce-scatter, per destination chunk; gathered parameters via leaf
                       forward pre-hooks; sharded gradient after backward; shards after the step).
  pipelining (pp2)     as E10 (driver: after the schedule's backward and after the tied-embedding
                       all-reduce written in the training script).
  every layout         optimizer step post-hook: step marker AND the optimizer hyperparameters
                       (per parameter group lr and momentum of the optimizer that applied the
                       update; ZeRO-1: the rank-local inner optimizer) as State rows at hook
                       fw.opt_hparam (objects "opt.g<i>.lr", "opt.g<i>.momentum").

Obj v3 (fw.obj payload), per object:
  {"obj", "base", "kind": "param"|"hparam", "requires_grad", "opt_group", "tied", "owner": rank|None,
   "red": {"op": "mean"|"sum", "dim": "DP"|"TIED", "from": hook, "to": hook} | None,
   "rest": {"shard_dims": [...], "slice": int, "hook": hook recording the at-rest state},
   "phases": {hook: {"shard_dims": [...], "slices": [...], "order": int, "dtype": "fp32"|"bf16",
                     "pending": None | {"op": "mean"|"sum", "dims": [...]}
                                     | {"op": "broadcast", "dims": [...], "from": "owner"}}}}
  pending = a collective that still has to run before the record is settled (pre-sync gradients:
  the DP reduction; ZeRO-1 before-broadcast parameters: the owner's broadcast). order = position
  of the hook inside one training step. The replica class computed at collection (cls) uses the
  phase's shard dims plus its pending dims.
"""
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .framework_hooks import (BUILD_STEP, HP_GATHER, HP_MARKER, HP_OBJ, HP_PARAM, HP_PLOCAL,  # noqa
                              HP_POST, HP_PRE, HP_SCALAR, HP_TOPO, FrameworkCollector,
                              install_fsdp2_reduce_capture, install_gathered_capture, tensor_stats)

HP_HPARAM = "fw.opt_hparam"
STATE_HOOKS_V3 = (HP_PRE, HP_POST, HP_PLOCAL, HP_PARAM, HP_GATHER, HP_HPARAM)
ORDER = {HP_GATHER: 0, HP_PRE: 1, HP_POST: 2, HP_PLOCAL: 3, HP_HPARAM: 4, HP_PARAM: 5}
HPARAM_KEYS = ("lr", "momentum")


class FrameworkCollectorV3(FrameworkCollector):
    def emit_obj(self, objects: List[Dict[str, Any]]) -> None:
        for o in objects:
            self.rest_dims[o["obj"]] = list(o["rest"]["shard_dims"])
            for h, ph in o.get("phases", {}).items():
                pend = ph.get("pending") or {}
                self.phase_dims[(o["obj"], h)] = sorted(d for d in set(ph["shard_dims"]) | set(pend.get("dims", []))
                                                       if d in self.coords)
        self.store.emit(HP_OBJ, {"objects": objects, "obj_schema": "v3"}, step=BUILD_STEP,
                        rank=self.rank)

    def add(self, hook: str, items: Iterable[Tuple[str, int, Optional[torch.Tensor]]],
            step: Optional[int] = None) -> None:
        if hook not in STATE_HOOKS_V3:
            raise ValueError(hook)
        s = self.step if step is None else int(step)
        if not self.active(s):
            return
        t0 = time.perf_counter()
        recs = []
        for obj, sl, t in items:
            if t is None:
                continue
            r = {"obj": obj, "slice": int(sl), "cls": self.cls_of(obj, hook)}
            r.update(tensor_stats(t, self.tier))
            recs.append(r)
        dt = time.perf_counter() - t0
        with self._lock:
            self._buf.setdefault((hook, s), []).extend(recs)
            self.timing["collect_s"] += dt

    def add_hparams(self, optimizer) -> None:
        """Per parameter group lr and momentum of `optimizer` (float64 scalars; checksum = sha256
        of the float64 bytes) at the current step."""
        items = []
        for i, g in enumerate(optimizer.param_groups):
            for k in HPARAM_KEYS:
                items.append((f"opt.g{i}.{k}", -1, torch.tensor([float(g.get(k, 0.0))], dtype=torch.float64)))
        s = self.step
        if not self.active(s):
            return
        recs = []
        import hashlib
        for obj, sl, t in items:
            v = float(t.item())
            recs.append({"obj": obj, "slice": -1, "cls": self.cls_of(obj, HP_HPARAM), "sum": v,
                         "l2": abs(v), "numel": 1, "nonfinite": not (v == v and abs(v) != float("inf")),
                         "cksum": hashlib.sha256(t.numpy().tobytes()).hexdigest() if self.tier == "cksum" else None})
        with self._lock:
            self._buf.setdefault((HP_HPARAM, s), []).extend(recs)


def hparam_objects(n_groups: int) -> List[Dict[str, Any]]:
    out = []
    for i in range(n_groups):
        for k in HPARAM_KEYS:
            out.append({"obj": f"opt.g{i}.{k}", "base": f"opt.g{i}.{k}", "kind": "hparam",
                        "requires_grad": False, "opt_group": i, "tied": None, "owner": None,
                        "red": None, "rest": {"shard_dims": [], "slice": -1, "hook": HP_HPARAM},
                        "phases": {HP_HPARAM: {"shard_dims": [], "slices": [-1], "pending": None,
                                               "order": ORDER[HP_HPARAM], "dtype": "fp64"}}})
    return out


class DDPGradProbes:
    """Pre-sync: post-accumulate-grad hooks (armed for the synchronized micro-batch only).
    Post-sync: param.grad after backward for the parameters DDP synchronizes."""

    def __init__(self, col: FrameworkCollectorV3, ddp, named_params: List[Tuple[str, torch.nn.Parameter]]):
        self.col = col
        self.armed = False
        self.named = [(n, p) for n, p in named_params if p.requires_grad]
        synced_ids = {id(p) for p in ddp._module_parameters}
        self.synced = [(n, p) for n, p in self.named if id(p) in synced_ids]
        self.not_synced = [n for n, p in self.named if id(p) not in synced_ids]
        self.handles = []
        for n, p in self.named:
            self.handles.append(p.register_post_accumulate_grad_hook(self._mk(n)))
        self.fired = 0

    def _mk(self, name):
        def hook(p):
            if self.armed and p.grad is not None:
                self.fired += 1
                self.col.add(HP_PRE, [(name, -1, p.grad)])
        return hook

    def arm(self, on: bool) -> None:
        self.armed = bool(on)

    def record_post(self) -> None:
        self.col.add(HP_POST, [(n, -1, p.grad) for n, p in self.synced])
