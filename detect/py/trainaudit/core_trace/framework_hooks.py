"""Framework-hook State collector (E10 / BENCH.md A11). New module; no existing hook is changed
(layout_state_hook.py, the E9 collector, is frozen and untouched).

Where E9's collector was called explicitly by a hand-written training loop, this collector is
attached to the frameworks' own extension points:

  torch DDP            a DDP communication hook that wraps the default all-reduce semantics
                       (divide the bucket by the process-group size, all-reduce SUM): the
                       per-parameter views of the bucket are recorded before the division
                       (grad_pre_sync) and the slices of the reduced bucket after the collective
                       (grad_post_sync, in the future's callback).
  ZeroRedundancyOptimizer   step post-hook of the rank-local inner optimizer (param_local_step:
                       after the owner's update, before the parameter broadcast) and of the
                       outer optimizer (step marker).
  torch FSDP2          a wrapper around FSDP2's foreach_reduce (the unsharded gradients that enter
                       the reduce-scatter, recorded per dim-0 chunk = per destination shard:
                       grad_pre_sync with slice = chunk index), forward pre-hooks on the leaf
                       modules (param_gathered: the all-gathered parameters FSDP2 installs for the
                       forward), the sharded gradients after backward (grad_post_sync, slice = own
                       shard) and the shards at rest after the step (param_after_step).
  DTensor TP + DDP     the DDP communication hook on the DP group (local TP shards).
  pipelining (PP)      the gradients after the schedule's backward (grad_pre_sync) and after the
                       tied-embedding all-reduce (grad_post_sync).
  every layout         an optimizer step post-hook emits the per-rank step marker; the training
                       loop calls end_step() at the end of every iteration (param_after_step and
                       the per-step flush).

Hookpoints (TraceStore events table; build-time rows have step = -1):

  fw.topo     {"layout", "world_size", "dims": {dim: {"size", "group", "coord", "members"}}}
  fw.obj      {"objects": [{"obj", "base", "kind": "param", "requires_grad", "opt_group",
                            "tied": group-id|None,
                            "rest": {"shard_dims": [...], "slice": int},     (at rest)
                            "phases": {hook: {"shard_dims": [...], "slices": [int, ...]}},
                            "red": {"op": "mean"|"sum", "dim": "DP"|"TIED"} | None}]}
              per object and per hook phase: along which dims the record of that phase is
              sharded and which slices (dim-0 chunk indices; -1 = the whole local tensor) this
              rank records. red = the framework's gradient reduction convention for the
              grad_post_sync record (Meta: used by G3).
  fw.grad_pre_sync / fw.grad_post_sync / fw.param_local_step / fw.param_after_step /
  fw.param_gathered
              {"tensors": [{"obj", "slice", "cls", "cksum", "sum", "l2", "numel", "nonfinite"}]}
              cls = replica class id computed AT COLLECTION from the rank's coordinates and the
              phase's shard dims (used by the shared replica-class scan; the views recompute
              it from Meta).
  fw.marker   {"step": s, "n": number of optimizer steps this rank has executed}
  fw.scalar   {"loss": float, ...}

Tiers: "cksum" = sha256 of the fp32 bytes + float64 sum and L2 (official runs); "suml2" = float64
sum and L2 only; "off" = no State rows (markers and Meta only). Sampling: State rows only at
steps with step % every == 0.
"""
import hashlib
import math
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

HP_TOPO = "fw.topo"
HP_OBJ = "fw.obj"
HP_PRE = "fw.grad_pre_sync"
HP_POST = "fw.grad_post_sync"
HP_PLOCAL = "fw.param_local_step"
HP_PARAM = "fw.param_after_step"
HP_GATHER = "fw.param_gathered"
HP_MARKER = "fw.marker"
HP_SCALAR = "fw.scalar"
STATE_HOOKS = (HP_PRE, HP_POST, HP_PLOCAL, HP_PARAM, HP_GATHER)
BUILD_STEP = -1


def tensor_stats(t: torch.Tensor, tier: str) -> Dict[str, Any]:
    x = t.detach()
    if x.device.type != "cpu":
        x = x.cpu()
    xd = x.double()
    s = float(xd.sum().item())
    l2 = float(xd.norm().item())
    out = {"sum": s, "l2": l2, "numel": int(x.numel()),
           "nonfinite": not (math.isfinite(s) and math.isfinite(l2))}
    if tier == "cksum":
        # clone: .numpy() on the original would pin its storage (FSDP2 frees/resizes the
        # storage of gathered parameters after forward)
        out["cksum"] = hashlib.sha256(x.float().contiguous().clone().numpy().tobytes()).hexdigest()
    else:
        out["cksum"] = None
    return out


class FrameworkCollector:
    def __init__(self, store, rank: int, tier: str = "cksum", every: int = 1):
        assert tier in ("cksum", "suml2", "off")
        self.store = store
        self.rank = int(rank)
        self.tier = tier
        self.every = max(1, int(every))
        self.step = 0
        self.n_steps = 0
        self._lock = threading.Lock()
        self._buf: Dict[str, List[Dict[str, Any]]] = {}
        self.coords: Dict[str, int] = {}
        self.phase_dims: Dict[Tuple[str, str], List[str]] = {}   # (obj, hook) -> shard dims
        self.rest_dims: Dict[str, List[str]] = {}
        self.timing = {"collect_s": 0.0}
        store.set_rank(self.rank)

    # ---- Meta -------------------------------------------------------------------------------
    def emit_topo(self, layout: str, world_size: int, dims: Dict[str, Dict[str, Any]]) -> None:
        self.coords = {d: int(v["coord"]) for d, v in dims.items()}
        self.store.emit(HP_TOPO, {"layout": layout, "world_size": int(world_size), "dims": dims},
                        step=BUILD_STEP, rank=self.rank)

    def emit_obj(self, objects: List[Dict[str, Any]]) -> None:
        for o in objects:
            self.rest_dims[o["obj"]] = list(o["rest"]["shard_dims"])
            for h, ph in o.get("phases", {}).items():
                self.phase_dims[(o["obj"], h)] = list(ph["shard_dims"])
        self.store.emit(HP_OBJ, {"objects": objects}, step=BUILD_STEP, rank=self.rank)

    def cls_of(self, obj: str, hook: str) -> str:
        dims = self.phase_dims.get((obj, hook), self.rest_dims.get(obj, []))
        return ",".join(f"{d}={self.coords.get(d, 0)}" for d in sorted(dims))

    # ---- State ------------------------------------------------------------------------------
    def active(self, step: Optional[int] = None) -> bool:
        s = self.step if step is None else step
        return self.tier != "off" and s % self.every == 0

    def set_step(self, step: int) -> None:
        self.step = int(step)
        self.store.set_step(self.step)

    def add(self, hook: str, items: Iterable[Tuple[str, int, Optional[torch.Tensor]]],
            step: Optional[int] = None) -> None:
        """items: (obj, slice, tensor). Thread-safe (DDP future callbacks may run on a
        communication thread). Rows are buffered and written by end_step()."""
        if hook not in STATE_HOOKS:
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

    def marker(self) -> None:
        """Called from an optimizer step post-hook."""
        self.n_steps += 1
        self.store.emit(HP_MARKER, {"step": self.step, "n": self.n_steps}, step=self.step,
                        rank=self.rank)

    def scalar(self, **values: Any) -> None:
        self.store.emit(HP_SCALAR, {k: (float(v) if isinstance(v, (int, float)) else v)
                                    for k, v in values.items()}, step=self.step, rank=self.rank)

    def end_step(self, param_items: Optional[Iterable[Tuple[str, int, torch.Tensor]]] = None) -> None:
        if param_items is not None:
            self.add(HP_PARAM, param_items)
        t0 = time.perf_counter()
        with self._lock:
            buf, self._buf = self._buf, {}
        for (hook, s), recs in sorted(buf.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if recs:
                self.store.emit(hook, {"tensors": recs}, step=s, rank=self.rank)
        self.timing["collect_s"] += time.perf_counter() - t0


# ---- DDP communication hook ------------------------------------------------------------------
class DDPCaptureHook:
    """Wraps the default DDP all-reduce hook semantics (torch.distributed.algorithms.
    ddp_comm_hooks.default_hooks.allreduce_hook: bucket /= group size; all_reduce SUM).

    `fault` (injection, None on clean runs) may change the semantics:
      {"kind": "sum_not_avg"}                      no division
      {"kind": "wrong_group", "group": pg, "size": n}   collective over another group
      {"kind": "skip_sync", "obj": name}           this rank keeps its local gradient of `obj`
                                                   (the reduced slice is overwritten with the local
                                                   value and no synchronized value is produced, so
                                                   no grad_post_sync row is recorded for it)
    fault_active() decides per step."""

    def __init__(self, col: FrameworkCollector, names_by_id: Dict[int, str], group, group_size: int,
                 fault: Optional[Dict[str, Any]] = None, fault_active=None):
        self.col = col
        self.names = names_by_id
        self.group = group
        self.size = group_size
        self.fault = fault
        self.fault_active = fault_active or (lambda: False)
        self.offsets_checked = False
        self.__name__ = "ta_capture_allreduce_hook"   # DDP._check_comm_hook reads hook.__name__

    def __call__(self, state, bucket):
        import torch.distributed as dist
        col = self.col
        step = col.step
        buf = bucket.buffer()
        params = bucket.parameters()
        grads = bucket.gradients()
        names = [self.names[id(p)] for p in params]
        # offsets of the per-parameter views inside the flat bucket buffer
        offs = []
        base = buf.data_ptr()
        es = buf.element_size()
        for g in grads:
            offs.append((g.data_ptr() - base) // es)
        if not self.offsets_checked:
            # the views must lie inside the buffer and not overlap
            ends = sorted((o, o + g.numel()) for o, g in zip(offs, grads))
            assert all(0 <= a and b <= buf.numel() for a, b in ends), "bucket view offsets"
            assert all(ends[i][1] <= ends[i + 1][0] for i in range(len(ends) - 1)), "overlap"
            self.offsets_checked = True
        col.add(HP_PRE, [(n, -1, g) for n, g in zip(names, grads)], step=step)
        f = self.fault if (self.fault and self.fault_active()) else None
        keep_local = {}
        if f and f["kind"] == "skip_sync":
            for n, g in zip(names, grads):
                if n == f["obj"]:
                    keep_local[n] = g.detach().clone()
        group, size, divide = self.group, self.size, True
        if f and f["kind"] == "sum_not_avg":
            divide = False
        if f and f["kind"] == "wrong_group":
            group, size = f["group"], f["size"]
        if divide:
            buf.div_(size)
        fut = dist.all_reduce(buf, group=group, async_op=True).get_future()

        def done(fut):
            out = fut.value()[0]
            items = []
            for n, g, o in zip(names, grads, offs):
                sl = out[o:o + g.numel()].view_as(g)
                if n in keep_local:
                    sl.copy_(keep_local[n])
                    continue
                items.append((n, -1, sl))
            col.add(HP_POST, items, step=step)
            return out

        return fut.then(done)


# ---- FSDP2 ------------------------------------------------------------------------------------
_FSDP_PATCHED = {"col": None, "names": None, "world": None}


def install_fsdp2_reduce_capture(col: FrameworkCollector, names_by_id: Dict[int, str], world: int):
    """Wrap FSDP2's foreach_reduce (as referenced from _fsdp_param_group) so that the unsharded
    gradients entering the reduce-scatter are recorded, chunked along dim 0 into the `world`
    destination shards (FSDP2 shards dim 0 with torch.chunk semantics)."""
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as PG
    _FSDP_PATCHED.update(col=col, names=names_by_id, world=world)
    if getattr(PG, "_ta_wrapped", False):
        return
    orig = PG.foreach_reduce

    def wrapped(fsdp_params, unsharded_grads, *args, **kwargs):
        c = _FSDP_PATCHED["col"]
        if c is not None and c.active():
            items = []
            for fp, g in zip(fsdp_params, unsharded_grads):
                n = _FSDP_PATCHED["names"][id(fp.sharded_param)]
                for j, ch in enumerate(torch.chunk(g, _FSDP_PATCHED["world"], 0)):
                    items.append((n, j, ch))
            c.add(HP_PRE, items)
        return orig(fsdp_params, unsharded_grads, *args, **kwargs)

    PG.foreach_reduce = wrapped
    PG._ta_wrapped = True


def install_gathered_capture(col: FrameworkCollector, model: torch.nn.Module, first_only=lambda: True):
    """Forward pre-hooks on every leaf module that owns parameters: inside an FSDP2 forward the
    module's parameters are the all-gathered (unsharded) tensors."""
    handles = []
    for mname, m in model.named_modules():
        own = list(m.named_parameters(recurse=False))
        if not own:
            continue

        def pre(mod, args, _mname=mname):
            if not first_only():
                return
            col.add(HP_GATHER, [((_mname + "." + pn) if _mname else pn, -1, p)
                                for pn, p in mod.named_parameters(recurse=False)])
        handles.append(m.register_forward_pre_hook(pre))
    return handles
