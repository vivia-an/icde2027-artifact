"""36-cell GPU detection on Megatron-LM.

The model is Megatron's GPTModel. Layouts are the 36-cell menu: data parallel
at two widths, ZeRO-1, FSDP2 around that model, tensor parallel, a two-stage
pipeline with tied embeddings, and the bf16 variants. Each run is 300 steps.
The trace is rank*.duckdb. score_guarded.py compiles the frozen specs and checks it.

Megatron is not copied into this tree. Set MEGATRON_HOME to the checkout.
This fork also imports vtimeline; set VTIMELINE_SRC if it is not already on PYTHONPATH.

  export MEGATRON_HOME=/path/to/Megatron-LM
  python detect/launch_megatron.py --layout ddp2 --injection f_no_accum_divisor --seed 0 --out /tmp/ta-run
  python detect/score_guarded.py --layout ddp2 --run /tmp/ta-run
"""
import argparse
import contextlib
import json
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "py"))

STEPS = 300
INJ_STEP = 5
LAYERS, HIDDEN, HEADS, SEQ, FFN, VOCAB = 2, 128, 4, 32, 256, 128
MICRO, MB = 2, 2
LR, MOM, WD = 0.05, 0.9, 1e-4
BUCKET_MB = 0.05

LAYOUTS = {
    "ddp2": dict(dp=2, tp=1, pp=1, base="ddp2", bf16=False),
    "ddp4": dict(dp=4, tp=1, pp=1, base="ddp4", bf16=False),
    "zero1": dict(dp=2, tp=1, pp=1, base="zero1", bf16=False),
    "fsdp2": dict(dp=2, tp=1, pp=1, base="fsdp2", bf16=False),
    "tp2dp2": dict(dp=2, tp=2, pp=1, base="tp2dp2", bf16=False),
    "pp2": dict(dp=1, tp=1, pp=2, base="pp2", bf16=False),
    "ddp2bf16": dict(dp=2, tp=1, pp=1, base="ddp2", bf16=True),
    "fsdp2bf16": dict(dp=2, tp=1, pp=1, base="fsdp2", bf16=True),
}
DDP_BASES = ("ddp2", "ddp4", "zero1", "tp2dp2")
TP_OWNED = ("ColumnParallelLinear", "RowParallelLinear", "VocabParallelEmbedding")

with open(os.path.join(ROOT, "catalog", "check_specs", "injections.json")) as _f:
    INJECTIONS = {i["id"]: i for i in json.load(_f)["injections"]}


def megatron_home():
    env = os.environ.get("MEGATRON_HOME", "").strip()
    if env and os.path.isdir(os.path.join(env, "megatron")):
        return env
    raise SystemExit("set MEGATRON_HOME to a Megatron-LM checkout")


def prepare_import_path():
    home = megatron_home()
    if home not in sys.path:
        sys.path.insert(0, home)
    vsrc = os.environ.get("VTIMELINE_SRC", "").strip()
    if not vsrc:
        sib = os.path.join(os.path.dirname(home), "VTimeline", "src")
        if os.path.isdir(sib):
            vsrc = sib
    if vsrc and vsrc not in sys.path:
        sys.path.insert(0, vsrc)
    return home


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def param_groups(named):
    g0 = [p for n, p in named if p.dim() >= 2]
    g1 = [p for n, p in named if p.dim() < 2]
    return [{"params": g0, "weight_decay": WD}, {"params": g1, "weight_decay": 0.0}]


def opt_group_of(shape_dim):
    return 0 if shape_dim >= 2 else 1


def user_comm_hook(group, size, faulty_from_step, step_ref):
    import torch.distributed as dist

    def user_allreduce_hook(state, bucket):
        buf = bucket.buffer()
        if step_ref() < faulty_from_step:
            buf.div_(size)
        return dist.all_reduce(buf, group=group, async_op=True).get_future().then(lambda f: f.value()[0])
    return user_allreduce_hook


def tp_owned_names(model):
    owned = set()
    for mod_name, mod in model.named_modules():
        if type(mod).__name__ not in TP_OWNED:
            continue
        for pn, _p in mod.named_parameters(recurse=False):
            owned.add(f"{mod_name}.{pn}" if mod_name else pn)
    return owned


def resolve_targets(model, kind, layout):
    names = [n for n, _p in model.named_parameters()]

    def pick(*parts):
        for n in names:
            if all(p in n for p in parts):
                return n
        return None

    if kind == "ddp_ignore_param":
        n = pick("layers.0", "linear_fc2.weight")
        if n is None:
            raise SystemExit("megatron linear_fc2.weight not found for g2")
        return [n]
    if kind == "rank_param_mod":
        if layout == "pp2":
            n = pick("word_embeddings.weight") or pick("output_layer.weight")
            return [n] if n else []
        n = pick("layers.0", "linear_fc1.weight")
        if n is None:
            raise SystemExit("megatron linear_fc1.weight not found for a")
        return [n]
    if kind == "tp_norm_partial":
        hits = [n for n in names if "layers.1" in n and "norm" in n]
        if not hits:
            hits = [n for n in names if "norm" in n]
        if not hits:
            raise SystemExit("megatron norm parameter not found for c")
        return hits
    if kind == "stale_gather":
        return ["1"]
    return []


def build_gpt(bf16, tp, pp, pre, post, tie):
    import torch
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    dt = torch.bfloat16 if bf16 else torch.float32
    cfg = TransformerConfig(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        ffn_hidden_size=FFN,
        params_dtype=dt,
        pipeline_dtype=dt,
        bf16=bf16,
        fp16=False,
        use_cpu_initialization=False,
        sequence_parallel=False,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=1,
    )
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        pre_process=pre,
        post_process=post,
        share_embeddings_and_output_weights=tie,
        position_embedding_type="rope",
        rotary_percent=1.0,
        parallel_output=False,
    )
    return model


def attn_mask(device):
    import torch
    mask = torch.tril(torch.ones(SEQ, SEQ, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    return ~mask


def batch(seed, step, dpc, micro, device):
    import torch
    g = torch.Generator(device="cpu").manual_seed(((seed * 1009 + step) * 101 + dpc) * 7 + micro)
    return torch.randint(0, VOCAB, (MB, SEQ), generator=g).to(device)


def worker(rank, world, layout, inj_id, seed, out, tier, every):
    import time
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from trainaudit.store import TraceStore
    from trainaudit.core_trace import framework_hooks_v3 as FH

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.manual_seed(1234 + seed)
    L = LAYOUTS[layout]
    DP, TP, PP = L["dp"], L["tp"], L["pp"]
    base, bf16 = L["base"], L["bf16"]
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=TP, pipeline_model_parallel_size=PP)
    model_parallel_cuda_manual_seed(seed)

    inj = INJECTIONS.get(inj_id) if inj_id != "clean" else None
    kind = inj["kind"] if inj else None
    inj_rank = world - 1 if inj else None
    dpc = parallel_state.get_data_parallel_rank()
    tpc = parallel_state.get_tensor_model_parallel_rank()
    ppc = parallel_state.get_pipeline_model_parallel_rank()
    dp_group = parallel_state.get_data_parallel_group()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    first = parallel_state.is_pipeline_first_stage()
    last = parallel_state.is_pipeline_last_stage()

    def fault_on(step):
        if inj is None:
            return False
        if kind == "skip_opt_step":
            return step == INJ_STEP
        if kind in ("tp_norm_partial", "ddp_ignore_param"):
            return True
        return step >= INJ_STEP

    dev = torch.device("cuda", rank)
    model = build_gpt(bf16, TP, PP, first, last, tie=(base == "pp2")).to(dev)
    model.train()
    tgt = resolve_targets(model, kind, layout) if kind else []
    owned = tp_owned_names(model)
    step_box = {"step": 0}
    if TP > 1:
        for n, p in model.named_parameters():
            if n in owned or not p.requires_grad:
                continue

            def _hook(grad, name=n):
                if kind == "tp_norm_partial" and name in tgt:
                    return grad
                dist.all_reduce(grad, group=tp_group)
                return grad

            p.register_hook(_hook)

    dims = {
        "DP": {"size": DP, "group": f"DP|TP={tpc}|PP={ppc}", "coord": dpc,
               "members": [tpc + TP * (d + DP * ppc) for d in range(DP)]},
        "TP": {"size": TP, "group": f"TP|DP={dpc}|PP={ppc}", "coord": tpc,
               "members": [t + TP * (dpc + DP * ppc) for t in range(TP)]},
        "PP": {"size": PP, "group": f"PP|DP={dpc}|TP={tpc}", "coord": ppc,
               "members": [tpc + TP * (dpc + DP * p) for p in range(PP)]},
    }
    store = TraceStore(os.path.join(out, f"rank{rank}.duckdb"))
    col = FH.FrameworkCollectorV3(store, rank, tier=tier if tier != "off" else "off", every=every)
    collect = tier != "off"
    col.emit_topo(layout, world, dims)

    shard_dims = {n: (["TP"] if n in owned else []) for n, _p in model.named_parameters()}
    base_named = list(model.named_parameters())
    names_by_id = {id(p): n for n, p in base_named}
    full_shapes = {n: list(p.shape) for n, p in base_named}
    ddp = None
    probes = None
    zero = None
    fsdp_units = []
    pdt = "bf16" if bf16 else "fp32"
    gdt = pdt
    if base in DDP_BASES:
        from torch.nn.parallel import DistributedDataParallel as DDP
        if kind == "ddp_ignore_param":
            DDP._set_params_and_buffers_to_ignore_for_model(model, list(tgt))
        ddp = DDP(model, process_group=dp_group, bucket_cap_mb=BUCKET_MB)
        if kind == "user_hook_sum":
            ddp.register_comm_hook(None, user_comm_hook(dp_group, DP, INJ_STEP, lambda: step_box["step"]))
        if collect:
            probes = FH.DDPGradProbes(col, ddp, base_named)
        fwd = ddp
    elif base == "fsdp2":
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.fsdp import fully_shard
        mesh = DeviceMesh.from_group(dp_group, "cuda")
        kw = {}
        if bf16:
            from torch.distributed.fsdp import MixedPrecisionPolicy
            kw["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
            pdt, gdt = "fp32", "bf16"
        layers = list(model.decoder.layers)
        for layer in layers:
            fully_shard(layer, mesh=mesh, **kw)
            fsdp_units.append(layer)
        fully_shard(model, mesh=mesh, **kw)
        fsdp_units.append(model)
        fwd = model
        base_named = list(model.named_parameters())
        names_by_id = {id(p): n for n, p in base_named}
        if collect:
            FH.install_fsdp2_reduce_capture(col, names_by_id, DP)
            fstate = {"first": True}
            FH.install_gathered_capture(col, model, first_only=lambda: fstate["first"])
    else:
        fwd = model

    if base == "zero1":
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(param_groups(base_named), optimizer_class=torch.optim.SGD,
                                      process_group=dp_group, lr=LR, momentum=MOM)
        zero = opt
        if collect:
            def _inner_post(o, a, k):
                col.add(FH.HP_PLOCAL, [(n, -1, p) for n, p in base_named])
                col.add_hparams(zero.optim)
            zero.optim.register_step_post_hook(_inner_post)
        owner = {}
        for r, groups in enumerate(zero._partition_parameters()):
            for g in groups:
                for p in g["params"]:
                    owner[names_by_id[id(p)]] = r
    else:
        opt = torch.optim.SGD(param_groups(base_named), lr=LR, momentum=MOM)
        owner = {}
    if collect:
        def _outer_post(o, a, k):
            col.marker()
            if base != "zero1":
                col.add_hparams(opt)
        opt.register_step_post_hook(_outer_post)

    O = FH.ORDER
    objs = []
    for n, p in base_named:
        red = {"op": "mean", "dim": "DP", "from": FH.HP_PRE, "to": FH.HP_POST}
        pend_pre = {"op": "mean", "dims": ["DP"]}
        if base == "pp2":
            red = {"op": "sum", "dim": "TIED", "from": FH.HP_PRE, "to": FH.HP_POST}
            pend_pre = {"op": "sum", "dims": ["PP"]}
        o = {"obj": n, "base": n, "kind": "param", "requires_grad": bool(p.requires_grad),
             "opt_group": opt_group_of(len(full_shapes.get(n, list(p.shape)))),
             "tied": n if base == "pp2" and ("word_embeddings" in n or "output_layer" in n) else None,
             "owner": owner.get(n), "red": red}
        if base == "fsdp2":
            o["rest"] = {"shard_dims": ["DP"], "slice": dpc, "hook": FH.HP_PARAM}
            o["phases"] = {
                FH.HP_PARAM: {"shard_dims": ["DP"], "slices": [dpc], "pending": None, "order": O[FH.HP_PARAM], "dtype": pdt},
                FH.HP_GATHER: {"shard_dims": [], "slices": [-1], "pending": None, "order": O[FH.HP_GATHER], "dtype": gdt},
                FH.HP_PRE: {"shard_dims": [], "slices": list(range(DP)), "pending": pend_pre, "order": O[FH.HP_PRE], "dtype": gdt},
                FH.HP_POST: {"shard_dims": ["DP"], "slices": [dpc], "pending": None, "order": O[FH.HP_POST], "dtype": pdt}}
            local = p.to_local() if hasattr(p, "to_local") else p
            o["shape"] = list(local.shape)
            o["full_shape"] = list(getattr(p, "shape", local.shape))
        else:
            sd = shard_dims.get(n, [])
            o["rest"] = {"shard_dims": sd, "slice": -1, "hook": FH.HP_PARAM}
            o["phases"] = {
                FH.HP_PARAM: {"shard_dims": sd, "slices": [-1], "pending": None, "order": O[FH.HP_PARAM], "dtype": pdt},
                FH.HP_PRE: {"shard_dims": sd, "slices": [-1], "pending": pend_pre, "order": O[FH.HP_PRE], "dtype": pdt},
                FH.HP_POST: {"shard_dims": sd, "slices": [-1], "pending": None, "order": O[FH.HP_POST], "dtype": pdt}}
            if base == "zero1":
                o["phases"][FH.HP_PLOCAL] = {"shard_dims": sd, "slices": [-1], "order": O[FH.HP_PLOCAL], "dtype": pdt,
                                             "pending": {"op": "broadcast", "dims": ["DP"], "from": "owner"}}
            o["shape"] = list(p.shape)
        objs.append(o)
    n_groups = len(opt.param_groups) if base != "zero1" else len(zero.optim.param_groups)
    objs += FH.hparam_objects(n_groups)
    col.emit_obj(objs)

    if rank == 0:
        gt = {"model": "megatron", "layout": layout, "base": base, "bf16": bf16, "injection": inj_id,
              "seed": seed, "world_size": world, "dp": DP, "tp": TP, "pp": PP, "tier": tier, "every": every,
              "steps": STEPS, "layers": LAYERS, "hidden": HIDDEN,
              "ddp_not_synced": (probes.not_synced if probes is not None else None)}
        if inj:
            targets = list(tgt)
            if kind == "zero_owner_lr":
                targets = sorted(n for n, r in owner.items() if r == inj_rank) + \
                    [f"opt.g{i}.lr" for i in range(len(zero.optim.param_groups))]
            if inj.get("all_params"):
                targets = "ALL"
            gt.update({"kind": kind, "class": inj["class"], "targets": targets,
                       "localized": bool(inj["localized"]),
                       "step": 0 if kind in ("tp_norm_partial", "ddp_ignore_param") else INJ_STEP,
                       "injected_rank": inj_rank, "description": inj["description"]})
        with open(os.path.join(out, "ground_truth.json"), "w") as f:
            json.dump(gt, f, indent=1)
        print(f"megatron steps={STEPS} inj_step={INJ_STEP} layout={layout} injection={inj_id} "
              f"layers={LAYERS} hidden={HIDDEN} sequence_parallel=0", flush=True)

    def tied_grads():
        buf = torch.zeros(VOCAB, HIDDEN, device=dev, dtype=next(model.parameters()).dtype)
        local = None
        for n, p in model.named_parameters():
            if "word_embeddings.weight" in n or "output_layer.weight" in n:
                local = p
                if p.grad is not None and p.grad.shape == buf.shape:
                    buf.copy_(p.grad.detach())
                elif p.grad is not None:
                    print(f"tied grad shape {tuple(p.grad.shape)} != {(VOCAB, HIDDEN)}", flush=True)
                break
        dist.all_reduce(buf, group=pp_group)
        if local is not None and local.grad is not None and local.grad.shape == buf.shape:
            local.grad.copy_(buf)

    times, losses_all = [], []
    for step in range(STEPS):
        t_it = time.perf_counter()
        step_box["step"] = step
        col.set_step(step)
        active = fault_on(step)
        if base == "fsdp2" and collect:
            fstate["first"] = True
        if active and step == INJ_STEP:
            if kind == "rank_param_mod" and rank == inj_rank:
                with torch.no_grad():
                    for n, p in base_named:
                        if n in tgt:
                            loc = p.to_local() if hasattr(p, "to_local") else p
                            g = torch.Generator(device=loc.device).manual_seed(4242)
                            loc.add_(torch.randn(loc.shape, generator=g, device=loc.device, dtype=loc.dtype) * 0.02)
            if kind == "zero_owner_lr" and rank == inj_rank:
                for g in zero.param_groups:
                    g["lr"] = LR * 10
            if kind == "user_hook_sum" and base == "fsdp2":
                for m in fsdp_units:
                    m.set_force_sum_reduction_for_comms(True)
                    m.set_gradient_divide_factor(1.0)
            if kind == "stale_gather" and base == "fsdp2":
                for t in tgt:
                    layer = model.decoder.layers[int(t)]
                    layer.set_reshard_after_forward(False)
                    layer.set_reshard_after_backward(False)
        div = 1 if (active and kind == "no_accum_divisor") else MICRO
        opt.zero_grad(set_to_none=True)
        losses = []
        if base == "pp2":
            nxt = parallel_state.get_pipeline_model_parallel_next_rank()
            prv = parallel_state.get_pipeline_model_parallel_prev_rank()
            for mi in range(MICRO):
                ids = batch(seed, step, dpc, mi, dev)
                pos = torch.arange(SEQ, device=dev).unsqueeze(0).expand(MB, SEQ)
                mask = attn_mask(dev)
                if first:
                    hidden = fwd(input_ids=ids, position_ids=pos, attention_mask=mask)
                    dist.send(hidden.detach(), dst=nxt)
                    grad = torch.empty_like(hidden)
                    dist.recv(grad, src=nxt)
                    hidden.backward(grad)
                else:
                    buf = torch.empty(SEQ, MB, HIDDEN, device=dev)
                    dist.recv(buf, src=prv)
                    inp = buf.detach().requires_grad_(True)
                    model.set_input_tensor(inp)
                    logits = fwd(input_ids=ids, position_ids=pos, attention_mask=mask)
                    loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), ids.reshape(-1))
                    (loss / div).backward()
                    dist.send(inp.grad, dst=prv)
                    losses.append(float(loss.item()))
            tied_grads()
            named = list(model.named_parameters())
            if collect:
                col.add(FH.HP_PRE, [(n, -1, p.grad) for n, p in named])
                col.add(FH.HP_POST, [(n, -1, p.grad) for n, p in named])
        else:
            for mi in range(MICRO):
                ids = batch(seed, step, dpc, mi, dev)
                pos = torch.arange(SEQ, device=dev).unsqueeze(0).expand(MB, SEQ)
                mask = attn_mask(dev)
                last_mb = mi == MICRO - 1
                if base == "fsdp2":
                    model.set_requires_gradient_sync(last_mb)
                ctx = ddp.no_sync() if (ddp is not None and not last_mb) else contextlib.nullcontext()
                if probes is not None:
                    probes.arm(last_mb)
                with ctx:
                    logits = fwd(input_ids=ids, position_ids=pos, attention_mask=mask)
                    loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), ids.reshape(-1))
                    (loss / div).backward()
                losses.append(float(loss.item()))
                if base == "fsdp2" and collect:
                    fstate["first"] = False
            if probes is not None:
                probes.arm(False)
                probes.record_post()
            if base == "fsdp2" and collect:
                col.add(FH.HP_POST, [(n, dpc, p.grad.to_local() if p.grad is not None else None)
                                     for n, p in base_named])
        if active and kind == "skip_param_group" and rank == inj_rank:
            for p in opt.param_groups[1]["params"]:
                p.grad = None
        if not (active and kind == "skip_opt_step" and rank == inj_rank):
            opt.step()
        if losses:
            col.scalar(loss=sum(losses) / len(losses))
            losses_all.append(sum(losses) / len(losses))
        if collect:
            if base == "fsdp2":
                items = [(n, dpc, p.to_local()) for n, p in base_named]
            else:
                items = [(n, -1, p) for n, p in base_named]
            col.end_step(items)
        times.append(time.perf_counter() - t_it)
    store.flush()
    store.close()
    with open(os.path.join(out, f"timing_r{rank}.json"), "w") as f:
        json.dump({"model": "megatron", "step_s": times, "collect_s": col.timing["collect_s"],
                   "losses": losses_all, "steps": STEPS,
                   "db_bytes": os.path.getsize(os.path.join(out, f"rank{rank}.duckdb"))}, f)
    dist.barrier()
    dist.destroy_process_group()


def _prepare(layout, inj_id, out):
    if inj_id != "clean" and layout not in INJECTIONS[inj_id]["targets"]:
        raise ValueError(f"{inj_id} not applicable to {layout}")
    os.makedirs(out, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True, choices=sorted(LAYOUTS))
    ap.add_argument("--injection", default="clean")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", default="cksum", choices=["cksum", "suml2", "off"])
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    home = prepare_import_path()
    L = LAYOUTS[a.layout]
    world = L["dp"] * L["tp"] * L["pp"]
    if "LOCAL_RANK" not in os.environ:
        _prepare(a.layout, a.injection, a.out)
        port = str(_free_port())
        cmd = ["torchrun", f"--nproc_per_node={world}", "--master_addr", "127.0.0.1",
               "--master_port", port, os.path.abspath(__file__), *sys.argv[1:]]
        print(f"megatron home={home} world={world} layout={a.layout} injection={a.injection} steps={STEPS}", flush=True)
        os.execvp(cmd[0], cmd)
    rank = int(os.environ["LOCAL_RANK"])
    worker(rank, world, a.layout, a.injection, a.seed, a.out, a.tier, a.every)


if __name__ == "__main__":
    main()
