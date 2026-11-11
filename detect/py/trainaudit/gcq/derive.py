"""Constraint generator (E12 / BENCH.md A12): derives the guarded constraint set of a layout from
its Meta, Topo and Obj v3 (core_trace/framework_hooks_v3.py), instead of authors choosing hooks per
layout. Output: GCQ specs (profile layout_v3; tuple / aggregate / completeness / aggregate_aligned
forms) and a derivation log that states, per rule template, the Meta facts that made it applicable
or not applicable.

Input
  topo  {rank: {dim: {"size", "group", "coord", "members"}}}
  objs  {rank: [Obj v3 object]}   (per object: kind, requires_grad, tied, owner, red{op,dim,from,to},
                                   rest{shard_dims, slice, hook}, phases{hook: {shard_dims, slices,
                                   pending, order, dtype}})

Templates (applied to every hook / object of the input; nothing is layout-specific):
  R  replica equality (G1 parameters, G2 gradients, G6 optimizer hyperparameters). For every phase
     hook h: the settled records at h (pending is None) are grouped into replica classes
     (rkey = tied group or object, slice, coordinates of the phase's shard dims); h is selected if
     at least one class has records from >= 2 ranks. Constraint: equal sha256 within a class.
  A  reduction consistency (G3). For every object with a reduction convention red = (op, dim,
     from, to) whose from- and to-phases both exist: post = red_scale * sum(pre) per (step, rkey,
     slice, reduction group), red_scale = 1/|dim| (mean) or 1 (sum). Tolerance by dtype of the
     compared phases: 1e-5 (fp32) or 1e-2 (any bf16 phase), relative to sqrt(numel) * red_scale *
     sum(l2_pre) (A12, fixed in advance).
  C  completeness of synchronized records (G4). For every hook that is the to-phase of a
     reduction: every (rank, obj, slice) declared at it with requires_grad has a row at every
     punctuated step.
  M  step-marker completeness (G5). Every rank of Topo has a marker at every punctuated step.
  S  gather consistency (G7, w=2). For every object sharded at rest (rest.shard_dims non-empty)
     with a settled phase h whose shard dims are a proper subset of the at-rest ones (a gathered
     copy): per (aligned step, rkey, persistent-shard key) the sum and the sum of squares of every
     gathered copy at h equal the sum / sum of squares over the at-rest shards (rest.hook), taken
     at the previous step if h comes before rest.hook in the step (order), else at the same step.
     Groups without all at-rest slices are skipped (completeness, not consistency).
  O  owner consistency (G8, w=2). For every object with a single owner and a phase h whose pending
     collective is a broadcast from the owner, followed in the step by the settled rest hook: on
     every non-owner rank, the record at h at step s equals the rest record at step s-1.

Modes (informative guard ablations, A12):
  guarded     as above
  dponly      replica classes from every dim except DP ("only DP replicates")
  rest        Obj phases rewritten to the at-rest sharding before deriving ("ignore phase-dependent
              sharding"); replica classes from the at-rest shard dims
  early       guarded set, then the phase guard moved one hook too early: G2 at the from-phase of
              the reduction (pre-sync), G1 at the before-broadcast phase where one exists (ZeRO-1);
              classes ignore the pending collective (the misspecified guard believes the earlier
              hook is settled)
"""
import copy
import json
from typing import Any, Dict, List, Tuple

PROFILE = "layout_v3"
TOL_FP32, TOL_BF16 = 1e-5, 1e-2
GRAD_HOOKS = {"fw.grad_pre_sync", "fw.grad_post_sync"}
HPARAM_HOOKS = {"fw.opt_hparam"}
MODES = ("guarded", "dponly", "rest", "early")


def A(x):
    return {"attr": x}


def op(o, *args):
    return {"op": o, "args": list(args)}


def fn(f, *args):
    return {"fn": f, "args": list(args)}


def S(x):
    return {"str": x}


def _kind_of_hook(h):
    if h in GRAD_HOOKS:
        return "G2"
    if h in HPARAM_HOOKS:
        return "G6"
    return "G1"


def _tol(dtypes):
    return TOL_BF16 if any(d == "bf16" for d in dtypes) else TOL_FP32


# ---- spec builders -------------------------------------------------------------------------------
def spec_replica(rid, hooks, grp, what, mode):
    return {"id": rid, "profile": PROFILE, "form": "tuple",
            "vars": [{"name": "t1", "rel": "State"}, {"name": "t2", "rel": "State"}],
            "phase": {"hooks": {"t1": hooks, "t2": hooks},
                      "steps": [{"var": "t2", "base": "t1", "offset": 0}],
                      "eq": [["t1.hook", "t2.hook"], ["t1.rkey", "t2.rkey"], ["t1.slice", "t2.slice"]]},
            "topo": {"dims": [grp], "atoms": [{"grp": grp, "vars": ["t1", "t2"], "ordered": True,
                                               "guard": "topo"}]},
            "schema": op("eq", A("t1.cksum"), A("t2.cksum")),
            "witness": ["t1.step", "t1.obj", "t1.hook", "t1.rank", "t2.rank"],
            "evidence": {"report_hook": hooks[0], "hook_in_witness": True, "mode": mode, "window": 1},
            "notes": f"{what} [{mode}]: records of the same (step, hook, rkey, slice) at "
                     f"{', '.join(hooks)} on two ranks of the same replica class ({grp}) have equal "
                     "sha256 checksums."}


def spec_reduction(rid, pre, post, tol, mode):
    pre_f = op("eq", A("t.hook"), S(pre))
    post_f = op("eq", A("t.hook"), S(post))
    aggs = {"pre_sum": {"fn": "sum", "arg": A("t.sum"), "filter": pre_f},
            "pre_l2": {"fn": "sum", "arg": A("t.l2"), "filter": pre_f},
            "pre_n": {"fn": "count", "arg": None, "filter": pre_f},
            "post_min": {"fn": "min", "arg": A("t.sum"), "filter": post_f},
            "post_max": {"fn": "max", "arg": A("t.sum"), "filter": post_f},
            "post_l2max": {"fn": "max", "arg": A("t.l2"), "filter": post_f},
            "numel": {"fn": "max", "arg": A("t.numel"), "filter": None},
            "scale": {"fn": "max", "arg": A("t.red_scale"), "filter": post_f}}
    scale = {"agg": "scale"}
    target = op("mul", scale, {"agg": "pre_sum"})
    bound = op("mul", tol, op("mul", fn("sqrt", {"agg": "numel"}), op("mul", scale, {"agg": "pre_l2"})))
    theta = op("and",
               op("le", fn("abs", op("sub", {"agg": "post_max"}, target)), bound),
               op("le", fn("abs", op("sub", {"agg": "post_min"}, target)), bound),
               op("le", {"agg": "post_l2max"}, op("mul", 1.0 + tol, op("mul", scale, {"agg": "pre_l2"}))))
    gb = ["t.step", "t.rkey", "t.slice", "t.red_grp"]
    return {"id": rid, "profile": PROFILE, "form": "aggregate", "vars": [{"name": "t", "rel": "State"}],
            "phase": {"hooks": {"t": [pre, post]}}, "group_by": gb, "aggregates": aggs, "theta": theta,
            "witness": gb, "extensions": ["E4-multi-aggregate"],
            "evidence": {"report_hook": post, "hook_in_witness": False, "tolerance": tol, "mode": mode,
                         "window": 1},
            "notes": f"Reduction consistency [{mode}]: per {gb}, every {post} sum equals red_scale x the "
                     f"sum of the {pre} sums within {tol} x sqrt(numel) x red_scale x sum(l2_pre), and "
                     f"l2_post <= (1+{tol}) x red_scale x sum(l2_pre). Groups without a {post} row or "
                     "without pre rows are not violations (completeness covers missing rows)."}


def spec_completeness(rid, hook, mode):
    return {"id": rid, "profile": PROFILE, "form": "completeness",
            "domain": {"vars": [{"name": "d", "rel": "Obj"}, {"name": "s", "rel": "Steps"}],
                       "where": op("and", op("eq", A("d.hook"), S(hook)), A("d.requires_grad"))},
            "target": {"var": "t", "rel": "State", "hooks": [hook]},
            "match": [["d.rank", "t.rank"], ["d.obj", "t.obj"], ["d.slice", "t.slice"], ["s.step", "t.step"]],
            "witness": ["s.step", "d.obj", "d.rank"],
            "evidence": {"report_hook": hook, "hook_in_witness": False, "mode": mode, "window": 1},
            "notes": f"Completeness of synchronized records [{mode}]: every (rank, obj, slice) declared "
                     f"at {hook} with requires_grad has a {hook} row at every punctuated step."}


def spec_markers(rid, mode):
    return {"id": rid, "profile": PROFILE, "form": "completeness",
            "domain": {"vars": [{"name": "k", "rel": "Ranks"}, {"name": "s", "rel": "Steps"}], "where": None},
            "target": {"var": "t", "rel": "Marker", "hooks": None},
            "match": [["k.rank", "t.rank"], ["s.step", "t.step"]],
            "witness": ["s.step", "k.rank"],
            "evidence": {"report_hook": "fw.marker", "hook_in_witness": False, "obj_in_witness": False,
                         "mode": mode, "window": 1},
            "notes": f"Step-marker completeness [{mode}]: every rank of Topo has a step marker at every "
                     "punctuated step."}


def spec_gather(rid, gh, rh, off, tol, mode):
    g_f = op("eq", A("t.hook"), S(gh))
    r_f = op("eq", A("t.hook"), S(rh))
    sq = op("mul", A("t.l2"), A("t.l2"))
    aggs = {"g_n": {"fn": "count", "arg": None, "filter": g_f},
            "g_min": {"fn": "min", "arg": A("t.sum"), "filter": g_f},
            "g_max": {"fn": "max", "arg": A("t.sum"), "filter": g_f},
            "g_sq_min": {"fn": "min", "arg": sq, "filter": g_f},
            "g_sq_max": {"fn": "max", "arg": sq, "filter": g_f},
            "numel": {"fn": "max", "arg": A("t.numel"), "filter": g_f},
            "s_n": {"fn": "count", "arg": None, "filter": r_f},
            "s_sum": {"fn": "sum", "arg": A("t.sum"), "filter": r_f},
            "s_l2": {"fn": "sum", "arg": A("t.l2"), "filter": r_f},
            "s_sq": {"fn": "sum", "arg": sq, "filter": r_f},
            "s_slices": {"fn": "count_distinct", "arg": A("t.slice"), "filter": r_f},
            "s_exp": {"fn": "max", "arg": A("t.rest_nslices"), "filter": r_f}}
    bound = op("mul", tol, op("mul", fn("sqrt", {"agg": "numel"}), {"agg": "s_l2"}))
    sqb = op("mul", tol, {"agg": "s_sq"})
    theta = op("or",
               op("eq", {"agg": "g_n"}, 0),
               op("lt", {"agg": "s_slices"}, {"agg": "s_exp"}),
               op("and",
                  op("le", fn("abs", op("sub", {"agg": "g_max"}, {"agg": "s_sum"})), bound),
                  op("le", fn("abs", op("sub", {"agg": "g_min"}, {"agg": "s_sum"})), bound),
                  op("le", fn("abs", op("sub", {"agg": "g_sq_max"}, {"agg": "s_sq"})), sqb),
                  op("le", fn("abs", op("sub", {"agg": "g_sq_min"}, {"agg": "s_sq"})), sqb)))
    gb = ["t.astep", "t.rkey", "t.gkey"]
    return {"id": rid, "profile": PROFILE, "form": "aggregate_aligned", "hooks": [gh, rh],
            "align": {rh: off} if off else {}, "group_by": gb, "aggregates": aggs, "theta": theta,
            "witness": gb, "extensions": ["E4-multi-aggregate", "E6-aligned-step"],
            "evidence": {"report_hook": gh, "hook_in_witness": False, "tolerance": tol, "mode": mode,
                         "window": 1 + off},
            "notes": f"Gather consistency [{mode}]: per (step s, rkey, persistent-shard key), the sum and "
                     f"the sum of squares of every gathered copy at {gh} (step s) equal the sum / sum of "
                     f"squares over the at-rest shards at {rh} (step s-{off}), within {tol} x sqrt(numel) "
                     f"x sum(l2_shards) and {tol} x sum(l2_shards^2). Groups with fewer at-rest slices "
                     "than Obj declares are skipped."}


def spec_owner(rid, h, rh, mode):
    return {"id": rid, "profile": PROFILE, "form": "tuple",
            "vars": [{"name": "t1", "rel": "State"}, {"name": "t2", "rel": "State"}],
            "phase": {"hooks": {"t1": [h], "t2": [rh]},
                      "steps": [{"var": "t2", "base": "t1", "offset": -1}],
                      "eq": [["t1.rank", "t2.rank"], ["t1.obj", "t2.obj"], ["t1.slice", "t2.slice"]]},
            "pre": op("and", op("is_not_null", A("t1.owner")), op("ne", A("t1.owner"), A("t1.rank"))),
            "schema": op("eq", A("t1.cksum"), A("t2.cksum")),
            "witness": ["t1.step", "t1.obj", "t1.hook", "t1.rank"],
            "evidence": {"report_hook": h, "hook_in_witness": True, "mode": mode, "window": 2},
            "notes": f"Owner consistency [{mode}]: on every rank that does not own the object, the "
                     f"record at {h} (before the owner's broadcast) at step s equals the record at {rh} "
                     "at step s-1 (only the owner may change it)."}


# ---- derivation ----------------------------------------------------------------------------------
def _coords(topo_r, dims):
    return tuple((d, topo_r[d]["coord"]) for d in sorted(dims) if d in topo_r)


def _rest_rewrite(objs):
    out = copy.deepcopy(objs)
    for r, lst in out.items():
        for o in lst:
            rd = list(o["rest"]["shard_dims"])
            for h, ph in o.get("phases", {}).items():
                ph["shard_dims"] = rd
    return out


def derive(topo: Dict[int, Dict[str, Any]], objs: Dict[int, List[Dict[str, Any]]],
           mode: str = "guarded") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if mode not in MODES:
        raise ValueError(mode)
    topo = {int(k): v for k, v in topo.items()}
    objs = {int(k): v for k, v in objs.items()}
    if mode == "rest":
        objs = _rest_rewrite(objs)
    log: List[Dict[str, Any]] = []
    specs: List[Dict[str, Any]] = []
    ranks = sorted(topo)
    hooks = sorted({h for r in objs for o in objs[r] for h in o.get("phases", {})})
    grp = {"guarded": "REPLICA", "dponly": "DPONLY", "rest": "REST", "early": "NOPEND"}[mode]

    def cls_dims(r, o, h):
        ph = o["phases"][h]
        if mode == "dponly":
            return [d for d in topo[r] if d != "DP"]
        if mode == "rest":
            return list(o["rest"]["shard_dims"])
        pend = ph.get("pending") or {}
        return list(ph["shard_dims"]) + list(pend.get("dims", []))

    # R: replica equality
    rep_hooks = {"G1": [], "G2": [], "G6": []}
    for h in hooks:
        classes, settled, pending = {}, 0, 0
        for r in ranks:
            for o in objs.get(r, []):
                ph = o.get("phases", {}).get(h)
                if ph is None:
                    continue
                if ph.get("pending"):
                    pending += 1
                    continue
                settled += 1
                rk = o.get("tied") or o["obj"]
                for sl in ph["slices"]:
                    classes.setdefault((rk, sl, _coords(topo[r], cls_dims(r, o, h))), set()).add(r)
        multi = sum(1 for v in classes.values() if len(v) >= 2)
        sel = multi > 0
        if sel:
            rep_hooks[_kind_of_hook(h)].append(h)
        log.append({"template": "R replica equality", "hook": h, "settled_records": settled,
                    "pending_records": pending, "classes": len(classes), "classes_with_2plus_ranks": multi,
                    "decision": f"emit ({_kind_of_hook(h)})" if sel else
                    ("not applicable: no settled record" if settled == 0 else
                     "not applicable: every replica class has one rank")})
    # A / C: reduction consistency and completeness of synchronized records
    pairs = {}
    for r in ranks:
        for o in objs.get(r, []):
            red = o.get("red")
            if not red or not red.get("from") or not red.get("to"):
                continue
            phs = o.get("phases", {})
            if red["from"] in phs and red["to"] in phs:
                e = pairs.setdefault((red["from"], red["to"]), {"objects": set(), "dtypes": set()})
                e["objects"].add(o["obj"])
                e["dtypes"] |= {phs[red["from"]].get("dtype"), phs[red["to"]].get("dtype")}
    red_specs, comp_specs = [], []
    for i, ((pre, post), e) in enumerate(sorted(pairs.items())):
        tol = _tol(e["dtypes"])
        sfx = "" if len(pairs) == 1 else f"_{i}"
        red_specs.append(spec_reduction(f"G3_reduction{sfx}", pre, post, tol, mode))
        comp_specs.append(spec_completeness(f"G4_completeness{sfx}", post, mode))
        log.append({"template": "A reduction consistency", "pair": [pre, post], "objects": len(e["objects"]),
                    "dtypes": sorted(x for x in e["dtypes"] if x), "tolerance": tol, "decision": "emit (G3)"})
        log.append({"template": "C completeness of synchronized records", "hook": post,
                    "decision": "emit (G4)"})
    if not pairs:
        log.append({"template": "A/C", "decision": "not applicable: no object declares a reduction "
                                                   "whose from- and to-phases are both recorded"})
    # early mode: move the phase guard one hook too early
    if mode == "early":
        moved = {}
        for (pre, post) in pairs:
            if post in rep_hooks["G2"]:
                moved[post] = pre
        rep_hooks["G2"] = sorted({moved.get(h, h) for h in rep_hooks["G2"]})
        g1 = []
        for h in rep_hooks["G1"]:
            earlier = sorted({hh for r in ranks for o in objs.get(r, [])
                              for hh, ph in o.get("phases", {}).items()
                              if (ph.get("pending") or {}).get("op") == "broadcast"
                              and o.get("phases", {}).get(h) is not None
                              and ph.get("order", 0) < o["phases"][h].get("order", 0)})
            g1 += earlier if earlier else [h]
            if earlier:
                moved[h] = earlier
        rep_hooks["G1"] = sorted(set(g1))
        log.append({"template": "early-phase ablation", "moved": moved})
    names = {"G1": ("G1_param_replica", "Parameter replica equality"),
             "G2": ("G2_grad_replica", "Synchronized-gradient replica equality"),
             "G6": ("G6_hparam_replica", "Optimizer-hyperparameter replica equality")}
    for k in ("G1", "G2", "G6"):
        if rep_hooks[k]:
            hs = sorted(rep_hooks[k], key=lambda h: h)
            specs.append(spec_replica(names[k][0], hs, grp, names[k][1], mode))
    specs += red_specs + comp_specs
    # M: markers
    if ranks:
        specs.append(spec_markers("G5_step_markers", mode))
        log.append({"template": "M step-marker completeness", "ranks": len(ranks), "decision": "emit (G5)"})
    # S: gather consistency
    gpairs = {}
    for r in ranks:
        for o in objs.get(r, []):
            rest = o.get("rest") or {}
            rdims = set(rest.get("shard_dims") or [])
            rh = rest.get("hook")
            if not rdims or not rh or rh not in o.get("phases", {}):
                continue
            ro = o["phases"][rh].get("order", 0)
            for h, ph in o.get("phases", {}).items():
                if h == rh or ph.get("pending") or o.get("kind", "param") != "param":
                    continue
                if set(ph["shard_dims"]) < rdims:
                    off = 1 if ph.get("order", 0) < ro else 0
                    e = gpairs.setdefault((h, rh, off), {"objects": set(), "dtypes": set()})
                    e["objects"].add(o["obj"])
                    e["dtypes"] |= {ph.get("dtype"), o["phases"][rh].get("dtype")}
    for i, ((h, rh, off), e) in enumerate(sorted(gpairs.items())):
        tol = _tol(e["dtypes"])
        sfx = "" if len(gpairs) == 1 else f"_{i}"
        specs.append(spec_gather(f"G7_gather_consistency{sfx}", h, rh, off, tol, mode))
        log.append({"template": "S gather consistency", "gathered_hook": h, "rest_hook": rh, "offset": off,
                    "objects": len(e["objects"]), "tolerance": tol, "decision": f"emit (G7, w={1 + off})"})
    if not gpairs:
        log.append({"template": "S gather consistency", "decision": "not applicable: no object is sharded "
                                                                  "at rest and recorded in a less-sharded settled phase"})
    # O: owner consistency
    opairs = {}
    for r in ranks:
        for o in objs.get(r, []):
            if o.get("owner") is None:
                continue
            rh = (o.get("rest") or {}).get("hook")
            phs = o.get("phases", {})
            if rh not in phs or phs[rh].get("pending"):
                continue
            for h, ph in phs.items():
                pend = ph.get("pending") or {}
                if pend.get("op") == "broadcast" and pend.get("from") == "owner" \
                        and ph.get("order", 0) < phs[rh].get("order", 0):
                    opairs.setdefault((h, rh), set()).add(o["obj"])
    for i, ((h, rh), objset) in enumerate(sorted(opairs.items())):
        sfx = "" if len(opairs) == 1 else f"_{i}"
        specs.append(spec_owner(f"G8_owner_consistency{sfx}", h, rh, mode))
        log.append({"template": "O owner consistency", "hook": h, "rest_hook": rh, "objects": len(objset),
                    "decision": "emit (G8, w=2)"})
    if not opairs:
        log.append({"template": "O owner consistency", "decision": "not applicable: no single-owner object "
                                                                 "with a pending broadcast phase"})
    return specs, log


def canonical(specs: List[Dict[str, Any]]) -> str:
    return json.dumps(sorted(specs, key=lambda s: s["id"]), sort_keys=True)
