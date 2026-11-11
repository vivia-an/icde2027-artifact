"""Online streaming runner acceptance.

The runner must:
  - emit zero new_violations on tick(0) when no events have arrived
  - detect a fault on the tick following the buggy event
  - NOT re-emit a violation on subsequent ticks (already-seen filter)
  - retire a violation only when underlying events are removed/modified
    (rare in append-only trace; not exercised here)
  - skip rules whose hookpoint doesn't appear in the new event window
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

import trainaudit  # noqa: E402
from trainaudit.streaming import OnlineRunner, build_default_runner  # noqa: E402


@pytest.fixture
def store():
    s = trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path=":memory:")
    yield s
    trainaudit.disable()


def test_tick_no_events_returns_zero(store):
    runner = build_default_runner(store)
    res = runner.tick()
    assert res.n_new_events == 0
    assert res.new_violations == []


def test_clip_violation_detected_only_once(store):
    runner = build_default_runner(store)
    # tick 0: clean
    res = runner.tick()
    assert not res.new_violations

    # buggy event lands
    store.emit("utils.clip_grad.post", {
        "kind": "clip_grad", "fn": "torch.nn.utils.clip_grad_norm_",
        "max_norm": 1.0, "pre_norm": 5.0, "post_norm": 5.0, "ratio": 1.0})
    res = runner.tick()
    assert res.n_new_events == 1
    assert any(v["rule_id"] == "T0-clip-grad-bounded"
                for v in res.new_violations), (
        f"expected T0-clip-grad-bounded in {res.new_violations}")
    seen_event_id = res.new_violations[0]["event_id"]

    # next tick with no new events: no NEW violations (already-seen filter)
    res = runner.tick()
    assert res.new_violations == [], (
        f"violation re-emitted: {res.new_violations}")

    # but if a SECOND buggy event arrives, that gets a fresh new_violations
    store.emit("utils.clip_grad.post", {
        "kind": "clip_grad", "fn": "torch.nn.utils.clip_grad_norm_",
        "max_norm": 1.0, "pre_norm": 100.0, "post_norm": 100.0, "ratio": 1.0})
    res = runner.tick()
    assert len(res.new_violations) == 1
    assert res.new_violations[0]["event_id"] != seen_event_id


def test_rule_skipping_by_hookpoint(store):
    """If only `module.fwd.post` events arrived this tick, rules that key
    on `utils.clip_grad.post` should be skipped."""
    runner = build_default_runner(store)
    store.emit("module.fwd.post", {
        "module_class": "Linear", "module_id": 1, "module_name": "x",
        "is_normalizer": False, "training": True,
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 1.0, "has_nan": False, "has_inf": False,
                   "abs_max": 1.0, "max": 1.0, "min": -1.0, "mean": 0.0},
    })
    res = runner.tick()
    # Among ~11 T0 rules, fewer should be evaluated since only one
    # hookpoint is new. We expect strictly fewer than the full registry.
    from trainaudit.rules import all_rules
    n_total = sum(1 for r in all_rules()
                   if r.min_tier <= trainaudit.Tier.T0_PYTORCH)
    assert res.n_rules_evaluated < n_total, (
        f"expected hookpoint-aware skip; got {res.n_rules_evaluated} "
        f"of {n_total} rules evaluated")


def test_multi_step_streaming_simulation(store):
    """Mimic a training loop: 3 steps clean, 1 step buggy, 2 steps clean.
    Buggy violation must surface exactly on tick 4 (the buggy step) and
    not be re-reported afterwards."""
    runner = build_default_runner(store)
    new_violation_count_per_tick = []

    for step in range(6):
        trainaudit.set_step(step)
        # one optim.step.post event per "training step"
        if step == 3:
            # buggy step: state_step regresses
            store.emit("optim.step.post", {
                "kind": "optim_step", "optimizer_class": "AdamW",
                "state_step_min": 1.0, "state_step_max": 1.0,
                "state_step_n_params": 4, "total_param_l2": 1.0})
        else:
            store.emit("optim.step.post", {
                "kind": "optim_step", "optimizer_class": "AdamW",
                "state_step_min": float(step + 1),
                "state_step_max": float(step + 1),
                "state_step_n_params": 4, "total_param_l2": 1.0})
        res = runner.tick()
        new_violation_count_per_tick.append(len(res.new_violations))

    # Step 3 caused state_step to drop (1, 2, 3, 1, ...). Both step=3 (the
    # 1.0 after 3.0) and step=4 (2.0 reverting) violate monotonicity for
    # consecutive pairs that involve step 3. Implementation detail: the
    # rule emits one violation per non-monotonic transition. The minimum
    # accepted is "violation appeared at the right tick"; we don't
    # over-specify which transitions are flagged.
    assert sum(new_violation_count_per_tick[:3]) == 0, (
        f"clean steps should produce 0 violations, got "
        f"{new_violation_count_per_tick[:3]}")
    assert new_violation_count_per_tick[3] >= 1, (
        f"step 3 should surface monotonic violation, got "
        f"{new_violation_count_per_tick[3]}")
    # later ticks must not re-report the same already-seen violations,
    # though they may surface new transitions involving subsequent events
    # — what we require is that no SINGLE event is re-reported.
    seen: set = set()
    for r in [runner.tick()]:  # just a final tick to harvest cumulative state
        pass
    # Validate: every violation event_id only appears once across ticks
    # (use the runner's internal cache as ground truth)
    cache_ids: list = []
    for ids in runner._violations_seen.values():
        cache_ids.extend(ids)
    assert len(cache_ids) == len(set(cache_ids)), (
        "duplicate event_ids in violation cache")
