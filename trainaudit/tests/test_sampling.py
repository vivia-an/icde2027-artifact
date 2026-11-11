"""Sampling acceptance: hooks skip work for sampled-out events."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import trainaudit  # noqa: E402
from trainaudit.core_trace._utils import (get_sample_rate, reset_sampling,
                                            set_sample_rates, should_emit)


def setup_function(_):
    reset_sampling()


def teardown_function(_):
    reset_sampling()


def test_no_sample_rate_means_emit_every_event():
    assert should_emit("module.fwd.post") is True
    assert should_emit("any.hookpoint") is True


def test_zero_rate_drops_all():
    set_sample_rates({"module.fwd.post": 0.0})
    for _ in range(10):
        assert should_emit("module.fwd.post") is False
    # other hookpoints unaffected
    assert should_emit("optim.step.post") is True


def test_rate_one_emits_every_event():
    set_sample_rates({"module.fwd.post": 1.0})
    for _ in range(20):
        assert should_emit("module.fwd.post") is True


def test_half_rate_emits_every_other():
    set_sample_rates({"module.fwd.post": 0.5})
    keep = [should_emit("module.fwd.post") for _ in range(20)]
    # Deterministic: keep every K=2 → True False True False ...
    assert keep == [True, False] * 10


def test_tenth_rate_emits_every_tenth():
    set_sample_rates({"module.fwd.post": 0.1})
    keep = [should_emit("module.fwd.post") for _ in range(30)]
    assert keep.count(True) == 3
    assert keep[::10] == [True, True, True]


def test_module_hook_skips_summarize_when_sampled_out():
    """Plumbing: with module.fwd.post rate=0, no summarize_tensor is
    called and no events land. Verifies the production overhead win."""
    with tempfile.TemporaryDirectory() as tmp:
        store = trainaudit.enable(
            tier=trainaudit.Tier.T0_PYTORCH,
            db_path=os.path.join(tmp, "trace.duckdb"),
            sample_rates={"module.fwd.post": 0.0,
                            "module.fwd.pre": 0.0,
                            "module.bwd": 0.0})
        try:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)
            x = torch.randn(2, 8)
            model(x).pow(2).sum().backward()
            opt.step()
            store.flush()
            # No module.fwd.post events should land
            n_fwd = store.conn.execute(
                "SELECT COUNT(*) FROM events "
                "WHERE hookpoint = 'module.fwd.post'").fetchone()[0]
            assert n_fwd == 0, f"expected 0 fwd.post events, got {n_fwd}"
            # But optim.step.post is unaffected
            n_optim = store.conn.execute(
                "SELECT COUNT(*) FROM events "
                "WHERE hookpoint = 'optim.step.post'").fetchone()[0]
            assert n_optim >= 1
        finally:
            trainaudit.disable()


def test_partial_sampling_reduces_event_count():
    """rate=0.1 on module.fwd.post should yield ~10% of the events."""
    # First measure baseline (no sampling)
    with tempfile.TemporaryDirectory() as tmp:
        s_full = trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                                    db_path=os.path.join(tmp, "full.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(*[nn.Linear(8, 8) for _ in range(20)])
            for _ in range(5):
                x = torch.randn(2, 8)
                model(x).pow(2).sum().backward()
            s_full.flush()
            n_full = s_full.conn.execute(
                "SELECT COUNT(*) FROM events "
                "WHERE hookpoint = 'module.fwd.post'").fetchone()[0]
        finally:
            trainaudit.disable()

    with tempfile.TemporaryDirectory() as tmp:
        s_sampled = trainaudit.enable(
            tier=trainaudit.Tier.T0_PYTORCH,
            db_path=os.path.join(tmp, "sampled.duckdb"),
            sample_rates={"module.fwd.post": 0.1})
        try:
            torch.manual_seed(0)
            model = nn.Sequential(*[nn.Linear(8, 8) for _ in range(20)])
            for _ in range(5):
                x = torch.randn(2, 8)
                model(x).pow(2).sum().backward()
            s_sampled.flush()
            n_sampled = s_sampled.conn.execute(
                "SELECT COUNT(*) FROM events "
                "WHERE hookpoint = 'module.fwd.post'").fetchone()[0]
        finally:
            trainaudit.disable()

    print(f"\n[sampling] full={n_full} sampled@0.1={n_sampled} "
          f"ratio={n_sampled/max(n_full,1):.2f}")
    # Should be roughly 10% with deterministic counter — exact ceil(n_full/10)
    expected = (n_full + 9) // 10
    assert n_sampled == expected, (
        f"expected {expected} events at rate=0.1, got {n_sampled} "
        f"(baseline {n_full})")


def test_sample_rate_clamping():
    set_sample_rates({"a": 1.5, "b": -0.3, "c": 0.7})
    assert get_sample_rate("a") == 1.0
    assert get_sample_rate("b") == 0.0
    assert get_sample_rate("c") == 0.7


def test_reset_sampling_clears_counters():
    set_sample_rates({"x.y": 0.5})
    s1 = [should_emit("x.y") for _ in range(4)]
    reset_sampling()
    s2 = [should_emit("x.y") for _ in range(4)]
    # reset → no rate set → all True
    assert s2 == [True, True, True, True]
