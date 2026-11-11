"""C0 acceptance: trace context metadata.

Goals:
  - >=90% module.fwd.post events carry module_class + module_id
  - Named modules (registered by snapshot_build) carry module_name
  - clip_grad_norm_ / F.softmax / checkpoint events carry callsite
    (file, line, function) of the user-code caller (best-effort)

These are prerequisites for C1 (diagnosis expander): without these, a
violation event_id can only be resolved to a module *class* (e.g.
RMSLayerNorm), not the specific named submodule (e.g. blocks.3.attn_norm).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

import trainaudit  # noqa: E402


def _events(store, hookpoint: str):
    store.flush()
    rows = store.conn.execute(
        "SELECT event_id, payload FROM events WHERE hookpoint = ? "
        "ORDER BY event_id", [hookpoint]).fetchall()
    return [(eid, json.loads(p)) for eid, p in rows]


def test_module_events_carry_module_id_and_class():
    """At least 90% of module.fwd.post events have module_class+module_id;
    named modules carry module_name."""
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(
                nn.Linear(8, 16),
                nn.LayerNorm(16),
                nn.GELU(),
                nn.Linear(16, 4),
            )
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)

            x = torch.randn(2, 8)
            model(x).pow(2).sum().backward()
            opt.step()

            store = trainaudit.get_store()
            posts = _events(store, "module.fwd.post")
            assert len(posts) > 0

            n_with_id = sum(1 for _, p in posts
                             if p.get("module_class") and p.get("module_id"))
            ratio = n_with_id / len(posts)
            assert ratio >= 0.9, (
                f"only {ratio:.1%} of module.fwd.post events have "
                f"module_class+module_id ({n_with_id}/{len(posts)})")

            # Named modules should resolve to their dotted names; nn.Sequential
            # gives '0', '1', '2', '3'.
            named = [p for _, p in posts if p.get("module_name")]
            assert named, "expected at least one event with module_name"
            assert any(p["module_name"] == "1" for p in named), (
                "expected the LayerNorm at index 1 to resolve to "
                f"module_name='1', got names {[p['module_name'] for p in named]}")
        finally:
            trainaudit.disable()


def test_clip_grad_event_carries_callsite():
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)
            x = torch.randn(2, 8)
            model(x).pow(2).sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            store = trainaudit.get_store()
            posts = _events(store, "utils.clip_grad.post")
            assert posts, "expected at least one clip_grad.post event"
            cs = posts[-1][1].get("callsite")
            assert cs is not None, f"clip_grad.post has no callsite: {posts[-1][1]}"
            assert "file" in cs and "line" in cs
            # callsite should point to *this* test file, not into trainaudit
            assert "trainaudit/" not in cs["file"] or "tests/" in cs["file"], (
                f"callsite should resolve past trainaudit wrapper, got {cs}")
        finally:
            trainaudit.disable()


def test_softmax_event_carries_callsite():
    import torch.nn.functional as F
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            x = torch.randn(2, 4)
            _ = F.softmax(x, dim=-1)
            store = trainaudit.get_store()
            posts = _events(store, "functional.softmax")
            assert posts, "expected functional.softmax event"
            cs = posts[-1][1].get("callsite")
            assert cs is not None, f"softmax event has no callsite: {posts[-1][1]}"
            assert "file" in cs and "line" in cs
        finally:
            trainaudit.disable()


def test_named_module_resolution_in_realistic_model():
    """Named submodules in a transformer-like model resolve to their
    dotted paths (blocks.0.attn_norm style)."""
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_norm = nn.LayerNorm(8)
            self.mlp = nn.Linear(8, 8)
        def forward(self, x):
            return self.mlp(self.attn_norm(x))

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(8, 8)
            self.blocks = nn.ModuleList([Block() for _ in range(3)])
            self.head = nn.Linear(8, 4)
        def forward(self, x):
            x = self.embed(x)
            for b in self.blocks:
                x = b(x)
            return self.head(x)

    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = TinyTransformer()
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)

            x = torch.randn(2, 8)
            model(x).pow(2).sum().backward()
            opt.step()

            store = trainaudit.get_store()
            posts = _events(store, "module.fwd.post")
            names = {p.get("module_name") for _, p in posts if p.get("module_name")}
            # We expect at least some of these to surface
            expected_some_of = {
                "embed", "head",
                "blocks.0", "blocks.0.attn_norm", "blocks.0.mlp",
                "blocks.1.attn_norm", "blocks.2.mlp",
            }
            overlap = names & expected_some_of
            assert overlap, (
                f"expected some named modules (e.g. blocks.0.attn_norm) in "
                f"events, got names: {sorted(names)}")
            # At least 5 distinct named modules
            assert len(names) >= 5, f"only {len(names)} distinct module_names"
        finally:
            trainaudit.disable()
