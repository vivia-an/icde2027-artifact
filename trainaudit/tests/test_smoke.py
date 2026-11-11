"""Phase 1 smoke test.

Goal: verify that on a clean toy training:
  (a) trace events are captured into DuckDB
  (b) all 7 T0 rules run without raising
  (c) no rule fires a false-positive violation
"""
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim

# Allow running from tests/ directly without install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import trainaudit  # noqa: E402


def _build_toy():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.LayerNorm(16),
        nn.GELU(),
        nn.Linear(16, 4),
    )
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


def test_smoke_clean():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "trace.duckdb")
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path=db)
        try:
            model, opt = _build_toy()
            trainaudit.snapshot_build(model, opt)

            for step in range(3):
                trainaudit.set_step(step)
                x = torch.randn(2, 8)
                y = model(x)
                loss = y.pow(2).sum()
                opt.zero_grad()
                loss.backward()
                # exercise clip_grad path (B11's hookpoint)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                opt.step()

            store = trainaudit.get_store()
            n = store.num_events()
            assert n > 0, "no events captured"
            print(f"[smoke] events captured: {n}")

            # By hookpoint
            counts = store.query("""
                SELECT hookpoint, COUNT(*) FROM events GROUP BY hookpoint ORDER BY hookpoint
            """)
            print("[smoke] events per hookpoint:")
            for hp, c in counts:
                print(f"           {hp:30s}  {c}")

            # Run rules
            results = trainaudit.run_rules()
            print(trainaudit.summarize(results))

            violated = [r for r in results if r.violated]
            assert not violated, f"unexpected violations on clean run: {violated}"
            assert len(results) >= 7, f"expected >=7 T0 rules, got {len(results)}"
        finally:
            trainaudit.disable()


def test_smoke_clip_grad_violation():
    """Inject a buggy clip path to verify B11 rule fires."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "trace.duckdb")
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path=db)
        try:
            model, opt = _build_toy()
            trainaudit.snapshot_build(model, opt)

            x = torch.randn(2, 8)
            (model(x).pow(2).sum() * 1e6).backward()

            # Manually emit a "buggy" clip event: post_norm > max_norm (mimic B11)
            from trainaudit.schema import HP_CLIP_GRAD_POST
            store = trainaudit.get_store()
            store.emit(HP_CLIP_GRAD_POST, {
                "kind": "clip_grad",
                "max_norm": 0.1,
                "pre_norm": 1234.5,
                "post_norm": 1234.5,
                "ratio": 1.0,
            })

            results = trainaudit.run_rules()
            clip_rule = next(r for r in results if r.rule_id == "T0-clip-grad-bounded")
            assert clip_rule.violated, f"expected clip rule to fire, got {clip_rule}"
            print(f"[smoke] B11 rule fired as expected: {clip_rule.message}")
        finally:
            trainaudit.disable()


if __name__ == "__main__":
    test_smoke_clean()
    print("\n--- clean smoke passed ---\n")
    test_smoke_clip_grad_violation()
    print("\n--- B11 violation smoke passed ---\n")
    print("All Phase 1 smoke tests passed.")
