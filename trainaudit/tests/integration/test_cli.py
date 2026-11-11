"""Smoke test the offline trainaudit CLI subcommands.

Generates a buggy trace via synthetic surrogate, then runs each CLI
subcommand against the saved .duckdb file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "trainaudit"))
import trainaudit  # noqa: E402


def _generate_buggy_trace(db_path: str) -> None:
    """Run a B11-style buggy clip_grad and persist the trace."""
    import torch.nn.utils as _nn_utils
    real_clip = _nn_utils.clip_grad_norm_

    def buggy_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        params = [p for p in parameters if p is not None and p.grad is not None]
        if not params:
            return torch.tensor(0.0)
        return torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad.detach().float(), norm_type)
             for p in params]), norm_type)

    _nn_utils.clip_grad_norm_ = buggy_clip
    try:
        store = trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                                   db_path=db_path)
        try:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)
            opt.zero_grad()
            x = torch.randn(2, 8)
            (model(x).pow(2).sum() * 1e6).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            store.flush()
        finally:
            trainaudit.disable()
    finally:
        _nn_utils.clip_grad_norm_ = real_clip


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "trainaudit"] + list(args),
        capture_output=True, text=True, cwd=str(REPO_ROOT / "trainaudit"),
        timeout=60)


@pytest.fixture
def buggy_trace():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "buggy.duckdb")
        _generate_buggy_trace(path)
        yield path


def test_cli_summary_lists_hookpoints(buggy_trace):
    proc = _run_cli("summary", buggy_trace)
    assert proc.returncode == 0
    assert "module.fwd.post" in proc.stdout
    assert "utils.clip_grad.post" in proc.stdout
    assert "build.snapshot" in proc.stdout
    assert "[summary] total events:" in proc.stdout


def test_cli_verify_returns_violation_exit_code(buggy_trace):
    proc = _run_cli("verify", buggy_trace, "--tier", "T0")
    # buggy trace → at least one rule fires → exit code 1 (violation)
    assert proc.returncode == 1, (
        f"expected exit 1 for buggy trace, got {proc.returncode}; "
        f"stdout={proc.stdout[-400:]}")
    assert "T0-clip-grad-bounded" in proc.stdout


def test_cli_verify_use_dsl_path(buggy_trace):
    proc = _run_cli("verify", buggy_trace, "--tier", "T0", "--use-dsl")
    assert proc.returncode == 1
    assert "T0-clip-grad-bounded" in proc.stdout, (
        f"DSL path should also detect; got {proc.stdout[-400:]}")


def test_cli_diagnose_emits_diagnosis_json(buggy_trace):
    proc = _run_cli("diagnose", buggy_trace, "--tier", "T0")
    assert proc.returncode == 1
    # JSON output should include suspect_module_class or callsite
    out = proc.stdout
    assert "T0-clip-grad-bounded" in out
    assert "callsite" in out or "hypothesis" in out


def test_cli_diagnose_rca_includes_llm_response(buggy_trace):
    proc = _run_cli("diagnose", buggy_trace, "--tier", "T0", "--rca")
    assert proc.returncode == 1
    assert "Suspect:" in proc.stdout or '"suspect"' in proc.stdout


def test_cli_replay_walks_forward(buggy_trace):
    proc = _run_cli("replay", buggy_trace, "--tier", "T0", "--every", "10")
    assert proc.returncode == 1
    assert "[replay]" in proc.stdout
    assert "tick" in proc.stdout
