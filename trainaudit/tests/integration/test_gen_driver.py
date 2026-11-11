"""Smoke test for benchmark/eval/gen_driver.py.

Verifies that on a real bug config (B11), the generator emits a
syntactically valid trainaudit_run.sh + trainaudit_driver.py with the
right framework-specific bits (DS_DIR env var, _t_api compat shim, etc.).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN_DRIVER = REPO_ROOT / "benchmark" / "eval" / "gen_driver.py"


def _run_gen_driver(args):
    return subprocess.run(
        [sys.executable, str(GEN_DRIVER)] + args,
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60)


def test_gen_driver_b11_smoke_ok():
    """Running gen_driver on B11's config produces an OK smoke status."""
    sys.path.insert(0, str(REPO_ROOT / "benchmark" / "eval"))
    import gen_driver  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory() as tmp:
        bug_dir = REPO_ROOT / "benchmark" / "bugs" / "B11"
        result = gen_driver.generate_for_bug(bug_dir)
        assert result.smoke_status == "PENDING", (
            f"unexpected pre-validation status: {result.smoke_status}")
        validated = gen_driver.smoke_validate(result, Path(tmp))
        assert validated.smoke_status == "OK", (
            f"smoke failed: {validated.smoke_status} {validated.smoke_detail}")


def test_gen_driver_includes_framework_specifics():
    sys.path.insert(0, str(REPO_ROOT / "benchmark" / "eval"))
    import gen_driver  # type: ignore[import-not-found]
    bug_dir = REPO_ROOT / "benchmark" / "bugs" / "B11"
    result = gen_driver.generate_for_bug(bug_dir)
    assert "DS_DIR" in result.run_sh, "DeepSpeed driver should reference DS_DIR"
    assert "torchrun" in result.run_sh
    assert "_t_api" in result.driver_py, (
        "DeepSpeed driver should include _t_api elastic-agent compat shim")
    assert "trainaudit.enable" in result.driver_py
    assert "[B11]" in result.driver_py, "contract output line must reference bug_id"


def test_gen_driver_megatron_includes_pythonpath():
    sys.path.insert(0, str(REPO_ROOT / "benchmark" / "eval"))
    import gen_driver  # type: ignore[import-not-found]
    bug_dir = REPO_ROOT / "benchmark" / "bugs" / "M-014"
    result = gen_driver.generate_for_bug(bug_dir)
    assert "MEGATRON_DIR" in result.run_sh
    assert "PYTHONPATH" in result.run_sh
    assert "CUDA_DEVICE_MAX_CONNECTIONS" in result.run_sh
    assert "--nproc_per_node=2" in result.run_sh, (
        "Megatron template should default to 2-rank")


def test_gen_driver_skips_when_no_fixed_commit():
    """Bugs whose config has no fixed_commit are SKIP, not failure."""
    sys.path.insert(0, str(REPO_ROOT / "benchmark" / "eval"))
    import gen_driver  # type: ignore[import-not-found]
    # O-006 has reproduction_status=reproduced but no fixed_commit
    bug_dir = REPO_ROOT / "benchmark" / "bugs" / "O-006"
    if not bug_dir.exists():
        pytest.skip("O-006 not present")
    result = gen_driver.generate_for_bug(bug_dir)
    assert result.smoke_status == "SKIP"
    assert "fixed_commit" in result.skipped_reason
