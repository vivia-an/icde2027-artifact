"""C1 acceptance: violation context expander produces structured
DiagnosisReport that resolves each bug to its source location.

For every real-bug E2E surrogate, after rules fire, `trainaudit.diagnose()`
must return at least one DiagnosisReport whose:
  - rule_id matches the failing rule
  - suspect_module / suspect_module_class is set (or callsite for
    function-patching bugs like B11)
  - bug_specific carries the diagnostic numbers
  - hypothesis is non-empty
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

import trainaudit  # noqa: E402


def _diagnosis_for(rule_id):
    rs = trainaudit.run_rules()
    reports = trainaudit.diagnose(rs)
    return [r for r in reports if r.rule_id == rule_id]


# === B11: clip_grad bypass — diagnosis must produce callsite + bug_specific ==


def test_diagnose_clip_grad_violation_locates_callsite():
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
        with tempfile.TemporaryDirectory() as tmp:
            trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                              db_path=os.path.join(tmp, "trace.duckdb"))
            try:
                torch.manual_seed(0)
                model = nn.Sequential(nn.Linear(8, 16), nn.GELU(),
                                       nn.Linear(16, 4))
                opt = optim.AdamW(model.parameters(), lr=1e-3)
                trainaudit.snapshot_build(model, opt)
                x = torch.randn(2, 8)
                (model(x).pow(2).sum() * 1e6).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                reports = _diagnosis_for("T0-clip-grad-bounded")
                assert reports, "expected diagnosis for clip_grad bug"
                r = reports[0]
                assert r.hookpoint == "utils.clip_grad.post"
                assert r.callsite is not None and "file" in r.callsite
                assert r.bug_specific.get("post_norm") is not None
                assert r.bug_specific.get("max_norm") == 0.1
                assert r.bug_specific.get("post_norm") > r.bug_specific["max_norm"]
                assert "clip" in r.hypothesis.lower()
                print(f"[diag B11] suspect={r.suspect_module}/{r.suspect_module_class} "
                      f"callsite={r.callsite['file'].split('/')[-1]}:{r.callsite['line']} "
                      f"hypothesis={r.hypothesis}")
            finally:
                trainaudit.disable()
    finally:
        _nn_utils.clip_grad_norm_ = real_clip


# === O-NEW-1: norm output bad RMS — diagnosis must locate the named module ==


class _BrokenRMSNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x) * 0.33


def test_diagnose_norm_rms_locates_named_module():
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            class TinyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Linear(8, 16)
                    self.attn_norm = _BrokenRMSNorm(16)  # buggy normalizer
                    self.head = nn.Linear(16, 4)
                def forward(self, x):
                    return self.head(self.attn_norm(self.embed(x)))

            torch.manual_seed(0)
            model = TinyModel()
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)

            for _ in range(2):
                x = torch.randn(2, 8)
                model(x).pow(2).sum().backward()
                opt.step()

            reports = _diagnosis_for("T0-norm-output-unit-rms")
            assert reports, "expected diagnosis for RMSNorm bug"
            r = reports[0]
            # Should resolve to attn_norm by name
            assert r.suspect_module == "attn_norm", (
                f"expected suspect_module='attn_norm', got {r.suspect_module}")
            assert r.suspect_module_class in ("_BrokenRMSNorm",), (
                f"unexpected class {r.suspect_module_class}")
            assert r.bug_specific.get("is_normalizer") is True
            assert "normalizer" in r.hypothesis.lower() or "rms" in r.hypothesis.lower()
            print(f"[diag O-NEW-1] suspect={r.suspect_module} ({r.suspect_module_class}) "
                  f"l2={r.bug_specific.get('l2_norm')} hypothesis={r.hypothesis}")
        finally:
            trainaudit.disable()


# === OC-NEW-2: state['step'] frozen — diagnosis identifies optimizer ===========


class _FrozenStepAdamW(optim.AdamW):
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.tensor(0.0)
                p.data.add_(p.grad, alpha=-group["lr"])
        return None


def test_diagnose_optim_step_counter_locates_optimizer():
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
            opt = _FrozenStepAdamW(model.parameters(), lr=1e-3)
            for p in model.parameters():
                opt.state[p]["step"] = torch.tensor(0.0)
            trainaudit.snapshot_build(model, opt)
            for _ in range(3):
                x = torch.randn(2, 8)
                model(x).pow(2).sum().backward()
                opt.zero_grad()
                opt.step()

            reports = _diagnosis_for("T0-optim-step-counter-monotonic")
            assert reports, "expected diagnosis for OC-NEW-2"
            r = reports[0]
            assert r.bug_specific.get("optimizer_class") == "_FrozenStepAdamW"
            assert r.bug_specific.get("state_step_max") == 0.0
            assert "step" in r.hypothesis.lower()
            print(f"[diag OC-NEW-2] {r.bug_specific} hypothesis={r.hypothesis}")
        finally:
            trainaudit.disable()


# === Replica cksum (M-005 surrogate): outlier rank identification ============


def test_diagnose_replica_cksum_finds_outlier_rank():
    """Synthesize a build.snapshot event with a 4-rank replica group where
    rank 2 holds a different cksum. C1 should identify outlier_rank=2."""
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T1_FW_METADATA,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            store = trainaudit.get_store()
            store.emit("build.snapshot", {
                "model": {"n_parameters": 4, "n_modules": 4},
                "cross_rank_cksums": [{
                    "name": "router.weight",
                    "group_size": 4,
                    "local_cksum": 100,
                    "gathered_cksums": [100, 100, 999, 100],
                    "all_equal": False,
                }],
            })

            reports = _diagnosis_for("T1-replica-cksum-equal")
            assert reports, "expected replica diagnosis"
            r = reports[0]
            assert r.bug_specific.get("param_name") == "router.weight"
            assert r.bug_specific.get("outlier_rank") == 2, (
                f"expected outlier_rank=2, got {r.bug_specific.get('outlier_rank')}")
            assert "rng" in r.hypothesis.lower() or "diverged" in r.hypothesis.lower()
            print(f"[diag M-005] {r.bug_specific} hypothesis={r.hypothesis}")
        finally:
            trainaudit.disable()


# === Clean run: diagnose() returns [] without crashing ========================


def test_diagnose_clean_run_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16),
                                   nn.GELU(), nn.Linear(16, 4))
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)
            for _ in range(3):
                x = torch.randn(2, 8)
                model(x).pow(2).sum().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                opt.step()

            reports = trainaudit.diagnose()
            assert reports == [], f"clean run should produce empty diagnosis, got {reports}"
        finally:
            trainaudit.disable()
