"""C2 acceptance: RCA agent wraps DiagnosisReport with LLM explanation.

Tests use the StubLLMClient so they're hermetic. A real LLM client (e.g.
Anthropic SDK pointing at claude-opus-4) would slot in via the
`llm_client=` kwarg. The structural contract this module enforces is:
  - prompt is well-formed JSON that round-trips
  - response parses to (suspect, cause, fix_hint)
  - RCAResult preserves the original DiagnosisReport for replay
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
from trainaudit.diagnosis import (DiagnosisReport, RCAResult, StubLLMClient,
                                    explain, explain_all)


def test_explain_with_stub_returns_structured_response():
    rep = DiagnosisReport(
        rule_id="T0-clip-grad-bounded",
        violation_event_id=42,
        hookpoint="utils.clip_grad.post",
        suspect_module=None,
        suspect_module_class=None,
        callsite={"file": "/u/proj/train.py", "line": 88, "function": "step"},
        bug_specific={"max_norm": 0.1, "post_norm": 1234.5,
                       "ratio": 12345.0,
                       "fn": "torch.nn.utils.clip_grad_norm_"},
        hypothesis="clip_grad_norm_ left grad norm at 1234.5 with max_norm=0.1",
    )
    res = explain(rep)
    assert isinstance(res, RCAResult)
    assert res.suspect, f"empty suspect in stub response: {res.llm_response!r}"
    assert res.cause, f"empty cause: {res.llm_response!r}"
    assert res.fix_hint, f"empty fix hint: {res.llm_response!r}"
    assert "T0-clip-grad-bounded" in res.fix_hint
    assert "train.py:88" in res.suspect, f"callsite missing in suspect: {res.suspect}"


def test_explain_carries_report_through_for_replay():
    rep = DiagnosisReport(rule_id="T0-no-nan-inf", violation_event_id=7,
                           hookpoint="module.fwd.post",
                           suspect_module="blocks.0.mlp",
                           suspect_module_class="Linear")
    res = explain(rep)
    assert res.report is rep
    d = res.to_dict()
    assert d["rule_id"] == "T0-no-nan-inf"
    assert d["report"]["suspect_module"] == "blocks.0.mlp"


def test_explain_e2e_through_real_diagnose():
    """End-to-end: real model + injected bug → trainaudit.diagnose() →
    RCA agent. Validates prompts contain real numbers from the trace."""
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
                opt.zero_grad()
                x = torch.randn(2, 8)
                (model(x).pow(2).sum() * 1e6).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                reports = trainaudit.diagnose()
                assert reports
                rcas = explain_all(reports)
                clip_rca = next(r for r in rcas
                                  if r.report.rule_id == "T0-clip-grad-bounded")
                # User prompt must contain the actual diagnostic numbers
                assert "0.1" in clip_rca.prompt_user, (
                    "max_norm=0.1 should appear in user prompt")
                assert clip_rca.suspect, "suspect should be filled"
                assert clip_rca.fix_hint, "fix_hint should be filled"
                print(f"[C2 E2E] suspect={clip_rca.suspect}")
                print(f"[C2 E2E] fix_hint={clip_rca.fix_hint}")
            finally:
                trainaudit.disable()
    finally:
        _nn_utils.clip_grad_norm_ = real_clip


def test_explain_handles_pluggable_llm_client():
    """Verify a custom LLM client is plumbed through unchanged — proves
    the production path of pointing at claude-proxy-v3 / Anthropic SDK
    is just a constructor swap."""
    captured = {}

    def my_client(system: str, user: str, *, max_tokens: int = 1024) -> str:
        captured["system"] = system
        captured["user"] = user
        captured["max_tokens"] = max_tokens
        return ("Suspect: blocks.5.attn_norm (myfile.py:42)\n"
                "Likely cause: bf16 RMSNorm misnamed eps placement\n"
                "Fix hint: cast eps to fp32 before division")

    rep = DiagnosisReport(rule_id="T0-norm-output-unit-rms",
                           violation_event_id=1,
                           hookpoint="module.fwd.post",
                           suspect_module="blocks.5.attn_norm")
    res = explain(rep, llm_client=my_client, framework_hint="OLMo bf16 path")
    assert captured.get("system") and captured.get("user")
    assert "OLMo bf16 path" in captured["user"]
    assert res.suspect == "blocks.5.attn_norm (myfile.py:42)"
    assert res.cause == "bf16 RMSNorm misnamed eps placement"
    assert res.fix_hint == "cast eps to fp32 before division"
