"""C2: LLM root-cause-analysis agent.

Wraps a `DiagnosisReport` (deterministic, from C1) with an LLM-generated
hypothesis paragraph that names the likely suspect code path and a fix
hint. The LLM is pluggable via `LLMClient` so this module is testable
without network access — the default `StubLLMClient` produces a
deterministic templated response shaped exactly like the production
output, and tests can assert structure without flakiness.

Production wiring: pass a callable that hits `claude-proxy-v3.py` (or
the Anthropic SDK directly) as `llm_client`. See `prompts/rca_v1.txt`
for the prompt template.

Acceptance (paper §3.4): of 8 paper bugs, target 5/8 hit on real
root-cause text. Hit-rate measurement is per-bug human eval against
`config.json:root_cause` — the harness here records prompt + LLM
output verbatim into the DiagnosisReport so the eval can replay later.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .report import DiagnosisReport


# ---- LLM client interface -----------------------------------------------


class LLMClient(Protocol):
    """A function that takes (system_prompt, user_prompt) → text."""

    def __call__(self, system: str, user: str,
                 *, max_tokens: int = 1024) -> str: ...


@dataclass
class StubLLMClient:
    """Deterministic stub used in tests / when no API key is set.

    Produces a structured response that mirrors a real LLM's contract:
      Suspect: <module/function>
      Likely cause: <one paragraph using DiagnosisReport.bug_specific>
      Fix hint: <one sentence>

    The stub mines the DiagnosisReport's bug_specific dict for the
    diagnostic numbers, so the response is faithful to the trace even
    without a real model — sufficient for harness self-test."""

    name: str = "stub"

    def __call__(self, system: str, user: str,
                 *, max_tokens: int = 1024) -> str:
        # Parse the user prompt's appended JSON for the report (we serialise
        # it at the end of the prompt). Tolerant: if JSON missing, return
        # a generic answer.
        report_json = _extract_trailing_json(user)
        if report_json is None:
            return ("Suspect: <unknown>\nLikely cause: <stub: no report>\n"
                     "Fix hint: <stub: provide DiagnosisReport JSON>")
        rule = report_json.get("rule_id", "<rule>")
        suspect = (report_json.get("suspect_module")
                    or report_json.get("suspect_module_class") or "<module>")
        cs = report_json.get("callsite") or {}
        cs_str = (f"{os.path.basename(cs.get('file', ''))}:{cs.get('line')}"
                   if cs else "")
        bs = report_json.get("bug_specific") or {}
        cause = report_json.get("hypothesis", "")
        return (
            f"Suspect: {suspect}{(' (' + cs_str + ')') if cs_str else ''}\n"
            f"Likely cause: {cause or 'see bug_specific'}\n"
            f"Fix hint: review {suspect} against expected invariant for "
            f"{rule}; check {sorted(bs.keys())[:3]}.\n"
            f"Diagnostics: {json.dumps(bs)[:240]}"
        )


def _extract_trailing_json(text: str) -> Optional[Dict[str, Any]]:
    """Find the last ```json ... ``` or trailing { ... } block."""
    # Try fenced first
    fence = text.rfind("```json")
    if fence != -1:
        end = text.find("```", fence + 7)
        if end != -1:
            try:
                return json.loads(text[fence + 7:end].strip())
            except Exception:
                pass
    # Fallback: last { ... } block
    last_open = text.rfind("{")
    if last_open != -1:
        try:
            return json.loads(text[last_open:])
        except Exception:
            pass
    return None


# ---- prompt templates ---------------------------------------------------


_SYSTEM_PROMPT = (
    "You are a root-cause analyst for distributed LLM training silent "
    "errors. Given a structured DiagnosisReport from a TrainAudit "
    "violation, name the suspect code path, explain the most likely "
    "cause in one paragraph, and give one sentence of fix guidance. "
    "Be specific — quote exact field names and numeric values from the "
    "report. Do not speculate beyond what the report supports.\n\n"
    "Output format (3 lines):\n"
    "Suspect: <module/function name + file:line if available>\n"
    "Likely cause: <one paragraph>\n"
    "Fix hint: <one sentence>")


def _build_user_prompt(report: DiagnosisReport,
                        framework_hint: str = "") -> str:
    body = json.dumps(report.to_dict(), indent=2, default=str)
    fw = f"\n(Framework hint: {framework_hint})" if framework_hint else ""
    return (
        f"DiagnosisReport for a TrainAudit violation:{fw}\n\n"
        f"```json\n{body}\n```\n\n"
        f"Identify the suspect code path and explain the cause."
    )


# ---- public API ---------------------------------------------------------


@dataclass
class RCAResult:
    report: DiagnosisReport
    llm_response: str
    prompt_user: str
    prompt_system: str = _SYSTEM_PROMPT
    suspect: str = ""
    cause: str = ""
    fix_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.report.rule_id,
            "violation_event_id": self.report.violation_event_id,
            "suspect": self.suspect,
            "cause": self.cause,
            "fix_hint": self.fix_hint,
            "llm_response": self.llm_response,
            "report": self.report.to_dict(),
        }


def explain(report: DiagnosisReport, *,
             llm_client: Optional[LLMClient] = None,
             framework_hint: str = "") -> RCAResult:
    """Wrap a DiagnosisReport with an LLM-generated explanation."""
    if llm_client is None:
        llm_client = StubLLMClient()
    user_prompt = _build_user_prompt(report, framework_hint=framework_hint)
    response = llm_client(_SYSTEM_PROMPT, user_prompt)
    suspect, cause, fix_hint = _parse_response(response)
    return RCAResult(report=report, llm_response=response,
                      prompt_user=user_prompt,
                      suspect=suspect, cause=cause, fix_hint=fix_hint)


def _parse_response(text: str) -> tuple:
    """Pull (suspect, cause, fix_hint) from the structured LLM output.
    Tolerant of small format drift."""
    suspect = cause = fix_hint = ""
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("suspect:"):
            suspect = s[len("suspect:"):].strip()
        elif s.lower().startswith("likely cause:"):
            cause = s[len("likely cause:"):].strip()
        elif s.lower().startswith("fix hint:"):
            fix_hint = s[len("fix hint:"):].strip()
    return suspect, cause, fix_hint


def explain_all(reports: List[DiagnosisReport], *,
                 llm_client: Optional[LLMClient] = None,
                 framework_hint: str = "") -> List[RCAResult]:
    return [explain(r, llm_client=llm_client, framework_hint=framework_hint)
            for r in reports]
