"""TrainAudit diagnosis (C1 + C2 of paper §3.4).

Given a RuleResult with violations, expand each violation event_id into
a structured DiagnosisReport:
  - which module/rank/step
  - what callsite (for function-patching events)
  - surrounding trace context (±N events on the same key)
  - cross-rank outlier identification (for replica-equality violations)
  - human-readable hypothesis sentence

C1 (this module) is deterministic — pulls from trace metadata only.
C2 (rca_agent.py, separate) wraps a DiagnosisReport with an LLM that
proposes the suspect code path and a fix.
"""
from .expander import expand_violation, expand_results
from .rca_agent import LLMClient, RCAResult, StubLLMClient, explain, explain_all
from .report import DiagnosisReport

__all__ = [
    "expand_violation", "expand_results", "DiagnosisReport",
    "LLMClient", "StubLLMClient", "RCAResult", "explain", "explain_all",
]
