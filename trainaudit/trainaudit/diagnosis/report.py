"""DiagnosisReport — structured output of C1 expander."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiagnosisReport:
    """One violation, located.

    All optional fields can be None if the trace event lacks the
    corresponding metadata (e.g. callsite for module events, or
    suspect_module for non-module events).
    """

    rule_id: str
    violation_event_id: int
    hookpoint: str

    # Source location
    suspect_module: Optional[str] = None       # e.g. 'blocks.3.attn_norm'
    suspect_module_class: Optional[str] = None  # e.g. 'RMSLayerNorm'
    suspect_module_id: Optional[int] = None
    suspect_rank: Optional[int] = None
    suspect_step: Optional[int] = None
    callsite: Optional[Dict[str, Any]] = None  # {file, line, function}

    # Bug-specific evidence
    bug_specific: Dict[str, Any] = field(default_factory=dict)
    """e.g. for B11: {pre_norm, post_norm, max_norm}.
    For M-005: {gathered_cksums, outlier_rank, param_name}."""

    # Surrounding trace context (±N events on the same suspect_module / hookpoint)
    context_events: List[Dict[str, Any]] = field(default_factory=list)

    # Free-form hypothesis (filled by C2 LLM agent if used; otherwise a
    # short deterministic summary)
    hypothesis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "rule_id": self.rule_id,
            "violation_event_id": self.violation_event_id,
            "hookpoint": self.hookpoint,
        }
        if self.suspect_module is not None:
            out["suspect_module"] = self.suspect_module
        if self.suspect_module_class is not None:
            out["suspect_module_class"] = self.suspect_module_class
        if self.suspect_module_id is not None:
            out["suspect_module_id"] = self.suspect_module_id
        if self.suspect_rank is not None:
            out["suspect_rank"] = self.suspect_rank
        if self.suspect_step is not None:
            out["suspect_step"] = self.suspect_step
        if self.callsite:
            out["callsite"] = self.callsite
        if self.bug_specific:
            out["bug_specific"] = self.bug_specific
        if self.context_events:
            out["context_events"] = self.context_events
        if self.hypothesis:
            out["hypothesis"] = self.hypothesis
        return out
