"""Rule registration + result schema."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ..tiers import Tier


@dataclass
class RuleResult:
    rule_id: str
    violated: bool = False
    message: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    rule_id: str
    min_tier: Tier
    families: List[str]
    evidence_bugs: List[str]
    description: str
    check: Callable  # (conn) -> RuleResult or List[RuleResult]


_REGISTRY: List[Rule] = []


def rule(rule_id: str,
         min_tier: Tier = Tier.T0_PYTORCH,
         families: Optional[List[str]] = None,
         evidence_bugs: Optional[List[str]] = None,
         description: str = ""):
    """Decorator: register a check function as a Rule."""
    def deco(fn: Callable):
        _REGISTRY.append(Rule(
            rule_id=rule_id,
            min_tier=min_tier,
            families=list(families or []),
            evidence_bugs=list(evidence_bugs or []),
            description=description,
            check=fn,
        ))
        return fn
    return deco


def all_rules() -> List[Rule]:
    return list(_REGISTRY)


def _normalize(r: Union[RuleResult, List[RuleResult]]) -> List[RuleResult]:
    if isinstance(r, list):
        return r
    if isinstance(r, RuleResult):
        return [r]
    return []
