"""TrainAudit Accept gate (Eq. accept).

Disabled unless SDC_PAPER_ALIGN=1. When enabled:
  Accept(c) <=> (forall ce: Verify(ce) != HOLDS) AND Conf(c) >= theta
  plus three-predicate checks, template_id, and healthy-run when SQL+DB exist.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOPO_KEYS = {
    "dp", "tp", "pp", "ep", "cp", "tpl", "topology", "parallel",
    "tensor_parallel", "pipeline_parallel", "data_parallel", "expert_parallel",
    "world_size", "zero_stage",
}
_PRECOND_KEYS = {
    "stage", "dtype", "requires_grad", "phase", "hook", "hookpoint",
    "precision", "bf16", "fp16", "micro_batch", "accumulation",
}


def paper_align_enabled() -> bool:
    return os.environ.get("SDC_PAPER_ALIGN", "").lower() in ("1", "true", "yes")


def theta_conf(no_adversarial: bool = False) -> float:
    if no_adversarial:
        raw = os.environ.get("SDC_MIN_CONF", "0")
    else:
        raw = os.environ.get("SDC_THETA_CONF", "0.8")
    try:
        return float(raw)
    except ValueError:
        return 0.0 if no_adversarial else 0.8


def extract_confidence(constraint: Dict[str, Any]) -> Optional[float]:
    for key in ("confidence", "Conf", "conf", "置信度", "conf_score"):
        if key in constraint and constraint[key] is not None:
            try:
                return float(constraint[key])
            except (TypeError, ValueError):
                pass
    meta = constraint.get("metadata") or constraint.get("mining_record") or {}
    if isinstance(meta, dict):
        for key in ("confidence", "Conf", "conf", "置信度"):
            if key in meta and meta[key] is not None:
                try:
                    return float(meta[key])
                except (TypeError, ValueError):
                    pass
    blob = " ".join(
        str(constraint.get(k, ""))
        for k in ("description", "logic", "name", "notes")
    )
    m = re.search(r"(?:confidence|Conf|置信度)\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)", blob, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def unwrap_constraint_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        return {}
    if "name" in raw or "logic" in raw or "pi_schema" in raw:
        return raw
    if len(raw) >= 1:
        inner = next(iter(raw.values()))
        if isinstance(inner, dict):
            return inner
    return raw


def check_three_predicates(constraint: Dict[str, Any]) -> Optional[str]:
    """Return None if ok, else reason. Maps applicable_conditions → π_topo/π_precond."""
    ac = constraint.get("applicable_conditions") or {}
    if not isinstance(ac, dict):
        ac = {}

    pi_schema = (
        constraint.get("pi_schema")
        or constraint.get("logic")
        or constraint.get("sql")
        or constraint.get("description")
    )
    if isinstance(pi_schema, str):
        pi_schema = pi_schema.strip()
    if not pi_schema:
        return "PI_SCHEMA_MISSING (need pi_schema|logic|sql|description)"

    pi_topo = constraint.get("pi_topo")
    if pi_topo is None:
        pi_topo = {k: v for k, v in ac.items() if str(k).lower() in _TOPO_KEYS}
    if isinstance(pi_topo, dict) and not pi_topo:
        pi_topo = None
    if not pi_topo:
        return "PI_TOPO_MISSING (need pi_topo or applicable_conditions dp/tp/pp/…)"

    pi_pre = constraint.get("pi_precond")
    if pi_pre is None:
        pi_pre = {k: v for k, v in ac.items() if str(k).lower() in _PRECOND_KEYS}
    if isinstance(pi_pre, dict) and not pi_pre:
        pi_pre = None
    if not pi_pre:
        return "PI_PRECOND_MISSING (need pi_precond or applicable_conditions stage/dtype/…)"

    return None


def discover_healthy_dbs() -> List[str]:
    env = os.environ.get("SDC_HEALTHY_DBS", "").strip()
    if env:
        parts = re.split(r"[:;,]", env)
        return [p.strip() for p in parts if p.strip() and os.path.exists(p.strip())]

    g = os.environ.get("SDC_HEALTHY_DB_GLOB", "").strip()
    if g:
        return sorted(p for p in glob(g) if os.path.isfile(p))[:8]

    # Prefer merged multi-rank clean DBs (healthy-run needs cross-rank columns).
    repo = Path(__file__).resolve().parents[2]
    preferred = [
        repo / "Megatron-LM/normal_db/tp_normal/Collector/merged_coredump.db",
        repo / "Megatron-LM/normal_db/dp_normal/Collector/merged_coredump.db",
        repo / "Megatron-LM/dp_normal_db/Collector/merged_coredump.db",
    ]
    found: List[str] = [str(p) for p in preferred if p.is_file()]
    if found:
        return found[:4]

    patterns = [
        str(repo / "Megatron-LM" / "normal_db" / "*" / "Collector" / "coredump_*.db"),
        str(repo / "Megatron-LM" / "dtype_test_db" / "Collector" / "coredump_*.db"),
        str(repo / "Megatron-LM" / "tp_router_test_db" / "Collector" / "coredump_*.db"),
    ]
    for pat in patterns:
        found.extend(sorted(glob(pat))[:2])
    return found[:6]


def allow_healthy_defer() -> bool:
    """Paper-align default: no defer (must execute healthy SQL). Legacy opt-in via env=1."""
    raw = os.environ.get("SDC_ALLOW_HEALTHY_DEFER")
    if raw is None:
        return False
    return raw.lower() in ("1", "true", "yes")


@dataclass
class AcceptContext:
    ce_verify_count: int = 0
    ce_holds_count: int = 0
    ce_briefs: List[str] = field(default_factory=list)
    fsm_stage: str = "S1"
    template_id: str = ""
    no_adversarial: bool = False
    evidence_hits: int = 0
    fsm: Any = None  # optional MiningFSM


@dataclass
class AcceptResult:
    ok: bool
    reason: str
    conf: Optional[float] = None
    theta: float = 0.8
    healthy: str = "skipped"


def check_accept(
    constraint: Dict[str, Any],
    ctx: AcceptContext,
    *,
    force: Optional[bool] = None,
) -> AcceptResult:
    enabled = paper_align_enabled() if force is None else force
    if not enabled:
        return AcceptResult(ok=True, reason="paper_align_off", healthy="n/a")

    theta = theta_conf(no_adversarial=ctx.no_adversarial)
    conf = extract_confidence(constraint)

    pred_err = check_three_predicates(constraint)
    if pred_err:
        return AcceptResult(ok=False, reason=pred_err, conf=conf, theta=theta)

    tid = constraint.get("template_id") or constraint.get("catalog_id") or ctx.template_id
    require_tid = os.environ.get("SDC_REQUIRE_TEMPLATE_ID", "1").lower() in ("1", "true", "yes")
    if require_tid and not tid:
        return AcceptResult(
            ok=False,
            reason="TEMPLATE_ID_MISSING",
            conf=conf,
            theta=theta,
        )

    if not ctx.no_adversarial:
        if ctx.ce_holds_count > 0:
            return AcceptResult(
                ok=False,
                reason=f"CE_HOLDS={ctx.ce_holds_count} (decisive reject)",
                conf=conf,
                theta=theta,
            )
        min_ce = int(os.environ.get("SDC_MIN_CE", "2"))
        if ctx.ce_verify_count < min_ce:
            return AcceptResult(
                ok=False,
                reason=f"CE_COUNT={ctx.ce_verify_count}<{min_ce}",
                conf=conf,
                theta=theta,
            )
        min_ev = int(os.environ.get("SDC_MIN_EVIDENCE_HITS", "0"))
        if min_ev and ctx.evidence_hits < min_ev:
            return AcceptResult(
                ok=False,
                reason=f"EVIDENCE_HITS={ctx.evidence_hits}<{min_ev}",
                conf=conf,
                theta=theta,
            )
    else:
        print("[trainaudit] CE gate skipped (SDC_ABLATION_NO_ADVERSARIAL)")

    if conf is None:
        return AcceptResult(
            ok=False,
            reason="CONF_MISSING",
            conf=None,
            theta=theta,
        )
    if conf < theta:
        return AcceptResult(
            ok=False,
            reason=f"CONF={conf}<theta={theta}",
            conf=conf,
            theta=theta,
        )

    healthy = _maybe_healthy_run(constraint, template_id=str(tid or ""))
    if healthy == "fail":
        return AcceptResult(
            ok=False,
            reason="HEALTHY_RUN_FP",
            conf=conf,
            theta=theta,
            healthy="fail",
        )
    # Healthy must pass unless SDC_ALLOW_HEALTHY_DEFER=1.
    strict = os.environ.get("SDC_HEALTHY_STRICT", "1").lower() in ("1", "true", "yes")
    if healthy in ("skipped", "deferred_nl", "error_skip", "no_sql") and strict and not allow_healthy_defer():
        return AcceptResult(
            ok=False,
            reason=f"HEALTHY_REQUIRED ({healthy})",
            conf=conf,
            theta=theta,
            healthy=healthy,
        )

    return AcceptResult(
        ok=True,
        reason="ACCEPT",
        conf=conf,
        theta=theta,
        healthy=healthy,
    )


def _maybe_healthy_run(constraint: Dict[str, Any], template_id: str = "") -> str:
    from healthy_sql import resolve_healthy_sql

    paths = discover_healthy_dbs()
    logic, src = resolve_healthy_sql(constraint, template_id=template_id)
    print(f"[trainaudit] healthy SQL source={src} chars={len(logic)}")

    if not logic:
        if allow_healthy_defer():
            print("[trainaudit] healthy-run DEFERRED (no SQL; SDC_ALLOW_HEALTHY_DEFER=1)")
            return "deferred_nl"
        print("[trainaudit] healthy-run NO_SQL")
        return "no_sql"

    if not paths:
        if allow_healthy_defer():
            print("[trainaudit] healthy-run SKIPPED (no DBs; defer allowed)")
            return "skipped"
        print("[trainaudit] healthy-run SKIPPED (no DBs; set SDC_HEALTHY_DBS)")
        return "skipped"

    try:
        import duckdb  # type: ignore
    except ImportError:
        print("[trainaudit] healthy-run SKIPPED (duckdb not installed)")
        return "error_skip"

    for path in paths:
        try:
            con = duckdb.connect(path, read_only=True)
            try:
                rows = con.execute(logic).fetchall()
            finally:
                con.close()
            if rows:
                print(f"[trainaudit] healthy-run FP on {path}: {len(rows)} rows")
                return "fail"
        except Exception as e:
            print(f"[trainaudit] healthy-run error on {path}: {e}")
            return "error_skip"

    print(f"[trainaudit] healthy-run PASS on {len(paths)} db(s) via {src}")
    return "pass"


def record_ce_from_brief(ctx: AcceptContext, research_brief: str) -> None:
    brief = research_brief or ""
    brief_l = brief.lower()
    if not (
        "反例" in brief
        or "验证反例" in brief
        or "counterexample" in brief_l
        or "verify_counterexample" in brief_l
    ):
        return
    ctx.ce_verify_count += 1
    ctx.ce_briefs.append(brief[:200])
    if re.search(r"\bHOLDS\b", brief, re.I):
        ctx.ce_holds_count += 1
    print(
        f"[trainaudit] S3 CE recorded count={ctx.ce_verify_count} "
        f"holds={ctx.ce_holds_count} brief={brief[:80]!r}"
    )


def set_fsm_stage(ctx: AcceptContext, stage: str, *, force: bool = False) -> None:
    if ctx.fsm is not None:
        try:
            ctx.fsm.advance(stage, force=force)
            ctx.fsm_stage = ctx.fsm.stage
            return
        except Exception as e:
            print(f"[trainaudit] FSM reject: {e}")
            raise
    ctx.fsm_stage = stage
    print(f"[trainaudit] FSM {stage}")


if __name__ == "__main__":
    os.environ.pop("SDC_PAPER_ALIGN", None)
    sample = {
        "name": "t",
        "logic": "SELECT step FROM coredump WHERE 1=0",
        "description": "schema witness cksum",
        "confidence": 0.9,
        "template_id": "T0_no_nan_inf",
        "applicable_conditions": {"dp": ">1", "stage": "model-after-optimizer-step"},
    }
    ctx = AcceptContext(ce_verify_count=2, template_id="T0_no_nan_inf")
    assert check_accept(sample, ctx).ok
    os.environ["SDC_PAPER_ALIGN"] = "1"
    # Prefer real clean DB; if missing, allow defer for this self-check only.
    if not discover_healthy_dbs():
        os.environ["SDC_ALLOW_HEALTHY_DEFER"] = "1"
    assert check_accept(sample, ctx).ok, check_accept(sample, ctx)
    assert not check_accept(sample, AcceptContext(ce_verify_count=1, template_id="T0_no_nan_inf")).ok
    bad = dict(sample, confidence=0.5)
    assert not check_accept(bad, AcceptContext(ce_verify_count=2, template_id="T0_no_nan_inf")).ok
    assert check_three_predicates({"logic": "x"}) is not None
    # NL path must ground via catalog/builtin probe when DBs exist
    nl = {
        "name": "layernorm cksum 跨rank一致",
        "logic": "自然语言：检查 layernorm cksum",
        "description": "schema witness",
        "confidence": 0.95,
        "template_id": "T0_no_nan_inf",
        "applicable_conditions": {"tp": ">1", "stage": "model-after-optimizer-step"},
    }
    if discover_healthy_dbs():
        os.environ.pop("SDC_ALLOW_HEALTHY_DEFER", None)
        r = check_accept(nl, AcceptContext(ce_verify_count=2, template_id="T0_no_nan_inf"))
        assert r.ok and r.healthy == "pass", r
    print("paper_accept_gate self-check OK")
