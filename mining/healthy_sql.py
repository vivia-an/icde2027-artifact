"""Resolve executable healthy-run SQL for the Accept gate.

Uses constraint SQL, catalog probes, or lightweight synthesis over coredump(step, stage, data JSON).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

from catalog_picker import get_template


def looks_like_sql(text: str) -> bool:
    return bool(re.search(r"\b(SELECT|WITH)\b", text or "", re.I))


def extract_sql_from_constraint(constraint: Dict[str, Any]) -> str:
    for key in ("sql", "logic", "query", "healthy_sql", "probe_sql"):
        val = constraint.get(key)
        if isinstance(val, str) and looks_like_sql(val):
            return val.strip()
    meta = constraint.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("sql", "logic", "healthy_sql"):
            val = meta.get(key)
            if isinstance(val, str) and looks_like_sql(val):
                return val.strip()
    return ""


# Probes are taken from the current rule file when it carries SQL.
_BUILTIN_PROBES = {}


def _stage_filter(constraint: Dict[str, Any], template: Optional[Dict[str, Any]]) -> str:
    ac = constraint.get("applicable_conditions") or constraint.get("pi_precond") or {}
    if isinstance(ac, dict):
        st = ac.get("stage") or ac.get("phase")
        if isinstance(st, str) and st.strip() and "SELECT" not in st.upper():
            # normalize "= 'x'" / "x" → LIKE
            m = re.search(r"['\"]([^'\"]+)['\"]", st)
            token = m.group(1) if m else st.replace("=", "").strip()
            if token and len(token) < 80:
                return token
    if template:
        pre = template.get("default_pi_precond") or {}
        st = pre.get("stage") if isinstance(pre, dict) else None
        if isinstance(st, str) and "optimizer" in st.lower():
            return "model-after-optimizer-step"
    return "model-after-optimizer-step"


def synthesize_from_nl(constraint: Dict[str, Any], template_id: str = "") -> str:
    """Best-effort SQL when LLM left NL / empty logic."""
    blob = " ".join(
        str(constraint.get(k, ""))
        for k in ("name", "description", "logic", "notes")
    ).lower()
    tid = (
        template_id
        or constraint.get("template_id")
        or constraint.get("catalog_id")
        or ""
    )
    tid = str(tid).strip()
    tmpl = get_template(tid) if tid else None
    if tmpl and tmpl.get("healthy_probe_sql") and looks_like_sql(tmpl["healthy_probe_sql"]):
        return tmpl["healthy_probe_sql"].strip()
    if tid in _BUILTIN_PROBES:
        return _BUILTIN_PROBES[tid]
    if tmpl and tmpl.get("id") in _BUILTIN_PROBES:
        return _BUILTIN_PROBES[tmpl["id"]]

    stage = _stage_filter(constraint, tmpl)
    if any(k in blob for k in ("cksum", "checksum", "一致", "replica", "跨rank", "跨 rank")):
        return f"""
WITH t AS (
  SELECT step,
    json_extract_string(data, '$.name') AS name,
    json_extract_string(data, '$.cksum') AS cksum
  FROM coredump
  WHERE stage = '{stage}'
    AND (
      lower(json_extract_string(data, '$.name')) LIKE '%layernorm%'
      OR lower(json_extract_string(data, '$.name')) LIKE '%layer_norm%'
      OR lower(json_extract_string(data, '$.name')) LIKE '%router%'
    )
)
SELECT name, step, COUNT(DISTINCT cksum) AS n_cksum
FROM t
GROUP BY name, step
HAVING COUNT(DISTINCT cksum) > 1
""".strip()
    if any(k in blob for k in ("dtype", "精度", "bf16", "fp16")):
        return f"""
SELECT step, json_extract_string(data, '$.name') AS name,
       json_extract_string(data, '$.type') AS typ
FROM coredump
WHERE stage = '{stage}'
  AND json_extract_string(data, '$.type') IS NOT NULL
  AND lower(json_extract_string(data, '$.type')) LIKE '%float64%'
LIMIT 20
""".strip()
    # Generic: impossible predicate → 0 rows on healthy (smoke that SQL path executes)
    return """
SELECT step, stage FROM coredump
WHERE json_extract_string(data, '$.cksum') = '__trainaudit_healthy_impossible__'
LIMIT 1
""".strip()


def resolve_healthy_sql(
    constraint: Dict[str, Any],
    *,
    template_id: str = "",
) -> Tuple[str, str]:
    """Return (sql, source) where source in sql_field|catalog|builtin|nl_synth|empty."""
    direct = extract_sql_from_constraint(constraint)
    if direct:
        return direct, "sql_field"
    tid = template_id or str(constraint.get("template_id") or "")
    tmpl = get_template(tid) if tid else None
    if tmpl and tmpl.get("healthy_probe_sql") and looks_like_sql(str(tmpl["healthy_probe_sql"])):
        return str(tmpl["healthy_probe_sql"]).strip(), "catalog"
    if tid in _BUILTIN_PROBES:
        return _BUILTIN_PROBES[tid], "builtin"
    if os.environ.get("SDC_HEALTHY_SYNTH", "1").lower() in ("1", "true", "yes"):
        return synthesize_from_nl(constraint, tid), "nl_synth"
    return "", "empty"
