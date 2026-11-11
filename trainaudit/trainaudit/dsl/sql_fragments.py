"""SQL helpers for DSL → DuckDB compilation.

Centralises JSON path resolution + value casting so the compiler stays
declarative. DuckDB JSON conventions:
  - `json_extract(payload, '$.foo')` → JSON value (still string-encoded for strings)
  - `json_extract_string(payload, '$.foo')` → string unquoted
  - `to_json(literal)` → JSON-encoded literal for `=` comparisons
"""
from __future__ import annotations

from typing import Any, List


def hookpoint_filter(hookpoints: List[str]) -> str:
    """`hookpoint = X` or `hookpoint IN (...)` clause."""
    if len(hookpoints) == 1:
        return f"hookpoint = '{hookpoints[0]}'"
    quoted = ", ".join(f"'{h}'" for h in hookpoints)
    return f"hookpoint IN ({quoted})"


def json_path_for(field: str) -> str:
    """Convert plain field name or '$.foo' to a `$.foo` JSON path."""
    f = field.strip()
    return f if f.startswith("$") else f"$.{f}"


def extract(source: str, field: str) -> str:
    """Return SQL expression `json_extract(<source>, '$.<field>')`."""
    return f"json_extract({source}, '{json_path_for(field)}')"


def extract_string(source: str, field: str) -> str:
    return f"json_extract_string({source}, '{json_path_for(field)}')"


def cast_num(expr: str) -> str:
    """Cast a JSON-extracted value to DOUBLE (preserves NULLs)."""
    return f"CAST({expr} AS DOUBLE)"


def cast_int(expr: str) -> str:
    return f"CAST({expr} AS BIGINT)"


def to_json_literal(value: Any) -> str:
    """Render a Python literal as a DuckDB to_json(...) expression so that
    equality against `json_extract` (which returns JSON) round-trips."""
    if value is True:
        return "to_json(true)"
    if value is False:
        return "to_json(false)"
    if value is None:
        return "NULL"
    if isinstance(value, str):
        # to_json a string includes the surrounding quotes
        esc = value.replace("'", "''")
        return f"to_json('{esc}')"
    if isinstance(value, (int, float)):
        return f"to_json({value})"
    raise ValueError(f"unsupported literal for to_json: {value!r}")
