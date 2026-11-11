"""S2 evidence retrieval: local grep/read over framework trees (no network)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

_MAX_HITS = 40
_MAX_FILE_BYTES = 200_000


def _roots() -> List[Path]:
    raw = os.environ.get("SDC_EVIDENCE_ROOTS", "")
    if raw:
        return [Path(p.strip()) for p in raw.split(os.pathsep) if p.strip()]
    # sensible defaults relative to repo
    here = Path(__file__).resolve()
    repo = here.parents[2]  # .../lsk
    candidates = [
        repo / "Megatron-LM",
        repo / "sdccheck",
        Path(os.environ.get("SDC_FRAMEWORK_ROOT", "")),
    ]
    return [p for p in candidates if p and p.exists()]


def grep_evidence(pattern: str, glob: str = "*.py", max_hits: int = _MAX_HITS) -> str:
    """Search evidence roots for pattern; return compact hit list."""
    if not pattern or len(pattern) < 2:
        return "ERROR: pattern too short"
    try:
        rx = re.compile(pattern, re.I)
    except re.error as e:
        return f"ERROR: bad regex: {e}"

    hits: List[str] = []
    for root in _roots():
        for path in root.rglob(glob):
            if not path.is_file():
                continue
            if any(x in path.parts for x in (".git", "node_modules", "__pycache__", "tools/texlive")):
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{path}:{i}: {line.strip()[:160]}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits)
    return "\n".join(hits) if hits else f"(no hits for {pattern!r} under {[str(r) for r in _roots()]})"


def read_evidence_file(rel_or_abs: str, max_chars: int = 8000) -> str:
    """Read a source file for evidence quotes."""
    path = Path(rel_or_abs)
    if not path.is_absolute():
        for root in _roots():
            cand = root / rel_or_abs
            if cand.exists():
                path = cand
                break
    if not path.exists():
        return f"ERROR: file not found: {rel_or_abs}"
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return f"ERROR: {e}"
    if len(data) > max_chars:
        return data[:max_chars] + f"\n...[truncated {len(data)-max_chars} chars]"
    return data


def format_evidence_bundle(pattern: str, paths: List[str] | None = None) -> str:
    parts = ["### grep", grep_evidence(pattern)]
    for p in paths or []:
        parts.append(f"### file {p}")
        parts.append(read_evidence_file(p))
    return "\n\n".join(parts)
