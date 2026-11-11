"""S1 PickPattern. Directions are the rule files in the frozen rules directory."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# The 18 directions. Identity and guide text come from these files.
RULE_FILES = (
    "T0_no_nan_inf.py",
    "T0_clip_grad_bounded.py",
    "T0_softmax_degenerate.py",
    "T0_token_id_range.py",
    "T0_initial_lr_present.py",
    "T0_norm_output_rms.py",
    "T0_dtype_propagation.py",
    "T0_optim_lr_positive.py",
    "T0_optim_step_counter.py",
    "T0_layer_count.py",
    "T0_checkpoint_preserve_rng.py",
    "T1_replica_cksum_equal.py",
    "T1_residual_stream.py",
    "T1_router_attribute.py",
    "T1_expert_bias_dtype.py",
    "T1_layer_count_strict.py",
    "T1_jitter_dtype.py",
    "T1_sqrt_decay_direction.py",
)


def rules_dir() -> Path:
    env = os.environ.get("SDC_RULES_DIR", "").strip()
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    return here.parents[1] / "catalog" / "frozen_20260712" / "rules"


def _template_from_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    stem = path.stem
    return {
        "id": stem,
        "name": stem,
        "relation": stem,
        "objects": path.name,
        "witness": text[:2000],
        "schema_predicate": "",
        "aliases": [],
        "source_file": str(path),
    }


def list_templates() -> List[Dict[str, Any]]:
    root = rules_dir()
    out: List[Dict[str, Any]] = []
    for name in RULE_FILES:
        path = root / name
        if not path.is_file():
            print(f"mining direction={Path(name).stem} missing={name}")
            continue
        out.append(_template_from_file(path))
    return out


def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    tid = (template_id or "").strip()
    if not tid:
        return None
    stem = tid[:-3] if tid.endswith(".py") else tid
    for t in list_templates():
        if t.get("id") == stem or t.get("objects") == tid:
            return t
    return None


def pick_pattern(library_categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """Current direction. SDC_TARGET_TEMPLATE selects one file; otherwise the first."""
    del library_categories
    templates = list_templates()
    if not templates:
        return {"id": "", "name": "unspecified", "relation": "", "aliases": []}
    target = os.environ.get("SDC_TARGET_TEMPLATE", "").strip()
    if target:
        t = get_template(target)
        if t:
            print(f"mining direction={t['id']} file={t['objects']}")
            return t
        print(f"mining direction={target} missing")
    t = templates[0]
    print(f"mining direction={t['id']} file={t['objects']}")
    return t


def prompt_block(template: Dict[str, Any]) -> str:
    if not template or not template.get("id"):
        return ""
    witness = (template.get("witness") or "").strip()
    lines = [
        "",
        "[Rule direction]",
        f"direction={template.get('id')} file={template.get('objects')}",
        "Propose rules only in this direction. Do not switch to another rule file.",
        "Then run counterexample checks. A rule is accepted only if those checks pass.",
        "Rule file:",
        witness,
        "",
    ]
    return "\n".join(lines)
