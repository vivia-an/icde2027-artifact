#!/usr/bin/env python3
"""Mine one frozen rule file at a time.

Each file in the rules directory is one direction. The expert proposes
inside that direction, counterexamples are checked, and a rule is written
only if the accept gate passes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)


def main() -> int:
    os.environ["SDC_PAPER_ALIGN"] = os.environ.get("SDC_PAPER_ALIGN", "1")
    from ag2_deepseek_thinking_patch import apply_ag2_reasoning_content_patch
    from catalog_picker import list_templates
    from workflow_task_generate_with_writeAgent import WorkflowTaskGenerateWithWriteAgent

    apply_ag2_reasoning_content_patch()
    directions = list_templates()
    only = os.environ.get("SDC_ONLY_DIRECTION", "").strip()
    if only:
        directions = [t for t in directions if t["id"] == only or t["objects"] == only]
    print(f"mining directions={len(directions)}")
    max_iterations = int(os.environ.get("SDC_MAX_ITERATIONS", "100"))
    constraints_file = os.environ.get(
        "SDC_CONSTRAINTS_OUTPUT",
        str(HERE / "predefined_constraints.json"),
    )
    wrote = 0
    for template in directions:
        os.environ["SDC_TARGET_TEMPLATE"] = template["id"]
        print(f"mining direction={template['id']} file={template['objects']}")
        generator = WorkflowTaskGenerateWithWriteAgent(
            constraints_file_path=constraints_file,
            max_iterations=max_iterations,
        )
        task = (
            "Collaborative constraint reasoning for distributed LLM training.\n"
            f"Direction file: {template['objects']}.\n"
            "Propose guarded constraints only in this direction, "
            "validate with counterexamples, and persist accepted rules via WRITE_CONSTRAINT.\n"
            "Routing: megatron_expert → coordinate → write_agent → JSON library.\n"
        )
        ctx = generator.run(task)
        wrote += int(ctx.get("constraints_written", 0) or 0)
    print(f"mining done directions={len(directions)} constraints_written={wrote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
