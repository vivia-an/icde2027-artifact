#!/usr/bin/env python3
"""
Entry point for agent workflow execution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# agents/ on path for local imports when launched from agents/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main execution function."""
    # DeepSeek V4 thinking: keep reasoning_content across ag2 tool rounds
    from ag2_deepseek_thinking_patch import apply_ag2_reasoning_content_patch
    apply_ag2_reasoning_content_patch()

    constraints_file = os.environ.get(
        "SDC_CONSTRAINTS_OUTPUT", "predefined_constraints.json"
    )
    max_iterations = int(os.environ.get("SDC_MAX_ITERATIONS", "100"))

    print("=" * 60)
    print("Self-Reasoning Workflow with Write Agent")
    print("Routing: megatron -> coordinate -> write_agent -> JSON")
    print("=" * 60)

    try:
        from workflow_task_generate_with_writeAgent import WorkflowTaskGenerateWithWriteAgent

        generator = WorkflowTaskGenerateWithWriteAgent(
            constraints_file_path=constraints_file, max_iterations=max_iterations
        )
        print("Generator initialized with 100 iteration limit")

        task = """
Collaborative constraint reasoning for Megatron-LM training:
Objective: Autonomously propose valid constraints from Megatron training workflow and build reasoning chains.

Constraint generation goals:
- Identify potential risk points in distributed training
- Based on official documentation, academic papers, and best practices
- Ensure practicality, innovation, and implementability
- Cover key checkpoints and exception handling

Key change: Use NEXT_ACTION: WRITE_CONSTRAINT to write directly to JSON file after validation.

Routing flow:
1. megatron_expert validates constraint completeness
2. Output NEXT_ACTION: WRITE_CONSTRAINT
3. coordinate_agent routes to write_agent
4. write_agent writes to predefined_constraints.json
5. write_agent reports completion

Output:
1) Constraints written to predefined_constraints.json
2) Complete reasoning chain with references
3) Next candidate constraint suggestions
        """

        target_category = os.environ.get("SDC_TARGET_CATEGORY", "").strip()
        if target_category:
            task += f"""
Current round target category: {target_category}
- First call analyze_existing_constraints(category_filter="{target_category}") to load this category's gaps.
- Propose and write constraints ONLY for category "{target_category}" in this round.
- If the category is empty, treat every training-stage checkpoint relevant to it as an uncovered gap.
"""

        print("\nStarting constraint generation workflow...")
        print("Task:", task.strip())
        print("\n" + "="*60)

        final_context = generator.run(task)

        print("\n" + "="*60)
        print("=== Execution Results ===")

        constraints_generated = final_context.get("constraints_generated", {})
        report_content = final_context.get("report_content", "")
        todolist = final_context.get("todolist", [])

        print(f"Constraints generated: {len(constraints_generated)}")
        if constraints_generated:
            print("Constraint details:")
            for key, value in constraints_generated.items():
                print(f"  {key}: {value}")

        print(f"\nReport length: {len(report_content)} characters")
        if report_content:
            print("Report summary:")
            print(f"  {report_content[:300]}...")

        print(f"\nTodo items: {len(todolist)}")
        if todolist:
            for i, todo in enumerate(todolist[:3], 1):
                print(f"  {i}. {todo}")

        print("\n=== Routing Verification ===")
        write_completed = final_context.get("write_completed", False)
        write_result = final_context.get("write_result", "")
        constraints_written = final_context.get("constraints_written", 0)

        print(f"Write agent status: {'completed' if write_completed else 'not executed'}")
        print(f"Write result: {write_result}")
        print(f"Constraints written: {constraints_written}")

        print("\n=== Loop Verification ===")
        loop_completed = final_context.get("loop_completed", False)
        stop_reason = final_context.get("stop_reason", "")
        workflow_finished = final_context.get("workflow_finished", False)

        print(f"Loop status: {'completed' if loop_completed else 'terminated early'}")
        print(f"Stop reason: {stop_reason if stop_reason else 'unknown'}")
        print(f"Workflow status: {'finished' if workflow_finished else 'incomplete'}")

        if write_completed and loop_completed:
            print("\nRouting test successful. Constraints written and loop completed.")
        elif write_completed:
            print("\nWrite function working, loop mechanism running.")
        else:
            print("\nWarning: Routing needs debugging, write_agent or loop not executing correctly.")

        return True

    except ImportError as e:
        print(f"Import failed: {e}")
        print("Ensure all dependency files are correctly created")
        return False
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\nSelf-Reasoning workflow test completed")
    else:
        print("\nTest failed")