#!/usr/bin/env python3
"""
Self-reasoning workflow with write-back agent.
Routing: megatron_expert -> coordinate_agent -> write_agent -> JSON file
"""

import json
import copy
import os
from typing import Any, Tuple, Dict, Optional
from datetime import datetime
from pathlib import Path

from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.group import (
    ReplyResult,
    ContextVariables,
    AgentTarget,
)
from context_variables import WorkflowContext
from llm_config import get_llm_config
from chinese_generator_no_unicode import ChineseConstraintGeneratorNoUnicode


def _safe_parse(payload: Any) -> Any:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return payload
    return payload



def _default_constraints_path() -> str:
    """Package-local default (no machine-specific absolute paths)."""
    return str(Path(__file__).resolve().parent / "predefined_constraints.json")

class WorkflowTaskGenerateWithWriteAgent:
    """
    Self-reasoning workflow with write-back agent.
    Routing: megatron_expert validates constraint -> coordinate_agent -> write_agent -> JSON file
    """

    def __init__(self, context_variables: ContextVariables | None = None, constraints_file_path: str = None, max_iterations: int = 100):
        self.context_variables = ContextVariables(WorkflowContext().model_dump()) if context_variables is None else context_variables
        self.constraints_file_path = constraints_file_path or _default_constraints_path()
        
        self.max_iterations = max_iterations
        self.current_iteration = 0
        print(f"[trainaudit] Initialized with max_iterations={self.max_iterations}")
        
        self.chinese_generator = ChineseConstraintGeneratorNoUnicode(self.constraints_file_path)
        self._create_agents()
        self.agents = [self.coordinate_agent, self.megatron_expert, self.aggregation_expert, self.write_agent, self.report_agent]

        self._last_write_payload = {}
        self._lineage_route = ['megatron_expert', 'coordinate_agent', 'write_agent']
        
        self.context_variables["max_iterations"] = self.max_iterations
        self.context_variables["current_iteration"] = self.current_iteration
        self._no_adversarial = os.environ.get(
            "SDC_ABLATION_NO_ADVERSARIAL", ""
        ).lower() in ("1", "true", "yes")
        if self._no_adversarial:
            print("[trainaudit] ABLATION: no-adversarial mining (skip VERIFY_COUNTEREXAMPLE)")
        self._holdout_excluded = self._load_holdout_exclusions()
        if self._holdout_excluded:
            print(f"[trainaudit] HOLDOUT: {len(self._holdout_excluded)} evaluation cases withheld from mining evidence")

        # Paper-align Accept gate (Eq. accept). Default OFF — legacy mining unchanged.
        from paper_accept_gate import AcceptContext, paper_align_enabled, set_fsm_stage
        from fsm_stages import MiningFSM
        from catalog_picker import pick_pattern, prompt_block

        self._paper_align = paper_align_enabled()
        self._catalog_template = pick_pattern() if self._paper_align else {}
        tid = (
            os.environ.get("SDC_TARGET_TEMPLATE", "")
            or (self._catalog_template.get("id") if self._catalog_template else "")
            or ""
        )
        self._accept_ctx = AcceptContext(
            no_adversarial=self._no_adversarial,
            template_id=tid,
            fsm=MiningFSM("S1") if self._paper_align else None,
        )
        self._catalog_prompt = prompt_block(self._catalog_template) if self._paper_align else ""
        if self._paper_align:
            print(
                "[trainaudit] SDC_PAPER_ALIGN=1: Accept+3predicates+catalog+FSM; "
                "healthy SQL grounded (NL→catalog/synth); no defer by default"
            )
            set_fsm_stage(self._accept_ctx, "S1", force=True)
        else:
            print("[trainaudit] SDC_PAPER_ALIGN=0 (default): legacy mining path, no hard Accept gate")

    def _load_holdout_exclusions(self):
        path = os.environ.get("SDC_HOLDOUT_EXCLUDE", "")
        if not path or not os.path.exists(path):
            return []
        import json as _json
        with open(path) as f:
            data = _json.load(f)
        out = []
        for c in data.get("cases", []):
            item = c.get("case_id", "")
            if c.get("fix_commit"):
                item += f" (fix commit {c['fix_commit']})"
            out.append(item)
        return out

    def _holdout_suffix(self) -> str:
        if not hasattr(self, "_holdout_excluded"):
            self._holdout_excluded = self._load_holdout_exclusions()
        if not self._holdout_excluded:
            return ""
        listing = "; ".join(self._holdout_excluded)
        return (
            "\n\n[HOLDOUT EXCLUSION] The following bug cases and their fix commits are "
            "withheld evaluation data. You must NOT cite, search for, or derive evidence "
            f"from any of them: {listing}. If external research surfaces one of these "
            "commits or cases, discard that evidence and use other sources.\n"
        )

    # Agent factory methods
    def _abl_suffix(self) -> str:
        if not getattr(self, "_no_adversarial", False):
            return ""
        min_conf = os.environ.get("SDC_MIN_CONF", "0")
        return (
            "\n\n[ABLATION no-adversarial] theta_conf=0: do NOT emit VERIFY_COUNTEREXAMPLE. "
            "After external research (CONTINUE_AGGREGATION), go directly to WRITE_CONSTRAINT. "
            f"Accept when confidence >= {min_conf}. Skip counterexample attack stage.\n"
        )

    def _create_agents(self) -> None:
        self.coordinate_agent = ConversableAgent(
            human_input_mode="NEVER",
            name="coordinate_agent",
            system_message=self._coordinate_prompt() + self._abl_suffix() + self._holdout_suffix(),
            llm_config=get_llm_config(model_specialization="default", temperature=0.1),
            functions=self._create_coordinate_functions(),
        )
        _catalog = getattr(self, "_catalog_prompt", "") if getattr(self, "_paper_align", False) else ""
        self.megatron_expert = ConversableAgent(
            human_input_mode="NEVER",
            name="megatron_expert",
            system_message=self._megatron_prompt() + self._abl_suffix() + self._holdout_suffix() + _catalog,
            llm_config=get_llm_config(model_specialization="default", temperature=0.1),
            functions=self._create_megatron_functions(),
        )
        _agg_extra = ""
        if getattr(self, "_paper_align", False):
            _agg_extra = (
                "\n\n[PAPER S2 TOOLS] Prefer tool_grep_evidence / tool_read_evidence "
                "to collect ≥2 independent source hits before returning research.\n"
            )
        self.aggregation_expert = ConversableAgent(
            human_input_mode="NEVER",
            name="aggregation_expert",
            system_message=self._aggregation_prompt() + self._holdout_suffix() + _agg_extra,
            llm_config=get_llm_config(model_specialization="default", temperature=0.2),
            functions=self._create_aggregation_functions(),
        )
        self.write_agent = ConversableAgent(
            human_input_mode="NEVER",
            name="write_agent",
            system_message=self._write_agent_prompt(),
            llm_config=get_llm_config(model_specialization="default", temperature=0.1),
            functions=self._create_write_agent_functions(),
        )
        self.report_agent = ConversableAgent(
            human_input_mode="NEVER",
            name="report_agent",
            system_message=self._report_prompt(),
            llm_config=get_llm_config(model_specialization="default", temperature=0.2),
            functions=self._create_report_functions(),
        )

        self.megatron_expert.handoffs.set_after_work(AgentTarget(self.coordinate_agent))
        self.aggregation_expert.handoffs.set_after_work(AgentTarget(self.coordinate_agent))
        self.write_agent.handoffs.set_after_work(AgentTarget(self.coordinate_agent))

    # Internal helper methods
    def _normalize_todolist_payload(self, payload: Any) -> list[str]:
        """Normalize todo payload inputs into a flat list of strings."""
        items: list[str] = []
        if payload is None or payload == '':
            return items

        parsed = _safe_parse(payload)
        queue: list[Any] = [parsed]

        while queue:
            current = queue.pop(0)
            if current is None:
                continue
            if isinstance(current, str):
                segments = current.splitlines() or [current]
                for segment in segments:
                    candidate = segment.strip()
                    if not candidate:
                        continue
                    candidate = candidate.lstrip('-*').strip()
                    if candidate:
                        items.append(candidate)
            elif isinstance(current, (list, tuple, set)):
                queue.extend(list(current))
            elif isinstance(current, dict):
                queue.extend(current.values())
            else:
                text = str(current).strip()
                if text:
                    items.append(text)

        normalized: list[str] = []
        for entry in items:
            if entry and entry not in normalized:
                normalized.append(entry)
        return normalized

    # Agent prompt definitions (English; see mining_prompts.py)
    def _coordinate_prompt(self) -> str:
        from mining_prompts import COORDINATE_PROMPT
        return COORDINATE_PROMPT

    def _megatron_prompt(self) -> str:
        from mining_prompts import MEGATRON_PROMPT
        return MEGATRON_PROMPT

    def _aggregation_prompt(self) -> str:
        from mining_prompts import AGGREGATION_PROMPT
        return AGGREGATION_PROMPT

    def _write_agent_prompt(self) -> str:
        from mining_prompts import WRITE_AGENT_PROMPT
        return WRITE_AGENT_PROMPT

    def _report_prompt(self) -> str:
        from mining_prompts import report_prompt
        return report_prompt(self.max_iterations)

    # Function factories
    def _create_report_functions(self):
        """Create report agent loop control functions."""
        workflow_instance = self
        
        def restart_analysis() -> ReplyResult:
            """Mark current iteration as completed."""
            print(f"[Report Agent] Round {workflow_instance.current_iteration} completed")
            print(f"[Progress] {workflow_instance.current_iteration}/{workflow_instance.max_iterations}")
            
            context_variables = ContextVariables()
            context_variables["workflow_finished"] = True
            context_variables["round_completed"] = True
            context_variables["current_iteration"] = workflow_instance.current_iteration
            
            return ReplyResult(
                message=f"Round {workflow_instance.current_iteration} completed\nworkflow_finished=True",
                context_variables=context_variables,
                target=AgentTarget(workflow_instance.report_agent)
            )
        
        return [restart_analysis]

    def _create_coordinate_functions(self):
        """Create coordinate agent functions"""
        workflow_instance = self

        def go_megatron(context_variables: ContextVariables = None) -> ReplyResult:
            if context_variables is None:
                context_variables = ContextVariables()
            return ReplyResult(message="route→megatron_expert", context_variables=context_variables, target=AgentTarget(workflow_instance.megatron_expert))

        def go_aggregation(research_brief: str = "") -> ReplyResult:
            context_variables = ContextVariables()
            context_variables["research_brief"] = research_brief
            print(f"[COORDINATE] Passing research_brief: {research_brief}")
            if getattr(workflow_instance, "_paper_align", False):
                from paper_accept_gate import record_ce_from_brief, set_fsm_stage
                from fsm_stages import FSMError
                record_ce_from_brief(workflow_instance._accept_ctx, research_brief or "")
                nxt = "S3" if workflow_instance._accept_ctx.ce_verify_count > 0 else "S2"
                try:
                    set_fsm_stage(workflow_instance._accept_ctx, nxt)
                except FSMError as e:
                    return ReplyResult(
                        message=f"fsm_reject: {e}",
                        context_variables=context_variables,
                        target=AgentTarget(workflow_instance.megatron_expert),
                    )
            return ReplyResult(message=f"route->aggregation_expert, research_brief: {research_brief}", context_variables=context_variables, target=AgentTarget(workflow_instance.aggregation_expert))

        def go_write_agent(constraint_json: str = "", category: str = "", next_candidate: str = "") -> ReplyResult:
            context_variables = ContextVariables()
            context_variables["constraint_json"] = constraint_json
            context_variables["category"] = category
            context_variables["next_candidate"] = next_candidate
            workflow_instance._last_write_payload = {
                "constraint_json": constraint_json,
                "category": category,
                "next_candidate": next_candidate,
            }
            workflow_instance.context_variables["next_candidate"] = next_candidate
            if getattr(workflow_instance, "_paper_align", False):
                from paper_accept_gate import set_fsm_stage
                from fsm_stages import FSMError
                try:
                    set_fsm_stage(workflow_instance._accept_ctx, "S4")
                except FSMError as e:
                    print(f"[trainaudit] block write: {e}")
                    return ReplyResult(
                        message=f"fsm_reject_write: {e}. Complete S2 evidence + S3 CE first.",
                        context_variables=context_variables,
                        target=AgentTarget(workflow_instance.megatron_expert),
                    )
            print(f"[COORDINATE] Passing constraint to write_agent: category={category}")
            return ReplyResult(message=f"route->write_agent, constraint_json: {constraint_json[:100]}...", context_variables=context_variables, target=AgentTarget(workflow_instance.write_agent))

        def go_report(context_variables: ContextVariables = None) -> ReplyResult:
            if context_variables is None:
                context_variables = ContextVariables()
            context_variables["workflow_finished"] = True
            return ReplyResult(message="route→report_agent", context_variables=context_variables, target=AgentTarget(workflow_instance.report_agent))

        return [go_megatron, go_aggregation, go_write_agent, go_report]

    def _create_megatron_functions(self):
        """Create megatron expert functions"""
        workflow_instance = self

        def req_research(research_brief: str) -> ReplyResult:
            context_variables = ContextVariables()
            context_variables["needs_research"] = True
            context_variables["research_brief"] = research_brief
            print(f"[MEGATRON] Requesting research: {research_brief}")
            return ReplyResult(message=f"request->aggregation_expert: {research_brief}", context_variables=context_variables, target=AgentTarget(workflow_instance.aggregation_expert))

        def write_constraint(constraint_json: str, category: str, next_candidate: str = "") -> ReplyResult:
            context_variables = ContextVariables()
            context_variables["write_constraint"] = True
            context_variables["constraint_json"] = constraint_json
            context_variables["category"] = category
            context_variables["next_candidate"] = next_candidate
            workflow_instance._last_write_payload = {
                "constraint_json": constraint_json,
                "category": category,
                "next_candidate": next_candidate,
            }
            workflow_instance.context_variables["next_candidate"] = next_candidate
            print(f"[MEGATRON] Submitting constraint write-back: {category}")
            return ReplyResult(message=f"write_constraint->write_agent: {category}", context_variables=context_variables, target=AgentTarget(workflow_instance.write_agent))

        def go_report_from_mega(summary: str = "") -> ReplyResult:
            context_variables = ContextVariables()
            context_variables["finalize_report"] = True
            context_variables["summary"] = summary
            return ReplyResult(message="finalize→report", context_variables=context_variables, target=AgentTarget(workflow_instance.report_agent))

        def analyze_existing_constraints(category_filter: str = "") -> ReplyResult:
            """Analyze existing constraint file and identify gaps in training process.
            
            Args:
                category_filter: Specify constraint category to analyze (e.g., "data_parallel"). Empty means analyze all categories.
            """
            context_variables = ContextVariables()
            
            try:
                # Read existing constraints file
                with open(workflow_instance.constraints_file_path, 'r', encoding='utf-8') as f:
                    constraints_data = json.load(f)
                
                existing_constraints = constraints_data.get('constraints', {})
                
                # Category filtering (if specified)
                if category_filter:
                    if category_filter not in existing_constraints:
                        return ReplyResult(
                            message=f"Category '{category_filter}' not found. Existing categories: {', '.join(existing_constraints.keys())}",
                            context_variables=context_variables
                        )
                    existing_constraints = {category_filter: existing_constraints[category_filter]}
                    print(f"[Analysis] Category filtered: {category_filter}")
                
                # Training stage keywords
                training_stages = {
                    "initialization": ["model init", "weight init", "random seed", "parameter initialization"],
                    "data_loading": ["data shard", "batch load", "augmentation", "dataloader", "batch"],
                    "forward_pass": ["forward pass", "activation", "attention", "forward"],
                    "backward_pass": ["backward pass", "grad compute", "chain rule", "backward"],
                    "gradient_processing": ["grad accum", "grad clip", "grad sync", "gradient"],
                    "optimizer_step": ["param update", "lr schedule", "momentum", "optimizer"],
                    "checkpointing": ["model save", "resume", "state persist", "checkpoint", "save"],
                    "validation": ["val eval", "metrics", "model val", "validation", "eval"],
                    "communication": ["AllReduce", "AllGather", "P2P", "communication"],
                    "mixed_precision": ["fp16 cast", "grad scale", "numeric stability", "mixed precision", "fp16"],
                    "memory_management": ["mem alloc", "cache mgmt", "gc", "memory", "cuda"],
                    "sequence_parallel": ["seq shard", "attn parallel", "long seq", "sequence parallel"]
                }
                
                # Analyze existing constraint coverage
                covered_areas = set()
                constraint_summary = {}
                
                for category, constraints in existing_constraints.items():
                    constraint_list = []
                    for key, constraint in constraints.items():
                        # Analyze training stages covered by constraints
                        desc_lower = constraint.get("description", "").lower()
                        name_lower = constraint.get("name", "").lower()
                        combined_text = desc_lower + " " + name_lower + " " + key.lower()
                        
                        covered_stages_for_constraint = []
                        for stage_key, keywords in training_stages.items():
                            if any(keyword.lower() in combined_text for keyword in keywords):
                                covered_areas.add(stage_key)
                                covered_stages_for_constraint.append(stage_key)
                        
                        constraint_list.append({
                            "key": key,
                            "name": constraint.get("name", ""),
                            "type": constraint.get("type", ""),
                            "covered_stages": covered_stages_for_constraint,
                            "description_summary": constraint.get("description", "")[:100] + "..."
                        })
                    
                    constraint_summary[category] = {
                        "count": len(constraint_list),
                        "constraints": constraint_list
                    }
                
                # Identify gaps
                uncovered_stages = set(training_stages.keys()) - covered_areas
                
                # Suggest new constraint categories
                suggested_categories = []
                for stage in uncovered_stages:
                    if stage in ["mixed_precision", "sequence_parallel", "memory_management"]:
                        suggested_categories.append(stage)
                    elif stage in ["initialization", "checkpointing"]:
                        suggested_categories.append("model_integrity")
                    elif stage in ["data_loading", "validation"]:
                        suggested_categories.append("training_progress")
                
                analysis_result = {
                    "total_constraints": sum(data["count"] for data in constraint_summary.values()),
                    "categories": list(existing_constraints.keys()),
                    "constraint_summary": constraint_summary,
                    "covered_training_stages": sorted(list(covered_areas)),
                    "uncovered_stages": sorted(list(uncovered_stages)),
                    "suggested_new_categories": list(set(suggested_categories)),
                    "priority_gaps": []
                }
                
                # Build detailed constraint list for deduplication comparison
                existing_constraints_detail = {}
                for category, constraints in existing_constraints.items():
                    existing_constraints_detail[category] = {}
                    for key, constraint in constraints.items():
                        existing_constraints_detail[category][key] = {
                            "name": constraint.get("name", ""),
                            "description": constraint.get("description", ""),
                            "type": constraint.get("type", ""),
                            "applicable_conditions": constraint.get("applicable_conditions", {})
                        }
                
                analysis_result["existing_constraints_detail"] = existing_constraints_detail
                
                # Sort gaps by priority
                priority_mapping = {
                    "mixed_precision": "high — common in mixed precision",
                    "checkpointing": "high — required for long runs",
                    "memory_management": "medium — important for large models",
                    "initialization": "medium — affects stability",
                    "sequence_parallel": "medium — long-sequence jobs",
                    "data_loading": "low — relatively mature",
                    "validation": "low — fewer failures"
                }
                
                for stage in uncovered_stages:
                    priority_info = priority_mapping.get(stage, "low")
                    analysis_result["priority_gaps"].append({
                        "stage": stage,
                        "priority": priority_info,
                        "keywords": training_stages[stage]
                    })
                
                context_variables["existing_constraints_analysis"] = analysis_result
                context_variables["constraints_loaded"] = True
                
                print(f"[Analysis] Constraint analysis completed: {analysis_result['total_constraints']} constraints, {len(uncovered_stages)} gaps")
                print(f"[Analysis] Covered stages: {', '.join(covered_areas)}")
                print(f"[Analysis] Gap stages: {', '.join(uncovered_stages)}")
                print(f"[Analysis] Suggested categories: {', '.join(suggested_categories)}")
                
                # Build detailed analysis report message including existing constraint list
                gap_summary = []
                for gap in analysis_result["priority_gaps"]:
                    gap_summary.append(f"- {gap['stage']}: {gap['priority']}")
                
                # Build existing constraint list summary
                constraints_by_category = []
                for category, constraints in existing_constraints.items():
                    constraint_names = [constraint.get("name", key) for key, constraint in constraints.items()]
                    constraints_by_category.append(f"**{category}** ({len(constraint_names)}):")
                    for name in constraint_names:
                        constraints_by_category.append(f"  - {name}")
                
                # Injection lineage timeline (for guiding constraint validation)
                injection_timeline = """
Error injection lineage (schedules.py:436-470):
- Injection location: within backward_step() function
- Injection timing: T3 (after backward computation -> inject parameters -> before dump)
- Injection target: param (model parameter weights)
- Injection operations: add / scale / zero
- Impact range: T4->T8 all subsequent stages

Injection lineage timeline:
| Time | Phase | Stage | Param | Grad | DP consistency |
|--------|------|-------|---------|---------|---------|
| T0 | train start | - | clean | empty | consistent |
| T1 | forward_step() | - | clean | empty | consistent |
| T2 | backward | - | clean | computed | consistent |
| T3 | inject | - | polluted | computed | inconsistent |
| T4 | dump | model-after-backward | polluted | computed | inconsistent |
| T5 | all-reduce | - | polluted | synced | inconsistent |
| T6 | dump | model-before-optimizer-step | polluted | synced | inconsistent |
| T7 | optimizer.step() | - | updated | synced | more inconsistent |
| T8 | dump | model-after-optimizer-step | updated+polluted | synced | inconsistent |

**Detectable points**: T4/T5/T6/T8 dumps can catch injection anomalies"""
                
                filter_info = f"[{category_filter}] " if category_filter else ""
                
                analysis_message = f"""{filter_info}Existing constraints and injection lineage loaded:

**Stats**:
- total constraints: {analysis_result['total_constraints']}
- categories: {len(analysis_result['categories'])} ({', '.join(analysis_result['categories'])})

**Library** (for equivalence checks):
{chr(10).join(analysis_result.get('constraint_summaries', [])[:40])}

**Your task** for {category_filter or 'all categories'}:
1. Observe where DP consistency changes along the timeline
2. Propose constraints that detect anomalies at those points
3. Align stage guards with detectable dump points
4. Explain how injection tests would falsify the rule

**Equivalence** (avoid these):
1. Same anomaly target (cksum ≈ mean ≈ distribution for DP consistency)
2. Same stage + param scope + parallel dims
3. Parent/child inclusion (prefer root constraints)

Judge non-equivalence against the {analysis_result['total_constraints']} existing rules."""
                
                return ReplyResult(message=analysis_message, context_variables=context_variables)
                
            except Exception as e:
                context_variables["constraints_load_error"] = str(e)
                print(f"[Analysis] ERROR - Constraint file analysis failed: {e}")
                return ReplyResult(message=f"Constraint file analysis failed: {e}", context_variables=context_variables)

        return [req_research, write_constraint, go_report_from_mega, analyze_existing_constraints]

    def _create_aggregation_functions(self):
        """Create aggregation expert functions"""
        workflow_instance = self

        def submit_research(research_content: str = "") -> ReplyResult:
            context_variables = ContextVariables()
            context_variables["research_submitted"] = True
            context_variables["research_content"] = research_content
            print(f"[AGGREGATION] Submitting research results: {research_content[:100]}...")
            return ReplyResult(message=f"research->coordinate: {research_content}", context_variables=context_variables, target=AgentTarget(workflow_instance.coordinate_agent))

        funcs = [submit_research]
        if getattr(workflow_instance, "_paper_align", False):
            from evidence_retrieval import grep_evidence, read_evidence_file

            def tool_grep_evidence(pattern: str, glob: str = "*.py") -> str:
                """S2: grep framework/source roots for evidence lines."""
                out = grep_evidence(pattern, glob=glob)
                hits = 0 if out.startswith("ERROR") or out.startswith("(no hits") else out.count("\n") + 1
                workflow_instance._accept_ctx.evidence_hits += max(hits, 0)
                print(f"[trainaudit] S2 grep hits≈{hits} total_evidence={workflow_instance._accept_ctx.evidence_hits}")
                return out

            def tool_read_evidence(path: str) -> str:
                """S2: read a source file excerpt for evidence quotes."""
                out = read_evidence_file(path)
                if not out.startswith("ERROR"):
                    workflow_instance._accept_ctx.evidence_hits += 1
                return out

            funcs.extend([tool_grep_evidence, tool_read_evidence])
        return funcs

    def _create_write_agent_functions(self):
        """Create write agent functions"""
        workflow_instance = self

        def actual_write_constraint(constraint_json: str = "", category: str = "", next_candidate: str = "") -> ReplyResult:
            """Execute actual constraint write operation."""
            context_variables = ContextVariables()
            print(f"[WRITE_AGENT] Starting write operation (iteration {workflow_instance.current_iteration})")
            print(f"[WRITE_AGENT] Category: {category}")
            print(f"[WRITE_AGENT] Constraint JSON length: {len(constraint_json)}")
            
            success_flag, write_result, constraints_written = workflow_instance.write_constraint_to_json(constraint_json, category)
            
            context_variables["write_completed"] = True
            context_variables["write_result"] = write_result
            context_variables["constraints_written"] = constraints_written
            context_variables["write_success"] = success_flag
            context_variables["next_candidate"] = next_candidate
            
            # cache on workflow instance
            workflow_instance.context_variables["write_completed"] = True
            workflow_instance.context_variables["write_result"] = write_result
            workflow_instance.context_variables["constraints_written"] = constraints_written
            workflow_instance.context_variables["write_success"] = success_flag
            
            if success_flag:
                print(f"[WRITE_AGENT] Successfully wrote {constraints_written} constraints to {category} category (iteration {workflow_instance.current_iteration})")
                lineage = workflow_instance._lineage_route + ["report_agent"]
                context_variables["workflow_finished"] = True
                context_variables["lineage_route"] = lineage
                return ReplyResult(message=f"write_success->report: {write_result}", context_variables=context_variables, target=AgentTarget(workflow_instance.report_agent))
            else:
                print(f"[WRITE_AGENT] Write failed: {write_result} (iteration {workflow_instance.current_iteration})")
                return ReplyResult(message=f"write_failed->coordinate: {write_result}", context_variables=context_variables, target=AgentTarget(workflow_instance.coordinate_agent))

        def write_complete(write_result: str = "", constraints_written: int = 0, success: Optional[bool] = None, todolist: Any = None) -> ReplyResult:
            context_variables = ContextVariables()

            if workflow_instance._last_write_payload:
                constraint_json = workflow_instance._last_write_payload.get("constraint_json", "")
                category = workflow_instance._last_write_payload.get("category", "")
                
                if constraint_json and category:
                    success_flag, write_result, constraints_written = workflow_instance.write_constraint_to_json(constraint_json, category)
                    print(f"[WRITE_AGENT] Write execution: success={success_flag}, result={write_result}")
                else:
                    success_flag = False
                    write_result = "Missing required write parameters"
                    constraints_written = 0
            else:
                success_flag = False
                write_result = "Write data not found"
                constraints_written = 0
            context_variables["write_completed"] = True
            context_variables["write_result"] = write_result
            context_variables["constraints_written"] = constraints_written

            workflow_instance.context_variables["write_completed"] = True
            workflow_instance.context_variables["write_result"] = write_result
            workflow_instance.context_variables["constraints_written"] = constraints_written

            if success is None:
                success_flag = constraints_written > 0
                if not success_flag and isinstance(write_result, str) and write_result:
                    lower_result = write_result.lower()
                    success_flag = ("success" in lower_result or "completed" in lower_result or "ok" in lower_result)
            else:
                success_flag = bool(success)

            context_variables["write_success"] = success_flag
            workflow_instance.context_variables["write_success"] = success_flag

            todo_items: list[str] = []
            if todolist:
                todo_items.extend(workflow_instance._normalize_todolist_payload(todolist))

            stored_next = workflow_instance._last_write_payload.get("next_candidate")
            if stored_next:
                todo_items.extend(workflow_instance._normalize_todolist_payload(stored_next))

            if todo_items:
                deduped: list[str] = []
                for item in todo_items:
                    if item and item not in deduped:
                        deduped.append(item)
                context_variables["todolist"] = deduped
                workflow_instance.context_variables["todolist"] = deduped

            if success_flag:
                lineage = workflow_instance._lineage_route + ["report_agent"]
                context_variables["workflow_finished"] = True
                context_variables["lineage_route"] = lineage
                workflow_instance.context_variables["workflow_finished"] = True
                workflow_instance.context_variables["lineage_route"] = lineage
                print(f"[WRITE_AGENT] Write-back successful: {write_result}")
                return ReplyResult(message=f"write_complete->report: {write_result}", context_variables=context_variables, target=AgentTarget(workflow_instance.report_agent))

            print(f"[WRITE_AGENT] Write-back failed: {write_result}")
            return ReplyResult(message=f"write_retry->coordinate: {write_result}", context_variables=context_variables, target=AgentTarget(workflow_instance.coordinate_agent))

        return [actual_write_constraint, write_complete]

    # Write Agent Implementation
    def write_constraint_to_json(self, constraint_json_str: str, category: str) -> tuple[bool, str, int]:
        """Write agent core functionality: write constraints back to JSON file."""
        try:
            # Parse constraint JSON
            raw_data = json.loads(constraint_json_str)
            
            original_key = None
            if "name" not in raw_data and len(raw_data) == 1:
                # Input format: {"constraint_key": {"name": "...", ...}}
                original_key = list(raw_data.keys())[0]
                constraint_data = raw_data[original_key]
                print(f"[WriteJSON] Detected input with key, extracting: {original_key}")
                print(f"[WriteJSON] Constraint name: {constraint_data.get('name', 'unknown')}")
            elif "name" not in raw_data and len(raw_data) > 1:
                # Input format: {"key1": {...}, "key2": {...}} multiple constraints
                print(f"[WriteJSON] Detected multiple constraints: {len(raw_data)}")
                original_key = list(raw_data.keys())[0]
                constraint_data = raw_data[original_key]
                print(f"[WriteJSON] Processing first constraint: {original_key}")
            else:
                # Input format: {"name": "...", ...}
                constraint_data = raw_data
                print(f"[WriteJSON] Parsed constraint JSON (no key format): {constraint_data.get('name', 'unknown')}")

            # Paper Eq. accept hard gate — only when SDC_PAPER_ALIGN=1
            if getattr(self, "_paper_align", False):
                from paper_accept_gate import check_accept, set_fsm_stage
                self._accept_ctx.no_adversarial = getattr(self, "_no_adversarial", False)
                constraint_data = dict(constraint_data)
                if not constraint_data.get("template_id") and self._accept_ctx.template_id:
                    constraint_data["template_id"] = self._accept_ctx.template_id
                # Normalize explicit π_* from applicable_conditions for auditability
                ac = constraint_data.get("applicable_conditions") or {}
                if isinstance(ac, dict):
                    if "pi_topo" not in constraint_data:
                        constraint_data["pi_topo"] = {
                            k: v for k, v in ac.items()
                            if str(k).lower() in {
                                "dp", "tp", "pp", "ep", "cp", "tpl", "topology",
                                "zero_stage", "world_size",
                            }
                        }
                    if "pi_precond" not in constraint_data:
                        constraint_data["pi_precond"] = {
                            k: v for k, v in ac.items()
                            if str(k).lower() in {
                                "stage", "dtype", "requires_grad", "phase", "hook",
                            }
                        }
                    if "pi_schema" not in constraint_data:
                        constraint_data["pi_schema"] = (
                            constraint_data.get("logic")
                            or constraint_data.get("sql")
                            or constraint_data.get("description")
                        )
                try:
                    set_fsm_stage(self._accept_ctx, "S4")
                except Exception as e:
                    return False, f"trainaudit REJECT: FSM {e}", 0
                gate = check_accept(constraint_data, self._accept_ctx)
                print(
                    f"[trainaudit] Accept gate: ok={gate.ok} reason={gate.reason} "
                    f"conf={gate.conf} theta={gate.theta} healthy={gate.healthy} "
                    f"ce={self._accept_ctx.ce_verify_count} "
                    f"evidence={self._accept_ctx.evidence_hits}"
                )
                if not gate.ok:
                    return False, f"trainaudit REJECT: {gate.reason}", 0

            # Category mapping and validation
            valid_categories = {
                # Core 6 categories
                "data_parallel", "tensor_parallel", "pipeline_parallel", 
                "zero_optimization", "model_integrity", "training_progress",
                # Extended categories based on training process gap analysis
                "mixed_precision",           # Mixed precision training (FP16/BF16, gradient scaling)
                "checkpointing_integrity",   # Checkpoint integrity (model save/restore, state persistence)
                "initialization_validation", # Initialization validation (weight init, random seed consistency)
                "sequence_parallel",         # Sequence parallelism (long sequence processing, attention parallel)
                "memory_optimization",       # Memory optimization (allocation strategy, cache management)
                "activation_checkpointing",  # Activation checkpointing (memory saving, recomputation)
                "data_loading_integrity",    # Data loading integrity (data sharding, batch consistency)
                "validation_consistency",    # Validation consistency (validation set processing, metrics)
                "communication_optimization", # Communication optimization (bandwidth, latency)
                "distributed_optimizer"      # Distributed optimizer (state distribution, sync strategy)
            }
            
            original_category = category
            if category not in valid_categories:
                # Infer category based on constraint content
                constraint_desc = constraint_data.get("description", "").lower()
                constraint_name = constraint_data.get("name", "").lower()
                
                if any(word in constraint_desc + constraint_name for word in ["mixed precision", "fp16", "bf16", "mixed", "precision", "grad scale"]):
                    category = "mixed_precision"
                elif any(word in constraint_desc + constraint_name for word in ["checkpoint", "model save", "resume", "state persist"]):
                    category = "checkpointing_integrity"
                elif any(word in constraint_desc + constraint_name for word in ["weight init", "random seed", "initialization", "init"]):
                    category = "initialization_validation"
                elif any(word in constraint_desc + constraint_name for word in ["sequence parallel", "long seq", "attn parallel", "sequence", "sp"]):
                    category = "sequence_parallel"
                elif any(word in constraint_desc + constraint_name for word in ["mem alloc", "cache mgmt", "memory", "cuda", "mem opt"]):
                    category = "memory_optimization"
                elif any(word in constraint_desc + constraint_name for word in ["activation checkpoint", "activation", "checkpoint", "recompute"]):
                    category = "activation_checkpointing"
                elif any(word in constraint_desc + constraint_name for word in ["data shard", "batch", "dataloader", "data load"]):
                    category = "data_loading_integrity"
                elif any(word in constraint_desc + constraint_name for word in ["validation set", "metrics", "validation", "eval"]):
                    category = "validation_consistency"
                elif any(word in constraint_desc + constraint_name for word in ["comm opt", "bandwidth", "latency", "communication"]):
                    category = "communication_optimization"
                elif any(word in constraint_desc + constraint_name for word in ["optimizer state", "distributed optimizer", "distributed_optimizer"]):
                    category = "distributed_optimizer"
                # existing category match
                elif any(word in constraint_desc + constraint_name for word in ["zero", "ZeRO"]):
                    category = "zero_optimization"
                elif any(word in constraint_desc + constraint_name for word in ["tensor", "tp", "tensor parallel"]):
                    category = "tensor_parallel"
                elif any(word in constraint_desc + constraint_name for word in ["pipeline", "pp", "pipeline parallel"]):
                    category = "pipeline_parallel"
                elif any(word in constraint_desc + constraint_name for word in ["data", "dp", "data parallel"]):
                    category = "data_parallel"
                elif any(word in constraint_desc + constraint_name for word in ["training progress", "loss", "learning rate", "grad norm"]):
                    category = "training_progress"
                else:
                    # Default to model_integrity
                    category = "model_integrity"
                    
            print(f"[WriteJSON] Category mapping: {original_category} -> {category}")

            # Convert to Chinese format
            chinese_constraint = self._convert_to_chinese_format(constraint_data, category, original_key=original_key)
            print(f"[WriteJSON] Conversion completed, generated {len(chinese_constraint)} constraints")
            if original_key:
                print(f"[WriteJSON] Using original key: {original_key}")

            # Load existing constraints
            if not self.chinese_generator.load_and_analyze_patterns():
                return False, "Failed to load existing constraints", 0

            # Build correct JSON structure
            new_constraints = {
                category: chinese_constraint
            }
            print(f"[WriteJSON] Built constraint structure for merge: {category}/{list(chinese_constraint.keys())}")

            # Merge and write back
            merged_data = self.chinese_generator.merge_with_existing_constraints(new_constraints)
            print(f"[WriteJSON] Merge with existing constraints completed")

            if not self.chinese_generator.save_constraints_to_json(merged_data):
                return False, "Failed to save constraints to JSON file", 0

            # Verify integrity
            if not self.chinese_generator.verify_json_integrity():
                return False, "JSON integrity verification failed", 0

            print(f"[WriteJSON] Successfully wrote constraints to {category} category")
            if getattr(self, "_paper_align", False):
                from paper_accept_gate import set_fsm_stage
                set_fsm_stage(self._accept_ctx, "S5")
            return True, f"Successfully wrote constraints to {category} category", len(chinese_constraint)

        except Exception as e:
            print(f"[WriteJSON] ERROR - Write-back process failed: {e}")
            return False, f"Write-back process failed: {e}", 0

    def _convert_to_chinese_format(self, constraint_data: Dict, category: str, original_key: str = None) -> Dict[str, Dict]:
        """Convert constraint data to Chinese format.
        
        Args:
            constraint_data: Constraint data content
            category: Constraint category
            original_key: Original constraint key (prioritized if provided)
        """
        chinese_constraints = {}

        # Single constraint
        if "name" in constraint_data:
            if original_key:
                chinese_key = original_key
                print(f"[Convert] Using original key: {original_key}")
            else:
                constraint_name = constraint_data.get("name", "")
                chinese_key = self._generate_chinese_key(constraint_name, category)
                print(f"[Convert] Generated new key: {chinese_key}")
            
            chinese_constraints[chinese_key] = self._format_chinese_constraint(constraint_data, category)

        # Constraint list
        elif "constraints" in constraint_data:
            for constraint in constraint_data["constraints"]:
                constraint_name = constraint.get("name", "")
                chinese_key = self._generate_chinese_key(constraint_name, category)
                chinese_constraints[chinese_key] = self._format_chinese_constraint(constraint, category)

        # Other format, try as single constraint
        else:
            if original_key:
                chinese_key = original_key
                print(f"[Convert] Else branch using original key: {original_key}")
            else:
                chinese_key = f"{category}_constraint_check"
                print(f"[Convert] Else branch generated default key: {chinese_key}")
            chinese_constraints[chinese_key] = self._format_chinese_constraint(constraint_data, category)

        return chinese_constraints

    def _generate_chinese_key(self, constraint_name: str, category: str) -> str:
        """Generate Chinese format constraint key."""
        name_lower = constraint_name.lower()

        # Keep legacy CJK library names as-is when already well formed
        if any("\u4e00" <= char <= "\u9fff" for char in constraint_name):
            if constraint_name.endswith(("_check", " check", "\u68c0\u67e5")):
                return constraint_name
            if " check" in constraint_name.lower() or "\u68c0\u67e5" in constraint_name:
                return constraint_name
            return f"{constraint_name} check"
        
        # Intelligently generate key based on category and name content
        if category == "data_parallel" or "dp" in name_lower:
            if "parameter" in name_lower or "param" in name_lower:
                return "DP param cross-rank consistency check"
            elif "gradient" in name_lower and "allreduce" in name_lower:
                return "DP grad AllReduce consistency check"
            elif "communication" in name_lower:
                return "DP post-comm param sync check"
            elif "requires_grad" in name_lower:
                return "DP requires_grad consistency check"
            else:
                return "DP param consistency check"
                
        elif category == "tensor_parallel" or "tp" in name_lower:
            if "boundary" in name_lower:
                return "TP shard boundary continuity check"
            elif "communication" in name_lower or "matrix" in name_lower:
                return "TP comm matrix dim match check"
            elif "shared_experts" in name_lower:
                return "shared_experts weight inconsistency check"
            elif "layernorm" in name_lower:
                return "LayerNorm weight consistency check"
            elif "router" in name_lower:
                return "Router weight consistency check"
            else:
                return "TP param shard consistency check"
                
        elif category == "pipeline_parallel" or "pp" in name_lower:
            if "activation" in name_lower:
                return "PP activation transfer integrity check"
            elif "forward" in name_lower and "gradient" in name_lower:
                return "forward-stage grad-empty check"
            elif "parameter" in name_lower and "invariant" in name_lower:
                return "forward param immutability check"
            else:
                return "PP pipeline sync check"
                
        elif category == "zero_optimization":
            if "parameter" in name_lower or "param" in name_lower:
                return "ZeRO param shard consistency check"
            elif "gradient" in name_lower or "grad" in name_lower:
                return "ZeRO grad accum consistency check"
            elif "optimizer" in name_lower:
                return "ZeRO optimizer-state shard check"
            else:
                return "ZeRO memory footprint check"

        elif category == "model_integrity":
            if "weight" in name_lower:
                return "model weight numeric stability check"
            elif "frozen" in name_lower:
                return "frozen param no-update check"
            elif "tying" in name_lower or "share" in name_lower:
                return "weight-tying consistency check"
            elif "grad_shape" in name_lower or "main_grad" in name_lower:
                return "grad/main_grad shape and dtype match"
            else:
                return "model integrity check"
                
        elif category == "training_progress":
            if "loss" in name_lower:
                return "training loss trend check"
            elif "learning" in name_lower or "lr" in name_lower:
                return "LR schedule consistency check"
            elif "gradient" in name_lower and "norm" in name_lower:
                return "grad-norm spike check"
            else:
                return "step-time stability check"

        # Extended categories key generation
        elif category == "activation_checkpointing":
            if "memory" in name_lower:
                return "activation-ckpt memory check"
            elif "recompute" in name_lower:
                return "activation recompute consistency check"
            else:
                return "activation checkpoint consistency check"
        elif category == "mixed_precision":
            if "overflow" in name_lower:
                return "mixed-precision grad overflow check"
            elif "scale" in name_lower:
                return "mixed-precision scale-factor check"
            else:
                return "mixed-precision numeric stability check"
        elif category == "sequence_parallel":
            if "attention" in name_lower:
                return "sequence-parallel attn shard check"
            elif "communication" in name_lower:
                return "sequence-parallel comm consistency check"
            else:
                return "sequence-parallel shard consistency check"
        elif category == "memory_optimization":
            if "cuda" in name_lower:
                return "CUDA memory usage check"
            elif "offload" in name_lower:
                return "param offload check"
            else:
                return "memory usage check"
        elif category == "communication_optimization":
            if "bandwidth" in name_lower:
                return "comm bandwidth utilization check"
            elif "overlap" in name_lower:
                return "compute-comm overlap check"
            else:
                return "comm optimization check"
        elif category == "distributed_optimizer":
            if "state" in name_lower:
                return "distributed optimizer state shard check"
            elif "synchronization" in name_lower:
                return "optimizer sync consistency check"
            else:
                return "distributed optimizer state check"
        
        # Default format - use category name
        category_chinese_map = {
            "data_parallel": "data parallel",
            "tensor_parallel": "tensor parallel", 
            "pipeline_parallel": "pipeline parallel",
            "zero_optimization": "ZeRO",
            "model_integrity": "model integrity",
            "training_progress": "training progress",
            "activation_checkpointing": "activation checkpointing",
            "mixed_precision": "mixed precision",
            "sequence_parallel": "sequence parallel",
            "memory_optimization": "memory optimization",
            "communication_optimization": "communication optimization",
            "distributed_optimizer": "distributed optimizer"
        }
        chinese_category = category_chinese_map.get(category, category)
        return f"{chinese_category} constraint check"

    def _format_chinese_constraint(self, constraint: Dict, category: str) -> Dict:
        """Format as Chinese constraint."""
        return {
            "name": constraint.get("name", f"{category} constraint check"),
            "description": constraint.get("description", f"Check consistency/validity for {category}"),
            "type": constraint.get("type", "consistency"),
            "logic": "",
            "tables": constraint.get("tables", ["coredump"]),
            "params": constraint.get("params", {}),
            "applicable_conditions": constraint.get("applicable_conditions", {})
        }

    # Public API
    def run(self, task: str):
        """Run constraint generation workflow."""
        self._last_write_payload = {}
        self.context_variables["lineage_route"] = list(self._lineage_route)
        
        print("\n" + "="*60)
        print("[Workflow] Starting constraint generation workflow with write-back agent")
        print(f"[Workflow] Loop configuration: max {self.max_iterations} iterations")
        print(f"[Workflow] Task: {task}")
        print("="*60)

        # Main loop: counter-based control
        final_context = None
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            print(f"\n{'='*60}")
            print(f"[Workflow] Iteration {self.current_iteration} starting")
            print(f"{'='*60}")

            # Fresh Accept evidence each round (paper-align only; no-op when off)
            if getattr(self, "_paper_align", False):
                from paper_accept_gate import AcceptContext, set_fsm_stage
                from fsm_stages import MiningFSM
                from catalog_picker import pick_pattern, prompt_block
                self._catalog_template = pick_pattern()
                tid = (
                    os.environ.get("SDC_TARGET_TEMPLATE", "")
                    or self._catalog_template.get("id", "")
                )
                self._catalog_prompt = prompt_block(self._catalog_template)
                self._accept_ctx = AcceptContext(
                    no_adversarial=self._no_adversarial,
                    template_id=tid,
                    fsm=MiningFSM("S1"),
                )
                set_fsm_stage(self._accept_ctx, "S1", force=True)
            
            # Recreate agents each iteration to avoid state accumulation
            print(f"[Workflow] Iteration {self.current_iteration}: Recreating agents, clearing state")
            self._create_agents()
            self.agents = [self.coordinate_agent, self.megatron_expert, 
                          self.aggregation_expert, self.write_agent, self.report_agent]
            
            # Update context with loop information
            self.context_variables["current_iteration"] = self.current_iteration
            self.context_variables["max_iterations"] = self.max_iterations
            
            # Start this iteration's chat session
            final_context = self._start_chat_session(task)
            
            # Check if iteration completed successfully
            if not final_context.get("write_success", False):
                print(f"[Workflow] Iteration {self.current_iteration} write failed, continuing to next iteration")
                continue
            
            print(f"[Workflow] Iteration {self.current_iteration} completed successfully")
        
        print(f"\n[Workflow] All {self.max_iterations} iterations completed")
        
        return self._finalize_context(final_context)

    def _start_chat_session(self, task: str):
        """Start chat session (common for first and subsequent iterations)."""
        print(f"[Chat Session] Creating iteration {self.current_iteration} chat session")
        
        # Create new user agent each iteration
        user = UserProxyAgent(
            name=f"user_round_{self.current_iteration}",
            llm_config=False,
            code_execution_config=False,
            is_termination_msg=lambda m: "workflow_finished" in str(m.get("content",""))
        )
        
        # Dynamic task description
        dynamic_task = self._get_dynamic_task_focus(task)
        
        seed = f"""
This is iteration {self.current_iteration}, focusing on new areas

{dynamic_task}

Execution flow:
1. megatron_expert calls analyze_existing_constraints() to analyze existing constraints
2. Identify gaps, select uncovered checkpoints
3. Request external research via CONTINUE_AGGREGATION
4. Derive new constraints and verify uniqueness
5. Write back to JSON using WRITE_CONSTRAINT
6. Generate final report

Please start analysis, megatron_expert!
        """.strip()
        
        pattern = DefaultPattern(
            initial_agent=self.megatron_expert,
            agents=self.agents,
            context_variables=self.context_variables,
            user_agent=user,
        )
        
        print(f"[Chat Session] Starting iteration {self.current_iteration} chat session from megatron_expert")
        chat_result, final_context, last_agent = initiate_group_chat(
            pattern=pattern, messages=seed, max_rounds=1200
        )
        
        print(f"[Chat Session] Iteration {self.current_iteration} session completed")
        print(f"[Chat Session] Return status: write_success={final_context.get('write_success', False)}")
        
        return final_context

    def _finalize_context(self, final_context):
        """Finalize context processing."""
        print("\n" + "="*60)
        print("[Workflow] Constraint generation workflow completed")
        print(f"[Workflow] Loop statistics: completed {self.current_iteration}/{self.max_iterations} iterations")
        
        # Ensure required keys exist
        if final_context is None:
            print("[WARNING] final_context is None, creating empty context")
            final_context = {}
            
        if "constraints_generated" not in final_context:
            final_context["constraints_generated"] = {}
        if "todolist" not in final_context:
            final_context["todolist"] = []
        if "report_content" not in final_context:
            final_context["report_content"] = ""

        if not final_context.get("todolist"):
            fallback_payload = self._last_write_payload.get("next_candidate")
            fallback_todos = self._normalize_todolist_payload(fallback_payload) if fallback_payload else []
            if fallback_todos:
                final_context["todolist"] = fallback_todos

        if final_context.get("workflow_finished") and "lineage_route" not in final_context:
            final_context["lineage_route"] = self._lineage_route + ["report_agent"]
        elif "lineage_route" not in final_context and self.context_variables.get("lineage_route"):
            final_context["lineage_route"] = list(self.context_variables.get("lineage_route"))
        
        final_context["total_iterations_completed"] = self.current_iteration
        final_context["loop_completed"] = True
        
        print("[Workflow] Final status:")
        print(f"  - workflow_finished: {final_context.get('workflow_finished', False)}")
        print(f"  - loop_completed: {final_context.get('loop_completed', False)}")
        print(f"  - total_iterations: {final_context.get('total_iterations_completed', 0)}")
        print(f"  - write_success: {final_context.get('write_success', False)}")
        print(f"  - constraints_written: {final_context.get('constraints_written', 0)}")
        print("="*60 + "\n")
        
        return final_context

    def _get_dynamic_task_focus(self, base_task: str) -> str:
        """Category focus, or template focus when SDC_TARGET_TEMPLATE is set (Phase-1 A/B-light)."""
        tmpl_id = os.environ.get("SDC_TARGET_TEMPLATE", "").strip()
        if tmpl_id:
            return self._get_template_task_focus(tmpl_id)

        focus_category = os.environ.get("SDC_TARGET_CATEGORY", "").strip() or "data_parallel"
        focus_descriptions = {
            "data_parallel": "Data parallel training constraints",
            "tensor_parallel": "Tensor parallel training constraints",
            "pipeline_parallel": "Pipeline parallel training constraints",
            "zero_optimization": "ZeRO/distributed-optimizer state constraints",
            "model_integrity": "Model integrity and checkpoint constraints",
            "training_progress": "Training progress and counter constraints",
        }
        focus_description = focus_descriptions.get(focus_category, f"{focus_category} constraints")
        
        print(f"[Dynamic Task] Iteration {self.current_iteration}: Focusing on {focus_category} category, reasoning based on injection lineage")
        
        dynamic_task = f"""
Iteration {self.current_iteration} - Focus on {focus_description}

Iteration objective: Generate verifiable constraints for {focus_category} category

Core requirements:

1. Focus category: Only focus on {focus_category}
   - Call: analyze_existing_constraints(category_filter="{focus_category}")
   - Only examine existing constraints and injection lineage for this category
   - Do not consider other categories

2. Reasoning based on injection lineage:
   - Understand injection lineage timeline (T0-T8)
   - Injection at T3 pollutes single DP rank parameter
   - Based on your Megatron training knowledge, reason which checkpoints are important
   - Reason what constraints {focus_category} needs at these points

3. Ensure verifiability:
   - Constraints must detect injection-induced anomalies (DP inconsistency)
   - Clearly specify which checkpoint the constraint applies to (stage condition)
   - Explain how to verify constraint through injection testing

4. Equivalence checking:
   - Compare with existing constraints in {focus_category} category
   - Check for equivalence (not simple text duplication, but semantic equivalence)
   
   Equivalence criteria:
   a) Detection target equivalence:
      Example: checksum detection of DP consistency ≈ mean detection ≈ distribution detection
   
   b) Detection scope equivalence:
      Example: both detect DP consistency for all parameters at after-backward
   
   c) Containment relationship equivalence (tree structure):
      Parent: Detect DP consistency for all parameters
        └─ Child: Detect DP consistency for attention parameters
      → If parent exists, no need for child; if child exists, no need for parent
   
   Strategy:
   - Prioritize generating "root constraints" (broad coverage)
   - Avoid "leaf constraints" (narrow coverage)
   - Find root constraints that don't currently exist
   - If equivalent constraint found, select other gaps

5. Continuous loop optimization:
   - Generate 1-2 constraints per iteration
   - Next iteration loads latest constraints, continues filling gaps
   - Continuously optimize {focus_category} category coverage

Strict limitations:
- Only generate constraints for {focus_category} category
- Do not generate other categories (tensor_parallel, pipeline_parallel, etc.)
- Constraints must relate to injection lineage and be verifiable through injection testing
        """.strip()
        
        return dynamic_task

    def _get_template_task_focus(self, tmpl_id: str) -> str:
        """Template-aligned round: evidence + counterexample differ from T3/cksum mining."""
        import json
        from pathlib import Path

        spec_path = Path(__file__).resolve().parents[1] / "config" / "template_mining_spec.json"
        focus = {
            "id": tmpl_id,
            "category": "model_integrity",
            "prompt_focus": f"Mine constraints for template {tmpl_id}",
            "counterexample": "template-specific — NOT T3 weight inject",
            "fields": [],
        }
        if spec_path.is_file():
            try:
                data = json.loads(spec_path.read_text())
                for item in data.get("phase1_templates", []):
                    if item.get("id") == tmpl_id:
                        focus.update(item)
                        break
            except Exception as e:
                print(f"[Dynamic Task] template spec load warn: {e}")

        cat = focus.get("category") or "model_integrity"
        print(
            f"[Dynamic Task] Iteration {self.current_iteration}: TEMPLATE {tmpl_id} "
            f"→ category={cat} (NOT injection-lineage)"
        )
        fields = ", ".join(focus.get("fields") or []) or "(see template)"
        return f"""
Iteration {self.current_iteration} - Template {tmpl_id}: {focus.get('name', tmpl_id)}

Iteration objective: Instantiate template {tmpl_id} into deployable constraints in category "{cat}".

Core requirements:
1. Call analyze_existing_constraints(category_filter="{cat}") first.
2. Template focus (mandatory):
   {focus.get('prompt_focus')}
3. Allowed evidence fields only: {fields}
4. Counterexample type: {focus.get('counterexample')}
   - Do NOT use T3 single-rank weight pollution as the verification story.
   - Do NOT emit generic cross-DP param cksum rules (those belong to injection rounds).
5. Write 1-2 non-equivalent constraints into category "{cat}" via WRITE_CONSTRAINT.
6. Equivalence: skip if an existing rule already checks the same fields/stage/logic.

Strict limitations:
- Output category MUST be "{cat}"
- Forbidden: inventing T3/injection-lineage cksum variants for this round
        """.strip()


def main():
    """Main execution function."""
    constraints_file = _default_constraints_path()

    print("Self-Reasoning Workflow with Write Agent")
    print("Routing: megatron -> coordinate -> write_agent -> JSON")
    print("=" * 60)

    generator = WorkflowTaskGenerateWithWriteAgent(constraints_file_path=constraints_file)

    task = """
Collaborative work as "Megatron-LM Forward Training Constraint Reasoner":
Objective: Autonomously propose plausible forward constraints from Megatron training workflow and build reasoning chains.

Focus on tensor parallel parameter sharding mechanism:
- Megatron-LM official documentation/papers on tensor parallel parameter sharding
- Related code snippets (e.g., parameter sharding implementation)
- Conference papers/official blogs on parameter sharding and communication

Key change: After constraint validation is complete, use NEXT_ACTION: WRITE_CONSTRAINT to write directly to JSON file.

Output:
1) Constraints written directly to predefined_constraints.json in the library JSON schema
2) Complete reasoning chain with reference report
3) Next candidate constraint suggestions
    """

    final_context = generator.run(task)

    print("\n=== Execution Results ===")
    print("Constraints written:", final_context.get("constraints_generated", {}))
    print("Report:", final_context.get("report_content", ""))
    print("Todo:", final_context.get("todolist", []))


if __name__ == "__main__":
    main()




