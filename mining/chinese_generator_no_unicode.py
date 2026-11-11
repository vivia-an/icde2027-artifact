#!/usr/bin/env python3
"""Constraint library helpers for Megatron-LM style JSON rules.

Keeps generated constraints compatible with predefined_constraints.json.
Legacy module/class names retained for import stability.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class ConstraintGenerator:
    """Analyze the library and merge English seed constraints."""

    def __init__(self, constraints_file_path: str):
        self.constraints_file_path = constraints_file_path
        self.constraints_data: Dict[str, Any] = {}
        self.patterns: Dict[str, Any] = {}

    def load_and_analyze_patterns(self) -> bool:
        try:
            with open(self.constraints_file_path, "r", encoding="utf-8") as f:
                self.constraints_data = json.load(f)
            self.patterns = self._analyze_patterns()
            return True
        except Exception as e:
            print(f"Failed to load constraints file: {e}")
            return False

    def _analyze_patterns(self) -> Dict[str, Any]:
        patterns: Dict[str, Any] = {
            "naming_patterns": {
                "key_suffixes": [],
                "name_patterns": [],
                "description_styles": [],
            },
            "technical_terms": {
                "english_terms": [],
            },
            "stage_references": [],
            "condition_patterns": [],
        }

        constraints = self.constraints_data.get("constraints", {})
        for _category_name, category_constraints in constraints.items():
            for constraint_key, constraint in category_constraints.items():
                if constraint_key.endswith(("_check", " check")):
                    patterns["naming_patterns"]["key_suffixes"].append("_check")

                name = constraint.get("name", "")
                if "check" in name.lower():
                    patterns["naming_patterns"]["name_patterns"].append(name)

                description = constraint.get("description", "")
                patterns["technical_terms"]["english_terms"].extend(
                    self._extract_english_terms(description)
                )

                applicable_conditions = constraint.get("applicable_conditions", {})
                if isinstance(applicable_conditions, dict):
                    stage = applicable_conditions.get("stage", "")
                    if stage and isinstance(stage, str):
                        patterns["stage_references"].append(stage)
                elif isinstance(applicable_conditions, list):
                    for condition in applicable_conditions:
                        if isinstance(condition, str) and (
                            "stage" in condition or "model-after" in condition
                        ):
                            patterns["stage_references"].append(condition)

        patterns["technical_terms"]["english_terms"] = self._safe_dedupe(
            patterns["technical_terms"]["english_terms"]
        )
        patterns["stage_references"] = self._safe_dedupe(patterns["stage_references"])
        return patterns

    def _safe_dedupe(self, items: List[Any]) -> List[Any]:
        result = []
        for item in items:
            if item not in result:
                result.append(item)
        return result

    def _extract_english_terms(self, text: str) -> List[str]:
        english_terms = []
        english_patterns = [
            r"model-after-\w+",
            r"grad_\w+",
            r"cksum",
            r"rank",
            r"DP",
            r"TP",
            r"PP",
            r"shared_experts",
            r"LayerNorm",
            r"Router",
            r"ZeRO",
            r"requires_grad",
        ]
        for pattern in english_patterns:
            english_terms.extend(re.findall(pattern, text))
        return english_terms

    def generate_constraints(self) -> Dict[str, Dict[str, Any]]:
        return {
            "zero_optimization": self._generate_zero_constraints(),
            "training_progress": self._generate_progress_constraints(),
            "data_parallel": self._generate_dp_extensions(),
            "tensor_parallel": self._generate_tp_extensions(),
            "pipeline_parallel": self._generate_pp_extensions(),
            "model_integrity": self._generate_integrity_extensions(),
        }

    # Backward-compatible alias used by older call sites
    def generate_chinese_constraints(self) -> Dict[str, Dict[str, Any]]:
        return self.generate_constraints()

    def _generate_zero_constraints(self) -> Dict[str, Any]:
        return {
            "ZeRO param shard consistency check": {
                "name": "ZeRO param shard consistency across ranks",
                "description": (
                    "When ZeRO is enabled, check that parameter shards are consistent "
                    "across ranks so each rank holds only its assigned shard."
                ),
                "type": "consistency",
                "logic": "",
                "tables": ["coredump"],
                "params": {},
                "applicable_conditions": {
                    "zero_stage": ">= 1",
                    "stage": "= 'model-after-optimizer-step'",
                },
            },
            "ZeRO grad accum consistency check": {
                "name": "ZeRO gradient accumulation consistency",
                "description": (
                    "Under ZeRO, check that gradient shards stay consistent across ranks "
                    "during accumulation and remain intact before AllReduce."
                ),
                "type": "consistency",
                "logic": "",
                "tables": ["coredump"],
                "params": {"min_accumulation_steps": 1},
                "applicable_conditions": {
                    "zero_stage": ">= 2",
                    "stage": "= 'main-grad-in-backward'",
                },
            },
            "ZeRO optimizer-state shard check": {
                "name": "ZeRO-2/3 optimizer-state shard check",
                "description": (
                    "In ZeRO-2/3, verify optimizer state (momentum, variance) is sharded "
                    "onto the owning ranks and cksum-consistent."
                ),
                "type": "partition",
                "logic": "",
                "tables": ["coredump"],
                "params": {},
                "applicable_conditions": {
                    "zero_stage": ">= 2",
                    "stage": "= 'model-after-optimizer-step'",
                },
            },
        }

    def _generate_progress_constraints(self) -> Dict[str, Any]:
        return {
            "training loss trend check": {
                "name": "Training loss decreasing-trend check",
                "description": (
                    "Check the overall decreasing trend of training loss and flag "
                    "abnormal spikes or divergence via a sliding window."
                ),
                "type": "validity",
                "logic": "",
                "tables": ["training_metrics"],
                "params": {"window_size": 10, "tolerance": 0.1},
                "applicable_conditions": {"step": "> 10"},
            },
            "LR schedule consistency check": {
                "name": "Learning-rate schedule execution consistency",
                "description": (
                    "Verify the LR scheduler matches the configured policy "
                    "(linear, cosine, step, …) and actual LR equals the expected value."
                ),
                "type": "consistency",
                "logic": "",
                "tables": ["training_metrics"],
                "params": {},
                "applicable_conditions": {"lr_scheduler": "!= 'none'"},
            },
            "grad-norm spike check": {
                "name": "Gradient-norm anomaly check",
                "description": (
                    "Monitor grad norm for explosion/vanishing; mark anomalies when "
                    "the norm exceeds the configured threshold."
                ),
                "type": "validity",
                "logic": "",
                "tables": ["training_metrics"],
                "params": {"grad_norm_threshold": 10.0},
                "applicable_conditions": {"stage": "= 'model-after-backward'"},
            },
        }

    def _generate_dp_extensions(self) -> Dict[str, Any]:
        return {
            "DP post-comm param sync check": {
                "name": "DP AllReduce post-comm param sync check",
                "description": (
                    "After AllReduce, verify all DP ranks have identical parameters "
                    "via cksum equality."
                ),
                "type": "consistency",
                "logic": "",
                "tables": ["coredump"],
                "params": {},
                "applicable_conditions": {
                    "dp": "> 1",
                    "stage": "= 'model-after-allreduce'",
                },
            }
        }

    def _generate_tp_extensions(self) -> Dict[str, Any]:
        return {
            "TP shard boundary continuity check": {
                "name": "TP tensor-shard boundary continuity check",
                "description": (
                    "Verify TP shard boundary indices are contiguous across ranks "
                    "so shards can be concatenated correctly."
                ),
                "type": "consistency",
                "logic": "",
                "tables": ["coredump"],
                "params": {},
                "applicable_conditions": {"tp": "> 1"},
            }
        }

    def _generate_pp_extensions(self) -> Dict[str, Any]:
        return {
            "PP activation transfer integrity check": {
                "name": "PP activation-transfer integrity check",
                "description": (
                    "Verify activations passed between pipeline stages have correct "
                    "shape, dtype, and numeric range."
                ),
                "type": "completeness",
                "logic": "",
                "tables": ["coredump"],
                "params": {},
                "applicable_conditions": {
                    "pp": "> 1",
                    "stage": "LIKE 'activation-transfer-%'",
                },
            }
        }

    def _generate_integrity_extensions(self) -> Dict[str, Any]:
        return {
            "model weight numeric stability check": {
                "name": "Model weight numeric stability check",
                "description": (
                    "Detect NaN/Inf and out-of-range weight distributions."
                ),
                "type": "validity",
                "logic": "",
                "tables": ["coredump"],
                "params": {"weight_range_threshold": 100.0},
                "applicable_conditions": {},
            }
        }

    def merge_with_existing_constraints(
        self, new_constraints: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        merged_data = copy.deepcopy(self.constraints_data)
        original_count = self._count_constraints(merged_data)

        if "constraints" not in merged_data:
            merged_data["constraints"] = {}
        if "metadata" not in merged_data:
            merged_data["metadata"] = {}

        for category, constraints in new_constraints.items():
            if category not in merged_data["constraints"]:
                merged_data["constraints"][category] = {}

            for constraint_key, constraint in constraints.items():
                if constraint_key not in merged_data["constraints"][category]:
                    merged_data["constraints"][category][constraint_key] = constraint
                    print(f"Added constraint: {category}::{constraint_key}")
                else:
                    print(f"Constraint exists, skipped: {category}::{constraint_key}")

        meta = merged_data.setdefault("metadata", {})
        meta["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        meta["version"] = "1.2"
        ctypes = meta.setdefault("constraint_types", {})
        ctypes.setdefault(
            "partition", "Partition constraint: check data sharding correctness"
        )
        ctypes.setdefault(
            "completeness", "Completeness constraint: check data integrity"
        )

        new_count = self._count_constraints(merged_data)
        print("\nConstraint merge completed:")
        print(f"  Original count: {original_count}")
        print(f"  New count: {new_count}")
        print(f"  Added: {new_count - original_count}")
        return merged_data

    def _count_constraints(self, data: Dict[str, Any]) -> int:
        total = 0
        for category in data.get("constraints", {}).values():
            total += len(category)
        return total

    def save_constraints_to_json(self, merged_data: Dict[str, Any]) -> bool:
        try:
            backup_path = (
                f"{self.constraints_file_path}.backup."
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            with open(self.constraints_file_path, "r", encoding="utf-8") as f:
                backup_content = f.read()
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(backup_content)
            print(f"Backup created at: {backup_path}")

            with open(self.constraints_file_path, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)

            print("Constraints successfully written to JSON file")
            return True
        except Exception as e:
            print(f"Failed to save constraints to JSON: {e}")
            return False

    def verify_json_integrity(self) -> bool:
        try:
            with open(self.constraints_file_path, "r", encoding="utf-8") as f:
                reloaded_data = json.load(f)

            for key in ("constraints", "metadata"):
                if key not in reloaded_data:
                    print(f"Missing required field: {key}")
                    return False

            constraints = reloaded_data.get("constraints", {})
            expected_categories = [
                "data_parallel",
                "tensor_parallel",
                "pipeline_parallel",
                "zero_optimization",
                "model_integrity",
                "training_progress",
            ]
            for category in expected_categories:
                if category not in constraints:
                    print(f"Missing constraint category: {category}")
                    return False

            total_constraints = sum(len(cat) for cat in constraints.values())
            print("JSON integrity verification passed")
            print(f"  Total constraints: {total_constraints}")
            return True
        except Exception as e:
            print(f"JSON integrity verification failed: {e}")
            return False

    def run_complete_generation_and_save(self) -> bool:
        print("Starting constraint generation and save workflow...")

        if not self.load_and_analyze_patterns():
            print("Pattern analysis failed")
            return False

        print("\nGenerating constraints...")
        new_constraints = self.generate_constraints()
        total_new = sum(len(c) for c in new_constraints.values())
        print(f"Successfully generated {total_new} constraints")

        print("\nMerging with existing constraints...")
        merged_data = self.merge_with_existing_constraints(new_constraints)

        print("\nWriting to JSON file...")
        if not self.save_constraints_to_json(merged_data):
            print("Write failed")
            return False

        print("\nVerifying JSON integrity...")
        if not self.verify_json_integrity():
            print("Integrity verification failed")
            return False

        print("\nConstraint generation and save workflow completed")
        return True


# Legacy alias — keep import path stable for the AutoGen workflow
ChineseConstraintGeneratorNoUnicode = ConstraintGenerator


def main() -> None:
    constraints_file = str(
        Path(__file__).resolve().parent / "predefined_constraints.json"
    )
    generator = ConstraintGenerator(constraints_file)
    success = generator.run_complete_generation_and_save()
    if success:
        print("\nAll tasks completed. Constraints successfully written to JSON file.")
    else:
        print("\nWorkflow execution failed")


if __name__ == "__main__":
    main()
