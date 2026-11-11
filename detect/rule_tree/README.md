# Rule-tree detection

This is the second checker. `run_rule_tree.py` calls `sdccheck.llm_orchestrator.LLMOrchestrator.run_with_hierarchy`.

The hierarchy builder groups constraints into a tree. Level 0 is the coarse rule. Finer rules are its children. `concurrent_executor.execute_tree_level_by_level` runs a level together. Only children of a failed node run at the next level. `root_cause_analyzer.py` reads that tree after the run.

Each node asks a model to write the SQL and to judge the rows. The model-free compile path stays in `detect/score_guarded.py`. No API key file is included. `--provider mock` does not call a model. A real provider needs the key in the environment and `llm_config.yaml` beside this package.

```bash
python detect/rule_tree/run_rule_tree.py /path/to/trace.duckdb --dp 2 --tp 2 --provider mock
```
