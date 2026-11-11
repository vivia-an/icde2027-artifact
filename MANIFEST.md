# Claim map

Start at the repository `README.md`. These paths are the paper objects.

| Paper object | Path |
|---|---|
| 36-cell Megatron job, 300 steps | `detect/launch_megatron.py` |
| Detection with the frozen guarded specs | `detect/score_guarded.py` |
| Coarse-to-fine rule tree | `detect/rule_tree/run_rule_tree.py` |
| 36 layout-fault cells | `catalog/check_specs/injections.json` |
| Guarded specs for those cells | `catalog/check_specs/specs/<layout>/guarded/` |
| Frozen rule modules | `catalog/frozen_20260712/rules/` |
| 155 fixes merged after the freeze | `catalog/postfreeze_155/` |
| 392-record corpus manifest | `corpus/silent_evidence_392.json` |

Run logs are not in this tree. A GPU rerun is `detect_gpu.py` followed by `score_guarded.py`.
