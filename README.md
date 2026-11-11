# TrainAudit ICDE 2027 supplemental material

Code, frozen rules, and case lists for the ICDE 2027 research-track paper *TrainAudit: Silent Error Detection for LLM Training via Admitted Guarded Relational Constraints*.

ICDE scores whether this material is available. The reproduction entries are the three commands below. `MANIFEST.md` maps each paper object to a path. Run logs are not in this repository. A rerun uses the scripts, the reader's Megatron checkout, and the reader's GPUs.

## Entries

Compiled SQL check. One of the 36 cells, 300 steps, then the frozen specs. `ddp2`, `zero1`, `fsdp2`, `pp2`, and the two bf16 layouts use 2 GPUs. `ddp4` and `tp2dp2` use 4.

```bash
export MEGATRON_HOME=/path/to/Megatron-LM
python detect/launch_megatron.py --layout ddp2 --injection f_no_accum_divisor --seed 0 --out /tmp/ta-run
python detect/score_guarded.py --layout ddp2 --run /tmp/ta-run
```

Rule tree. Coarse rules run first. Children of a failed node run next. `--provider mock` does not call a model.

```bash
python detect/rule_tree/run_rule_tree.py /path/to/trace.duckdb --dp 2 --tp 2 --provider mock
```

Mining. One frozen rule file is one direction. The accept gate needs `DEEPSEEK_API_KEY` and healthy traces, which are not in this repository.

```bash
export DEEPSEEK_API_KEY=...
python mining/run_miner.py
```

## Layout

| Path | Role |
|---|---|
| `detect/launch_megatron.py` | 36-cell Megatron job, 300 steps |
| `detect/score_guarded.py` | Compile the frozen specs to SQL and check the trace |
| `detect/rule_tree/` | Coarse-to-fine rule tree |
| `mining/run_miner.py` | Walk the frozen rule files and accept rules |
| `catalog/frozen_20260712/rules/` | Frozen rule modules, including the 18 mining directions |
| `catalog/check_specs/` | The 36 cells and their guarded specs |
| `catalog/postfreeze_155/` | 155 fixes merged after the freeze |
| `corpus/silent_evidence_392.json` | 392-record corpus manifest |

`trainaudit/` is an earlier library tree. The commands above do not start there.

## Not included

Megatron-LM itself, trace logs, healthy-trace databases, and the later catalog update that scores the same 155 fixes. Set `MEGATRON_HOME` to a local Megatron checkout. This fork also imports `vtimeline`; set `VTIMELINE_SRC` if that package is not already on `PYTHONPATH`.

## License

MIT. See `LICENSE`.
