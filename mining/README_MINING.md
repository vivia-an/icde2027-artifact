# Mining

`run_miner.py` walks the rule files in `catalog/frozen_20260712/rules/`. Each file is one direction. The expert proposes inside that direction, counterexamples are checked, and a rule is kept only if the accept gate passes.

```bash
export DEEPSEEK_API_KEY=...
python mining/run_miner.py
```

One direction only: `SDC_ONLY_DIRECTION=T0_no_nan_inf`.
