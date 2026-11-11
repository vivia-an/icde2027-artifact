# Post-freeze fixes (155)

Fixes merged after the frozen rule directory (`frozen_20260712`, rule files last modified 2026-07-12) and before the 2026-10-25 checksum. Repositories: Megatron-LM 93, DeepSpeed 51, PyTorch 9, OLMo-core 2.

These 155 did not go into the frozen catalog and were not inputs to the later catalog version. The frozen check on this list is 10/155. The next version, read from later framework source, is 54/155 on the same list.

`cases.json` is the 155 merged fixes. `static.json` is the same 155 with the static reading: 16 not silent, 36 invisible to the hooks, 83 change recorded state with no frozen rule, 13 match a frozen rule, 7 undecidable. Replay logs are not in this directory.
