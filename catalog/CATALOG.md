# Frozen rule directory

This is the rule directory frozen on 2026-07-12. The archive checksum is in `frozen_20260712/SHA256SUMS`. No run logs are included.

The files are named `T0_*` and `T1_*`. There are 32 rule modules. A separate directory named T01–T35 was not in the local tree. Do not rename these files to T01–T35.

`postfreeze_155/` is the 155 fixes merged after this freeze. They are not part of the frozen rule modules.

`check_specs/injections.json` is the 36-cell list. Nine fault mechanisms are placed only on the layouts where they apply:

| Fault | Layouts | Cells |
|---|---|---:|
| d2 sum without divide | ddp2, ddp4, zero1, tp2dp2, fsdp2, ddp2bf16, fsdp2bf16 | 7 |
| g2 parameter left out of DDP | ddp2, ddp4, zero1, tp2dp2 | 4 |
| a rank-local parameter change | ddp2, ddp4, zero1, fsdp2, tp2dp2, pp2, ddp2bf16, fsdp2bf16 | 8 |
| b skip a parameter group | ddp2, ddp4, tp2dp2 | 3 |
| b2 ZeRO-1 owner learning rate | zero1 | 1 |
| c tensor-parallel norm left partial | tp2dp2 | 1 |
| f missing loss divisor | ddp2, ddp4, zero1, fsdp2, tp2dp2, pp2 | 6 |
| h skip optimizer step | ddp2, ddp4, fsdp2, tp2dp2, pp2 | 5 |
| i FSDP2 stale gather | fsdp2 | 1 |
| **Total** | | **36** |

The reported result is 30 of these 36 cells. The six misses are all `f`. Seeds score each cell; they are not another 36. The 32 `T0_*` and `T1_*` files are the checker, not these 36 cells. `check_specs/specs/` holds the check templates for the eight layouts. These files are templates, not execution logs.

A GPU rerun is `detect/launch_megatron.py` (Megatron GPTModel, 300 steps; set `MEGATRON_HOME`), then `detect/score_guarded.py` on the trace it writes. The second script applies these guarded specs. This package does not contain Megatron or the traces. `detect/detect_gpu.py` only starts that Megatron job.
