# Scout: verify-forward MoE GEMM occupancy on GB10 — CLOSED (little headroom)

**Status: investigation closed 2026-05-29. The verify MoE GEMMs are already well-tuned;
the obvious occupancy lever is empirically a 17% regression. Do not re-open without a
new angle (see "If anyone re-opens this").**

## Why we looked

Across the broader perf work the consistent finding was that MTP decode tps lives in the
**n_tok=3 combined verify forward**, and within it the MoE GEMMs dominate (an internal
`DS4_CUDA_MOE_PROFILE` bracket put gate-up + down at ~94% of the MoE call). Earlier
profiling rejected graph-capture / device-accept / WMMA. The MoE GEMMs were the last
plausible decode-tps lever, so this scout asked: **are the verify MoE GEMMs
occupancy-limited, and would raising occupancy help?**

Answer: they are low-occupancy (~33%) **by design**, and raising it makes them slower.

## What actually runs at verify (corrected twice)

Two wrong turns worth recording so nobody repeats them:

1. **ncu cannot profile these kernels.** Its kernel-replay segfaults the ds4 process
   mid-stream (`==ERROR== returned an error code (11)`). A clean `-c 10` capture only
   succeeds on the *earliest* launches — which are the **n_tok=1** decode path
   (`moe_gate_up_mid_decode_lut_qwarp32`, grid `(16,6)`, wave 0.33, grid-starved). That is
   **not** the verify path and its occupancy story does not transfer. `--launch-skip` to
   reach steady state re-triggers the crash.
2. **The verify kernel is not `sorted_p2`.** At n_tok=3, `use_expert_tiles` is on by
   default (`ds4_cuda.cu:10628`: `use_sorted_pairs && !getenv("DS4_CUDA_MOE_NO_EXPERT_TILES")`
   — *not* gated on n≥128), so the gate-up dispatch (`ds4_cuda.cu:10772`) takes the
   **`moe_gate_up_mid_expert_tile8_row32_kernel`** branch and down takes
   **`moe_down_expert_tile8_row32_kernel`**. `sorted_p2` is only the fallback when expert
   tiles are disabled.

**Method that worked:** `ptxas -v` (compile-time registers/smem, no runtime profiling) +
reading the dispatch + an env-flag A/B for the decode-tps comparison.

## Occupancy of the verify kernels

`ptxas -v` on `ds4_cuda.cu`, GB10 limits (48 warps/SM, 65536 regs/SM, block = 256 = 8 warps,
~100 KB smem/SM):

| Verify kernel | regs/thr | smem | blocks/SM | theo occ | limiter |
| --- | --- | --- | --- | --- | --- |
| `moe_gate_up_mid_expert_tile8_row32` | 126 | **38.6 KB** | 2 | **33%** | smem (regs co-limit) |
| `moe_down_expert_tile8_row32` | 128 | 18.3 KB | 2 | **33%** | registers |
| lean alt `moe_gate_up_mid_sorted_p2_qwarp32` | 62 | 0 | 4 | 67% | registers |
| lean alt `moe_down_sorted_p2_qwarp32` | 63 | 0 | 4 | 67% | registers |

Gate-up is **smem-capped at 2 blocks/SM**: 38.6 KB × 3 > ~100 KB, so even zero register
pressure could not raise its occupancy without cutting the shared-memory stage.

## The decisive A/B — occupancy is a dead lever

Same prompt, ctx-alloc 200000, kv-restore 9045 tok, MTP draft 2, gen 256, temp 0,
`DS4_CUDA_MOE_NO_ATOMIC_DOWN=1`, 3 runs each:

| Path | env | occupancy | decode tps (3 runs) | mean |
| --- | --- | --- | --- | --- |
| default `expert_tile8` | — | 33% | 20.42 / 21.15 / 19.87 | **20.48** |
| lean `sorted_p2` | `DS4_CUDA_MOE_NO_EXPERT_TILES=1` | 67% | 16.76 / 17.56 / 16.81 | **17.04** |

**Doubling occupancy is ~17% slower.** The `expert_tile8` kernel spends its 38.6 KB smem
staging weights for reuse across the expert tile; that reuse cuts the bandwidth that
matters more than extra warps hide latency. The default dispatch is the right choice —
do not set `DS4_CUDA_MOE_NO_EXPERT_TILES=1`.

## Verdict

❨`✗`❩ **Raise occupancy** — closed. Gate-up is smem-bound (can't); going lean is −17% (A/B).
❨`✗`❩ **WMMA / tensor cores** — closed earlier (compute SOL 10-20%; can't accelerate unused compute, and dequant to f16 would inflate the binding bandwidth).
❨`~`❩ **cp.async double-buffer on the down kernel** — the one genuinely-unexplored lever (overlap next tile's weight load with current compute). But reuse already captures most of the bandwidth benefit, occupancy is only 2 blocks/SM, and ncu can't measure it here (replay crash). Frame as a research spike with low expected payoff, not a likely win.
❨`~`❩ **Register-shave the down kernel** (128 regs → ~85 for 3 blocks/SM) — down is register-limited (not smem), so this *could* lift its occupancy, but it's bandwidth-sensitive like gate-up so the upside is likely single-digit-% and risks spills.

**The verify-forward MoE GEMMs have little headroom left on GB10.** They run the right
kernel and the obvious lever regresses. This matches the session-wide conclusion: ds4 on
GB10 is bandwidth-bound and already well-tuned — the easy decode-tps wins are not here.

## If anyone re-opens this

A real attempt needs a *new* angle, not occupancy:
- A cp.async double-buffer prototype on `moe_down_expert_tile8_row32`, validated by the
  deterministic token-diff gate (`DS4_CUDA_MOE_NO_ATOMIC_DOWN=1`) + an A/B decode bench —
  treat as a spike; revert unless it clears run-to-run variance (~1 tok/s here).
- A weight-layout change that raises reuse density without growing the 38.6 KB stage.
- Measuring whether the down kernel's register pressure is spill-driven and shaveable
  without losing throughput (ptxas `-v` + a `__launch_bounds__(256, 3)` A/B).

Do NOT: re-run ncu expecting clean verify-kernel occupancy (replay crash), propose WMMA,
or disable expert tiles.

## Scope of this scout (for the record)

- Files touched: none (read-only scout + env-flag A/B; one throwaway `ptxas -v` compile).
- Named units examined: `moe_gate_up_mid_expert_tile8_row32`, `moe_down_expert_tile8_row32`,
  `moe_gate_up_mid_sorted_p2_qwarp32`, `moe_down_sorted_p2_qwarp32`,
  `moe_gate_up_mid_decode_lut_qwarp32` (n_tok=1), dispatch `ds4_cuda.cu:10628-10905`.
- Verification: `ptxas -v` register/smem; 3-run A/B decode bench; dispatch read.

## References

- Memory `[[moe-gemm-occupancy-bound]]` — the corrected finding (this doc's TL;DR).
- Memory `[[mtp-graph-capture-rejected]]` — verify forward is ~99% of the spec-iter.
- Memory `[[tokendiff-needs-no-atomic-down]]` — the deterministic-gate requirement used here.
- `misc/DS4-IMPROVEMENTS-CATALOG.md` — the broader catalog this closes out the C11/GEMM line of.

Reproduce the A/B:
```
DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 ./ds4-bench --cuda --warm-weights -m ds4flash.gguf \
    --mtp /home/trevor/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
    --kv-restore ~/.ds4/kvcache/b9dbb307b5f4150cf3b1925c92441a015734989c.kv \
    --ctx-alloc 200000 --gen-tokens 256 --temp 0
```
Append `DS4_CUDA_MOE_NO_EXPERT_TILES=1` (prefix) for the lean-kernel arm.
