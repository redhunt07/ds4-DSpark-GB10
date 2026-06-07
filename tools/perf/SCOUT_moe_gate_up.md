# SCOUT — moe_gate_up occupancy (P2)

Target: `moe_gate_up_mid_expert_tile8_row32_kernel` (`ds4_cuda.cu:8997`, launched
`:10389` as `<<<tgrid, 256>>>`). 2nd decode lever.

## Gamut signal

16% of decode time · ~35% of peak read BW · ncu stall long_scoreboard 50% ·
~31% occupancy.

## Findings

- **38.6 KB smem is 94.5% `sxq[8][16]` — staged *activations*, not weights**
  (`:9024`). Gate (`gr`) and up (`ur`) weights stream straight from global
  (`:9052-9062`); smem only caches the per-token activation quant blocks for
  reuse across the 32 row-groups. Two small LUTs (`s_iq2_grid` 2 KB,
  `s_iq2_signs` 128 B, `:9025-9026`) are negligible.
- **It's the decode path** (`use_gate_row2048` needs `n_tokens ≥ 128`; decode
  falls to tile8, `:10387`).
- **One `__syncthreads()` (`:9048`)**, guards the smem stage — barriers
  negligible here (unlike moe_down).
- **Reduction is shuffle** (`quarter_warp_sum_f32`, 8-lane) — bit-identity hinges
  on block→lane assignment + the 4→2→1 shuffle tree; don't reorder.

## ncu limiter confirmation (measured)

`launch__occupancy_limit_*` (blocks/SM) on the real decode launches:

| limiter | moe_gate_up | reading |
| ------- | ----------: | ------- |
| registers | **2** | **113 regs/thread** (ncu — higher than the doc's 96) |
| shared mem | **2** | 38.6 KB |
| warps | 6 | warp cap (48 warps/SM) |
| blocks (HW) | 24 | not binding |
| **achieved occ** | **31%** | ≈ 2 blocks × 8 warps / 48 = 33% |

**Co-limited: registers AND smem both cap at 2 blocks/SM.** Like moe_down, a
one-sided fix stalls — cutting smem to allow 3 blocks does nothing while regs
still cap at 2, and vice-versa. To reach 3 blocks (50%) you need **both**
`__launch_bounds__(256,3)` (≤85 regs) **and** a smem cut below ~⅓ of the pool.

## Recommendation (ranked)

1. **Drop the `sxq` activation stage** (read activations from global) → 38.6 KB →
   ~2.2 KB. Bit-safe (byte copy; weights dominate BW so the extra activation
   reads are cheap). Lifts the smem limit well past the register limit.
2. **Pair with `__launch_bounds__(256,3)`** (≤85 regs) so the register limit also
   reaches 3 blocks → ~50% occupancy. One without the other ≈ no gain.
3. Cheaper-but-weaker: switch decode to the **tile4 sibling** (`:8917`,
   `sxq[4][16]`, no LUT smem ≈ 18 KB) — halves smem but also halves token reuse →
   more weight re-reads; net likely negative given weight-BW dominance. Lower
   priority.

`__launch_bounds__` alone is **not** sufficient (smem co-binds); smem cut alone
is **not** sufficient (regs co-bind). Do both.

**Scope** — Files: `ds4_cuda.cu` (drop `sxq` staging block + read-from-global in
the dot loop, ~`+10/-15`; add 1 attribute). Verify: `make cuda-spark` ·
`./ds4_test` bit-exact · `gamut.py --ncu` (occ 31→~50%, long_scb%↓, ms↓). Risk:
public API no · migration no · cross-module no · reversible yes · bit-identity yes.
