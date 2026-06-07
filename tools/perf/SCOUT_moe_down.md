# SCOUT — moe_down occupancy (P1)

Target: `moe_down_expert_tile8_row32_kernel` (`ds4_cuda.cu:9671`, launched `:10579`
as `<<<tgrid, 256>>>`). The #1 decode lever per the gamut report.

## Gamut signal

19% of decode time · **15% of peak read BW** · 168 regs → ~12–16% occupancy ·
ncu stalls: **barrier 53%**, long_scoreboard 31%. Not bandwidth-limited (~6×
headroom to the 236 GB/s wall) — occupancy-starved and barrier-bound.

## Findings

- **The reduction is already warp-shuffle.** `quarter_warp_sum_f32`
  (`:8406`) does an 8-lane `__shfl_down_sync` (intra-warp, no smem, no barrier).
  So "replace the smem tree-reduction with shuffles" is a **no-op — it's already
  done.** The sibling tile4/tile16/row2048 variants all use the same shuffle.
- **The single `__syncthreads()` (`:9709`) guards the activation smem stage**, not
  a reduction: it cooperatively loads the per-token quant blocks into
  `__shared__ cuda_block_q8_K sxq[8][8]` (`:9693`, 18.25 KiB) so the dot loop
  reads activations from smem. It dominates stalls only **because** 8 resident
  warps can't hide it.
- **168 regs root cause** = `dev_dot_q2_K_q8_K_block8` (`:8353`) inlined with
  8-wide live accumulators: `int isum[8] + int summs[8] + float acc[8] +
  const cuda_block_q8_K *ys[8]` (`:8368-8373`), all live across the hot
  `for (b=lane; b<midq_blocks; b+=8)` loop (`:9715`). The `[8]` tile width *is*
  the register pressure; the whole `moe_down_expert_*` family shares it.
- **Occupancy arithmetic** (sm_121a / GB10: 64K regs/SM, **48 warps/SM = 1536
  threads/SM**). 168 regs × 256 = 43,008 regs/block → ⌊65536/43008⌋ = 1 block by
  registers. 1 block × 8 warps / 48 = 16.7%.

## ncu limiter confirmation (measured — `tools/perf/runs/`)

`launch__occupancy_limit_*` (blocks/SM each resource permits) for the real
decode launches:

| limiter | moe_down | reading |
| ------- | -------: | ------- |
| registers | **1** | 168 regs pins it |
| shared mem | **1** | 18.25 KB pins it *too* |
| warps | 6 | (6 blocks × 8 warps = 48 = the warp cap) |
| blocks (HW) | 24 | not binding |
| **achieved occ** | **16%** | ≈ 1 block resident |

The `occlim_smem=1` reading looked like a smem co-limit — **but it was a
carveout artifact, not a hard limit** (the experiment below disproved it). With
168 regs the driver only provisioned smem for 1 block; `__launch_bounds__` told
the compiler to target 2, the carveout grew, and `occlim_smem` jumped to 5.
Registers were the sole binder all along.

## Result (measured) — `__launch_bounds__(256, 2)` ALONE

| metric | before | after | |
| --- | ---: | ---: | --- |
| regs/thread | 168 | **128** | capped |
| achieved occupancy | 16% | **31.5%** | ≈2× (2 blocks/SM) |
| `occlim_smem` (blocks) | 1 | **5** | carveout grew — never a real limit |
| barrier stall | 53% | **42%** | more warps hide the sync |
| long_scoreboard | 31% | 22% | |
| decode t/s (knight) | 18.72 | **19.73** (median/3) | **+5.4%** |

`./ds4_test --long-context` → **OK** (bit-safe; decode output identical). The
smem-drop (fix C) is **not needed for this gain** — it's now only relevant for
pushing past 31.5% (e.g. `(256,3)` → 50%).
- **Decode uses this kernel** (confirmed): `use_down_row2048`/`tile16` require
  `n_tokens ≥ 128` (prefill); decode (n_tok=2–3 under MTP) falls to the `tile8`
  branch (`:10579`).

## Fix (A) — `__launch_bounds__(256, N)` · RECOMMENDED, bit-safe

No `__launch_bounds__` exists in the file yet.

Register limit only (must be paired with the smem cut — see above). Occupancy %
is vs 48 warps/SM:

| directive | reg ceiling = 65536/(256·N) | reg-limit blocks/SM | occ if smem also freed |
| --------- | --------------------------: | ------------------: | ---------------------: |
| none      | 168 | 1 | 16% |
| `(256,2)` | ≤128 | 2 | **33%** — first try (gentle cut, low spill) |
| `(256,3)` | ≤85  | 3 | 50% (spill likely — live state ~40 regs) |

**Bit-identity: SAFE.** `__launch_bounds__`/register count never reorder float
adds; spills store/reload the same bits. The accumulation order in
`dev_dot_q2_K_q8_K_block8` (`:8391-8393`) is untouched.

## Fix (C) — drop/shrink the smem activation stage · REQUIRED to unpin

Since smem co-limits at 1 block, (A) needs (C). The `sxq[8][8]` stage (`:9693`)
caches activation quant blocks; with `midq_blocks ≤ 8` they're tiny and
L2-resident. **Dropping the stage** (read `xqb` from global, skip the
`__syncthreads`) zeroes the smem → smem limit jumps, register limit (with (A))
binds at 2–3 blocks. Bit-safe: `sxq[p][b]=xqb[p][b]` is a byte copy; the dot
consumes identical bytes in identical order. Trade: a re-read of small
activations (weights dominate BW, so cheap).

## Fix (B) — shuffle reduction · N/A (already shuffle-based)

Nothing to do. If barriers still dominate after (A), the bit-safe alternative is
to **drop the smem activation stage** (read `xqb` from global; activations are
tiny, `midq_blocks ≤ 8`) — removes the barrier entirely, trades a re-read. Or
borrow the **row-span amortization** from `tile16_row2048` (`:9846`): one stage
barrier across 64 row-groups instead of per 32 rows.

**Never** widen/reorder the shuffle (e.g. 32-lane) or change
`dev_dot_q2_K_q8_K_block8`'s accumulation order — that breaks byte-identical decode.

## Plan

1. ✅ **DONE**: `__launch_bounds__(256, 2)` → 16%→31.5% occ, +5.4% decode,
   bit-safe. (commit on the gb10-decode-perf stack.)
2. ❌ **REJECTED**: `(256, 3)` (≤85 regs) → heavy spill, `moe_down` ms +142%,
   decode −18% (see `SQUEEZE_LOG.md` #2). `(256,2)` is the sweet spot.
3. ❌ Family rollout unsafe: `tile16*` use `[16]`-wide accumulators (>168 regs);
   `(256,2)`'s ≤128 ceiling would spill them like #2.
4. `moe_gate_up` is reg+smem co-locked at 2 blocks — see `SCOUT_moe_gate_up.md`.

The launch_bounds occupancy lever is fully explored; `moe_down(256,2)` is the
banked win. Deeper gains need kernel restructuring — see `SQUEEZE_LOG.md`.

**Scope** — Files: `ds4_cuda.cu` (`+1` attribute/kernel). Verify: `make
cuda-spark` · `./ds4_test` bit-exact decode · `make cuda-regression` ·
`gamut.py --ncu` recheck (occ↑, barrier%↓, ms↓). Risk: public API no · migration
no · cross-module no · reversible yes · bit-identity yes.
