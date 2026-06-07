# GB10 decode performance — research notes

Working notes from optimizing DeepSeek-V4-Flash `--mtp` decode on the DGX Spark
(GB10, sm_121a). Captures the profiling method, the bottleneck diagnosis, every
lever tried (banked / dead / pending), upstream ideas worth stealing, and two
designed-but-unbuilt directions (adaptive cascade depth, CUDA capture graphs).

Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8` + MTP support GGUF.
Hardware: GB10 / sm_121a, ~270 GB/s LPDDR5X, 48 SMs, driver 580.142, CUDA 13.0.

## TL;DR

- Decode is **memory-latency-bound at ~34% of peak bandwidth** — not compute-,
  tensor-core-, or launch-issue-bound. The warps are resident but stalled on
  weight reads.
- The wins that landed (PR #15): **cascaded N=3 MTP** + **share-warp Q8 weight
  hoist** → 18.85 → 20.5 t/s (knight), 17.45 → 22.2 t/s (long context),
  byte-identical output.
- Most "obvious" further levers are **dead ends** here (Q8_K already banked,
  Q8_0 wide loads falsified by misalignment, expert HBM residency thrashes,
  MoE occupancy is bandwidth-capped). Independently confirmed by upstream
  cghart's `gb10-cuda-notes.md` (PR #236).
- The two live directions are **adaptive cascade depth** (robustness, designed
  below) and a **`__launch_bounds__` probe on `moe_down`** (pending Spark time).

## The bottleneck

Measured with `nsys --gpu-metrics-set=gb20b` during decode (GB10 reports as
"Blackwell GB20B"):

| metric | value | reading |
|---|---:|---|
| SMs Active | 84% | SMs occupied |
| SM Issue | **8%** | …but issuing instructions only 8% of cycles → **stalled** |
| Tensor Active | 0.8% | not tensor-core bound |
| Compute Warps in Flight | 34% | moderate occupancy |
| GPU busy (kernel coverage) | 71% | ~29% idle = launch/sync gaps |

Effective bandwidth ≈ **90 GB/s = ~34% of the ~270 GB/s peak** (weight bytes
moved per forward ÷ GPU-busy time). We are **not** at the bandwidth wall — the
stall is memory *latency* at moderate occupancy, plus per-token CPU/sync gaps.

### Per-kernel register / occupancy (sm_121a, `nvcc -Xptxas=-v`)

| kernel | %GPU | regs/thread | smem | occupancy | limiter |
|---|---:|---:|---:|---:|---|
| `matmul_q8_0_preq_batch_share_warp<2/3>` | 22.6% | 48 | 0 | ~83% | bandwidth/latency (not occupancy) |
| `moe_down_expert_tile8_row32` | 18.3% | **168** | 18.7KB | **~17%** | registers |
| `moe_gate_up_mid_expert_tile8_row32` | 15.7% | 96 | 39.5KB | ~33% | shared memory |
| cuBLAS f16 (KV/indexer, n_tok>1) | ~19% | — | — | — | tensor-core, bandwidth on weight read |
| `matmul_q8_0_preq_warp8` (n_tok=1) | 5.8% | 48 | 0 | ~83% | MTP-draft GEMV |

Key surprise: the **share-warp kernel is already ~83% occupancy** (48 regs) — it
is *not* occupancy-limited, so split-K / CTA-parallel rewrites of it gain
nothing and would break bit-identity. The genuinely occupancy-starved kernel is
**`moe_down` (168 regs → 17%)**.

## Profiling toolbox (reproducible)

```
# kernel + API summary
nsys profile -o /tmp/p -t cuda --sample none ./ds4 -m ds4flash.gguf --mtp <mtp> -p knight -n 48 --temp 0 --nothink -sys ""
nsys stats --report cuda_gpu_kern_sum /tmp/p.nsys-rep
# (cuda_api_sum for launch/sync counts)

# GPU hardware metrics (SM issue / occupancy / tensor)  -- GB10 set is gb20b, NOT gb10b
nsys profile --gpu-metrics-devices=0 --gpu-metrics-set=gb20b --gpu-metrics-frequency=20000 -o /tmp/gm ...

# per-kernel register counts
/usr/local/cuda/bin/nvcc -O3 --use_fast_math -gencode=arch=compute_121a,code=sm_121a -Xptxas=-v -c -o /tmp/x.o ds4_cuda.cu
```

- `ncu` lives at `/usr/local/cuda/bin/ncu` (not on PATH) but its hardware
  counters hit a profiling-permission error here (needs admin counter access).
  The `gb20b` nsys metric set is the workable substitute.
- glint's `scripts/perf/{hbm_report,roofline,nsys_top}.py` (sibling repo) port
  cleanly for per-kernel %-of-peak headroom if adapted to ds4 kernel names.
- MTP accept telemetry: `DS4_MTP_TIMING=1` prints `combined drafted=N committed=M`
  per iter — the source for tokens/iter and accept-rate analysis.

## Lever ledger

### Banked (PR #15)

- **Cascaded N=3 combined-forward** — verify `[first, draft0, draft1]` in one
  batched pass; +prefix2 rollback; per-row argmax fix. 1.79 → 2.24 tok/iter.
- **share-warp Q8 weight hoist** — load each row's 8 int32 weight words once per
  block, reuse across N_TOK tokens. Bit-identical.

### Dead ends (measured)

| lever | result | why |
|---|---|---|
| Q8_K activations | already in ds4 | `q8_K_quantize_kernel` used by both decode + batch MoE; no main/MTP mismatch to fix (v1's +12-18pp came from fixing that mismatch) |
| Q8_0 weight wide loads (uint4) | neutral, reverted | `qs` at byte offset 2 in the 34-byte block → unaligned. Repack to aligned split layout was byte-identical but **neutral**: warps stall on the DRAM fetch, not the byte-assembly. cghart's notes call the same idea "falsified." |
| Routed-expert HBM residency | 20.5 → **10.7** t/s | caching the 65 GiB of experts evicts the reclaimable mmap page cache → cold reads fault from disk |
| MoE tile4 (more occupancy) | 18.7 → 16.4 t/s | smaller tile = less weight reuse = more bytes; MoE is bandwidth-bound |
| F16 share-warp small-N kernel | -1.3 t/s | 1-warp-per-row starves occupancy on small out_dim |
| share-warp CTA-parallel/split-K | not viable | already ~83% occupancy (48 regs); would break bit-identity for no gain |
| `cudaDeviceScheduleSpin` | neutral | synchronous `cudaMemcpy` readbacks already block; wakeup latency isn't the cost |
| `end_commands` sync removal | neutral | redundant drains; readbacks self-sync via `cudaMemcpy` |

### Pending (needs Spark time)

- **`__launch_bounds__(256, 2)` on `moe_down`** — it runs at 168 regs → 17%
  occupancy. This is the clean, unconfounded occupancy test (tile4 confounded
  occupancy with bytes). Bit-identical. Mirrors upstream PR #145. Expected
  modest — MoE is likely bandwidth-saturated — but it definitively settles
  "latency vs bandwidth" for the MoE kernels.

### Settled — fusion mostly already done

ds4 already implements v1's Tier-1 fusion stack: `moe_gate_up_mid_*` (gate+up),
`qkv_rms_fused` (Q/K/V pair), `matmul_q8_0_hc_expand_*` (attn-out + HC expand),
`hc_split_weighted_sum_norm_fused` (HC sum+norm). Remaining gaps (norm+quantize,
draft-local `repeat+add`) are marginal and aimed at launch-count, which isn't
the bottleneck (latency-bound, kernels issue ahead on the default stream).

## Upstream ideas worth stealing (antirez/ds4)

Both #236 and #144 are closed as *reference branches*, not rejections.

- **#236 — CUDA: GB10 decode optimizations** (closed, reference). Ships
  `docs/hardware/gb10-cuda-notes.md` whose findings match ours exactly
  (Q8_0 uint4 falsified, MoE tile/CTA not the lever, F16 GEMV at 44% peak
  recovered by uint4). Its kernels — **q8_0 GEMV CTA-parallelism** and
  **F16 GEMV uint4/vec8** — target **single-token (n_tok=1) decode**, which in
  our cascade is *only* the MTP-draft path (the n_tok=3 verify uses cuBLAS +
  share-warp). So they'd speed the drafts (byte-safe: emitted stream is
  verifier-defined) but not the verify hot path.
- **#121 — skip ordered f16 matmul on Blackwell** (+14/-2). On sm≥11 the
  256-thread F16 reduction beats the 32-thread ordered path. Trivial, A/B-gated,
  hits the n_tok=1 F16 matmuls (our MTP drafts, 2×/iter). The cheapest probe of
  the draft-GEMV slice.
- **#144 — Gx10 cuda graph decode** (closed by mistake). A real, compact (+1089)
  CUDA-graph decode implementation — the reference for the capture-graph work.
- **#243 — turbo3 3-bit KV cache**. 4.75× smaller KV rows; +1.9% decode at 8K,
  grows with context. Helps long-context (KV-bandwidth-bound).
- Skip for decode: **#187** (mmq + VMM) is 5.9× *prefill* on discrete Blackwell
  but the author notes GB10's LPDDR5X is bandwidth-limited so it doesn't
  translate to decode.

## Direction 1: adaptive cascade depth (designed, unbuilt)

The cascade does N=3 unconditionally, but `draft1` only accepts ~58% (lower on
unpredictable prose). When acceptance is low the 2nd MTP draft + 3rd verify row
cost ~13 ms/iter for ~0 extra tokens — *slower* than N=2.

### Break-even

With p0 = draft0 accept, p1 = draft1 conditional accept, Δ = N=3's extra cost,
C₂ = N=2 iter cost:

> N=3 wins ⟺ **p1 > p1\* = (1 + p0)·Δ / (p0·C₂)**

Measured (p0=0.79, Δ≈13 ms, C₂≈95 ms): **p1\* ≈ 0.31**. We're at p1≈0.58 → N=3
wins. On content where p1 < 0.31, N=2 wins.

### Mechanism

- Track an EWMA of p1 (sample = `[commit == 2]` on iters that ran N=3), α≈0.1.
- `draft_cap = (ewma_p1 > 0.35) ? 2 : 1` at the decision site (`ds4.c:~18611`),
  threshold a margin above the 0.31 break-even (hysteresis).
- **Periodic re-probe**: in N=2 mode you stop sampling p1, so force one N=3 iter
  every ~16 to keep the estimate live (amortized ~0.8 ms/iter). Cold start: N=3.
- Refinement: Δ and C₂ grow with context; a v2 could measure them at runtime
  (we already have per-iter timing) for a self-calibrating threshold.

### Properties

- **Bit-safe**: N=2 and N=3 both emit the verifier's greedy stream (already
  verified byte-identical), so interleaving them doesn't change the output —
  only the speed.
- **~30-40 LOC**, `ds4.c` only, no kernels. Flag-gated (`DS4_MTP_CASCADE_ADAPTIVE`
  opt-in; `DS4_MTP_NO_CASCADE` stays the kill switch).
- **Value is robustness, not headline throughput**: on the benched prompts
  (high p1) it just stays at N=3. It makes the cascade *strictly never worse
  than N=2*, so the cascade becomes safe to default-enable on any workload, and
  it wins on mixed (code+prose) content. Validating the fallback needs a
  genuinely low-accept prompt + Spark time.

## Direction 2: CUDA capture graphs (scout)

Targets the ~29% launch/sync idle. Mechanism proven on this chip: glint's
`conditional_graph_sm121` spike replayed `cudaGraphCondTypeIf` **1000/1000
drift-free, p50 16 µs** on GB10/CUDA 13.0.

- **MoE is *not* the blocker** I feared: `tile_capacity` is a function of
  `pair_count = n_tokens × n_expert_used` (fixed per n_tokens), the grid
  launches at that fixed max, and kernels early-return on a device-read
  `*tile_total`. So the MoE runs on-device with fixed grids → capturable.
- **Real obstacles**: (1) 8 cuBLAS call sites need algo pre-select/cache
  (v1 ate a segfault from allocating inside a captured stream); (2) default
  stream + synchronous `cudaMemcpy` everywhere needs a capture-stream + async
  restructure; (3) **captured-graph drift on sm_121a** is documented (v1
  single-digit %), and breaks the byte-identical guarantee the stack rests on —
  mitigating it is the whole `platform_sm121` toolkit (warmup, grid.y=1 splits,
  ScheduleSpin, pinned ReplaySlots).
- **Gain**: bounded by the 29% idle, of which only launch-bubble + sync is
  reclaimable. v1 shipped captured-graph spec decode for **+1.03× over plain** —
  a subsystem's worth of work for a modest, drift-risky win.
- **Recommendation**: not part of the current push. If pursued, the decisive
  cheap test is a measurement-only prototype — capture just the N=3 verify
  forward (Option B: host keeps sample/accept), run the byte-identical/drift
  gate + bench — before committing to the full subsystem. PR #144 is the
  reference.

## Open questions

- Is `moe_down` latency-bound at 17% occupancy (→ `__launch_bounds__` helps) or
  bandwidth-saturated (→ it doesn't)? The pending probe settles it.
- Does adaptive cascade depth actually recover the N=2 advantage on a real
  low-accept prompt? Needs a prose/unpredictable workload + Spark.
- Residual sm_121a captured-graph drift source (open upstream too).
