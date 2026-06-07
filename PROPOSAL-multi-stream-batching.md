# Proposal: multi-stream batched decode

*Draft for upstream discussion — measurements from DGX Spark (GB10), 2026-06-09.*

## The observation

Single-stream decode is memory-bandwidth-bound: one decode step reads the
model's hot weight set once, whether it serves one sequence or five. On GB10
(~236 GB/s sustained, measured with `tools/perf/membw.cu`) plain greedy decode
sits at **12.6 t/s** — the bandwidth wall, confirmed repeatedly and not
reachable by kernel work (occupancy, WMMA, graph capture were each measured
and ruled out on this box).

But the wall is per *stream*, not per *token*. The engine already proves it:
the MTP combined verify runs 2–5 rows through one batched forward
(`metal_graph_encode_layer_batch`), and its cost grows far slower than
linearly with rows.

## Measurement

New instrumentation (in this branch): `ds4_session_eval_batch_replay`
teacher-forces n known tokens through the production combined-verify forward,
and `ds4-bench --batch-cost` times widths 1–5 round-robin (warmup discarded,
mean±std over 16 steps/width). Rows are consecutive positions of one stream,
so these are the *optimistic* bound for multi-session batching (see Caveats).

GB10, `DS4_CUDA_FAST_VERIFY=1`, IQ2_XXS chat-v2 weights, std ≤ 2.7 ms:

| ctx | width | ms/step | aggregate t/s | × vs width 1 |
| --- | ----- | ------- | ------------- | ------------ |
| 4k  | 1 | 58.2  | 17.2 | 1.00 |
| 4k  | 2 | 79.1  | 25.3 | 1.47 |
| 4k  | 3 | 90.3  | 33.2 | 1.93 |
| 4k  | 4 | 100.7 | **39.7** | **2.31** |
| 4k  | 5 | 134.9 | 37.1 | 2.16 |
| 32k | 1 | 72.0  | 13.9 | 1.00 |
| 32k | 4 | 140.8 | **28.4** | **2.05** |
| 32k | 5 | 183.3 | 27.3 | 1.96 |

Plain-decode baseline, same binary/config/prompt: **12.6 t/s (79.4 ms)** at 4k.
So four rows through one forward is **~3.1× plain single-stream aggregate**
at 4k and ~2× at 32k. Width 5 hits a marginal-cost cliff at both depths
(+34%/row vs +12% at width 4); width 4 is the optimum on this hardware.

Two findings from the same run worth flagging independently:

1. **The width-1 batch path is ~27% faster than the plain decode path**
   (58.2 vs 79.4 ms for the same single token, same config), and it is
   **argmax-exact**: `ds4-bench --batch-check` teacher-forced 240 positions
   and compared greedy argmax plain-vs-batch at widths 2–5 — 0 mismatches in
   308 comparisons, under both the deterministic and the fast verify configs
   (MoE down atomics off, the usual token-diff rule). nsys attribution shows
   the two paths run different kernel families end-to-end: plain decode uses
   the decode-specialist set (`moe_*_decode_lut_qwarp32`, Q8 GEMV `preq`
   family, `attention_indexed_mixed`), while the batch path uses the verify
   set (`moe_*_expert_tile8/16` weight-reuse MoE, `heads8_online` attention,
   cuBLAS f16 GEMMs). This is now implemented behind an opt-in gate
   (`DS4_BATCH_DECODE=1` routes `ds4_session_eval` through the n=1 batch
   forward) and measured end-to-end: **+21.0% at 4k (12.67 → 15.33 t/s,
   256 greedy tokens) and +19.2% at 32k (11.64 → 13.87)**, with the decoded
   token stream identical to plain decode. Under the deterministic-verify
   default the gated path is bit-reproducible run-to-run; under fast verify
   it inherits the same cuBLAS reduction-order nondeterminism the MTP
   combined path already has. (The result also implies the batch path's
   per-token traffic is lower, ~15.5 effective GB/token vs ~18.9 — the
   "wall" was a property of the plain kernel set, not of the model.)
2. The width-5 cliff is unexplained (kernel shape/occupancy boundary
   somewhere in the batch path) — fixing it would extend the scaling.

## What multi-session batching would take

The forward is already row-batched. What's missing is letting rows belong to
*different sequences*:

- **Per-row positions.** The batch path derives row positions as
  `pos0 + row_index` (RoPE tail, SWA ring offset, indexer). Needs a
  `positions[]` array instead — touches ~4 kernel families.
- **Per-row KV bases.** Rows currently share one KV ring per layer. Separate
  sessions need per-row base pointers (or one ring partitioned per session).
- **Server scheduling.** `ds4-server` worker is a FIFO that serializes
  generation through one session. Dequeue-N into one fused step; sessions
  whose queues are empty just don't occupy a row.
- **Already row-independent:** MoE routing, expert kernels, logit readback,
  sampling (host-side per row).

MTP composes per stream: a batched verify over s streams × k rows is the same
mechanism with a wider window, so speculative decode and multi-session
batching stack rather than compete.

## Caveats on the numbers

- Rows in the measurement share one KV at consecutive positions. Real
  multi-session rows attend to disjoint KVs — more traffic at long context
  (the 32k 2.05× will degrade more than the 4k 2.31×) plus the indirection
  cost in the kernels. The go/no-go question was whether the weight-read
  amortization is large enough to pay for that overhead; at 2–3× headroom,
  it is.
- Memory: each extra session adds its KV allocation (a few GB at moderate
  context on this model) — fine for batch 2–4 on 128 GB unified memory.
- Token-exactness held at 100% in the gate above (4k context, 240
  positions); longer contexts and adversarial near-tie prompts should be
  re-gated before relying on it as an invariant.

## Suggested staging

1. Land the instrumentation (`--batch-cost`, `--batch-check`) — measurement
   only, no inference changes.
2. Per-row `positions[]` in the batch forward (mechanical; verify path keeps
   working with `positions[i] = pos0 + i`).
3. Per-row KV bases in the attention/indexer kernels.
4. Server dequeue-N + a second session; A/B aggregate t/s vs serialized.

Each stage is independently testable against the existing correctness gates
(token-diff, tensor-equivalence, perplexity regression).
