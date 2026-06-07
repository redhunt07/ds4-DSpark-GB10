# SQUEEZE LOG — GB10 decode perf

Running log of perf experiments driven by the gamut suite. Each row is a
measured change (or a measured rejection). Baseline = `gb10-on-upstream`
(upstream + the 3 GB10 PRs), knight prompt, MTP draft=2.

Bottleneck (from the gamut report): decode is **memory-latency-bound at moderate
occupancy** — SM-issue ~8%, ~38% of the measured 236 GB/s read ceiling. The
lever is occupancy (hide weight-read latency), not bandwidth.

## Experiments

| # | change | result | verdict |
|---|--------|--------|---------|
| 1 | `moe_down` `__launch_bounds__(256,2)` | regs 168→128, occ 16→31%, barrier 53→42%, `moe_down` −20% ms, **decode +3–5%** | ✅ **KEEP** (bit-safe, `--long-context` OK) |
| 2 | `moe_down` `__launch_bounds__(256,3)` | regs→≤85 → heavy spill; occ 31→46% but `moe_down` ms 307→743 (+142%), stall flips barrier→lg_throttle, **decode −18%** | ✗ reject — spill |
| 3 | server `--mtp-draft` default 1→2 | enables combined-forward on server (was off); output-identical | ✅ keep (free) |
| 4 | narrow-accumulator `moe_down` via tile4 (`DS4_CUDA_MOE_TILE4=1`) | **−19%** decode (16.06 vs 19.78 t/s); output bit-identical to tile8 | ✗ reject — 2× weight reads |

### #4 detail — why narrow accumulators don't help

tile4 is *also* 128 regs (the `[4]` accumulators help less than hoped — pointers
+ dot temps dominate), so **no occupancy gain** over bounded tile8, while halving
pairs/block doubles the blocks → **2× expert-weight reads** (the large weights
evict L2, so it's not absorbed). The genuine "narrow accumulators, same weight
traffic" variant is blocked structurally: `acc[8]` must persist across the whole
K-reduction loop, so shrinking to `[4]` forces either a 2nd weight pass (=tile4)
or smem-resident accumulators (overhead). **`moe_down(256,2)` is the occupancy
optimum for this dot-over-K structure.** Bit-identity confirmed (tiling only
repartitions work across blocks; per-output float order is unchanged).

## Ruled out (with evidence — launch_bounds avenue exhausted)

- **`moe_gate_up`** — occupancy-quantization-locked. 113 regs → 2 blocks; smem
  38.6 KB → 2 blocks (co-limited). Reaching 3 blocks needs ≤85 regs **and** a
  smem cut — compound spill risk; expected EV negative after experiment #2.
- **`moe_down_expert_*` family rollout** — the `tile16*` variants use `[16]`-wide
  accumulators (more regs than tile8's 168), so `(256,2)`'s ≤128 ceiling would
  spill them like #2. Not safe.
- **Other decode kernels** — `share_warp<3>` (48 regs, ~83% occ, at the 86%-BW
  wall), `preq_warp8` / `attention_decode_mixed` (40 regs, 75% occ), `rms_norm`
  (100% occ) have no register headroom. Only `moe_down`/`gate_up` were
  register-starved; #1 banked the one safe win.

## Prefill (profiled ctx 8192, promessi — `gamut --phase prefill`)

Different regime from decode: **SM-issue 33%, tensor 4.4%, SMs-active 99.6%** —
compute-occupied, not memory-latency-stalled. Top kernels: `moe_gate_up
…rowspan<1024>` (33%), `moe_down_expert_tile16_row2048` (26%, **232 regs → 12%
occ**), `attention_indexed_mixed<8,16>` (10%, already 67% warps). An f16 path is
present (`f32_to_f16` + `nvjet`/cuBLAS + cutlass), tensor cores underused (4.4%).

No easy launch_bounds win: `tile16_row2048` at 232 regs would need ≤128 (a
104-reg cut) for 2 blocks — far past the 168→85 cut that spilled in decode #2,
and prefill is more compute-occupied (SMiss 24.8%) so occupancy is a weaker
lever. Prefill is at a structural limit too. (Tooling note: `roofline_estimate`
lacks prefill kernel-name matchers, so prefill `%peakBW` shows `—`.)

## CUDA-graph decode (greenlit — in progress)

Goal: kill the per-token `argmax→embed` host round-trip (~7.7% idle). Built as a
**non-strict, gated (`DS4_GRAPH_DECODE=1`) plain-greedy-only** path (strict + MTP
untouched).

- **Stage 0** (done): `--default-stream per-thread` → all launches on a
  capturable `cudaStreamPerThread`. Verified bit-identical (`long-context` OK),
  ~neutral perf (19.58 vs 19.76 t/s).
- **Stage 1** (done): per-token `cudaStreamBeginCapture(forward)→instantiate→
  replay` in `metal_graph_eval_token_raw_swa`. **Result: capture fires (nsys: 15
  instantiate+launch for n=16), zero errors, output BIT-IDENTICAL to eager —
  no sm_121a captured-graph drift for plain greedy.** This clears the make-or-break
  drift gate. No speedup yet (per-token re-instantiate ~1.67 ms + still host-syncs).
- **Stage 2** (done): device-resident token feedback — argmax→`comp_selected`,
  `embed_tokens_hc`(device) + router device-token (kernels already read
  `tokens[t]`), forward chained through device buffer *views* so the host
  pipelines the whole burst with **one** sync (no per-token sync), then reads the
  token log. **Result: BIT-IDENTICAL to eager, but no speedup (16.47 vs 16.47).**

### CORRECTED finding: decode idle IS host-launch latency — graphs win +6.5%

Plain-greedy decode is 93.4% kernel / **6.6% idle**, and the idle is **73,232
sub-5µs gaps across 1562 kernels/token** — host-launch latency, not GPU stalls.
Decisive timing test (capture forward ONCE, launch the cached exec per token =
1 host submit vs 1562 eager launches): **16.49 → 17.55 t/s, +6.5%.**

My earlier "dead end" was WRONG. Stage 1 missed this because it re-instantiated
(1.67ms) AND re-captured per token — re-capture re-issues all 1562 launches, so
no win. Stage 2 (device-token, no per-token sync) only removed the ~80µs/token
boundary gap (2% of idle), not the intra-forward launch latency. The win needs
**capture-once + launch-cached** (proven above).

**DONE — correct +5% landed.** The winning recipe (`DS4_GRAPH_DECODE=1`,
plain-greedy): each token, re-capture the device-token forward, patch the cached
exec via **`cudaGraphExecUpdate`** (cheap — falls back to re-instantiate only on
topology change: compressor emit / indexer growth), launch as ONE submit, no
per-token sync. Two surprises that broke my earlier "dead end":
1. Recording 1562 kernels in capture mode is **cheaper** than eager-launching them.
2. `ExecUpdate` (vs the 1.67ms instantiate Stage 1 used) is cheap enough per token.

Measured: knight n=48 **16.50→17.41 t/s (+5.5%)**; essay n=192
**16.04→16.86 (+5.1%)** — both **BIT-IDENTICAL** to eager. Ceiling (launch cached,
no re-capture, wrong output) is 17.55 (+6.5%); re-capture costs ~1%.

**Lesson: I twice declared "dead end" prematurely.** Stage 1 (re-instantiate/token)
and Stage 2-eager (device-token, no graph) each tested only half; the win needed
capture-once-style *launch* (graph collapses 73k host-launch gaps) + cheap
per-token ExecUpdate. Scope to extend: the MTP/session decode path (this is wired
into the plain-greedy CLI loop only).

### Stage 3 — MTP combined-forward capture: bit-correct, but not yet a gain

Reviewed ds4-spark/glint/llama.cpp first (the sm_121a captured-graph "drift bible"
warns capture-replay drifts and greedy masks it). **Verified my own greedy win is
NOT masked: 31/31 logit vectors bit-EXACT vs eager (worst |Δ|=0).** My re-capture
(resets executor state, the drift-free property ds4-spark found) + cheap ExecUpdate
(not their slow re-instantiate) is why.

Wired capture into the n_tok=3 combined verify forward (`metal_graph_verify_suffix_tops`).
Two capture-blockers fixed (both sync ops → async, stream-ordered so production
unchanged): `routed_moe sorted counts clear` cudaMemset→cudaMemsetAsync; the
prefix-snapshot `tensor_copy` cudaMemcpy→cudaMemcpyAsync. cuBLAS captures fine in
`cudaStreamCaptureModeRelaxed` with the handle bound to the per-thread stream.

**Result: tokens AND accept-rate identical (24/44 eager == graph) — bit-correct
even on the accept-sensitive MTP path (no drift). But −2.3% (18.94 vs 19.38).**
The n_tok=3 verify isn't launch-latency-bound (bigger kernels), and the real MTP
idle is BETWEEN spec-iters (eager draft-gen + host accept), not in the verify.
Gated off (`DS4_GRAPH_MTP_VERIFY`, opt-in); kept as drift-validation + scaffolding.

**The real MTP gain needs the whole-spec-iter capture** (draft-gen + verify +
sample as one graph, device-side accept via `cudaGraphCondTypeIf` per glint) so
the per-iter host gap is reclaimed. Large build; next stage. Capture-infra (async
clears, Relaxed mode, cuBLAS stream bind) is landed and reusable for it.

### Stage 3 build — scout + two validation spikes (foundation laid)

Scouted the exact device-residency surface for the whole-spec-iter capture. The
per-iter host round-trip is **one** point, three coupled ops:
1. **readback** `ds4_gpu_tensor_read(comp_selected → row_tops)` (`ds4.c:14338`),
2. **accept compare** `commit` loop `row_tops[i-1]!=drafts[i]` (`ds4.c:19253-57`),
3. **state advance** `checkpoint.len` / `mtp_n_raw` (`DS4_MTP_KEEP_ACCEPTED`) /
   prefix1·prefix2 commit copies (`ds4.c:19300-19365`).

Good news: the MTP verify path **already passes `pos0` as a kernel arg** where it
matters — indexed-attn (`ds4_cuda.cu:7408+`), KV-store-batch (`6714`), compressor
(`6770`). Only RoPE (`rope_tail_kernel`, 4 sites `ds4.c:9688/9715/9924/10076`)
bakes scalar `pos0` and needs device-pointer conversion. The one *structural*
blocker is the compressor count: `layer_n_comp[il]` is host-incremented after a
**host-gated** emit (`emit=((pos+1)%ratio)==0`, `ds4_cuda.cu:6804` → `ds4.c:9816`),
so device-side `pos` advance requires moving emit to a device `atomicAdd` counter
(Phase 3, mandatory + invasive).

Two spikes now validate the device-accept mechanism end-to-end on GB10/sm_121a,
**before** touching the forward (production untouched):
- **Phase 0** `tools/perf/cond_graph_spike.cu` — `cudaGraphCondTypeIf` +
  device `cudaGraphSetConditional` fire drift-free (gated body exact over replays).
- **Phase 1 keystone** `tools/perf/set_cond_from_accept_spike.cu` — the
  `set_cond_from_accept_kernel` mirrors the host accept loop on-device (reads
  device `row_tops[]`/`drafts[]`, computes `commit`, drives the IF cond from
  `commit==draft_n`). **Differential vs host: commit-sum bit-exact + full-accept
  tail gated exactly, over 10k replays (N=2 and N=3); compute-sanitizer memcheck
  0 errors / racecheck 0 hazards.** This is the keystone that lets a captured
  spec-iter decide accept-vs-rollback with no host round-trip.

Next: device-resident `pos`/`mtp_n_raw`/`layer_n_comp[]` (ReplaySlot) + RoPE
device-pos conversion + device-gated compressor emit, then wire the accept kernel
+ cond-node into the live (gated) MTP path, holding logit-exactness per phase.

### Stage 3 — PROFILE recalibration: device-accept is NOT the gain lever

Before building the device-resident refactor, profiled the live eager MTP decode
(`nsys -t cuda`, knight n=48, 19.63 t/s, ~104 ms/iter, ~2 tok/iter). Steady-state
idle is **15.7% (470 ms)**, and it splits decisively:

- **One-time cuBLAS-f16 warmup — ~218 ms, ~128 per-layer gaps (~1.7 ms each), ALL
  in the first 494 ms of decode, none after.** Cause: `cuda_q8_f16_ptr`
  (`ds4_cuda.cu:6200`) lazily dequantizes each Q8_0 weight → cached f16 buffer on
  first f16-GEMM use (alloc + `dequant_q8_0_to_f16_kernel`). Amortized over any
  real multi-response session; pre-warming only relocates it load→decode, no
  wall-clock gain. **Not a recurring lever.**
- **Recurring launch latency — 156 ms = 6.3%** (71.8k sub-5 µs gaps, post-warmup),
  spread across the WHOLE iter: `quantize_q8_0` 18.5, cutlass 15.3, cublasLt
  splitKreduce 15.0, `share_warp` 13.7, `f32_to_f16` 9.6, rope/rms/compressor …
  i.e. both draft-gen AND the N=3 verify GEMM-prep sequences.
- **Host accept round-trip — small** (only 14 of 48 token boundaries showed a big
  `argmax→embed` gap; one end-of-iter sync).

**Conclusion: the whole-spec-iter device-accept (cond-node, Phases 1-6) targets
only the residual one-sync-per-iter — not justified.** The real 6.3% is plain
launch latency, reclaimable with the SAME re-capture + `ExecUpdate` machinery that
landed the +5% plain-greedy win — drift-free (re-capture resets the executor),
bit-correct (host accept unchanged), no device-resident `pos`/cond-node needed.
The earlier verify-only capture got −2.3% because it wrapped just the verify
layer-batch (few, large kernels → ExecUpdate overhead > gap savings); capturing
the WHOLE iter amortizes ExecUpdate over draft-gen's many small kernels too.

Prereq: draft-gen does a host `top_id` readback BETWEEN draft steps (`ds4.c:13442`)
to chain — that mid-iter sync blocks one-graph capture. Step 1 = device-chain
draft-gen (Stage-2 device-token trick: argmax→`comp_selected`→device embed/router,
read drafts back once at iter end). Independently a bit-correct win (removes N-1
syncs/iter). Then capture the whole iter (draft-gen + verify) with re-capture.

Phase-0/Phase-1 cond-node spikes stay as validated foundation for the residual
sync IF a future workload (large N, batched serving) makes the accept round-trip
dominate — but this profile says it doesn't today.

### Stage 3 — MEASURED root cause of the verify-capture −2.3% (DS4_GRAPH_CAPTURE_STATS)

Instrumented `ds4_gpu_graph_capture_update_launch` to tally ExecUpdate-vs-reinstantiate
and print the `cudaGraphExecUpdateResultInfo.result`. Ran MTP verify-capture
(`DS4_GRAPH_MTP_VERIFY=1`, knight n=48):

**`update_ok=0  reinstantiate=21  cold=1` — ExecUpdate NEVER succeeds.**
Failure reason: **`result=3` = `cudaGraphExecUpdateErrorNodeTypeChanged`.**

So the −2.3% is 100% re-instantiate (~21 × ~1.67 ms ≈ 35 ms, ~1.4 % of the run)
that the capture can never amortize — NOT capture overhead per se. Root cause
(`ds4.c:12150-12209`, the per-token `else` branch the n_tok=3 verify always
takes since `n_tokens % ratio != 0`): the compressor emit is **host-conditional**
— `compressor_update` runs every token, but `fp8_kv_quantize` (kernel) +
`commit_attn_comp_stage` (memcpy) + `n_comp++` only fire `if (emit)` at
ratio-aligned positions. The verify advances `pos` by up to 3/iter, so which of
the 3 rows hits a ratio boundary shifts almost every iter → the quantize/commit
nodes land at different sequence indices → a kernel sits where a memcpy did →
`NodeTypeChanged` → re-instantiate.

**To make ExecUpdate succeed: stabilize the captured node structure** — emit must
become **unconditional** (always launch quantize+commit; redirect the write to the
not-yet-valid row `n_comp` so it's harmlessly overwritten at the real emit; keep
`n_comp++` host-conditional). That touches the shared compressor hot path
(`metal_graph_encode_layer_attention_batch`, used by prefill too) so it must be
capture-mode-gated to stay bit-exact, AND it's only the FIRST node-structure
variation — the indexer compressor (ratio==4) has the same pattern, and attention
kernel *selection* (online vs non-online) + grid dims shift as n_raw/n_comp grow
(grid changes ExecUpdate tolerates; selection changes it does not).

**Attempted the fix + measured (not inferred):** added `DS4_GRAPH_STABLE_EMIT`
— always-launch the compressor quantize+commit to the not-yet-live row `n_comp`
(harmless: attention reads only `[0,n_comp)`; row overwritten until a real emit
increments), `n_comp++` stays emit-only.  Bit-identical confirmed (knight n=24).
**Result: still `update_ok=0` / `NodeTypeChanged`** — stabilizing the emit does
NOT make ExecUpdate succeed.  At least one more per-iter node-type source remains
(NOT the prefix1/prefix2 snapshots — those are unconditional memcpys at fixed t).
So topology-stabilization is an **open-ended multi-fix cascade**, not a single
change: each barrier needs a fix+rebuild+measure cycle, and the residual ~−0.9 %
(verify recording > its launch idle) caps the outcome near-neutral regardless.

**Verdict (now empirical, not theorized):** the decode-replay capture is blocked
by per-iter node-structure drift; stabilizing it is a multi-week, correctness-
critical refactor of the shared compressor/attention path for a marginal ceiling
(the verify isn't launch-bound; even with ExecUpdate fixed the upside is ~0–2 %).
Measured the first fix (emit) — insufficient; the cascade has more barriers.
The achievable, landed MTP gains are the **+10.9 % cascade** (N=3 combined-forward,
already in baseline) and the **f16 prewarm ~2× prefill TTFT**. The cond-node
device-accept (Phases 4-6) targets an even smaller slice. Recommendation: bank
this precise scoping; pursue the topology-stabilization only if a future workload
raises the ceiling. The `DS4_GRAPH_CAPTURE_STATS` instrumentation stays as the
diagnostic to re-check after any stabilization attempt.

### Stage 3 — CORRECTION: the warmup is PREFILL, not decode (+97% prefill landed)

The first profile windowed "steady decode" as "after the last >20 ms gap" — but
there's no >20 ms gap between model-cache-prep and the prefill forward, so the
window **included prefill's tail**.  Re-windowed strictly after the first
`embed_token_hc` (the real first decode token, 1098 ms in): **all 344
`dequant_q8_0_to_f16` fire in prefill (367–1095 ms), ZERO in decode.** True
decode idle is 7.7 % (192 ms), 156 ms of it sub-5 µs launch latency — the
whole-iter-capture lever, unchanged conclusion.

The 344 f16 dequants are **prefill** lazily filling the Q8->f16 dense-weight
cache (`cuda_q8_f16_ptr`: per-weight `cudaMalloc`+dequant, first n_tok>4 use).
Decode-verify runs at n_tok<=4 → share-warp path → never touches f16.  So the
warmup is pure **time-to-first-token**, not decode.

**Landed: `metal_graph_prewarm_dense_f16`** (engine-open, CUDA, gated on
`warm_weights` — server sets it unconditionally, CLI/bench opt in via
`--warm-weights`, so default CLI keeps fast launch).  Eagerly populates the f16
cache for every dense Q8_0 weight via a new
`ds4_gpu_prewarm_q8_f16(…, label)` that calls `cuda_q8_f16_ptr` directly (no
GEMM/scratch).  Passing the weight's real name as the label is required: the
admission policy `cuda_q8_f16_cache_allowed` keys on substrings
(`attn_output_a`/`attn_q_b`/`ffn_*_shexp`) — the generic "q8_0" label admitted
only 258/344 (the dims-allowlisted ones), missing the 86 attn_output_a/b
(4096×8192 / 8192×4096, not in the dims list).  With real names: 344/344, 0 lazy.

**Result: prefill 7.6 → 15.0 t/s (+97 %), MTP and non-MTP alike; decode
unchanged (19.8); output bit-identical (n=24).**  Cost: ~2.8 s one-time at load
(front-loads the same 10.8 GB f16 cache prefill builds lazily anyway — no extra
HBM, just moved off the timed prefill).  Clear win for the server (cold-start /
TTFT) and long-context prefill; opt out with DS4_NO_F16_PREWARM.

## Bigger levers (owner-level — proposals, not done)

The readily-available occupancy squeeze is extracted. Further decode gains need
structural change (surface as proposals per the contributor role):

- **Narrower-accumulator decode kernel** — a `[4]`-wide `moe_down` (≈half the
  regs) could hit 3 blocks without spilling, at the cost of more weight re-reads;
  bit-identity must be checked (different reduction grouping).
- **Launch-gap elimination** — `argmax → embed_token_hc` is 0.6–1.2 ms/token
  (~7.7% idle); a CUDA-graph or async-sample decode loop would reclaim it.
- **MTP adaptive cascade depth** (N=2↔3) — robustness on low-accept prose; the
  ~54% accept ceiling itself is MTP-head-quality bound (needs model work).
- **Prefill** — separate regime (n_tok≥128, `tile16_row2048`); not yet profiled.

## Method note

`(256,3)` raised *occupancy* but tanked throughput — the canonical reminder that
occupancy is a means, not the goal. The gamut diff caught it instantly: occ↑ but
ms↑↑, %peakBW↓, stall reason flipped to `lg_throttle` (spill traffic). Always
read ms + stall-reason alongside occupancy.

## Prod-flag matrix — accept regime, depth, sampling (2026-05-27)

All prior rows profiled the *synthetic* capture (knight, greedy, shallow ctx).
This matrix profiled the axes that actually vary in production: **content/accept**
(code = high accept, prose = low), **depth** (managed KV at `--ctx 524288`,
~14k filled), and **temp** (greedy/MTP vs sampled/plain). Decode t/s below are
the **clean** (non-nsys) timing-run numbers; the deep cells' per-kernel tables
are nsys+managed-KV perturbed (`accept-code-deep` trace 11.0 vs measured 19.76 —
trust measured t/s + accept, not deep kernel ms). Runs in `runs/{accept-*,
prod-*,gb10-postrebase-*}`.

Throughput ladder (measured decode t/s):

| config | accept | tok/iter | t/s |
|--------|-------:|---------:|----:|
| code, greedy, shallow (high accept) | 68.8% | 2.38 | 21.68 |
| knight, greedy, shallow | 54.5% | 2.09 | 19.61 |
| code, greedy, **deep** | 70.0% | 2.40 | 19.76 |
| prose, greedy, shallow (low accept) | 43.6% | 1.86 | 17.49 |
| diverse, greedy, deep | 56.8% | 2.14 | 16.86 |
| sampled, deep, top_p 1.0 | — | — | 12.11 |
| **sampled, deep, top_p 0.95** (prod chat) | — | — | **10.66** |

Real-prod span is **10.66 → 21.68 = 2.0×**; the knight baseline we'd been tuning
sits near the top, so it is ~1.6–2× optimistic for sampled chat.

Findings:

- **Accept is the master throughput variable, and it is content-driven, not
  engine.** Shallow greedy is monotonic in accept (prose 17.49 → knight 19.61 →
  code 21.68) with a **near-identical per-kernel table** — the whole spread is
  tokens/iter. The R1 false-bisect rule in spades: low accept looks like a
  regression, no kernel moved.
- **Accept holds across depth; depth is a separate ~10–14% managed-KV tax**
  (orthogonal). Code accept 68.8% → 70.0% shallow→deep. So a high-accept coding
  agent stays fast at depth (19.76); diverse/low-accept chat is slower before you
  even sample.
- **Prose sits at the cascade break-even** (1.86 tok/iter vs the ~1.9 where N=3
  stops paying). Slightly harder content pushes MTP net-negative → the lever
  there is **adaptive cascade depth (N=3→2)**, not kernels.
- **Sampling (temp>0) drops MTP entirely** → plain N=1 kernel path (no
  `share_warp<3>`): −26% at shallow (20.33 → 15.09).
- **NEW — qsort sampler tax (greedy-only profiling never saw this).**
  `top_p < 1` with `top_k == 0` takes `sample_full_vocab`'s sort branch, which
  **`qsort`s all 129k logits per token** — a fixed **~11 ms/token = −14%**
  (shallow 15.09 → 13.01). `top_p == 1.0` takes the cheap no-sort branch and
  hides it, which is why the matrix's first sampled cells missed it. Production
  chat sends `top_p≈0.95`, so every sampled token pays it.

### Verdict — where the squeeze actually is

The greedy path is exhausted (kernel ledger above + the landed +5% graph-capture).
The remaining squeeze is almost entirely on the **sampled path**, invisible to the
greedy-only profiling:

1. **qsort sampler — LANDED, +14.8%.** `sample_full_vocab`'s `top_p<1` path now
   maintains a bounded top-K=1024 partial-select over the same full-softmax
   denominator instead of `qsort`-ing all ~129k logits (exact-sort fallback kept
   for the near-flat/high-top_p edge). Verified **bit-identical** token stream vs
   the exact sort at a fixed seed; **13.05 → 14.98 t/s at top_p=0.95** (recovers
   the full ~11 ms/token tax, matches the top_p=1.0 no-sort baseline). Also drops
   a ~2 MB/token malloc. Commit `4748b3a`.
2. **MTP-for-sampling — LANDED (env-gated `DS4_MTP_SAMPLE`).** Recover MTP for
   temp>0 via the distribution-preserving rejection sampler (`spec_sample_step` +
   `ds4_session_eval_speculative_sample`, forked from the combined-forward argmax
   path).  Drafts are sampled from the MTP draft dist `q`, accepted vs target `p`
   with `min(1, p_trunc/q_trunc)`, residual `max(0,p_trunc-q_trunc)` on reject
   (one extra main forward to commit the correction); bonus deferred to next-iter
   first_token on full accept.  Preserves the truncated target so output ==
   plain truncated sampling.  **Measured (temp 0.7, top_p 0.95): code 14.94 →
   19.48 t/s (+30%, high accept), prose 14.95 → 16.16 (+8%, low accept)** — win
   scales with accept, as predicted.  Verified coherent, deterministic (seeded),
   and reduces to greedy in the low-temp limit.  Stacks on the qsort win for the
   whole sampled-chat path.  **Default-on** (the `DS4_MTP_SAMPLE` enable-gate was
   removed; `DS4_MTP_SPEC_DISABLE` is the off-switch).  Math verified exact by
   `./ds4_test --spec-sampling` (host-only: TV(out,p)<0.004 and empirical accept ==
   Σmin(p,q) across q==p / sharper / flatter / disjoint-modes / random), and a
   content×temp×ctx soak found **every cell ≥ plain, no slower-than-plain regime,
   no crash** — code +26–41%, prose +14–22%, deep (managed KV) +14–26%.
3. **Depth** — managed-KV tax ~14–19%; config lever (right-size `--ctx`) or an
   owner-level managed-KV optimization.

Realistic chat (~10.7 t/s) could reach ~14+ with (1)+(2) — both engine-side, both
missed by every prior greedy capture.
