# Handoff: CUDA-graph / device-accept for MTP decode — INVESTIGATED, REJECTED

**Status: closed as a perf lever for GB10 long-context. Read this before re-opening it.**

This doc previously proposed extending CUDA-graph decode capture to the MTP spec
path ("wrap the verifier forward / move accept device-side, expected payoff >
plain's +5%"). That proposal was wrong for our target. The verify-forward capture
was already built, measured, and shelved; whole-spec-iter "device-accept" capture
was then evaluated against a real profile and rejected. The evidence is below so
nobody burns another cycle rediscovering it.

## TL;DR verdict

On GB10 long-context decode, **the verify forward is ~99% of every spec-iter and is
compute-bound.** Everything graph-capture/device-accept could reclaim (draft-gen
launch latency + per-iter host syncs + accept logic) is ~1% of the iter wall, before
subtracting capture overhead. Net-zero-to-negative. It gets *worse* as context grows,
which is exactly our regime. Do not pursue it. The real decode lever is the verify
forward itself (long-context attention / combined-forward matmuls), not host
orchestration.

## The decisive measurement

From `tools/perf/runs/profile-77k/mtp.nsys-rep` (`ds4-bench`, `--warm-weights`,
ctx=200000, 77,111 tokens restored from KV, `--mtp ...Flash-MTP-Q4K-Q8_0-F32.gguf
--mtp-draft 2`, GB10 sm_121). The in-tree `DS4_MTP_TIMING` line carries the split:

```
mtp timing micro: draft=4.416 ms  verify=327.579 ms  snapshot=0.000 ms  total=332.068 ms
```

Per spec-iter at 77k context:

| Component | Time | Share | Capturable by device-accept? |
| --- | --- | --- | --- |
| Verify forward (full model, combined N=3 rows) | ~328 ms | ~99% | No — compute-bound; capture already tried = −2% |
| Draft-gen (two MTP-block forwards) | ~4.4 ms | ~1.3% | Yes, but it's 1% of the wall |
| Snapshot HC + host accept compare + frontier rollback | ~0 ms | ~0% | Yes, but already free |

Steady-state combined iters run ~130–180 ms (the 332 ms line is the first micro-iter,
which folds in some warmup). The ratio holds: draft-gen and host orchestration are in
the noise next to the verify forward, and the gap *widens* with KV length because
verify attention scales with context while draft-gen + accept are constant.

## GPU idle is warmup, not spec-iter bubbles

The profile shows 19.7% GPU idle (1858 ms / 9421 ms span), which is what made the old
"79% SM, reclaim the headroom" framing tempting. It decomposes to one-time startup:

- All **158 `>1300µs` gaps** and all **344 `dequant_q8_0_to_f16` kernels** fall in the
  first 20% of the run — 80.76 GiB page-warm + prewarm of 344 dense-weight f16 caches +
  first prefill. (Verified: `first_20pct(prefill)|344`, zero in steady state.)
- The `>1300µs` bucket alone is 1267 ms = 68% of all idle, and is **not MTP-specific**:
  the `nomtp.nsys-rep` plain-decode run has the same profile (151 gaps / 1157 ms).
- In steady decode the GPU is **never starved** — 140 ms of solid verify compute per
  iter gives the host ample slack to issue the next iter's launches. The
  "between-spec-iters idle" the in-tree comment cites is real but tiny and early-clustered
  (600–1300µs bucket: 114 gaps in the first half, 14 in the second).

## Why verify-forward capture was −2% (consistent, not a fluke)

The verify-only capture path exists in tree, gated `DS4_GRAPH_MTP_VERIFY` (NOT
`DS4_GRAPH_DECODE` — that flag is plain-greedy only). See the comment at
`ds4.c:14657-14667`:

> Verify-only capture is bit-correct (tokens + accept-rate identical) but a net loss
> (~−2%): the n_tok>1 verify isn't launch-latency-bound, and the real MTP idle is
> BETWEEN spec-iters (eager draft-gen + host accept), not inside the verify.

That result is now fully explained by the 99% number: capturing the compute-bound
verify hides no idle (launches already overlap GPU work) and adds re-capture +
`cudaGraphExecUpdate` overhead per iter. Landed in `4fe87ba "cuda: MTP graph-capture
wiring + cuBLAS-safe capture infra (Stage 3)"`. Keep it as drift-validation
scaffolding only.

## The 93%-one-core CPU finding, reconciled

Earlier measurement: one worker thread pegged ~93% on a single Grace core during
sustained decode. That's genuine `cudaLaunchKernel` issue cost (141k launches =
3469 ms host time in this profile). But at 77k+ context it does **not** gate
throughput — the launch cost is hidden behind the long verify. It would only become
the wall in a launch-bound regime (short context, small KV), which is not our target.
So the CPU saturation is real but not the bottleneck for GB10 long-context.

## The one regime where device-accept could matter (and why it's not ours)

Device-accept only pays off when verify is cheap relative to host overhead — i.e.
**short context** (small KV → verify drops toward draft+sync magnitude). Our GB10
mandate is the opposite: long-context, 1M-fits, where verify dominates *more* as KV
grows. The optimization is anti-correlated with the target, and it's gated behind the
hardest change in the tree: device-side data-dependent accept + frontier rollback
(prefix1/prefix2 selection) under capture, which needs CUDA conditional graph nodes.
Worst cost/benefit ratio on the board. Skip it.

## If you want decode tok/s on GB10, look here instead

The 99% says the lever is the **verify forward itself**, not orchestration:

- Long-context attention kernel efficiency (the combined-forward attention over 77k+ KV
  rows is where the 328 ms lives).
- Combined-forward matmuls (MoE gate/up/down, the q8_0 paths).
- KV layout / compression costs at long context.

Orchestration-level work (graph capture, device-accept, launch-bound micro-opts) is
provably in the noise at the contexts we care about. The squeeze memory's history
already showed `(256,3)` launch_bounds and eager-stage-2 mods gave ~0–1% — same lesson.

## Validation discipline (unchanged, still mandatory for any kernel change)

This is the hill the row128 patch died on. Read
`feedback_logit-equality-must-exercise-kernels.md` before writing any
kernel-affecting code. For any verify-forward kernel work, the primary correctness
probe is the `DS4_BENCH_TOKEN_DUMP` greedy token-diff — `make test` /
logprob-vectors will NOT catch kernel drift (metal-tensor-equivalence skips on CUDA;
logprob vectors are coarse and skip short_code_completion for API-drift reasons).

```sh
DS4_BENCH_TOKEN_DUMP=/tmp/base.tokens ./ds4-bench --cuda --warm-weights \
    -m ds4flash.gguf --mtp <mtp.gguf> --mtp-draft 2 \
    --kv-restore ~/.ds4/kvcache/b9dbb307b5f4150cf3b1925c92441a015734989c.kv \
    --ctx-alloc 200000 --gen-tokens 32 --temp 0

# after your change:
DS4_BENCH_TOKEN_DUMP=/tmp/new.tokens ./ds4-bench ...   # same flags
diff /tmp/base.tokens /tmp/new.tokens   # MUST be empty

# live sampled sanity (the row128 attempt passed throughput, failed here):
./ds4-agent --cuda -c 100000 --warm-weights --power 85 \
    --mtp <mtp.gguf> --mtp-draft 2 --non-interactive --nothink --tokens 600 \
    -p "Write a 500-word essay about lighthouses. Just write directly."
# Watch for BOS spam / repetition / incoherent text.
```

## Code map (for whoever works the verify forward instead)

- `ds4.c:19665` `ds4_session_eval_speculative_argmax_combined` — the N=K+1 cascaded
  verify-forward + host accept. drafts via `metal_graph_eval_mtp_draft_from_hc`
  (`ds4.c:13732`) + chained `_draft_n_from_hc`; verify via
  `metal_graph_verify_suffix_tops`. Accept = `while (row_tops[c]==drafts[c]) c++`.
  Frontier rollback: prefix1 (commit==0) / prefix2 (commit==1).
- `ds4.c:19840` `ds4_session_eval_speculative_sample_combined` — sampled counterpart.
- `ds4.c:19992-20034` / `20628-20664` — the public `_argmax` / `_sample` wrappers.
- `ds4_cuda.cu:1419-1500` — plain-greedy graph capture machinery (`g_decode_exec`,
  `cudaGraphExecUpdate`). The `+5% bit-identical` plain path (`82be3d3`).
- `ds4.c:14657-14674` — the `DS4_GRAPH_MTP_VERIFY` verify-capture branch + the −2% comment.
- `Makefile:33-36` — `-DDS4_GRAPH_DECODE_BUILD` for the CUDA target. (The old doc cited
  `ds4_cuda.cu:33-36` here — that's an unrelated attention-cap enum. It's the Makefile.)

## Profiles backing this verdict

- `tools/perf/runs/profile-77k/mtp.nsys-rep` + `.sqlite` — the MTP run analyzed above.
- `tools/perf/runs/profile-77k/nomtp.nsys-rep` — plain-decode control (same warmup idle).

To reproduce the per-iter split, grep the bench stderr for `mtp timing`, or for the
GPU-idle decomposition query the sqlite (`CUPTI_ACTIVITY_KIND_KERNEL`, gap histogram by
`ROW_NUMBER() OVER (ORDER BY start)`).

## Reviewed against tree as of

```
4977dce8  test: skip metal-tensor-equivalence on CUDA builds   <- main HEAD
...
4fe87ba   cuda: MTP graph-capture wiring + cuBLAS-safe capture infra (Stage 3)  <- DS4_GRAPH_MTP_VERIFY, the -2% path
```
