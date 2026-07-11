> **This is a fork.** [`TrevorS/ds4`](https://github.com/TrevorS/ds4) is a GB10 / DGX Spark (NVIDIA `sm_121a`) performance fork of [`antirez/ds4`](https://github.com/antirez/ds4). Fork `main` carries the changes below on top of upstream `main`, rebased on upstream periodically (currently on [`59d9bc7`](https://github.com/antirez/ds4/commit/59d9bc7830f2); the rebase is verified bit-identical — perplexity Δ=0, prefill/decode neutral). The speed/accuracy commits live on the [`land`](https://github.com/TrevorS/ds4/tree/land) branch — the upstream-candidate set; [`main`](https://github.com/TrevorS/ds4/tree/main) stacks runtime steering and the in-process Python binding on top. `FORK_RELEASE.md` summarizes the public-facing delta and the measured GB10 gains. Everything below this section is the upstream README, unchanged.

## Publication Snapshot

- The repository is now trimmed for publication: local virtualenvs, downloaded HF weights, and throwaway test binaries are not part of the tree.
- The releaseable source stays focused on the ds4 fork itself: CUDA backend, DSpark runtime, quantization path, server/agent integration, perf tooling, and release docs.
- For model acquisition, keep using `download_model.sh` and `quantize_dspark.sh`; the weights are intentionally fetched on demand instead of committed.

## What's different in this fork

- **GB10 / DGX Spark CUDA backend** — the full model kept resident in the unified GPU memory pool (no per-step weight streaming) — the real GB10 win. Built via PTX→`sm_121` JIT (`make cuda-spark`, the default empty `CUDA_ARCH`), **not** native `sm_121a` cubins: native-arch SASS measured **−5.5% decode** on GB10 for these kernels (the PTX-JIT path schedules better here), so the JIT path is the validated default — don't set `CUDA_ARCH=sm_121`.
- **MTP speculative decode (greedy + sampling)** — combined-forward, cascaded N=3 (two draft tokens verified in a single forward pass); `--mtp-draft` defaults to 2. **~1.5× sustained decode uplift over plain** on GB10 across 4k–32k ctx (greedy; see Benchmarks). Now covers **temperature sampling** too, via distribution-preserving rejection sampling (output is identical to plain sampling, just faster: 1.25–1.43× measured); on by default for `temp>0`, `DS4_MTP_SPEC_DISABLE=1` turns it off. **MTP is now wired in `ds4-agent`** — was previously the only DS4 binary that loaded the MTP gguf without ever calling `ds4_session_eval_speculative_*`; the agent's worker decode now spec-decodes too, lifting agent decode tps from ~11 to ~15-18 (+39% to +64%) on chat-formatted prompts.
- **CUDA kernel & graph perf** — share-warp Q8 (load weight quants once across tokens), `__launch_bounds__` tuning (+5.5% decode), bit-identical CUDA-graph decode (+5%), a Q8→f16 dense-weight prewarm (~2× prefill TTFT), and a bounded top-K nucleus sampler that drops the per-token 129k qsort (+14.8% on the sampled path).
- **Deterministic MTP verify (default) + `DS4_CUDA_FAST_VERIFY` fast mode** — the combined-forward verify is **bit-exact by default** (`mtp-correctness` gate: `worst_rms=0`; `mtp-selfconsistency`: run-to-run `maxabs=0`), so it's mergeable and reproducible. `DS4_CUDA_FAST_VERIFY=1` is the single "Spark fast mode" switch: it bypasses *all* the determinism machinery (fast batched `heads8` attention, n=1 GEMM → cuBLAS, skips the comp-row bitonic sort) for faster decode while staying self-consistent (MTP-fast output == plain-fast). Greedy quality is unaffected on IQ2-XXS.
- **GB10 weight-residency: HBM device-copy of the hot dense spans** — the startup cache stages the hot Q8 weights into device memory (2 MiB pages) instead of serving them over 4 KiB-page host-registered UVA; without it the Q8 decode matmuls run **~25%/call slower** (a regression an upstream rebase had silently dropped). `DS4_CUDA_NO_HBM_CACHE=1` opts out.
- **Runtime directional steering** — per-request scale plus an admin endpoint, named profiles, and model-name steering where `model:profile:tier` selects a steering vector and strength.
- **In-process Python binding** — `make libds4.so` builds an fPIC shared library, and `python/ds4.py` is a ctypes wrapper over the full `ds4.h` API (engine/session lifecycle, tokenization, sampling/logprobs, MTP decode, steering, KV persistence). No server process needed.
- **Perf & debug tooling — the `gamut` suite** — `tools/perf/gamut/` is a clean Python package (one CLI, no bash): `gamut-cli capture` nsys-traces a run and joins ptxas regs / gb20b occupancy / ncu stalls / MTP accept+verify timing into one report (md/json/html) and a SQLite run-store; `gamut-cli bench --matrix` runs the 3-cell decode matrix (plain · MTP-greedy · MTP-sample) under a GPU/thermal monitor; `gamut-cli db list|show|compare` reviews/diffs historical runs. Plus a streaming-reasoning client, a mobile-friendly LAN log viewer, and a `DS4_MTP_TV` acceptance probe (`1 − TV`) — see `docs/mtp-nongreedy-sampling.md`.

### Recommended GB10 settings

What we run for best perf on a DGX Spark (128 GB unified memory) with DeepSeek-V4-Flash IQ2-XXS. **Same core flags across binaries** — `--cuda --warm-weights --mtp DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 --power 85` — then add binary-specific tail.

```sh
# ds4-server (HTTP)
DS4_METAL_PREFILL_CHUNK=2048 ./ds4-server -m ds4flash.gguf \
  --cuda --warm-weights --power 85 \
  --mtp DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
  --ctx 524288

# ds4 (interactive chat)
./ds4 -m ds4flash.gguf \
  --cuda --warm-weights --power 85 \
  --mtp DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
  -c 524288

# ds4-agent (native coding agent)
./ds4-agent -m ds4flash.gguf \
  --cuda --warm-weights --power 85 \
  --mtp DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
  -c 524288

# ds4-bench (perf measurement)
./ds4-bench -m ds4flash.gguf \
  --cuda --warm-weights --power 85 \
  --mtp DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 4096 --ctx-max 32768 --step-mul 2 --gen-tokens 128 --temp 1
```

What each flag buys:

- `--mtp` + `--mtp-draft 2` — combined-forward speculative decode; **+50-65% sustained decode** over plain. Works in all four binaries as of `mqoyowyo` (the agent was the last holdout — it loaded the MTP gguf but never called the speculative path; that's fixed now).
- `--warm-weights` — front-loads the Q8→f16 dense-weight cache; **~2× prefill TTFT**.
- `--power 85` — caps GPU duty cycle at 85%. **Slightly faster than 100%** at sustained load on Spark because the firmware-level thermal throttle never kicks in. Measured 18.99 vs 17.52 t/s at draft=2 sampled (the 100% cell hit the silent throttle floor mid-run).
- `-c 524288` / `--ctx 524288` — keeps KV cache device-resident; the model's full **1M** context also fits via demand-paged managed memory.
- `DS4_METAL_PREFILL_CHUNK=2048` — frees context-buffer headroom at no throughput cost (the `4096` default wastes memory on this model). Name is historical, applies to CUDA too.

Flags **not** to pass for everyday use:

- `--quality` — disables TF32, the WMMA indexer, and the Q8→f16 dequant cache. Costs **~37% decode tps** with no observable quality gain on IQ2-XXS weights (the precision was already burned at quantization time). Only useful for cross-backend numerical-drift debugging.
- `--think-max` — generates extra reasoning tokens; same per-token speed but longer time-to-final-answer on routine prompts. Use only when a task warrants the extra thinking budget.

### Fork knobs

CLI flags (server + bench, fork additions to upstream):

| flag                         | default      | what it does                                                                       |
| ---------------------------- | ------------ | ---------------------------------------------------------------------------------- |
| `--mtp PATH`                 | off          | load MTP draft model; enables combined-forward speculative decode                  |
| `--mtp-draft N`              | 2            | draft tokens verified per forward (cascaded N=3 verifier)                          |
| `--warm-weights`             | off          | front-load Q8→f16 dense-weight cache (~2× prefill TTFT)                            |
| `--power N`                  | 100          | cap GPU duty cycle (%). **85** is slightly faster than 100 on Spark — keeps the firmware throttle from biting; thermally safer for sustained sessions |
| `--ctx N`                    | 4096         | KV cache size; `524288` keeps device-resident, full 1M fits via demand-paged       |
| `--temp F` (bench)           | 0 (greedy)   | `>0` measures the sampled decode path (spec-sampling when `--mtp` set)             |
| `--top-p`/`--min-p`/`--seed` | 0.95 / 0 / 1234 | sampler params for the sampled bench cell                                       |

Environment variables (perf / observability):

| env                                | default | effect                                                                  |
| ---------------------------------- | ------- | ----------------------------------------------------------------------- |
| `DS4_METAL_PREFILL_CHUNK=N`        | 4096    | prefill chunk tokens; **2048** frees ctx-buffer headroom (zero TPS cost on DS4-Flash). Name is historical — applies to CUDA too. |
| `DS4_MTP_SPEC_DISABLE=1`           | off     | disable speculative sampling (greedy MTP still active)                  |
| `DS4_GRAPH_DECODE=1`               | off     | enable bit-identical CUDA-graph decode (+5% sustained, greedy-only)     |
| `DS4_CUDA_DIRECT_MODEL=1`          | off     | register model mapping without the startup prewarm scan (debug)         |
| `DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB=N`| 1024    | Q8→f16 dense-weight preload span size                                   |
| `DS4_LOG_MEM=1`                    | off     | KV/buffer memory log lines on session open                              |
| `DS4_MTP_TV=1`                     | off     | speculative-sampling acceptance probe (`1 − TV`); see `docs/mtp-nongreedy-sampling.md` |
| `DS4_MTP_SPEC_LOG=1`               | off     | log every spec-decode miss/verifier fallback                            |
| `DS4_MTP_TIMING=1`                 | off     | per-spec-step stderr: `drafted=N committed=N total=X ms` — feed into `tools/perf/mtp/parse_timing.py` for accept-rate + committed-distribution stats |
| `DS4_MTP_MIN_MARGIN=F`             | 0       | reject drafts whose verifier margin is below F. Sweep showed mostly noise across prompt classes; only `analytical-qa` saw a clear +8% at `F=0.5`. Not a universal lever |
| `DS4_MTP_NO_CASCADE=1`             | off     | force single-draft window (no draft-conditioning). Costs throughput; debug only |
| `DS4_MTP_STRICT=1`                 | off     | exact verification path (no margin-skip). Same kill-switch `--quality` activates |
| `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1`    | off     | force the deterministic (non-atomic) MoE down-projection accumulation, even at large prefill chunks. Set this for bit-reproducible greedy decode (see caveat below) |
| `DS4_CUDA_MOE_ATOMIC_DOWN=1`       | off     | force the atomic-accumulate MoE down-projection on, regardless of chunk size. Faster at large prefill but scheduling-order-dependent |
| `DS4_CUDA_FAST_VERIFY=1`           | off     | **Spark fast mode** — bypass all deterministic-verify machinery (heads8 batched attention, n=1 GEMM → cuBLAS, skip comp-row sort). Faster MTP/decode; greedy-quality-neutral on IQ2-XXS. Default off keeps the bit-exact verify (the `mtp-correctness`/`mtp-selfconsistency` gates run without it) |
| `DS4_CUDA_NO_HBM_CACHE=1`          | off     | skip the startup HBM device-copy of hot weight spans (serve them over UVA instead). Costs ~25%/call on the Q8 decode matmuls — debug only |
| `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=N` | 96      | device-resident weight-cache budget ceiling (the HBM device-copy stops once spans reach this) |

> **MoE atomic-down nondeterminism (issue #244):** the MoE down-projection auto-enables atomic accumulation for prefill chunks `>= 128` tokens (`routed_moe` dispatch in `ds4_cuda.cu`). Atomic-add accumulation order is scheduling-dependent, so f32 rounding differs run-to-run; this can flip the greedy argmax during large prefill, picking a valid-but-different next token. For bit-reproducible greedy output set `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1` (forces the deterministic path); `DS4_CUDA_MOE_ATOMIC_DOWN=1` forces it on for any chunk size.

`gamut-cli bench` knobs: `--label NAME`, `--matrix`, `--iter N`, `--no-mtp`, `--temp` (sampled cell), `--fast` (`DS4_CUDA_FAST_VERIFY=1`), `--ctx-start/--ctx-max/--gen-tokens`, `-m MODEL`, `--prompt-file FILE`. (The old `tools/perf/bench-with-monitor.sh` and flat scripts now live under `tools/perf/legacy/`.)

### Benchmarks (GB10 / DGX Spark)

#### DSpark long-context profile

For the 131072-token context configuration, use the explicit DSpark fast
verify profile. It keeps the deterministic MoE down path enabled and leaves
the normal verifier available for A/B and token-identity checks:

```sh
tools/perf/dspark/run-17tps.sh tests/long_code_audit.txt
```

The launcher uses the release GGUF, `--ctx 131072`, `--tokens 32768`,
`-t 10`, and `--prefill-chunk 2048`; SSD streaming is not enabled. On the
GB10 validation prompt this measured 17.59 generated tok/s and 18.11 effective
tok/s, versus 14.53 generated tok/s with safe verify. The run reported 94.6%
combined coverage, `p0=0.854`, and `p1=0.752`. These are decode measurements
after prefill; results vary with prompt shape, thermal state, and acceptance.

Reproduce the full 8-cell decode matrix — four paths (plain / MTP × greedy / sample) crossed with the accuracy dial (high-accuracy deterministic verify vs the `DS4_CUDA_FAST_VERIFY` Spark fast path) — at 4k–32k context. The matrix sets the accuracy mode per-cell and cools the GPU to ≤55 °C between cells (anti-soak), so one run captures the whole grid:

```sh
tools/perf/gamut-cli bench --label decode-matrix --matrix --iter 1 --gen-tokens 256
```

Or a single-cell sanity check (plain, fast path, same ctx sweep):

```sh
tools/perf/gamut-cli bench --label sanity --no-mtp --fast
```

Defaults the wrapper bakes in: model `DeepSeek-V4-Flash-IQ2XXS` chat-v2, draft `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32` with `--mtp-draft 2`, core flags `--cuda --warm-weights --power 85`, prompt `tests/long_context_story_prompt.txt` (auto-stitched to fit `--ctx-max`), ctx sweep `--ctx-start 4096 --ctx-max 32768 --step-mul 2`; sample cells add `--temp 1.0 --top-p 0.95 --seed 1234`. Per-cell GPU/CPU/throttle monitors stream into `tools/perf/runs/<label>/` alongside the bench CSVs. (`--no-cooldown` / `--cooldown-c N` tune the between-cell cooldown.)

**Decode t/s** (`--gen-tokens 256` steady-state, cooled between cells, `tools/perf/runs/readme-matrix/`). MTP `(N×)` is the speedup over plain in the same mode.

Fast path (`DS4_CUDA_FAST_VERIFY=1` — the Spark default):

| context | plain-greedy | plain-sample | mtp-greedy   | mtp-sample   |
| ------: | -----------: | -----------: | -----------: | -----------: |
|   4,096 |         12.7 |         12.6 | 21.3 (1.68×) | 18.8 (1.49×) |
|   8,192 |         12.6 |         12.4 | 21.2 (1.68×) | 18.8 (1.52×) |
|  16,384 |         12.4 |         12.2 | 20.7 (1.67×) | 18.9 (1.55×) |
|  32,768 |         11.6 |         11.4 | 19.9 (1.72×) | 17.9 (1.57×) |

High-accuracy path (deterministic bit-exact verify — the default; gates run here):

| context | plain-greedy | plain-sample | mtp-greedy   | mtp-sample   |
| ------: | -----------: | -----------: | -----------: | -----------: |
|   4,096 |          9.7 |          9.6 | 18.8 (1.94×) | 16.0 (1.67×) |
|   8,192 |          9.5 |          9.5 | 18.7 (1.97×) | 16.1 (1.69×) |
|  16,384 |          9.4 |          9.4 | 18.3 (1.95×) | 15.7 (1.67×) |
|  32,768 |          8.9 |          8.8 | 16.9 (1.90×) | 15.1 (1.72×) |

The fast path buys **~30 % plain decode, ~13 % MTP decode, and ~55 % prefill** over deterministic verify (greedy output is bit-identical between the two — determinism only changes sampled-path fidelity, validated by the `mtp-correctness`/`mtp-selfconsistency` gates). MTP-greedy holds **~1.7× over plain** across the whole ctx range on the fast path (and ~1.9× on the accuracy path, where the determinism tax falls harder on plain). Plain decode is memory-bandwidth-bound (≈12.7 t/s × ~18.9 GB/token ≈ the measured 236 GB/s LPDDR5X read ceiling) — MTP is the only lever past it, by amortizing the per-token weight read across accepted drafts. Prefill is identical across decode paths (~350 t/s fast, ~228 t/s deterministic at 4k–32k) — MTP doesn't affect prefill.

**Long-context decode** (MTP-greedy, `--mtp-draft 2 --power 85`, KV-restore single-context points, 2026-05-29):

| context | decode t/s |
| ------: | ---------: |
|  43,643 |       19.7 |
| 147,877 |       17.6 |

Decode degrades gracefully far past the 32k sweep: ~22.6 t/s peak at 4k → 17.6 t/s at 148k, so ~36× context costs only ~22% decode. The attention + compressed-KV path is bandwidth-bound but holds up at long context (F16 comp-KV above 131k ctx-alloc keeps the cache footprint halved — a memory win, decode-neutral).

Those are *prose* numbers (the bench greedily continues an Italian novel). MTP shines harder on low-entropy output: on real **code or structured generation** the draft accept rate climbs and sustained decode clears **23+ t/s** greedy. Reproducible via the server — AVL-tree codegen:

```sh
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"deepseek-v4-flash","reasoning_effort":"none","temperature":0,"max_tokens":1200,
  "messages":[{"role":"user","content":"Write a complete Python implementation of a balanced AVL tree: node class, insert, delete, search, rotations, in-order traversal, and height-balancing. Full docstrings. Output only code, no prose."}]
}'
# server log -> gen=1200 ... avg~=23.3 t/s   (JSON-array generation lands ~25; prose ~15)
```

#### Per-binary decode tps (same prompt, same flags)

Lighthouse-essay prompt (300 token cap, sampled `--temp 1`, `--mtp-draft 2 --power 85`) across all four binaries:

| binary       | decode tps | what's measured                                        |
| ------------ | ---------:| ------------------------------------------------------ |
| **ds4 (CLI)**  | **18.11** | chat-prefilled, sampled MTP via `eval_speculative_sample` |
| **ds4-bench**  |     17.79 | clean kernel decode, KV-restored (no chat prefill)     |
| **ds4-server** |     15.91 | HTTP request → chat → MTP decode (server stderr avg)  |
| **ds4-agent**  |     14.96 | one-shot non-interactive, DSML tool framing + MTP decode |

CLI > bench because the bench measures pure decode on a warm KV while CLI's MTP head sees the chat-formatted essay context, which happens to land more `committed=2` draft hits than the bench's prose-continuation; server takes a ~12% hit from HTTP/JSON + worker-thread coordination; agent takes another ~6% from DSML/tool-parser framing and stream-renderer overhead. None of these were "bugs" — they're the natural cost of each binary's job.

Reproduce the per-binary comparison:

```sh
tools/perf/legacy/all-binary-bench.sh
```

#### MTP acceptance by prompt class (agent)

Same agent binary, same flags, varying the prompt content (`--mtp-draft 2 --temp 1 --power 85`):

| class              | accept | decode tps | committed dist (c=0/1/2) |
| ------------------ | -----:| ---------:| ------------------------: |
| code-generation    | **79.9%** | **19.62** |  9 / 36 / 55 |
| prose-continuation |   75.0%   |     n/a*  | 10 / 30 / 60 |
| analytical-qa      |   61.8%   |   17.29†  | 20 / 37 / 43 |
| structured-list    |   49.8%   |    14.94  | 27 / 47 / 27 |
| chat-essay         |   47.7%   |    15.43  | 32 / 41 / 27 |

\* prose model stopped after ~25 generated tokens.  † with `DS4_MTP_MIN_MARGIN=0.5` (analytical was the only class where the margin lever helped). Reproduce: `tools/perf/mtp/baseline_run.sh` and `margin_sweep.sh`.

#### How to measure your own workload

```sh
# one-shot agent profile (nsys kernel trace + MTP acceptance + decode tps)
tools/perf/agent/profile_run.sh \
    --prompt "your prompt here" \
    --tokens 400 --label my-workload

# server-style HTTP request with avg t/s in server logs
./ds4-server --cuda --warm-weights --power 85 \
    --mtp DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
    --port 8000 -m ds4flash.gguf &
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"deepseek-v4-flash","max_tokens":400,"messages":[{"role":"user","content":"..."}]}'

# bench against a warm KV (skip prefill, isolate decode kernels)
./ds4-bench --cuda --warm-weights --power 85 \
    --mtp DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
    --kv-restore ~/.ds4/kvcache/<sha>.kv --ctx-alloc 200000 \
    --gen-tokens 256 --temp 1 --csv /tmp/bench.csv
```

For the agent path specifically, `+DWARFSTAR_METRICS` on `--non-interactive` exit reports `decode_tps` (clean) and `avg_tps` (including tool-call drag) — so a tool-heavy session can show a misleading mid-flight number while the actual kernel decode is fine.

---

# DwarfStar

**DwarfStar** is a small native inference engine optimized first for
**DeepSeek V4 Flash**, with support for **DeepSeek V4 PRO** on very high-memory
machines. It is
intentionally narrow: not a generic GGUF runner, not a wrapper around another
runtime: it is completely self-contained. Other than running the model in a
correct and fast way, the project goal is to provide DeepSeek specific loading,
prompt rendering, tool calling, KV state handling (RAM and on-disk), server
API and integrated coding agent, all ready to work with coding agents or with
the provided CLI interface. There are also tools for GGUF and imatrix generation,
and for quality and speed testing.

We support the following backends:
* **Metal** is our primary target. Starting from MacBooks with 96GB of RAM (or less, using SSD streaming).
* **NVIDIA CUDA / DGX Spark**, CUDA with special care for the DGX Spark.
* **Strix Halo (ROCm)**, systems like the Framework Desktop and other systems based on the same GPU and unified RAM design.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

**Note that DeepSeek v4** is not our only target. Right now Flash and PRO are the
perfect fit because of capabilities, size, KV cache efficiency. If tomorrow a
better open weight model is released for the 128GB size, we could switch, the same
for other important size classes like 512GB of RAM. The project is stictly
opportunistic depending on what open weight models exist in a given moment.
If a new model will be supported, the old one may be removed completely and
no longer supported, unless there is some kind of overlap of abilities.

## Motivations

* Very capable open weight models finally exist. DeepSeek v4 Flash feels quasi-frontier. The PRO is even better. Both resist 2 bit quantization very well.
* Very capable computers like MacBooks, the DGX Spark now exist.
* DeepSeek v4 kv cache design makes it pratical to run very big contexts. Other vendors are using this approach.
* This few hundred billions models are strictly better than smaller (even if dense) models, regardless of what benchmarks say.

That said, a few important things about this project:

* The local inference landscape contains many excellent projects, but new models are released continuously, and the attention immediately gets captured by the next model to implement. This project takes a deliberately narrow bet: one model at a time, official-vector validation (logits obtained with the official implementation), long-context tests, and enough agent integration to know if it really works. The exact model may change as the landscape evolves, but the constraint remains: local inference credible on high end personal machines or Mac Studios, starting from 96/128GB of memory.
* This software is developed with **strong assistance from GPT 5.5** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.
* This implementation is based on the idea that compressed KV caches like the one of DeepSeek v4 and the fast SSD disks of modern MacBooks should change our idea that KV cache belongs to RAM. **The KV cache is actually a first-class disk citizen**. Fast SSD disks also changed the inference game from the point of view of "model needs to fit RAM": while having more RAM the the model size is still preferred, SSD streaming allows to turn the available amount of RAM from a hard cutoff (can I run this model or not?) to continuous spectrum of speed levels.
* Our vision is that local inference should be a set of three things working well together, out of the box: A) inference engine with HTTP API + B) GGUF specially crafted to run well under a given engine and given assumptions + C) testing and validation with coding agents implementations. D) Purpose built agents for specific models and execution environments. DwarfStar only runs with the GGUF files provided. It gets tested against officially obtained logits at different context sizes. This project exists because we wanted to make one local model feel finished end to end, not just runnable. However this is beta quality code, so probably we are not still there, especially since recently we introduced large new features: distributed inference, SSD streaming, and other minor improvements.
* The optimized graph path targets **Metal on macOS** and **CUDA on Linux**. The CPU path is only for correctness checks and model/tokenizer diagnostics. For CPU-only Linux builds, use `make cpu`; it builds the normal `./ds4` and `./ds4-server` binaries without CUDA or Metal. On macOS, **warning: current macOS versions have a bug in the virtual memory implementation that will crash the kernel** if you try to run the CPU code. Remember? Software sucks. It was not possible to fix the CPU inference to avoid crashing, since each time you have to restart the computer, which is not funny. Help us, if you have the guts.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The code and GGUF files are to be considered of **beta quality** because
inference and model serving is a complicated matter and all this exists
only for a few days. It will take months to reach a more stable form.
However, we try to keep the project in a usable state, and we are making
progress. If you have issues, make sure to use `--trace` to log the
sessions, and open issues including the full trace.

The `ds4-agent` is alpha quality, the project was later added.

## More Documentation

If you are looking for very specific things, we have other
sub-README files. Otherwise for normal usage keep reading the
next sections.

- [CONTRIBUTING.md](CONTRIBUTING.md): correctness and speed regression testing
  guide for contributors. **Read this before sending a pull request**.
- [gguf-tools/README.md](gguf-tools/README.md): offline GGUF generation,
  imatrix collection, quantization tooling, and quality checks.
- [gguf-tools/imatrix/README.md](gguf-tools/imatrix/README.md): how the
  routed-MoE imatrix is collected and used.
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md):
  how the calibration prompt corpus is generated.
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md):
  how local GGUFs are scored against official DeepSeek V4 Flash/PRO continuations.
- [dir-steering/README.md](dir-steering/README.md): directional steering data,
  vector generation, and usage.
- [speed-bench/README.md](speed-bench/README.md): benchmark commands, charts,
  and CSV generation.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): official
  continuation vectors used for regression checks.

## Model Weights

This implementation only works with the DeepSeek V4 Flash and PRO GGUFs published for
this project. It is not a general GGUF loader, and arbitrary DeepSeek/GGUF files
will not have the tensor layout, quantization mix, metadata, or optional MTP
state expected by the engine. The 2 bit quantizations provided here are not
a joke: they behave well, work under coding agents, call tools in a reliable way.
The 2 bit quants use a very asymmetrical quantization: only the routed MoE
experts are quantized, up/gate at `IQ2_XXS`, down at `Q2_K`. They are the
majority of all the model space: the other components (shared experts,
projections, routing) are left untouched to guarantee quality.

Download one main model. **Prefer the imatrix versions.**

```sh
./download_model.sh q2-imatrix   # 96/128 GB RAM machines, imatrix-tuned q2
./download_model.sh q2-q4-imatrix  # 96/128 GB RAM machines, q2 with last 6 layers q4
./download_model.sh q4-imatrix   # >= 256 GB RAM machines, imatrix-tuned q4
./download_model.sh pro-q2-imatrix  # 512 GB RAM machines, PRO q2 imatrix quant
```

If you are using DSpark, download the abliterated DSpark Hugging Face source
checkpoint and quantize it locally:

```sh
./download_model.sh dspark-source
./quantize_dspark.sh --out gguf/DeepSeek-V4-Flash-DSpark-Abliterated-Q2.gguf
```

For the full PRO Q4 distributed run, download one half on each machine:

```sh
./download_model.sh pro-q4-layers00-30      # first half of PRO Q4 split
./download_model.sh pro-q4-layers31-output  # second half of PRO Q4 split
```

The script downloads from `https://huggingface.co/antirez/deepseek-v4-gguf`,
stores files under `./gguf/`, resumes partial downloads with `curl -C -`, and
updates `./ds4flash.gguf` to point at the selected main model.
The `pro-q4-layers00-30`, `pro-q4-layers31-output`, and `pro-q4-split` targets
download distributed PRO Q4 pieces and do not update `./ds4flash.gguf`.
Authentication is optional for public downloads, but `--token TOKEN`,
`HF_TOKEN`, or the local Hugging Face token cache are used when present.

If you want to regenerate GGUF files or collect a new imatrix, see
[gguf-tools/README.md](gguf-tools/README.md). Those tools are meant for offline
model-building work and can take a long time on the full DeepSeek V4 Flash
weights. Flash GGUF generation is supported by the local tools. PRO GGUF
production currently still depends on the external `llama.cpp`-based workflow;
native tooling can be added later.

`./download_model.sh mtp` fetches the optional speculative decoding support
GGUF for Flash. It can be used with q2-imatrix, q2-q4-imatrix, and q4-imatrix,
but must be enabled explicitly with `--mtp`. The current MTP/speculative
decoding path is still experimental: it is correctness-gated and currently
provides at most a slight speedup, not a meaningful generation-speed win.
On our GB10 agent workload, the picture is mixed: code-heavy prompts do benefit
from `--mtp` while prose-heavy prompts can still favor the base path, so keep
the service on MTP when the target traffic is agent/code generation.
The long-context KV-disk cache also matters: our repeated 9.3k-token code prompt
went from about `24.1s` cold prompt time to `3.2s` once `8192` tokens were
reused from disk, so the service budget is now set to `131072` MiB instead of
the previous `98304` MiB.

The DSpark branch in this clone accepts `--dspark` as a flag on the main DSpark
GGUF. Unlike legacy `--mtp`, DSpark is not a second small model path: the DSpark
worker tensors live inside the converted base model.

For conversion/quantization, the source checkpoint we want is
`Valent1qw/DeepSeek-V4-Flash-DSpark-Abliterated`, which contains the abliterated
DSpark safetensors shards plus the DSpark worker tensors. Use `dspark-source` to
download it, then feed the local directory into `gguf-tools/deepseek4-quantize`.
If you want the whole flow in one command, use `./quantize_dspark.sh` after the
source checkpoint is present.

The `abliterated` part is a model-selection requirement, not a runtime flag:
pick an abliterated DSpark checkpoint before conversion. Run it as:

```sh
./ds4 --model gguf/DeepSeek-V4-Flash-DSpark-Abliterated-Q2.gguf --dspark
```

Then build:

```sh
make                  # macOS Metal
make cuda-spark       # Linux CUDA, DGX Spark / GB10
make cuda-generic     # Linux CUDA, other local CUDA GPUs
make cpu              # CPU-only diagnostics build
```

`./ds4flash.gguf` is the default model path used by both binaries. Pass `-m` to
select another supported GGUF from `./gguf/`. Run `./ds4 --help` and
`./ds4-server --help` for the full flag list.

## Speed

These are single-run Metal CLI numbers with `--ctx 32768`, `--nothink`, greedy
decoding, and `-n 256`. The short prompt is a normal small Italian story
prompt. The long prompts exercise chunked prefill plus long-context decode.
Q4 requires the larger-memory machine class, so M3 Max Q4 numbers are `N/A`.

| Machine | Quant | Prompt | Prefill | Generation |
| --- | ---: | ---: | ---: | ---: |
| MacBook Pro M3 Max, 128 GB | q2 | short | 58.52 t/s | 26.68 t/s |
| MacBook Pro M3 Max, 128 GB | q2 | 11709 tokens | 250.11 t/s | 21.47 t/s |
| MacBook Pro M3 Max, 128 GB | q4 | short | N/A | N/A |
| MacBook Pro M3 Max, 128 GB | q4 | long | N/A | N/A |
| MacBook Pro M5 Max, 128 GB | q2 | short | 87.25 t/s | 34.27 t/s |
| MacBook Pro M5 Max, 128 GB | q2 | 11707 tokens | 463.44 t/s | 25.90 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | short | 84.43 t/s | 36.86 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | 11709 tokens | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | short | 78.95 t/s | 35.50 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | 12018 tokens | 448.82 t/s | 26.62 t/s |
| Mac Studio M3 Ultra, 512 GB | PRO q2 | 32768 tokens | 138.82 t/s | 9.56 t/s |
| DGX Spark GB10, 128 GB | q2 | 7047 tokens | 343.81 t/s | 13.75 t/s |

![M3 Max t/s](speed-bench/m3_max_ts.svg)
![PRO model M3 Ultra t/s](speed-bench/pro_model_m3_ultra_ts.svg)

## Running models larger than RAM

The normal Metal path tries to make the model resident in GPU-addressable
memory. This is the fastest path and should remain your default when the model
fits. When it does not fit, DwarfStar also has a Metal-only **SSD streaming**
capacity mode. In this mode the non-routed model weights stay resident, while
routed MoE experts are kept in an in-memory cache and loaded from the GGUF file
on cache misses.

Streaming is not as fast as fitting the full model in RAM. It still needs memory
for non-routed weights, KV cache, graph scratch, activations, and the routed
expert cache. It is useful because routed experts dominate model size and modern
Mac SSDs are fast enough to make cache misses tolerable. Long prefills can still
be fast; generation is more sensitive to cache misses because every new token
routes through experts again.

Start with the automatic cache budget:

```sh
./ds4 -m ./ds4flash.gguf --ssd-streaming
```

If startup reports that the expert cache is too large, or if you want to reserve
more memory for context, set the routed expert cache explicitly:

```sh
./ds4 -m ./ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB
```

The `32GB` value is a memory budget for complete routed experts, not a generic
byte cache. DwarfStar converts it to the number of full experts that fit for the
current GGUF. Non-routed weights, KV cache, graph scratch, and activations need
additional memory. Only the automatic cache budget does the subtraction for you:
it takes 80% of the Metal recommended working set, subtracts non-routed weights,
then uses the rest for routed experts. Leave the hot expert preload enabled for
normal use; use `--ssd-streaming-cold` and `--ssd-streaming-preload-experts N`
only for measurements.

### Practical SSD streaming examples

On 64GB MacBooks, start with the 2-bit Flash GGUF and a moderate expert cache:

```sh
./download_model.sh q2-imatrix

./ds4 \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 32GB \
  --ctx 32768 \
  --nothink
```

On 128GB MacBooks, PRO q2 streaming is experimental but usable for inspection
and occasional work when you accept slow generation. Start with `--nothink`:

```sh
./download_model.sh pro-q2-imatrix

./ds4 \
  -m gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf \
  --ssd-streaming \
  --ctx 32768 \
  --nothink
```

On an M5 Max with 128GB of RAM, a short PRO q2 streaming decode benchmark found
the automatic budget best: it selected about `59GB` of routed expert cache.
Manual `64GB` to `75GB` caches were close on that machine. Larger explicit
`NGB` requests are capped before inference so the expert buffers remain
lockable instead of falling into macOS paging. If the system is under extra
memory pressure and `mlock` still fails, ds4 refuses to install pageable
expert-cache entries and releases a locked-cache margin before continuing with
the measured lockable cache size. Prefer the automatic budget; if setting the
cache manually on this class of machine, start around `48GB` to `64GB`, then
increase only while the startup log reports a lockable cache. Once the machine
is stable, re-enable thinking with a conservative generation limit:

```sh
./ds4 \
  -m gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf \
  --ssd-streaming \
  --ctx 32768 \
  --think \
  --tokens 1500
```

The important startup line is the cache report. Start conservative, then
increase the cache if the machine has headroom.

## Distributed Inference

Distributed inference lets DwarfStar **run a model that is too large for one machine** by
splitting transformer layers across multiple machines. The main example is the
full 4-bit Flash quant across two 128 GB MacBooks: each process maps only its
own layer slice, activations are sent over TCP, and the coordinator keeps normal
CLI/API behavior.

Distributed inference also allows to **speed up prefill** by
using multiple GPUs at the same time to process different micro-batches at
different layers, like in an assembly line. Only prefill can be accelerated this
way. Generation is purely autoregressive: each token must finish across the
route before the next token can start. The model work is the same as a single
process, plus coordination latency, so distributed generation is slower.

To build an initial mental model, here are the high level concepts:

1. You put the GGUF on every machine, but each one loads just a subset. `--layers` controls which tensors are mapped, so a worker with `--layers 20:output` does not load the earlier layers.
2. Layer ranges are inclusive: `10:20` means layers 10, 11, ..., 20. `N:output` means layer `N` through the final layer plus the output head.
3. You assign one of the machines the role of `coordinator`, the others the roles of `workers`. Workers will connect to the coordinator and will tell they are there and which layers they are able to process.
4. Each worker keeps its slice of the KV cache.
5. Communication is worker-to-worker, there is no need to use the coordinator as relay, so if your coordinator is `A`, and you make a request, activations will flow in `A -> B -> C -> back to A`.

### How it works and how to configure it

The prefill path is pipelined (this is why it can go faster than in a single machine).
For large prompts the coordinator can run its
slice on chunk N+1 while the worker is running its slice on chunk N. The
distributed rows below were measured with two M5 Max 128 GB MacBooks connected
by Thunderbolt 5, using the Q4 Flash GGUF and the default 4096-token
distributed prefill chunk. The single-process column is a reference run with
the Q2 GGUF on a single machine, so it actually is a bit faster since
the routed MoEs are smaller.

| Prompt | Single-process reference | Two MacBooks | Speedup |
| ---: | ---: | ---: | ---: |
| 9421 tokens | 421.70 t/s | 582.22 t/s | 1.38x |
| 28684 tokens | 405.30 t/s | 674.16 t/s | 1.66x |
| 63819 tokens | 353.62 t/s | 654.79 t/s | 1.85x |

Generation is different. **It is strictly autoregressive**: token N+1 cannot start
until token N has produced logits and sampling has selected the next token. That
means distributed generation cannot use the long prefill pipeline. It pays at
least one cross-machine activation hop per generated token, so generation is
slower than a single local process. On the same two-Mac Thunderbolt setup, a
12k-context control run with the 91 GB Flash quant went from 30.59 t/s
single-process to 24.67 t/s distributed, a 19.4% loss. Distributed inference is
therefore mainly for fitting larger models and speeding up long prefills, not
for making decode faster.

### Full DeepSeek V4 PRO Q4 on two Mac Studios

The full-size PRO Q4 GGUF can be run across two 512 GB Mac Studio M3 Ultra
machines by giving the coordinator layers `0:30` and the worker
`31:output`. Use the split GGUF files so each side maps only the tensors it
needs:

```sh
# Coordinator machine.
./download_model.sh pro-q4-layers00-30

# Worker machine.
./download_model.sh pro-q4-layers31-output
```

The two files are:

```text
gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf
gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf
```

This is a capacity use case: each process maps only its own half of the model,
while the worker owns the output head and returns logits.

The current PRO Q4 Metal path uses queue-resident exact expert tables for the
large routed experts. This avoids the broad multi-GiB routed-tensor bindings
that made early distributed PRO Q4 attempts either run very slowly or hit Metal
memory accounting limits. In a short greedy smoke test over the direct
`192.168.0.182` / `192.168.0.183` link, the model generated coherent text and
measured 11.47 t/s generation after startup. Per-token telemetry was balanced:
local layers were around 39-43 ms, remote layers around 44-49 ms, for total
token times around 84-92 ms. Expect a slow startup while each side maps and
makes its half of the model resident. Long-context PRO Q4 prefill and decode
performance still needs separate benchmarking.

The measurements above use a Thunderbolt 5 cable. The implementation is plain
TCP and also works over slower links, including WiFi, but fast Ethernet or
Thunderbolt networking is strongly recommended. Slow links mostly hurt
generation latency and short prefills; large prefills can still benefit when
the layer split is balanced. In the normal performance path, the last worker
owns the output head and returns logits directly.

Minimal two-host configuration:

```sh
# Machine A: coordinator, owns tokenization, sampling, the prompt, and layers 0..30.
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf \
  --role coordinator \
  --layers 0:30 \
  --listen 169.254.43.68 1234

# Machine B: worker, connects to A and owns layers 31..output.
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf \
  --role worker \
  --layers 31:output \
  --coordinator 169.254.43.68 1234
```

Normally the final worker should own the output head too, for example
`--layers 20:output`. This avoids returning a full final hidden-state batch
after prefill and lets the final worker produce the logits directly. On very
slow or metered links, `--layers 20:42` is also supported: the coordinator will
load the output head and compute logits locally, trading extra coordinator work
for smaller per-token replies.

### Network Link Comparison

The table below shows the same two M5 Max hosts, the same 91 GB Flash quant,
coordinator `--layers 0:19`, worker `--layers 20:output`, an 8192-token prompt
from `speed-bench/promessi_sposi.txt`, and 128 generated tokens. WiFi and
Internet numbers vary with local conditions, but the shape is the important
part: high latency hurts generation directly, while lower bandwidth also pulls
down long-prefill speed.

| Link | Addresses | Ping avg | Prefill | Generation |
| --- | --- | ---: | ---: | ---: |
| Thunderbolt 5 | `169.254.43.68` -> `169.254.12.245` | 0.45 ms | 582.99 t/s | 25.09 t/s |
| WiFi | `192.168.1.57` -> `192.168.1.95` | 77.20 ms | 250.70 t/s | 10.70 t/s |
| Internet / VPN | `10.77.0.4` -> `10.77.0.3` | 152.10 ms | 114.88 t/s | 3.63 t/s |

The Internet/VPN case is not meant to be a good interactive experience. It is
still useful for collective testing: multiple people can temporarily combine
machines to run a larger model that would not fit on any single host, accepting
slow decode in exchange for being able to inspect the model at all.

Use the coordinator exactly like normal `./ds4`: interactive chat, `/read`,
and ordinary generation go through the same high-level session API. The same
distributed options are also wired into `ds4-agent`, `ds4-eval`, and
`ds4-bench`. For benchmarks, workers should already be running; `ds4-bench`
waits until a complete route is available.

Useful tuning and diagnostics:

```sh
./ds4-bench \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 \
  --ctx-max 65536 \
  --step-incr 32768 \
  --gen-tokens 0 \
  --role coordinator \
  --layers 0:19 \
  --listen 169.254.43.68 1234 \
  --debug
```

`--debug` on the coordinator prints route formation and per-hop telemetry:
layer range, token span, local evaluation time, downstream wait time, socket
send time, and input/output byte counts. This is the current profiling tool for
deciding whether a split is balanced. `--dist-prefill-window N` controls how
many prefill chunks may be in flight end-to-end; the default is conservative
and bounded. `--dist-prefill-chunk N` exists for experiments, but the default
4096-token chunk is the canonical setting and should be used unless you are
explicitly validating a different chunk size.

By default DwarfStar sends hidden-state activations as 32-bit floats. To reduce
traffic, pass `--dist-activation-bits 16` or `--dist-activation-bits 8` on the
coordinator. This changes only the transport format between machines, not the
model weights or KV cache. 16-bit transport halves activation traffic and is the
first option to try on Ethernet or WiFi. 8-bit transport is more aggressive and
should be treated as an approximate/experimental mode unless you have validated
the output for your use case. However experimentally reduction activation
size didn't provide a significant improvement, so this option may be removed
in the future.

**If a worker disconnects, the coordinator removes that worker from the active
route**. The request already in flight can fail, and later calls report an
incomplete route until a compatible worker reconnects and sends a new
registration. For live sessions, the coordinator keeps the token history and can
rebuild worker KV state by replaying the prefix when the route is available
again. Workers also validate a rolling 64-bit token-prefix hash on every work
item, so a restarted worker at position 0 cannot silently accept work for
position N; it reports the mismatch and the coordinator replays the current
transcript. Ctrl+C in the CLI and agent is cooperative: DwarfStar waits for the
current distributed token or prefill chunk to drain before returning control,
which avoids coordinator-caused KV splits. Saved agent/server sessions use the
same KV file format as single-machine sessions: during save the coordinator
fetches worker-owned layer tensors and serializes one normal payload; during
load it splits that payload over the currently registered route.

### Distributed protocol overview

At the protocol level there are two kinds of connections. Workers keep a
control TCP connection open to the coordinator and send a `HELLO` with their
model ID, model family, quant profile, layer slice, context capacity, and data
port. The coordinator uses these registrations to build a route that covers all
layers. Work then moves over low-latency TCP data connections: the coordinator
computes the first slice, sends a `WORK` frame with session ID, token positions,
rolling token-prefix hashes before and after the span, route information, and
hidden-state payload, and each worker computes its slice. Middle workers can
forward directly to the next worker. The final worker returns logits to the
coordinator, or ACKs for non-final prefill chunks so the prefill pipeline can
stay full. `RESULT` frames echo the request ID and the post-span hash. A worker
status error is handled differently from a socket failure: KV/hash mismatch can
be recovered by replaying the token history on the same route, while transport
failure drops the route and waits for a replacement worker. For persistent KV,
the coordinator opens worker data connections and sends snapshot save/load
messages for each worker-owned layer range; the disk payload remains a single
agent/server cache file. The protocol has no
encryption or authentication, and is not release-stable yet; coordinator and
workers should be built from the same commit and used on trusted machines and
trusted networks.

## Reducing heat, power usage and fan noise

Long local inference runs can keep the GPU busy for extended periods. If you
care more about heat, fan noise, battery life on MacBooks, or reducing thermal
stress on the hardware than about maximum throughput, use `--power N`.

`--power 100` is the default and means full speed. Lower values ask DwarfStar to target
that percentage of GPU usage: `--power 70` targets about 70%, `--power 50`
targets about half usage, and so forth. DwarfStar does this by measuring GPU work time
and inserting small sleeps between work units: during prefill it sleeps between
layers, and during generation it sleeps between decoded tokens. This reduces
sustained load without changing model output.

The option is available on the CLI, server, agent, eval, and benchmark tools,
for example:

```sh
./ds4 --power 50
./ds4-agent --power 70
./ds4-server --power 40 --ctx 100000
```

## Native agent

DwarfStar features a native coding agent that works in a different way
than most other systems: the inference is controlled from within the agent
itself, without socket/API boundaries, so the session is represented
by the on-disk KV cache itself. Moreover the tools and the system prompt
are all designed vertically for DeepSeek v4 Flash and PRO. This provides a
few advantages:

* Low latency experience, bounded mainly by the prefill speed limits. Displaying of generated text, tool calling, start of a new session are always instantaneous.
* Live progress bar during prefill time.
* No DSML tool calling conversion, the tools are handled natively in the LLM format.
* KV cache mismatch are impossible by construction, the current state is always the truth.
* Everything is tuned for this model.
* Ability to switch saved sessions with `/list` and `/switch`; full KV sessions resume without a prefill stage.

Agent sessions are stored in `~/.ds4/kvcache`. Use `/save` to persist the
current session, `/list` to show saved sessions sorted by recent update time,
and `/switch <sha>` to resume one of them. The session ID is stable across
future saves and is derived from the first user prompt and creation time.
`/del <sha>` removes a saved session. `/strip <sha>` keeps the rendered
conversation text and title but removes the heavy KV payload; switching to a
stripped session rebuilds the KV cache by prefilling the saved text.

Use `--chdir /path/to/ds4` when launching `ds4-agent` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

However while the system already works, there is a lot of work to do
in order to make it ready for prime time. When finally the agent will reach
the wanted shape, we will *likely* split the server and the client creating a stateful
session-based protocol that can recreate all that in a client-server way.

## Benchmarking

`ds4-bench` measures instantaneous prefill and generation throughput at context
frontiers instead of reporting one whole-run average. It loads the model once,
walks a fixed token sequence to frontiers such as 2048, 4096, 6144, and uses
incremental prefill so each row measures only the newly-added token interval.
After each frontier it saves the live KV state to memory, generates a fixed
greedy non-EOS probe, restores the memory snapshot, and continues prefill.

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

The example file is a cleaned public-domain Project Gutenberg text of
Alessandro Manzoni's *I Promessi Sposi* (ebook #45334), with the Gutenberg
header and footer removed: <https://www.gutenberg.org/ebooks/45334>.

Use `--step-incr N` for different linear spacing, or `--step-mul F` for
exponential sweeps. Output is CSV with one row per frontier: latest prefill
interval tokens/sec, generation tokens/sec at that frontier, and
`kvcache_bytes`.

Sessions prefill long prompts in 4096-token chunks by default. Set
`DS4_METAL_PREFILL_CHUNK=N` to compare another chunk size, for example `2048`
to match the strict official-vector checkpoint path, or
`DS4_METAL_PREFILL_CHUNK=0` to prefill a prompt as one whole batch when memory
allows. Changing the chunk changes the KV checkpoint/logit path, so compare it
as an explicit run configuration.
Chunked Metal prefill reuses the same range-capable layer-major graph for each
chunk, preserving absolute compressor/indexer boundaries while avoiding the old
per-layer chunk dispatch path.

## Capability Evaluation

`ds4-eval` is a small real-model integration benchmark. It is not a leaderboard
runner and should not be reported as an official GPQA, SuperGPQA, AIME, or
security benchmark score: the questions are an embedded 92-item subset chosen
to make local regression testing useful and visually inspectable. The program
loads the real GGUF, renders DeepSeek chat prompts, streams sampled tokens in a split-screen TUI, grades
the final answer, and prints a per-question report with prompt tokens,
generated tokens, pass/fail state, the model answer, and the correct answer.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
```

The default run uses `--tokens 16000`, thinking mode enabled, and a soft/hard
`</think>` budget cutoff so the model has room to produce a visible answer.
`ds4-eval` sizes the context internally from the largest selected prompt plus
the generation budget, and refuses runs that would need more than 1M context
tokens. Press `p` to pause, `q` to exit and print the report, Up/Down to
inspect or select another question, and Enter to run the selected question next.
`--plain` disables the TUI.

Use `--regrade-trace /path/to/trace.txt` to replay the current answer
extractor and scorer against a prior `--trace` file without loading the model
or regenerating tokens. This is useful when auditing evaluator changes: it
shows which cases changed, the old picked answer, the new picked answer, and a
pass/fail summary.

For inference changes that can affect generation drift, keep this deterministic
q1..q4 token-count gate in the test plan:

```sh
./ds4-eval \
  -m ds4flash.gguf \
  --plain \
  --questions 4 \
  --tokens 2048 \
  --temp 0 \
  --seed 1
```

The generated-token counts must stay aligned with the baseline:

| Question | Expected state | Expected generated tokens | Expected given/correct |
|---:|---|---:|---|
| 1 | `PASSED` | 2048 | `B` / `B` |
| 2 | `PASSED` | 438 | `C` / `C` |
| 3 | `PASSED` | 666 | `70` / `70` |
| 4 | `FAILED` | 2048 | `A` / `C` |

The first 75 embedded questions are interleaved as 25 GPQA Diamond, 25 audited
SuperGPQA, and 25 AIME 2025 problems. The final 17 are an audited COMPSEC
subset of reduced single-function C/C++ vulnerability-localization questions.
The model is asked for the single best source line, or the smallest exact line
set only when the bug cannot be localized to one line; the scorer accepts small
audited ranges only when adjacent lines are equivalent locations for the same
bug. The order is
intentionally progressive: early questions are useful smoke tests, while later
questions are hard enough that a strong reasoning model should still miss some
of them. The SuperGPQA slice is curated rather than blind: upstream rows with
wrong keys, missing figures, or underspecified prompts are replaced with cleaner
rows.

The set should be treated as a hard capability regression suite rather than
a pass/fail unit test.

- **GPQA Diamond** contributes graduate-level science questions with
  multiple-choice answers. DeepSeek's model card reports strong results
  on full GPQA Diamond in thinking mode, but individual items still require
  careful physics, chemistry, or biology reasoning and are easy to lose with a
  small prompt/rendering or sampling regression.
- **SuperGPQA** contributes broad specialist knowledge and domain-transfer
  questions. The model-card SuperGPQA number is much lower than GPQA Diamond,
  so these items are expected to be uneven: some look mundane, others require
  niche professional knowledge or exact interpretation of a translated-style
  exam question.
- **AIME 2025** contributes exact-answer contest math. These are often the most
  unforgiving items in the set: no multiple-choice prior, no partial credit, and
  a single arithmetic or algebraic slip changes the grade.
- **COMPSEC** contributes single-function C/C++ security reasoning items
  reduced from public CVE writeups. These are not exploit prompts: the task is
  to identify the best source line where the defensive code flaw is introduced,
  or return `0` for a safe function.

In practice this means `ds4-eval` should not be expected to produce a perfect
92/92 run. It is meant to answer a more useful engineering question: after a
kernel, quantization, prompt-rendering, KV-cache, or tool-streaming change, does
DeepSeek V4 Flash still solve a representative mix of hard science, broad
knowledge, exact math, and security-code problems while using the same inference
path users run?

## CLI

One-shot prompt:

```sh
./ds4 -p "Explain Redis streams in one paragraph."
```

No `-p` starts the interactive prompt:

```sh
./ds4
ds4>
```

The interactive CLI is a real multi-turn chat. It keeps the rendered chat
transcript and the live graph KV checkpoint, so each turn extends the previous
conversation. Useful commands are `/help`, `/think`, `/think-max`, `/nothink`,
`/ctx N`, `/read FILE`, and `/quit`. Ctrl+C interrupts the current generation
and returns to `ds4>`.

The CLI defaults to thinking mode. Use `/nothink` or `--nothink` for direct
answers. `--mtp MTP.gguf --mtp-draft 2` enables the optional MTP speculative
path; it is useful only for greedy decoding, currently uses a confidence gate
(`--mtp-margin`) to avoid slow partial accepts, and should be treated as an
experimental slight-speedup path.

`--model gguf/DeepSeek-V4-Flash-DSpark-Abliterated-Q2.gguf --dspark` enables
the DSpark metadata path on a GGUF converted from the DSpark checkpoint. DSpark
is not a second small model like legacy `--mtp`: the DSpark tensors live inside
the same base model under the `mtp.*` namespace. The CUDA runtime executes the
official `main_hidden → main_proj → DSpark stages → Markov → confidence`
pipeline, then verifies the scheduled drafts with the target microbatch path.
For greedy generation the cost controller selects a one- or two-draft verifier
from measured acceptance and iteration cost; set
`DS4_DSPARK_NO_COST_ADAPTIVE=1` to disable it.
Deterministic verification is the default. `DS4_CUDA_FAST_VERIFY=1` is an
explicit throughput mode and may produce a different greedy stream; do not set
it when token identity with canonical target decode is required.

The GB10 target-forward optimizations are independently reversible for A/B
tests: `DS4_CUDA_NO_HOT_TARGET_CACHE=1` disables the bounded output/attention/
router cache, `DS4_CUDA_HOT_TARGET_CACHE_MB=N` changes its 4096 MiB admission
budget, and `DS4_CUDA_NO_ATTENTION_OUTPUT_A_SHARE=1` disables shared-weight
attention output-A for N=2/N=3. `DS4_DSPARK_TIMING=1` logs per-iteration
decisions; session shutdown prints aggregate target/effective throughput,
acceptance, verifier depth, fallback rate, and phase timings.

## Server

Start a local OpenAI/Anthropic-compatible server:

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Use `--chdir /path/to/ds4` when launching `ds4-server` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

The server keeps one mutable backend/KV checkpoint in memory,
so stateless clients that resend a longer version of the same prompt can reuse
the shared prefix instead of pre-filling from token zero.

Request parsing and sockets run in client threads, but inference itself is
serialized through one graph worker. The current server does not batch multiple
independent requests together; concurrent requests wait their turn on the single
live graph/session.

Supported endpoints:

- `GET /v1/models`
- `GET /v1/models/deepseek-v4-flash`
- `GET /v1/models/deepseek-v4-pro`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

The Flash and PRO model endpoints are compatibility aliases. They both report
the model currently loaded from the GGUF passed with `-m`; the endpoint name does
not select a different model.

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, `stream`, `stream_options.include_usage`, `tools`, and `tool_choice`.
Tool schemas are rendered into DeepSeek's DSML tool format, and generated DSML
tool calls are mapped back to OpenAI tool calls.

`/v1/responses` accepts OpenAI Responses-style `input`, `instructions`,
`tools`, `tool_choice`, `max_output_tokens`, `temperature`, `top_p`, `stream`,
and `reasoning`. It is the preferred endpoint for Codex CLI. The server keeps
Responses continuations bound to live state when possible, and can fall back to
the same DSML rendering and KV prefix reuse used by chat completions.

`/v1/messages` is the Anthropic-compatible endpoint used by Claude Code style
clients. It accepts `system`, `messages`, `tools`, `tool_choice`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`, and thinking
controls. Tool uses are returned as Anthropic `tool_use` blocks.

Default sampled API generation uses `temperature=1`, `top_p=1`, and
`min_p=0.05`, so the default filter is relative probability rather than
nucleus mass. In thinking mode DwarfStar uses those fixed sampling defaults and
ignores client sampling knobs, matching DeepSeek's fixed-thinking API behavior.

The chat, Responses, and Anthropic endpoints support SSE streaming. In thinking
mode, reasoning is streamed in the native API shape instead of being mixed into
final text. OpenAI chat streaming
also streams tool calls as soon as the DSML invocation is recognized: the tool
header is sent first, then parameter bytes are forwarded as
`tool_calls[].function.arguments` deltas while generation continues. The
Anthropic endpoint streams thinking and text live, then emits structured
`tool_use` blocks when the generated tool block is complete.
The Responses endpoint streams the Responses event lifecycle expected by Codex,
including `response.output_text.delta`, function-call argument events, and
terminal `response.completed` / `response.incomplete` / `response.failed`
events.

For browser JavaScript clients served from another origin, start the server with
`--cors` to emit `Access-Control-Allow-*` headers. This only changes HTTP
headers; it does not expose the server on the LAN. Use `--host 0.0.0.0`
explicitly when remote machines should be able to connect.

### Tool call handling and canonicalization

DeepSeek V4 emits tool calls as [DSML text](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/README.md). Agent clients do not send that
same text back on the next request: they send normalized OpenAI/Anthropic JSON
tool-call objects. **If the server re-rendered those objects slightly
differently, the rendered byte prefix would no longer match the live KV
checkpoint** and the next turn would have to be rebuilt.

The first line of defense is exact replay. Every tool call gets an unguessable
API tool ID, and the server remembers `tool id -> exact sampled DSML block` in
a bounded in-memory map backed by radix trees. When the client later sends that
tool ID back, the prompt renderer uses the exact DSML bytes the model sampled,
not a freshly formatted approximation. This map can also be saved inside KV
cache files, so exact replay survives server restarts for cached histories.

**Canonicalization is only the backup path**. If the exact DSML block is missing,
or exact replay is disabled with `--disable-exact-dsml-tool-replay`, the server
renders a deterministic DSML form from the JSON tool object. After a tool-call
turn, it compares the live sampled token stream with the prompt that the next
client request will render. If needed, it rewrites the live checkpoint, or
falls back to an older disk KV snapshot and replays only the suffix. This keeps
the model continuation aligned with the stateless API transcript.

During generation, the server also treats DSML syntax differently from payload.
When the model is emitting stable protocol structure such as DSML tags,
parameter headers, JSON punctuation, or closing markers, sampling is forced to
`temperature=0` so the tool call stays parseable. This greedy mode does **not**
apply to argument payloads: `string=true` parameter bodies and JSON string
values, including file contents and edit text, use the request's normal sampling
settings. That separation is important: deterministic decoding is helpful for
syntax, but can create repeated text when applied to long code or file bodies.

Minimal OpenAI example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

### Agent Client Usage

`ds4-server` can be used by local coding agents that speak OpenAI-compatible
chat completions. Start the server first, and set the client context limit no
higher than the `--ctx` value you started the server with:

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

You can use larger context and larger cache if you wish. Full context of
1M tokens is going to use more or less 26GB of memory (compressed indexer
alone will be like 22GB), so configure a context which makes sense in
your system. With 128GB of RAM you would run the 2-bit quants, which are
already 81GB, 26GB are going to be likely too much, so a context window
of 100~300k tokens is wiser. However users reported being able to run 2bit
quants with 250k ctx window in a Macs with just 96GB of system memory: make sure
to kill processes that use too much memory, if you plan doing so ;)

The `384000` output limit below avoids token caps since the model is able
to generate very long replies otherwise (up to 384k tokens). The server
still stops when the configured context window is full.

For **opencode**, add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "ds4.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "limit": {
            "context": 100000,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "ds4": {
      "description": "DeepSeek V4 Flash served by local ds4-server",
      "model": "ds4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

For **Pi**, add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ds4": {
      "name": "ds4.c local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "deepseek-v4-flash",
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 100000,
          "maxTokens": 384000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionally make it the default Pi model in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "ds4",
  "defaultModel": "deepseek-v4-flash"
}
```

For **Codex CLI**, use the Responses wire API:

```toml
[model_providers.ds4]
name = "DS4"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

Then run:

```sh
codex --model deepseek-v4-flash -c model_provider=ds4
```

For **Claude Code**, use the Anthropic-compatible endpoint. A wrapper like this
matches the local `~/bin/claude-ds4` setup:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${DS4_ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}"
export ANTHROPIC_AUTH_TOKEN="${DS4_API_KEY:-dsv4-local}"
export ANTHROPIC_MODEL="deepseek-v4-flash"

export ANTHROPIC_CUSTOM_MODEL_OPTION="deepseek-v4-flash"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="DeepSeek V4 Flash local ds4"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="ds4.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code may send a large initial prompt, often around 25k tokens, before it
starts doing useful work. Keep `--kv-disk-dir` enabled: after the first expensive
prefill, the disk KV cache lets later continuations or restarted sessions reuse
the saved prefix instead of processing the whole prompt again.

## Thinking Modes

DeepSeek V4 Flash has distinct non-thinking, thinking, and Think Max modes.
The server defaults to direct replies unless thinking is explicitly enabled.
`reasoning_effort=max` requests Think Max, but it is only applied when the
context size is large enough for the model card recommendation; smaller
contexts fall back to normal thinking. OpenAI `reasoning_effort=xhigh` maps to
normal thinking.

For direct replies, use `thinking: {"type":"disabled"}`, `think:false`, or
`reasoning_effort:low`.

## Disk KV Cache

Chat/completion APIs are stateless: agent clients usually resend the whole
conversation every request. `ds4-server` first tries the cheap exact token-prefix
check, then falls back to comparing rendered prompt bytes with decoded
checkpoint bytes. The live in-memory checkpoint covers the current session; the
disk KV cache makes useful prefixes survive session switches and server
restarts.

For RAM reasons there is currently only one live KV cache in memory. When a new
unrelated session replaces it, the old checkpoint can only be resumed without
re-processing if it was written to the disk KV cache. In other words, memory
cache handles the active session; disk cache is the resume mechanism for
different sessions.

Enable it with:

```sh
./ds4-server --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

The cache key is the SHA1 of the rendered byte prefix, and files are named
`<sha1>.kv`. The DS4 payload still stores the exact token IDs and graph state
for that prefix. This matters for continued chats: the model may have generated
one token whose decoded text is later sent back by a client as two canonical
prompt tokens. A rendered byte-prefix hit can still reuse the checkpoint and
tokenize only the new suffix.
The file is intentionally written with ordinary `read`/`write` I/O, not
`mmap`, so restoring cache entries does not add more VM mappings to a process
that already maps the model.

Tool calls also keep a bounded exact-DSML replay map keyed by unguessable tool
IDs, so client JSON history can be rendered back to the exact sampled text. The
RAM map keeps up to 100000 IDs by default; tune it with `--tool-memory-max-ids`.
Use `--disable-exact-dsml-tool-replay` to disable this and fall back to
canonical JSON-to-DSML rendering.

On disk, a cache file is:

```text
KVC fixed header, 48 bytes
u32 rendered_text_bytes
rendered_text_bytes of UTF-8-ish token text
DS4 session payload, payload_bytes from the KVC header
optional tool-id map section
```

The fixed header is little-endian:

```text
0   u8[3]  magic = "KVC"
3   u8     version = 1
4   u8     routed expert quant bits, currently 2 or 4
5   u8     save reason: 0 unknown, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     extension flags, bit 0 = appended tool-id map
7   u8     reserved
8   u32    cached token count
12  u32    hit count
16  u32    context size the snapshot was written for
20  u8[4]  reserved
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    DS4 session payload byte count
```

The rendered text is the tokenizer-decoded text for the cached token prefix.
It is both the human-inspectable prefix and the lookup identity: its SHA1 is
the filename, and a file is reusable only when those bytes are a prefix of the
incoming rendered prompt. After load, the exact checkpoint tokens from the DS4
payload remain authoritative, and only the incoming text suffix after the cached
bytes is tokenized.

The optional tool-id map is present only when header extension bit 0 is set.
Appended sections use fixed bit order, so future extension bits can add fields
without ambiguity. The map stores unguessable API tool call IDs back to the
exact DSML block the model sampled. Only mappings whose DSML block is present
in the rendered cached text are stored. This lets restarted servers render
later client history byte-for-byte like the original model output, even if the
client reorders JSON arguments.

The current tool-id map section is:

```text
0   u8[3]  magic = "KTM"
3   u8     version = 1
4   u32    entry count

For each entry:
0   u32    tool id byte length
4   u32    sampled DSML byte length
8   bytes  tool id
... bytes  exact sampled DSML block
```

The section is auxiliary replay memory, not model state. A cache hit restores
the session payload first, then loads the map if present. Before rendering a
request, the server can also scan cache files for the tool IDs present in the
client history and load just those mappings, so an exact DSML replay can survive
server restarts even when the matching KV snapshot is not the one ultimately
used for the rendered-prefix hit.

The DS4 session payload starts with thirteen little-endian `u32` fields:

```text
0   magic = "DSV4"
1   payload version = 2
2   saved context size
3   prefill chunk size
4   raw KV ring capacity
5   raw sliding-window length
6   compressed KV capacity
7   checkpoint token count
8   layer count
9   raw/head KV dimension
10  indexer head dimension
11  vocabulary size
12  live raw rows serialized below
```

Then it stores:

- `u32[token_count]` checkpoint token IDs.
- `float32[vocab_size]` logits for the next token after that checkpoint.
- `u32[layer_count]` compressed attention row counts.
- `u32[layer_count]` ratio-4 indexer row counts.
- For every layer: the live raw sliding-window KV rows, written in logical
  position order rather than physical ring order.
- For compressed layers: live compressed KV rows and compressor frontier
  tensors.
- For ratio-4 compressed layers: live indexer compressed rows and indexer
  frontier tensors.

The logits are raw IEEE-754 `float32` values from the host `ds4_session`
buffer. They are saved immediately after the checkpoint tokens so a loaded
snapshot can sample or continue from the exact next-token distribution without
running one extra decode step. MTP draft logits/state are not persisted; after
loading a disk checkpoint the draft state is invalidated and rebuilt by normal
generation.

Distributed coordinator sessions use the same `DSV4` payload. Worker-owned
layer tensors are pulled during save and merged into the normal layer-ordered
tensor stream; during load the coordinator splits that stream into the current
route and pushes the relevant layer tensors back to the workers. The saved file
does not retain the distributed topology.

The tensor payload is DS4-specific KV/session state, not a generic inference
graph dump. It is expected to be portable only across compatible `ds4.c`
builds for this model layout.

The cache stores checkpoints at four moments:

- `cold`: after a long first prompt reaches a stable prefix, before generation.
- `continued`: when prefill or generation reaches the next absolute aligned frontier.
- `evict`: before an unrelated request replaces the live in-memory session.
- `shutdown`: when the server exits cleanly.

Cold saves intentionally trim a small token suffix and align down to a prefill
chunk boundary. This avoids common BPE boundary retokenization misses when a
future request appends text to the same prompt. The defaults are conservative:
store prefixes of at least 512 tokens, cold-save prompts up to 30000 tokens,
trim 32 tail tokens, and align to 2048-token chunks. The important knobs are:

Continued saves use the same alignment and are written only when the live graph
naturally reaches an absolute frontier. With the defaults this means roughly
every 10k tokens, independent of where the first cold checkpoint landed, so long
generations leave restart points behind without persisting the fragile final few
tokens.

- `--kv-cache-min-tokens`
- `--kv-cache-cold-max-tokens`
- `--kv-cache-continued-interval-tokens`
- `--kv-cache-boundary-trim-tokens`
- `--kv-cache-boundary-align-tokens`
- `--tool-memory-max-ids`
- `--disable-exact-dsml-tool-replay`

By default, checkpoints may be reused across the 2-bit and 4-bit routed-expert
variants if the rendered prefix matches. Use `--kv-cache-reject-different-quant`
when you want strict same-quant reuse only.

The cache directory is disposable. If behavior looks suspicious, stop the
server and remove it. You can investigate what is cached with hexdump as
the kv cache files include the verbatim prompt cached.

## Backends

The default graph backend is Metal on macOS and CUDA in CUDA builds:

```sh
./ds4 -p "Hello" --metal
./ds4 -p "Hello" --cuda
```

On Linux, plain `make` prints the available build targets instead of selecting a
CUDA target implicitly. Use `make cuda-spark` for DGX Spark / GB10. It omits an
explicit `nvcc -arch` because that is currently the fastest path on GB10. Use
`make cuda-generic` for a normal local CUDA build, or set `CUDA_ARCH` explicitly
when cross-building or when you need a known target:

```sh
make cuda CUDA_ARCH=sm_120
make cuda CUDA_ARCH=native
```

There is also a CPU reference/debug path:

```sh
./ds4 -p "Hello" --cpu
make cpu
./ds4
./ds4 -p "Hello"
```

Do not treat the CPU path as the production target. The CLI and `ds4-server`
support the CPU backend for reference/debug use and share the same KV session
and snapshot format as Metal and CUDA, but normal inference should use Metal or
CUDA.

## Steering

This project supports steering with single-vector activation directions; see the
`dir-steering` directory for more information. This follows the core idea of the
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
paper. You can use it to make the model more or less verbose, less likely to
answer programming questions if it is a chatbot for your car rental web site,
and so forth, much faster than fine-tuning.
This is also useful for cybersecurity researchers who want to reduce a model's
willingness to provide dual-use or offensive security guidance.

## Test Vectors

`tests/test-vectors` contains short and long-context continuation vectors
captured from the official DeepSeek V4 Flash API. The requests use
`deepseek-v4-flash`, greedy decoding, thinking disabled, and the maximum
`top_logprobs` slice exposed by the API. Local vectors are generated with
`./ds4 --dump-logprobs` and compared by token bytes, so tokenizer/template or
attention regressions show up before they become long generation failures. The
C runner pins `DS4_METAL_PREFILL_CHUNK=2048` for this strict API-vector
comparison.

All project tests are driven by the C runner, with a small `ds4-eval`
extractor self-test run first:

```sh
make test                  # ./ds4-eval --self-test-extractors && ./ds4_test --all
./ds4_test --logprob-vectors
./ds4_test --server
```

## Debugging Notes

When a generation looks wrong, three small tools are usually enough to get a
first answer:

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits /tmp/logits.json --metal --nothink --prompt-file prompt.txt
./ds4-server --trace /tmp/ds4-trace.txt ...
```

- `--dump-tokens` tokenizes the `-p` or `--prompt-file` string exactly as
  written, recognizes DS4 protocol specials, and then exits before inference
  starts. For example, the DSML tool close marker starts as two tokens: `</`
  and `｜DSML｜`.
- `--dump-logprobs` stores a greedy continuation with the top local
  alternatives at each step, which helps separate sampling choices from
  logit/model issues.
- `ds4-server --trace` writes the rendered prompts, cache decisions, generated
  text, and tool-parser events for a whole agent session.
