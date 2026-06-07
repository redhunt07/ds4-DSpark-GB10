# Gamut report — design notes

A single joined performance report for ds4 decode on GB10. One command, one
document, every relevant metric from the (extensive) perf suite pulled together
and cross-keyed so a human can read the whole picture without opening six tool
outputs and joining them by eye.

Status: **built** — `tools/perf/{perflib,gpu_metrics,gamut}.py` + `membw.cu`.
See `tools/perf/README.md` for the workflow and `tools/perf/runs/` for a real
report. The mockup below was the target; the shipped report matches it (with
real per-kernel registers/occupancy/HW-metrics joined in). Remaining follow-ups
are noted at the end.

## Why

The perf suite is wide and growing — `nsys_top`, `roofline`, `hbm_report`,
`gaps`, `gpu_metrics` (the missing one), `bench-v2`, `profile_diff`, ptxas
register dumps. Each emits its own table keyed on its own slice. Reading them
means running 5–6 commands and mentally joining on kernel name + run identity.

That disjointness has already cost us: per `ds4-spark/tools/perf/PLAN_tool_improvements.md`,
a bisect chased a "regression" that was environmental, because **throughput and
accept-rate were never shown together** — we couldn't see that accept-rate was
flat while t/s moved. The gamut report's job is to make that class of mistake
impossible: every number that explains a decode result, on one page, from one
run-set.

### Requirements (each maps to a past pain point)

- **R1** — accept-rate beside throughput (the false-bisect fix; PLAN T.1.1)
- **R2** — steady-state vs warmup separated; wall / kernel / idle split (PLAN T.1.2, T.2.2)
- **R3** — host-overhead fraction visible (gap total, not buried; PLAN T.2.2)
- **R4** — per-kernel occupancy + limiter beside time (answers "why is `moe_down` slow")
- **R5** — per-kernel roofline %-of-peak beside time (answers "how much headroom")
- **R6** — global HW-metric verdict: SM-issue / occupancy / achieved BW (stalled? bandwidth-bound?)
- **R7** — optional A/B delta vs a baseline tag, with a noise floor (regression gate; PLAN T.3.3)

## The join model

Two tiers, because not everything is keyed the same way.

### Tier A — run-set scalars (header + verdict)

One value per run-set: build change-id, model + quant, MTP model, GB10 / driver
/ CUDA / clocks / power-cap, prompt + ctx, prefill t/s, decode t/s,
accept-rate, tokens/iter, kvcache bytes, wall / kernel / idle split, and the
windowed GPU HW metrics (SM issue, SMs active, tensor, compute-warps, achieved
GB/s). These come from `bench-v2` (throughput/accept) + the `gb20b` capture
(HW metrics, windowed to the decode phase) + static run metadata.

### Tier B — per-kernel spine (the joined table)

**Join key = canonical kernel slug.** Every kernel-keyed tool contributes
columns:

| source | columns it owns | keyed on |
| ------ | --------------- | -------- |
| plain cuda trace (`nsys_top`) | launches, total ms, avg µs, **%time** | demangled name |
| `roofline` | est BW GB/s, %HBM peak, est TOPS, %compute peak, class | demangled name |
| `hbm_report` | grid, block, smem, floor µs, %peak headroom | demangled name |
| ptxas (`nvcc -Xptxas=-v`) | regs/thread, smem, **theoretical occupancy**, limiter | mangled name |
| `gb20b` trace (windowed) | **per-kernel SM-issue %, achieved occupancy** | demangled name |

Two non-obvious decisions baked in here:

1. **Timing comes from the plain trace, not the `gb20b` run.** The `gb20b`
   metric set perturbs kernel timing, so `%time` / ms must come from a clean
   `-t cuda` capture. The two captures are joined by kernel slug, not by
   timeline (their timelines differ).

2. **Per-kernel HW metrics are sourced *within* the `gb20b` run** by windowing
   its own `GPU_METRICS` samples to each kernel's `[start,end]`. At 20 kHz
   (~50 µs/sample) this is only trustworthy for the big kernels: `moe_down`
   (~470 µs → ~9 samples) is solid; `rms_norm` (~16 µs → <1 sample) is not.
   **Rule: show per-kernel SM-issue/occupancy only when sample-count ≥ 4, else
   `—`.** This is the one genuinely novel join — nobody in glint / ds4-spark /
   llama.cpp / vllm does per-kernel HW-metric attribution — and the sample-floor
   keeps it honest.

### Kernel-name normalization (the crux)

The slug functions differ across tools (`nsys_top.short_name`,
`roofline.short`, `hbm` uses raw). The gamut tool owns **one** canonical slug
and applies it uniformly; it joins on the full demangled name and slugs only
for display.

- Templated kernels (`matmul_q8_0_preq_batch_share_warp<2>` vs `<3>`) stay
  **distinct rows** — different instantiations have different regs/occupancy.
- ptxas names are mangled; reconcile via `c++filt` or the nsys `StringIds`
  table (which carries both `shortName` and `demangledName`).
- Offer an `--alias` fold (PLAN T.4.3) for when you *want* `mmq_q8.*` collapsed
  to one row.

### Run-set manifest

A gamut report joins data from ≥2 nsys captures + 1 ptxas compile + 1 bench run.
They're only comparable if they're the **same build, model, prompt**. A tiny
manifest (JSON) records `{change_id, model, mtp, prompt, ctx, gpu, driver,
cuda, captures:{plain, gb20b}, ptxas, bench_tag}` and is the unit `gamut.py`
ingests. This is also the diff unit for A/B (R7).

## Mockup (grounded in today's `gb10-on-upstream` run)

```
# ds4 gamut — gb10-on-upstream @ c5b39429

run-set  knight / ctx2048 / MTP draft=2        2026-05-25 19:30
build    c5b39429 (upstream ad0209f6 + PRs #13/#14/#15)
model    DeepSeek-V4-Flash-IQ2XXS-w2Q2K …  + MTP-Q4K-Q8_0-F32
hw       GB10 sm_121a · drv 580.142 · CUDA 13.0 · ~270 GB/s · power 100%

THROUGHPUT                                  GPU HW (decode-windowed)
  prefill   408.9 t/s                         SMs active     88.0%
  decode     16.3 t/s   (IQR 16.2–16.4)       SM issue        8.6%   ← stalled
  accept     88.2%   tokens/iter 1.76         tensor          0.8%
  kvcache    52.2 MB                          compute warps  36.8%
                                              achieved BW    ~90 GB/s = 34% peak
TIME SPLIT
  wall 3.06 s · kernel 0.74 s (24%) · idle 2.32 s (76%)   ← host/launch gaps

VERDICT  memory-latency-bound at moderate occupancy. Not compute/tensor/issue-
         throughput bound. Top lever: moe_down occupancy (168 regs → 17%).

PER-KERNEL  (time ← plain trace · HW ← gb20b windowed, ≥4 samples)
| kernel                        | %t   | ms    | calls | regs | occ→ach | %HBM | class      | SMiss |
|-------------------------------|-----:|------:|------:|-----:|--------:|-----:|------------|------:|
| matmul_q8_0…share_warp<3>     | 22.5 | 585.3 |  6644 |   48 | 83→79%  |  61% | mem-bound  |  9.1% |
| moe_down_expert_tile8_row32   | 18.4 | 477.8 |   989 |  168 | 17→16%  |  44% | UNDER-util |  6.8% |
| moe_gate_up_mid…tile8_row32   | 15.6 | 406.3 |   989 |   96 | 33→31%  |  52% | mem-bound  |  7.4% |
| cutlass f16 wmma 32x32        | 12.3 | 318.6 |   989 |   —  |   —     |   —  | tensor     |   —   |
| matmul_q8_0_preq_warp8        |  5.8 | 149.9 |   227 |   48 | 83→—    |  58% | mem-bound  |   —   |

TOP GAPS (steady-state)
  142 µs  between embed_token … and rms_norm_plain   ← per-token launch gap
   …
A/B  vs phase-c.1:  decode +1.2% ✓ · accept +0.0pp · no kernel >2% shift
```

(Numbers are illustrative where a source isn't wired yet — `regs`/`occ` and
per-kernel `SMiss` need the ptxas + windowing passes; `class`/`%HBM` need the
roofline estimators confirmed against current kernel names.)

## Orchestration

`tools/perf/gamut.py` — ingest-and-join, not a re-profiler:

1. **collect** (optional) — run the three captures for a run-set: plain
   `-t cuda` trace, `gb20b` metrics trace, ptxas `-Xptxas=-v` compile; plus a
   `bench-v2` run for throughput/accept. Writes the manifest.
2. **join** — slug-normalize, join Tier B on kernel slug, roll up Tier A,
   window `gb20b` per-phase and per-kernel (sample-floor gate).
3. **emit** — one Markdown report (TUI-readable, status-marks style) + a JSON
   sidecar (the diff unit for A/B / regression history).

Reuses the existing tools as libraries/subprocesses where possible
(`nsys_top`, `roofline`, `hbm_report`, `gaps`, `bench-v2`); adds only the
join + the two missing inputs.

## New pieces this needs (don't exist yet)

- **`gpu_metrics.py`** — extract gb20b metrics (IDs 7/8/9/16) from the sqlite,
  windowed to a phase or a kernel `[start,end]`. Prototyped already (the query
  behind today's HW-metric table). This is the Tier-1 gap from the survey.
- **NVTX phase scopes** — push/pop around prefill vs decode (and ideally each
  captured graph; PLAN T.3.2). Without these, phase windowing falls back to a
  timestamp heuristic. ~30 LOC C in the decode loop.
- **bench JSON output** — `bench-v2` already has the sqlite; emit a JSON row so
  `gamut.py` doesn't re-parse stdout (PLAN T.4.1).
- **manifest writer** — small; ties a run-set together.

## Resolved (questions from the design phase)

Home = mainline `ds4/tools/perf` (accumulate in the fork). Built single-run
first; per-kernel HW metrics are in (windowed, sample-gated); Markdown + JSON
sidecar from the start. A/B-delta over the JSON sidecar is the next mode.

## CUDA-tooling survey findings (2026-05-25)

What the GB10 actually exposes, established by probing — relevant because it
bounds what the report can ever show:

- **`gb20b` is the limited iGPU metric set: 19 metrics, no DRAM-throughput and
  no L2 counters.** So device-wide *measured* achieved bandwidth is not
  available from nsys here — the report's `%peakBW` is necessarily a byte-model
  estimate vs the synthetic `membw` read ceiling.
- **ncu hardware counters work** (no permission error), incl. warp **stall
  reasons** and achieved occupancy — but ncu's `dram__throughput` returns `n/a`
  on GB10 too. So no profiler gives measured DRAM BW on this chip; `membw.cu`
  is the only bandwidth ground truth.
- **ncu default kernel-replay segfaults on ds4's real kernels** (the 80 GB
  HBM-resident VMM model breaks save/restore). `--replay-mode application`
  works but re-runs the whole app per pass (≈3–4 min per kernel) — opt-in only.
- Cross-check captured: `moe_down` stall breakdown is **long_scoreboard 15.39
  (~67%)** vs lg_throttle 4.47, mio 0.0 → **memory-latency-bound, not
  bandwidth-throttled**, at 16.0% achieved occupancy. ncu's occupancy (16.0%)
  matches the gb20b-windowed `compute_warps` (15.9%), validating the windowing.

## Next additions (ranked)

1. **`ncu_stalls.py` (opt-in)** — app-replay the top N kernels, emit the stall
   breakdown; `gamut --ncu` joins a `stall` column. Highest diagnostic value
   (latency-vs-BW is the whole question); slow, so off the default path.
2. **`gamut_diff.py`** — A/B over two JSON sidecars, regression-flagged.
3. **NVTX phase + captured-graph scopes** (~30 LOC C) — retire the
   `embed_token_hc` timestamp heuristic; enable per-graph rollups (PLAN T.3.2).
4. **Cheap report columns** — headroom multiplier (glint `hbm_report`),
   arithmetic intensity (extend `roofline_estimate` to return flops), and a
   steady-state launch-gap section (`gaps.py --steady-state`).
5. **Tighten roofline byte models** — the `%peakBW` estimate is only as good as
   the per-kernel byte models in `perflib.roofline_estimate`.

## Scope (rough, for the v1 single-run report)

- Files: `tools/perf/gamut.py` (new, ~250–350 LOC), `tools/perf/gpu_metrics.py`
  (new, ~80 LOC), `~30 LOC` C NVTX scopes in the decode loop, `~15 LOC` JSON
  emit in `bench-v2`.
- Named units: 1 join driver, 1 metric extractor, 1 manifest schema, 2 new
  ingest paths (ptxas regs, gb20b windows); reuses 5 existing tools.
- Verification: run on `gb10-on-upstream` knight/ctx2048 (the run mocked above)
  and eyeball against the individual tool outputs for agreement.
- Risk: public API no · data migration no · cross-module no (tooling only) ·
  reversible yes · external blocker no (depends on porting ds4-spark suite up,
  if mainline home is chosen).
```
