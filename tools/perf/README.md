# tools/perf — GB10 decode perf suite

CLI tooling for profiling ds4 decode on the DGX Spark (GB10, sm_121a). The
headline is **`gamut.py`** — one joined report that pulls kernel time, GPU
hardware metrics, register/occupancy, and roofline onto a single page so you
don't hand-join six tool outputs (the disjointness that once cost us a false
regression bisect — see `docs/gamut-report-design.md`).

Pure stdlib Python (sqlite3) + one CUDA file. No external deps.

## Pieces

| file | what |
| ---- | ---- |
| `membw.cu` | synthetic memory-bandwidth probe → the roofline denominator |
| `perflib.py` | shared primitives: slug, phase windows, sqlite/metric extraction, ptxas parse, roofline, launch gaps |
| `gpu_metrics.py` | extract gb20b GPU HW metrics (SM-issue/occupancy/tensor), phase-windowed or per-kernel |
| `ncu_stalls.py` | per-kernel warp **stall reasons** via Nsight Compute (opt-in, slow) → `gamut --ncu` |
| `gamut.py` | the joined report (Markdown + JSON sidecar + HTML) |
| `gamut_html.py` | render a report JSON as a self-contained HTML page (inline SVG, no deps) |

## Calibrate the roofline peak (once per machine)

```
/usr/local/cuda/bin/nvcc -O3 --use_fast_math \
  -gencode=arch=compute_121a,code=sm_121a -o /tmp/membw tools/perf/membw.cu
/tmp/membw
```

On this GB10 the DRAM plateau is **read ~236 GB/s** (87% of the 273 GB/s LPDDR5X
theoretical; ignore the in-L2 spike at ≤16 MB). `perflib.HW.hbm_read_gbps`
holds this measured read ceiling — it's the denominator `gamut.py` uses for
`%peakBW`, not a spec guess. (`roofline.py`'s old 546 was a 2× error.)

## Capture the inputs

```
M=ds4flash.gguf
MTP=/home/trevor/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
ARGS="-m $M --mtp $MTP -p knight -n 48 --temp 0 --nothink -sys \"\""

# 1. plain CUDA trace (clean kernel time)
nsys profile -o /tmp/p -t cuda --sample none --force-overwrite true ./ds4 $ARGS
# 2. gb20b hardware metrics (GB10 set is gb20b, NOT gb10b)
nsys profile --gpu-metrics-devices=0 --gpu-metrics-set=gb20b \
  --gpu-metrics-frequency=20000 -o /tmp/gm --force-overwrite true ./ds4 $ARGS
# 3. register / occupancy
/usr/local/cuda/bin/nvcc -O3 --use_fast_math \
  -gencode=arch=compute_121a,code=sm_121a -Xptxas=-v -c -o /tmp/x.o ds4_cuda.cu 2>/tmp/ptxas.txt
# 4. accept-rate telemetry
DS4_MTP_TIMING=1 ./ds4 $ARGS >/tmp/accept_run.txt 2>&1
```

`nsys stats`/`gamut` auto-export the `.nsys-rep` to `.sqlite` on first use; or
pass an already-exported `.sqlite`.

## Streaming TTFT / throughput diagnostic

For the OpenAI-compatible server, use the helper below to measure client TTFT,
prompt timing from `ds4-server` logs, and a basic GPU/RAM/KV snapshot:

```sh
python3 tools/perf/gb10_stream_diag.py \
  --warmup \
  --prompt "Rispondi con 128 parole ciao separate da spazi e senza punteggiatura." \
  --max-tokens 128 \
  --json /tmp/ds4-stream-diag.json
```

It prints a Markdown summary plus a log tail and a quick readout telling you
whether DSpark's speculative timing path was observed.

## Run the report

```
tools/perf/gamut.py \
  --plain /tmp/p.sqlite --metrics /tmp/gm.sqlite \
  --ptxas /tmp/ptxas.txt --accept /tmp/accept_run.txt \
  --prefill-tps 408.9 --decode-tps 16.32 --kvcache-mb 52.2 \
  --label "gb10-on-upstream @ c5b39429" --json /tmp/gamut.json
```

See `runs/gb10-on-upstream.md` for an example. The `--json` sidecar is the unit
for A/B / regression diffing (a `gamut_diff.py` is the natural next tool).

The per-kernel table also carries **AI** (flop/byte: <~1 mem-bound, >~10
compute-bound) and a **launch-gaps** section (steady-decode host/scheduling
idle). `headroom` (measured/floor µs) is in the JSON sidecar.

### HTML report

Add `--html PATH` to `gamut.py` (or run `gamut_html.py report.json -o out.html`
on a saved sidecar). It's a **self-contained** page — inline SVG charts, no CDN
or JS deps — so it serves and renders offline:

```
tools/perf/gamut.py --plain … --metrics … --html /tmp/gamut.html --json /tmp/gamut.json
( cd $(dirname /tmp/gamut.html) && python3 -m http.server 8009 --bind 127.0.0.1 )
# open http://127.0.0.1:8009/gamut.html  (ssh -L 8009:127.0.0.1:8009 if remote)
```

Charts (drawn only where the data supports them): kernel time distribution,
GPU HW verdict, theoretical-vs-achieved occupancy, a roofline scatter (AI vs
% peak BW with the 236 GB/s ceiling), warp-stall composition (with `--ncu`),
and top launch gaps.

### Stall reasons (opt-in, slow — needs ncu)

The gb20b set can't say *why* warps stall. ncu can, but its default
kernel-replay segfaults on ds4's HBM-resident VMM model, so `ncu_stalls.py`
forces application-replay (re-runs the app per pass, minutes):

```
tools/perf/ncu_stalls.py --out /tmp/ncu.json \
  --kernels "moe_down_expert_tile8_row32|matmul_q8_0_preq_batch_share_warp|moe_gate_up_mid_expert_tile8_row32" \
  --launch-skip 200 --launch-count 12 -- \
  ./ds4 -m ds4flash.gguf --mtp $MTP -p knight -n 24 --temp 0 --nothink -sys ""
```

Then add `--ncu /tmp/ncu.json` to `gamut.py` for a `stall` column (dominant
issue-stall reason + %). `long_scoreboard` = memory-latency-bound;
`lg_throttle`/`mio_throttle` = bandwidth-throttled; `barrier` = sync-bound.
Re-parse an existing report with `ncu_stalls.py --import-rep X.ncu-rep`.

## How the join works (the non-obvious bits)

- **Join key = canonical kernel slug** (`perflib.canon_slug`), never truncated;
  truncation is display-only (`disp`) so two kernels can't collide into a row.
  nsys renders templates as `<(int)3>`, c++filt as `<3>` — both normalize to `<3>`.
- **Timing from `--plain`, HW from `--metrics`, joined by slug — never by
  timeline.** The gb20b set perturbs kernel timing, so ms/%t must come from the
  clean trace; the two captures only share kernel *names*.
- **Per-kernel HW metrics** are windowed inside the gb20b run's own timeline and
  gated at ≥4 samples (20 kHz can't resolve a 16 µs kernel; those show `—`).
- **Phase windowing** keys on `embed_token_hc` (one launch per decode token —
  the engine has no NVTX yet). Steady state skips the first N warmup tokens.
- **`%peakBW` is estimated** from per-kernel byte models for the kernels with a
  matcher in `perflib.roofline_estimate`; others show `—`. The byte models are
  the part most worth tightening as kernels change.
