# tools/perf/longctx — long-context decode improvement harness

Built on top of `tools/perf/` (gamut + ds4-bench). Adds the context-slope
measurement loop: how does decode tok/s degrade from 8k → 130k, and which
kernels are responsible.

Headline metric is **slope of decode tok/s per ctx doubling**, fit as
`tps(ctx) = a − b·log2(ctx/8192)`. Two scalars:

- `tps@8k = a` — short-ctx ceiling (weights/dense regime).
- `slope_per_doubling = b` — tok/s lost per ctx doubling.

## Pieces

| file | role |
| ---- | ---- |
| `thermal_guard.sh` | `cooldown_wait` to <55C + `thermal_snapshot` JSON; sourced by every script. |
| `prime_kv.sh` | One-shot: prefill to long ctx + `/save`. Sha is keyed to a layout fingerprint so a KV-format change auto-invalidates. |
| `ablate.sh` | Inner loop. `/switch <sha>` + decode-only, one toggle at a time. Fast. |
| `sweep.sh` | Outer loop. ds4-bench across 6 frontiers, MTP-on + MTP-off, nsys + gb20b metrics per frontier, then `gamut-cli report` per frontier. |
| `prompts/ensure_longform.sh` | Builds `prompts/longform.txt` (gitignored, ~240k tokens) from the in-tree story prompt; sweep/prime auto-run it. |
| `gamut_sweep_diff.py` | Baseline vs candidate slope fit + per-kernel growth + flatten/lift/trap classification. |

## End-to-end

```
# 0. once per machine (calibrate gamut roofline)
/usr/local/cuda/bin/nvcc -O3 --use_fast_math \
    -gencode=arch=compute_121a,code=sm_121a -o /tmp/membw tools/perf/membw.cu

# 1. prime a 65k KV once (reused across ablation runs on the same build)
#    (prompts/longform.txt is built automatically on first use)
sha=$(tools/perf/longctx/prime_kv.sh --ctx 65536)

# 2. inner ablation matrix at 65k (seconds per toggle)
tools/perf/longctx/ablate.sh --kv-sha "$sha" \
    --out tools/perf/runs/ablate-$(date +%s)

# 3. outer ctx sweep, both MTP variants (slow; nsys per frontier)
tools/perf/longctx/sweep.sh --out tools/perf/runs/sweep-baseline

# ... make a change, rebuild, sweep again ...
tools/perf/longctx/sweep.sh --out tools/perf/runs/sweep-cand

# 4. diff
tools/perf/longctx/gamut_sweep_diff.py \
    --baseline tools/perf/runs/sweep-baseline \
    --candidate tools/perf/runs/sweep-cand \
    --mtp-tag nomtp
```

## Cooldown discipline

Every script in this directory `source`s `thermal_guard.sh` and calls
`cooldown_wait` before each measurement (and `thermal_snapshot pre/post` to
write a per-iter thermal JSON next to the artifact). Default target is
55°C; tune via `THERMAL_TARGET_C=...`. Per
`memory/project_spark-ebf0-thermal-shutoff.md`, long unmoderated sweeps have
hard-powered-off the board; the harness defaults are conservative on purpose.

## Verdict classification (`gamut_sweep_diff.py`)

- **PARETO**: tps@8k up ≥5% AND slope flatter ≥15%.
- **LIFT**: tps@8k up ≥5%, slope ~unchanged.
- **FLATTEN**: slope flatter ≥15%, tps@8k ~unchanged.
- **TRAP**: tps@8k up but slope worse (or vice versa) — the harness
  surfaces these explicitly so a short-ctx win doesn't hide a long-ctx loss.

## Status

Source-audited against the binaries (2026-06-09): `ds4-bench --csv` column
order, `--kv-restore`, the agent's `/save` flush + `+DWARFSTAR_SAVED` marker,
and the gamut JSON keys (`throughput.decode_tps`, kernel rows `kernel`/`avg_us`)
are all wired to what the code actually emits. `ablate.sh` reads `ds4-bench
--csv` (not `--trace`). `sweep.sh` exports the nsys `.nsys-rep` to sqlite
itself before invoking `gamut-cli report`.

Not yet exercised end-to-end on hardware — the first full sweep may still
shake out runtime issues (nsys overhead at the 131k frontier, gb20b capture
size). Run a single-frontier smoke first:

```
tools/perf/longctx/sweep.sh --frontiers 8192 --out /tmp/sweep-smoke
```
