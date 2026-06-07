"""gamut.bench — throughput matrix under a GPU/thermal monitor (replaces bench-with-monitor.sh).

Runs ds4-bench across a matrix of decode paths (plain / mtp-greedy / mtp-sample)
x N iters, back-to-back under one continuous GpuMonitor stream so per-cell
thermal/throttle signals and the transitions between them are visible. Each cell
is one ds4-bench invocation (a ctx-frontier sweep); results parse out of the
per-cell bench.csv. No bash, no respawn loops, no quoting traps.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .capture import ROOT, PERF
from .monitor import GpuMonitor

MODEL_DEFAULT = str(Path.home() / "models/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf")
MTP_DEFAULT = str(Path.home() / "models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf")
PROMPT_DEFAULT = str(ROOT / "tests/long_context_story_prompt.txt")

# (name, use_mtp, use_temp, fast_verify)
# 4 decode paths × high-accuracy (deterministic verify, FAST_VERIFY off) vs
# fast (DS4_CUDA_FAST_VERIFY=1). Grouped path-then-accuracy so each pair reads
# adjacently in the summary. FAST_VERIFY also reshapes the *plain* path (heads8
# dispatch + n=1 GEMM → cuBLAS), so it's a real dimension for plain too.
MATRIX_CELLS = [
    ("plain-greedy-acc",   False, False, False),
    ("plain-greedy-fast",  False, False, True),
    ("plain-sample-acc",   False, True,  False),
    ("plain-sample-fast",  False, True,  True),
    ("mtp-greedy-acc",     True,  False, False),
    ("mtp-greedy-fast",    True,  False, True),
    ("mtp-sample-acc",     True,  True,  False),
    ("mtp-sample-fast",    True,  True,  True),
]


@dataclass
class BenchCfg:
    label: str
    model: str = MODEL_DEFAULT
    mtp: str = MTP_DEFAULT
    prompt_file: str = PROMPT_DEFAULT
    matrix: bool = False
    use_mtp: bool = True
    use_temp: bool = False
    iters: int = 1
    warmup: int = 0               # extra leading iters per cell, excluded from stats
    ctx_start: int = 4096
    ctx_max: int = 32768
    step_mul: int = 2
    gen_tokens: int = 32
    fast_verify: bool = False
    prewarm: bool = True          # read model+MTP into page cache before the cells
    cooldown: bool = True         # wait for the GPU to cool between cells (anti-soak)
    cooldown_c: int = 55          # cool to <= this (°C) before the next cell starts
    cooldown_max_s: int = 240     # cap the wait so a stuck sensor can't hang the run
    extra_env: dict | None = None


def _cell_args(cfg: BenchCfg, use_mtp: bool, use_temp: bool, csv_path: str) -> list[str]:
    a = ["--cuda", "--warm-weights", "--power", "85",
         "--prompt-file", cfg.prompt_file, "-m", cfg.model,
         "--ctx-start", str(cfg.ctx_start), "--ctx-max", str(cfg.ctx_max),
         "--step-mul", str(cfg.step_mul), "--gen-tokens", str(cfg.gen_tokens)]
    if use_mtp:
        a += ["--mtp", cfg.mtp, "--mtp-draft", "2"]
    if use_temp:
        a += ["--temp", "1.0", "--top-p", "0.95", "--seed", "1234"]
    a += ["--csv", csv_path]
    return a


def _parse_bench_csv(path: str) -> list[dict]:
    try:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(open(path))]
    except OSError:
        return []


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return x


def _fit_prompt(prompt_file: str, ctx_max: int, out_dir: Path) -> str:
    """ds4-bench refuses to start when the prompt tokenizes to fewer tokens than
    --ctx-max ('prompt has N tokens, need at least --ctx-max=M'). The story prompt
    is ~30.5k tokens, short of the 32k frontier, so stitch copies into the run dir
    until it's long enough (rough ~5 bytes/token + slack). Ported from the legacy
    bench-with-monitor harness."""
    need_bytes = ctx_max * 5 + 8192
    try:
        src_bytes = os.path.getsize(prompt_file)
    except OSError:
        return prompt_file
    if src_bytes >= need_bytes or src_bytes == 0:
        return prompt_file
    copies = (need_bytes + src_bytes - 1) // src_bytes
    stitched = out_dir / "prompt.txt"
    data = Path(prompt_file).read_bytes()
    stitched.write_bytes(data * copies)
    print(f"## prompt stitched: {copies}x copies → {stitched} "
          f"({stitched.stat().st_size} bytes, target≈{ctx_max} tok)", flush=True)
    return str(stitched)


def _prewarm_cache(paths: list[str]) -> None:
    """Sequentially read the model/MTP files into the OS page cache once, so the
    per-cell ds4-bench reloads hit cache instead of faulting ~80 GB from disk.
    The matrix spawns a fresh process per cell (model loads each time); without
    this the first cell pays a ~5-minute cold device-copy and later cells re-fault
    if the cache got churned. Turns the 9x-cold tax into 1x. Best-effort."""
    seen = set()
    for p in paths:
        rp = os.path.realpath(p)
        if not p or rp in seen or not os.path.exists(rp):
            continue
        seen.add(rp)
        gib = os.path.getsize(rp) / (1 << 30)
        t0 = time.time()
        try:
            with open(rp, "rb", buffering=0) as f:
                while f.read(1 << 24):   # 16 MiB chunks; kernel reads ahead into cache
                    pass
        except OSError as e:
            print(f"## prewarm skipped {os.path.basename(rp)}: {e}", flush=True)
            continue
        print(f"## prewarmed {os.path.basename(rp)} ({gib:.1f} GiB) into page cache "
              f"in {time.time() - t0:.1f}s", flush=True)


def _gpu_temp() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _cooldown(mon, target_c: int, max_s: int) -> None:
    """Block until the GPU cools to <= target_c (capped at max_s). Back-to-back
    cells otherwise heat-soak: the 8-cell matrix climbs to ~88°C and the firmware
    throws sw_thermal throttle on the later cells, depressing their t/s. Each
    cell's own ~4-min startup (warm-weights + device-copy) is only light load, so
    it doesn't shed enough heat on its own — hence an explicit gate here."""
    t = _gpu_temp()
    if t is None or t <= target_c:
        return
    mon.set_stage("cooldown")
    t0 = time.time()
    while time.time() - t0 < max_s:
        time.sleep(5)
        t = _gpu_temp()
        if t is None or t <= target_c:
            break
    print(f"## cooldown {time.time() - t0:.0f}s → {t}°C (target {target_c})", flush=True)
    mon.set_stage("idle")


def run(cfg: BenchCfg) -> dict:
    out = PERF / "runs" / cfg.label
    out.mkdir(parents=True, exist_ok=True)
    ds4_bench = str(ROOT / "ds4-bench")
    Path("/tmp/ds4.lock").unlink(missing_ok=True)
    cfg.prompt_file = _fit_prompt(cfg.prompt_file, cfg.ctx_max, out)
    if cfg.prewarm:
        _prewarm_cache([cfg.model] + ([cfg.mtp] if cfg.use_mtp or cfg.matrix else []))

    cells = MATRIX_CELLS if cfg.matrix else [("single", cfg.use_mtp, cfg.use_temp, cfg.fast_verify)]
    base_env = dict(os.environ)
    if cfg.extra_env:
        base_env.update(cfg.extra_env)

    results: dict = {"label": cfg.label, "cells": {}}
    with GpuMonitor(str(out)) as mon:
        for ci, (name, use_mtp, use_temp, fast) in enumerate(cells):
            if ci > 0 and cfg.cooldown:
                _cooldown(mon, cfg.cooldown_c, cfg.cooldown_max_s)
            env = dict(base_env)
            env.pop("DS4_CUDA_FAST_VERIFY", None)
            if fast:
                env["DS4_CUDA_FAST_VERIFY"] = "1"
            cell_rows = []
            total_iters = cfg.warmup + cfg.iters
            for idx in range(1, total_iters + 1):
                is_warmup = idx <= cfg.warmup
                it = idx if is_warmup else idx - cfg.warmup
                # cool between every iteration (not just cells): back-to-back
                # iters of the same cell heat-soak identically, and the whole
                # point of N runs is iid samples, not a thermal staircase.
                if cfg.cooldown and (ci > 0 or idx > 1):
                    _cooldown(mon, cfg.cooldown_c, cfg.cooldown_max_s)
                stage = f"{name}#{'w' if is_warmup else ''}{it}"
                mon.set_stage(stage)
                multi = cfg.matrix or total_iters > 1
                cell_dir = out / name / f"iter-{'w' if is_warmup else ''}{it:03d}" if multi else out
                cell_dir.mkdir(parents=True, exist_ok=True)
                csv_path = str(cell_dir / "bench.csv")
                log_path = str(cell_dir / "bench.log")
                print(f"## [{stage}] {time.strftime('%H:%M:%S')} "
                      f"(verify={'fast' if fast else 'det'})"
                      f"{' [warmup — discarded]' if is_warmup else ''}", flush=True)
                with open(log_path, "w") as lf:
                    rc = subprocess.run([ds4_bench, *_cell_args(cfg, use_mtp, use_temp, csv_path)],
                                        env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    print(f"## [{stage}] FAILED rc={rc} — continuing", flush=True)
                if not is_warmup:
                    cell_rows.append({"iter": it, "rows": _parse_bench_csv(csv_path), "rc": rc})
                mon.set_stage("idle")
            results["cells"][name] = _aggregate_cell(cell_rows)

    summ = mon.summary()
    results["monitor"] = summ
    (out / "summary.json").write_text(json.dumps({"results": results, "monitor": summ}, indent=2))
    (out / "summary.txt").write_text(_render_summary(results, summ))
    return results


def _aggregate_cell(cell_rows: list[dict]) -> dict:
    """Mean ± sample-std gen_tps / prefill_tps per ctx across measured iters."""
    by_ctx: dict[int, dict[str, list[float]]] = {}
    for cr in cell_rows:
        for row in cr["rows"]:
            ctx = int(row.get("ctx_tokens", 0))
            d = by_ctx.setdefault(ctx, {"gen": [], "pf": []})
            if isinstance(row.get("gen_tps"), float):
                d["gen"].append(row["gen_tps"])
            if isinstance(row.get("prefill_tps"), float):
                d["pf"].append(row["prefill_tps"])
    out = {}
    for ctx, d in sorted(by_ctx.items()):
        out[ctx] = {"gen_tps": _mean(d["gen"]), "gen_std": _std(d["gen"]),
                    "prefill_tps": _mean(d["pf"]), "prefill_std": _std(d["pf"]),
                    "n": len(d["gen"])}
    return out


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def _std(xs):
    return round(statistics.stdev(xs), 2) if len(xs) >= 2 else None


def _render_summary(results: dict, mon: dict) -> str:
    L = [f"# gamut bench · {results['label']}", ""]
    for name, ctxs in results["cells"].items():
        parts = []
        for c, d in ctxs.items():
            pm = f"±{d['gen_std']}" if d.get("gen_std") is not None else ""
            parts.append(f"{c // 1024}k:{d['gen_tps']}{pm}(n{d['n']})")
        L.append(f"{name:18} decode  " + "  ".join(parts))
    L.append("")
    g = mon.get("busy") or {}
    if g:
        L.append(f"GPU busy: sm_mean={_f(g.get('sm_mean'))}MHz peak={_f(g.get('sm_peak'))} "
                 f"power_mean={_f(g.get('power_mean'))}W temp_peak={_f(g.get('temp_peak'))}C")
        thr = g.get("throttled") or {}
        if thr:
            L.append("throttle (busy samples): " + ", ".join(f"{k}×{v}" for k, v in thr.items()))
    return "\n".join(L)


def _f(x):
    return f"{x:.0f}" if isinstance(x, (int, float)) else "—"
