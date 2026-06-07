"""gamut.metrics — gb20b GPU metrics join · ptxas regs/occupancy · roofline · ncu stalls.

The analysis layer that turns raw kernel timings into a bottleneck read:
  - gb20b GPU_METRICS time-series joined to kernels by timestamp window
  - `nvcc -Xptxas=-v` register/occupancy parse
  - best-effort roofline (bytes/flops/%peakBW/headroom) per decode kernel
  - ncu warp-stall reasons (opt-in, slow) parsed from an .ncu-rep
"""

from __future__ import annotations

import bisect
import csv
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import hw, trace

NCU = "/usr/local/cuda/bin/ncu"


# ---- gb20b GPU hardware metrics ---------------------------------------------

def has_gpu_metrics(con: sqlite3.Connection) -> bool:
    return trace.has_table(con, "GPU_METRICS")


def busy_metric_avgs(con: sqlite3.Connection, win: tuple[int, int],
                     keys: list[str]) -> dict[str, float]:
    """Average each metric over *busy* samples (SMs Active > threshold) in win,
    excluding inter-token launch gaps so the numbers describe compute time."""
    lo, hi = win
    out: dict[str, float] = {}
    busy_sub = ("SELECT timestamp FROM GPU_METRICS WHERE metricId=? AND value>? "
                "AND timestamp BETWEEN ? AND ?")
    for k in keys:
        r = con.execute(
            f"SELECT AVG(value) v FROM GPU_METRICS WHERE metricId=? "
            f"AND timestamp IN ({busy_sub})",
            (hw.METRIC[k], hw.METRIC["sms_active"], hw.BUSY_SMS_ACTIVE, lo, hi),
        ).fetchone()
        out[k] = float(r["v"]) if r["v"] is not None else float("nan")
    return out


def per_kernel_metric(con: sqlite3.Connection, win: tuple[int, int],
                      intervals: dict[str, list[tuple[int, int]]],
                      key: str, min_samples: int = 4) -> dict[str, float]:
    """Average one metric within each kernel's launch windows (same run). Only
    kernels with >= min_samples hits (short kernels get too few @ 20 kHz)."""
    lo, hi = win
    rows = con.execute(
        "SELECT timestamp, value FROM GPU_METRICS WHERE metricId=? "
        "AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (hw.METRIC[key], lo, hi),
    ).fetchall()
    ts = [int(r["timestamp"]) for r in rows]
    val = [float(r["value"]) for r in rows]
    out: dict[str, float] = {}
    for slug, ivs in intervals.items():
        tot, cnt = 0.0, 0
        for st, en in ivs:
            i = bisect.bisect_left(ts, st)
            j = bisect.bisect_right(ts, en)
            for v in val[i:j]:
                tot += v; cnt += 1
        if cnt >= min_samples:
            out[slug] = tot / cnt
    return out


# ---- ptxas register / occupancy ---------------------------------------------

@dataclass
class PtxasInfo:
    regs: int = 0
    smem: int = 0           # bytes
    spills_store: int = 0
    spills_load: int = 0


def parse_ptxas(text: str) -> dict[str, PtxasInfo]:
    """Parse `nvcc -Xptxas=-v` stderr into slug -> PtxasInfo."""
    out: dict[str, PtxasInfo] = {}
    cur: str | None = None
    entry_re = re.compile(r"Compiling entry function '([^']+)'")
    regs_re = re.compile(r"Used (\d+) registers")
    smem_re = re.compile(r"(\d+) bytes smem")
    spill_re = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
    for line in text.splitlines():
        m = entry_re.search(line)
        if m:
            cur = trace.canon_slug(trace.demangle(m.group(1)))
            out.setdefault(cur, PtxasInfo())
            continue
        if cur is None:
            continue
        info = out[cur]
        if (m := regs_re.search(line)):
            info.regs = max(info.regs, int(m.group(1)))
        if (m := smem_re.search(line)):
            info.smem = max(info.smem, int(m.group(1)))
        if (m := spill_re.search(line)):
            info.spills_store = int(m.group(1)); info.spills_load = int(m.group(2))
    return out


def theoretical_occupancy(regs: int, smem: int, threads: int = 256,
                          h: hw.HW | None = None) -> float:
    """Crude theoretical occupancy from regs/thread + smem/block (reg/smem only)."""
    h = h or hw.HW()
    if regs <= 0:
        return float("nan")
    warps_per_block = max(1, threads // 32)
    reg_warps = (h.regs_per_sm // (regs * 32))
    reg_warps = (reg_warps // warps_per_block) * warps_per_block
    smem_warps = h.max_warps if smem <= 0 else ((h.smem_per_sm // smem) * warps_per_block)
    return min(h.max_warps, reg_warps, smem_warps) / h.max_warps


# ---- roofline (decode byte models; best-effort) -----------------------------

@dataclass
class Roofline:
    bytes_per_launch: int = 0
    flops_per_launch: int = 0
    kind: str = "unknown"     # i8-dp4a | f16-tc | f32 | unknown


def roofline_estimate(slug: str, m: hw.Model) -> Roofline:
    """Best-effort per-launch (bytes, flops, class). Decode is weight-streaming:
    byte traffic ~= the weight matrix read once; flops ~= 2*M*N*K. Assumes the
    dominant matmul shape per family (CUPTI gives no per-launch M/N/K), so it
    classifies mem- vs compute-bound — not exact accounting."""
    s = slug
    K = m.n_embd
    if "share_warp" in s and ("q8" in s.lower() or "preq" in s):
        N = m.n_embd
        return Roofline(int(N * K * m.bits_q8 / 8), 2 * N * K, "i8-dp4a")
    if "moe_gate_up" in s:
        N = 2 * m.n_expert_used * m.n_ff_exp
        return Roofline(int(N * K * m.bits_q2k / 8), 2 * N * K, "i8-dp4a")
    if "moe_down" in s:
        N, Kd = m.n_expert_used * m.n_embd, m.n_ff_exp
        return Roofline(int(N * Kd * m.bits_q2k / 8), 2 * N * Kd, "i8-dp4a")
    if "cutlass" in s.lower() or "wmma" in s.lower():
        return Roofline(0, 0, "f16-tc")
    if "rms_norm" in s or "quantize" in s:
        return Roofline(int(m.n_embd * 4 * 2), m.n_embd * 4, "f32")
    return Roofline(0, 0, "unknown")


def achieved_gbps(bytes_per_launch: int, launches: int, total_ns: int) -> float:
    if total_ns <= 0 or bytes_per_launch <= 0:
        return float("nan")
    return (bytes_per_launch * launches) / (total_ns * 1e-9) / 1e9


def arithmetic_intensity(rf: Roofline) -> float:
    if rf.bytes_per_launch <= 0 or rf.flops_per_launch <= 0:
        return float("nan")
    return rf.flops_per_launch / rf.bytes_per_launch


def bw_headroom(bytes_per_launch: int, avg_ns: float, peak_gbps: float) -> float:
    """measured / floor — how many× faster the kernel could run at the BW ceiling.
    >1 means slack (fixable); ~1 means it's at the wall."""
    if bytes_per_launch <= 0 or avg_ns <= 0:
        return float("nan")
    floor_ns = (bytes_per_launch / (peak_gbps * 1e9)) * 1e9
    return avg_ns / floor_ns if floor_ns > 0 else float("nan")


# ---- ncu warp-stall reasons (opt-in, slow) ----------------------------------

STALLS = {
    "long_scoreboard": "long_scb",    # global/local load latency -> LATENCY
    "lg_throttle": "lg_throttle",      # LSU pipe throttle          -> BANDWIDTH
    "mio_throttle": "mio_throttle",
    "short_scoreboard": "short_scb",
    "wait": "wait",
    "barrier": "barrier",
    "not_selected": "not_selected",
}
_NCU_METRICS = ([f"smsp__average_warps_issue_stalled_{k}_per_issue_active.ratio"
                 for k in STALLS]
                + ["sm__warps_active.avg.pct_of_peak_sustained_active"])


def run_ncu(cmd: list[str], kernels: str, skip: int, count: int, out_base: str) -> Path:
    """Application-replay ncu (GB10: default kernel-replay segfaults on the 80 GB
    VMM model). Slow — re-runs the app once per metric pass."""
    rep = Path(out_base + ".ncu-rep")
    argv = [NCU, "--replay-mode", "application", "-o", out_base, "-f",
            "--metrics", ",".join(_NCU_METRICS), "--kernel-name", f"regex:{kernels}",
            "--launch-skip", str(skip), "--launch-count", str(count), *cmd]
    subprocess.run(argv, check=True)
    return rep


def parse_ncu(rep: Path) -> dict:
    """Import the .ncu-rep as raw CSV and aggregate stall reasons by slug."""
    out = subprocess.run([NCU, "--import", str(rep), "--csv", "--page", "raw"],
                         capture_output=True, text=True, check=True)
    rows = list(csv.DictReader(out.stdout.splitlines()))
    if not rows:
        return {}
    kn = next(c for c in rows[0] if c and "Kernel Name" in c)
    acc: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if not (r.get("ID") or "").strip():
            continue
        slug = trace.canon_slug(r[kn])
        bucket = acc.setdefault(slug, {})
        for col, val in r.items():
            if not col or not val:
                continue
            try:
                bucket.setdefault(col, []).append(float(val.replace(",", "")))
            except ValueError:
                pass
    res = {}
    for slug, mtr in acc.items():
        def avg(metric: str) -> float:
            xs = mtr.get(metric, [])
            return sum(xs) / len(xs) if xs else 0.0
        stalls = {short: avg(f"smsp__average_warps_issue_stalled_{lng}_per_issue_active.ratio")
                  for lng, short in STALLS.items()}
        tot = sum(stalls.values()) or 1.0
        dom = max(stalls, key=lambda k: stalls[k])
        res[slug] = {
            "stalls": stalls,
            "dominant": dom,
            "dominant_pct": 100.0 * stalls[dom] / tot,
            "occupancy_pct": avg("sm__warps_active.avg.pct_of_peak_sustained_active"),
        }
    return res
