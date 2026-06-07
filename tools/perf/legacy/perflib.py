#!/usr/bin/env python3
"""tools/perf/perflib.py — shared primitives for the ds4 GB10 perf suite.

Everything the gamut report joins on lives here so the slug, the phase
windows, and the metric IDs are defined exactly once. Pure stdlib (sqlite3).

Data sources and how they key:
  - kernel time   : CUPTI_ACTIVITY_KIND_KERNEL in an nsys-exported sqlite,
                    keyed by demangled name -> canon_slug().
  - GPU HW metric : GPU_METRICS time-series in a `gb20b` capture, joined to
                    kernels by timestamp window (same run only).
  - ptxas regs    : `nvcc -Xptxas=-v` stderr, mangled names -> demangle -> slug.

Phase windowing has no NVTX to lean on (the engine isn't instrumented yet),
so we key on `embed_token_hc` — it launches exactly once per decode token,
which makes it a clean per-token boundary marker.

GB10 hardware constants are calibrated by tools/perf/membw.cu, not guessed:
the LPDDR5X read ceiling measures ~236 GB/s (87% of the 273 GB/s theoretical).
Use the measured read ceiling as the roofline denominator for decode.
"""

from __future__ import annotations

import bisect
import re
import sqlite3
import subprocess
from dataclasses import dataclass


# ---- hardware (GB10, calibrated by membw.cu) --------------------------------

@dataclass
class HW:
    name: str = "GB10 sm_121a"
    hbm_read_gbps: float = 236.0   # MEASURED sustained read ceiling (membw.cu)
    hbm_theoretical_gbps: float = 273.0  # LPDDR5X spec peak
    f32_tflops: float = 31.0
    f16_tc_tflops: float = 125.0
    i8_dp4a_tops: float = 250.0
    n_sms: int = 48


# gb20b GPU_METRICS metricId -> short key. Confirmed against
# TARGET_INFO_GPU_METRICS on this nsys version (2025.3).
METRIC = {
    "gpc_clock_mhz": 0,
    "gr_active": 6,
    "sms_active": 7,
    "sm_issue": 8,
    "tensor_active": 9,
    "compute_warps": 16,
}
# Metrics we surface in the verdict, in display order.
VERDICT_METRICS = ["sms_active", "sm_issue", "tensor_active", "compute_warps"]

# A sample counts as "busy" (inside real compute, not an inter-token launch
# gap) when SMs Active exceeds this. Matches the windowing that reproduced the
# decode doc's numbers.
BUSY_SMS_ACTIVE = 40.0

TOKEN_MARKER = "embed_token_hc"  # launches once per decode token


# ---- kernel-name canonicalization -------------------------------------------

def canon_slug(full: str) -> str:
    """Canonical join key for a demangled C++ kernel name. NOT truncated —
    truncation is display-only (see disp()), so it can never collide two
    distinct kernels into one row.

    Keeps template params (`<3>`) — different instantiations have different
    registers/occupancy and must stay distinct rows in the join.
    """
    # Drop C-style cast in template args first: nsys renders `<(int)3>` while
    # c++filt renders `<3>`; normalize both to `<3>` so they join.
    full = re.sub(r"\((?:unsigned\s+|signed\s+)?(?:int|long|short|char)\)", "", full)
    paren = full.find("(")
    if paren >= 0:
        full = full[:paren]
    full = re.sub(r"^void\s+", "", full)
    full = re.sub(r"ds4::<unnamed>::|ds4::|\(anonymous namespace\)::", "", full)
    return full.strip()


_CUTLASS_RE = re.compile(r"cutlass.*?_(f16|bf16|tf32|f32|s8)_(\d+x\d+)", re.I)


def disp(slug: str, max_len: int = 46) -> str:
    """Display form of a slug. Cutlass library kernels get a compact label
    that keeps their distinguishing dtype+tile (e.g. `cutlass·wmma·f16·32x32`);
    everything else keeps head+tail so a distinguishing suffix survives."""
    if "cutlass" in slug.lower():
        m = _CUTLASS_RE.search(slug)
        if m:
            return f"cutlass·wmma·{m.group(1)}·{m.group(2)}"
    if len(slug) <= max_len:
        return slug
    head = max_len - 17
    return slug[:head] + "…" + slug[-16:]


def demangle(mangled: str) -> str:
    try:
        out = subprocess.run(["c++filt", mangled], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or mangled
    except Exception:
        return mangled


# ---- sqlite access ----------------------------------------------------------

def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def kernel_span(con: sqlite3.Connection) -> tuple[int, int]:
    have = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='CUPTI_ACTIVITY_KIND_KERNEL'"
    ).fetchone()
    if not have:
        raise SystemExit(
            "perflib: this sqlite has no CUPTI_ACTIVITY_KIND_KERNEL table — the nsys "
            "capture was truncated or had no CUDA kernels (commonly: a second GPU job "
            "ran concurrently, or `nsys profile` was killed). Re-capture serialized "
            "(capture.sh now takes an flock) and confirm the .nsys-rep is non-empty.")
    r = con.execute(
        "SELECT MIN(start) lo, MAX(end) hi FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    if r is None or r["lo"] is None:
        raise SystemExit(
            "perflib: CUPTI_ACTIVITY_KIND_KERNEL is empty (0 kernels traced). The "
            "profiled run captured no GPU work — check the nsyslog for a CUDA-injection "
            "error and re-capture.")
    return int(r["lo"]), int(r["hi"])


def token_starts(con: sqlite3.Connection) -> list[int]:
    """Sorted start timestamps of the per-token marker kernel."""
    rows = con.execute(
        f"""SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName)
            WHERE s.value LIKE '{TOKEN_MARKER}%' ORDER BY k.start"""
    ).fetchall()
    return [int(r["start"]) for r in rows]


@dataclass
class Windows:
    prefill: tuple[int, int]
    decode: tuple[int, int]
    steady: tuple[int, int]   # decode minus warmup tokens
    n_tokens: int
    skip: int


def phases(con: sqlite3.Connection, skip_warmup: int = 8) -> Windows:
    lo, hi = kernel_span(con)
    toks = token_starts(con)
    if not toks:
        # No decode marker: treat the whole run as one window.
        return Windows((lo, lo), (lo, hi), (lo, hi), 0, 0)
    skip = min(skip_warmup, max(0, len(toks) - 1))
    return Windows(
        prefill=(lo, toks[0]),
        decode=(toks[0], hi),
        steady=(toks[skip], hi),
        n_tokens=len(toks),
        skip=skip,
    )


@dataclass
class Kernel:
    slug: str
    full: str
    launches: int
    total_ns: int
    avg_ns: float


def kernels_in(con: sqlite3.Connection, win: tuple[int, int]) -> list[Kernel]:
    """Per-kernel time accounting for kernels fully inside `win`, slug-folded."""
    lo, hi = win
    rows = con.execute(
        """SELECT s.value name, COUNT(*) n, SUM(k.end-k.start) tot
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName)
           WHERE k.start >= ? AND k.end <= ?
           GROUP BY s.value""",
        (lo, hi),
    ).fetchall()
    agg: dict[str, list] = {}
    for r in rows:
        slug = canon_slug(r["name"])
        a = agg.setdefault(slug, [r["name"], 0, 0])
        a[1] += int(r["n"]); a[2] += int(r["tot"])
    out = [Kernel(s, v[0], v[1], v[2], v[2] / v[1] if v[1] else 0.0)
           for s, v in agg.items()]
    out.sort(key=lambda k: -k.total_ns)
    return out


def kernel_intervals(con: sqlite3.Connection, win: tuple[int, int]
                     ) -> dict[str, list[tuple[int, int]]]:
    """slug -> list of (start,end) for launches inside `win`."""
    lo, hi = win
    rows = con.execute(
        """SELECT s.value name, k.start st, k.end en
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName)
           WHERE k.start >= ? AND k.end <= ?""",
        (lo, hi),
    ).fetchall()
    out: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        out.setdefault(canon_slug(r["name"]), []).append((int(r["st"]), int(r["en"])))
    return out


# ---- GPU hardware metrics (gb20b capture only) ------------------------------

def has_gpu_metrics(con: sqlite3.Connection) -> bool:
    r = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='GPU_METRICS'"
    ).fetchone()
    return r is not None


def busy_metric_avgs(con: sqlite3.Connection, win: tuple[int, int],
                     keys: list[str]) -> dict[str, float]:
    """Average each metric over *busy* samples (SMs Active > threshold) in win.

    Excludes inter-token launch gaps so the numbers describe time spent
    actually computing, which is what the stall verdict is about.
    """
    lo, hi = win
    out: dict[str, float] = {}
    busy_sub = (
        "SELECT timestamp FROM GPU_METRICS WHERE metricId=? AND value>? "
        "AND timestamp BETWEEN ? AND ?"
    )
    for k in keys:
        r = con.execute(
            f"SELECT AVG(value) v FROM GPU_METRICS WHERE metricId=? "
            f"AND timestamp IN ({busy_sub})",
            (METRIC[k], METRIC["sms_active"], BUSY_SMS_ACTIVE, lo, hi),
        ).fetchone()
        out[k] = float(r["v"]) if r["v"] is not None else float("nan")
    return out


def per_kernel_metric(con: sqlite3.Connection, win: tuple[int, int],
                      intervals: dict[str, list[tuple[int, int]]],
                      key: str, min_samples: int = 4) -> dict[str, float]:
    """Average one metric within each kernel's launch windows (same run).

    Returns slug -> avg, only for kernels with >= min_samples hits. At 20 kHz
    (~50 us/sample) short kernels get too few samples to trust, so they're
    dropped (caller shows '—').
    """
    lo, hi = win
    rows = con.execute(
        "SELECT timestamp, value FROM GPU_METRICS WHERE metricId=? "
        "AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (METRIC[key], lo, hi),
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


# ---- ptxas register / occupancy parsing -------------------------------------

@dataclass
class PtxasInfo:
    regs: int = 0
    smem: int = 0           # bytes
    spills_store: int = 0
    spills_load: int = 0


def parse_ptxas(text: str) -> dict[str, PtxasInfo]:
    """Parse `nvcc -Xptxas=-v` stderr into slug -> PtxasInfo.

    Blocks look like:
      ptxas info : Compiling entry function '_Z..mangled..' for 'sm_121a'
      ptxas info : Used 48 registers, 0 stack, 0 bytes smem, ...
    """
    out: dict[str, PtxasInfo] = {}
    cur_slug: str | None = None
    entry_re = re.compile(r"Compiling entry function '([^']+)'")
    regs_re = re.compile(r"Used (\d+) registers")
    smem_re = re.compile(r"(\d+) bytes smem")
    spill_re = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
    for line in text.splitlines():
        m = entry_re.search(line)
        if m:
            cur_slug = canon_slug(demangle(m.group(1)))
            out.setdefault(cur_slug, PtxasInfo())
            continue
        if cur_slug is None:
            continue
        info = out[cur_slug]
        if (m := regs_re.search(line)):
            info.regs = max(info.regs, int(m.group(1)))
        if (m := smem_re.search(line)):
            info.smem = max(info.smem, int(m.group(1)))
        if (m := spill_re.search(line)):
            info.spills_store = int(m.group(1)); info.spills_load = int(m.group(2))
    return out


def theoretical_occupancy(regs: int, smem: int, threads: int = 256,
                          regs_total: int = 65536, smem_total: int = 232448,
                          max_warps: int = 64) -> float:
    """Crude theoretical occupancy from regs/thread + smem/block.

    sm_121a: 64K 32-bit regs/SM, ~227KB smem/SM, 64 warps/SM max.
    Returns fraction 0..1. (Block-size-limited terms ignored; reg/smem only.)
    """
    if regs <= 0:
        return float("nan")
    warps_per_block = max(1, threads // 32)
    reg_warps = (regs_total // (regs * 32)) // 1  # warps the reg file allows
    reg_warps = (reg_warps // warps_per_block) * warps_per_block
    smem_warps = max_warps if smem <= 0 else \
        ((smem_total // smem) * warps_per_block)
    return min(max_warps, reg_warps, smem_warps) / max_warps


# ---- roofline (decode byte models; best-effort) -----------------------------

@dataclass
class Model:
    """DeepSeek-V4-Flash shapes (decode). Override per model if needed."""
    n_embd: int = 4096
    n_ff_exp: int = 2048
    n_expert_used: int = 6
    bits_q8: float = 8.5      # Q8_0 incl. scales (8 + 16/32)
    bits_q2k: float = 2.6     # Q2_K incl. scales/mins
    bits_iq2: float = 2.6     # IQ2_XXS


@dataclass
class Roofline:
    bytes_per_launch: int = 0
    flops_per_launch: int = 0
    kind: str = "unknown"     # i8-dp4a | f16-tc | f32 | unknown


def roofline_estimate(slug: str, m: Model) -> Roofline:
    """Best-effort per-launch (bytes, flops, class) for the current decode
    kernels. Decode is weight-streaming, so byte traffic ≈ the weight matrix
    each call reads once; flops ≈ 2·M·N·K. Kernels without a matcher return
    kind='unknown' (caller shows '—').

    CAVEAT: CUPTI gives no per-launch M/N/K, so these assume the dominant
    matmul shape per family. They classify mem- vs compute-bound and give an
    order-of-magnitude %peakBW — not exact accounting. The measured signals
    (ncu stall reasons, gb20b occupancy) are the ground truth; this is colour.
    """
    s = slug
    K = m.n_embd
    if "share_warp" in s and ("q8" in s.lower() or "preq" in s):
        # Batched dense Q8_0 projection, weights read once per call. Assume a
        # square n_embd×n_embd projection (o_proj-class) as the dominant shape.
        N = m.n_embd
        return Roofline(int(N * K * m.bits_q8 / 8), 2 * N * K, "i8-dp4a")
    if "moe_gate_up" in s:
        # gate+up for the active experts: two [n_ff_exp × n_embd] reads each.
        N = 2 * m.n_expert_used * m.n_ff_exp
        return Roofline(int(N * K * m.bits_q2k / 8), 2 * N * K, "i8-dp4a")
    if "moe_down" in s:
        # down: [n_embd × n_ff_exp] per active expert.
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
    """flops per byte. <~1 = memory-bound, >~10 = compute-bound."""
    if rf.bytes_per_launch <= 0 or rf.flops_per_launch <= 0:
        return float("nan")
    return rf.flops_per_launch / rf.bytes_per_launch


def bw_headroom(bytes_per_launch: int, avg_ns: float, peak_gbps: float) -> float:
    """measured / floor — how many× faster the kernel *could* run if it hit the
    BW ceiling. >1 means slack (fixable); ≈1 means it's at the wall."""
    if bytes_per_launch <= 0 or avg_ns <= 0:
        return float("nan")
    floor_ns = (bytes_per_launch / (peak_gbps * 1e9)) * 1e9
    return avg_ns / floor_ns if floor_ns > 0 else float("nan")


def launch_gaps(con: sqlite3.Connection, win: tuple[int, int], top: int = 8,
                exclude_us: float = 50_000.0) -> list[tuple[float, str, str]]:
    """Top inter-kernel idle gaps inside `win` (host/launch stalls).

    Returns (gap_us, prev_slug, next_slug). Gaps > exclude_us are dropped as
    warmup / inter-rep outliers so the per-token tail surfaces (PLAN T.1.4).
    """
    lo, hi = win
    rows = con.execute(
        """SELECT k.start st, k.end en, s.value name
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName)
           WHERE k.start >= ? AND k.end <= ? ORDER BY k.start""",
        (lo, hi),
    ).fetchall()
    gaps = []
    for a, b in zip(rows, rows[1:]):
        gap = b["st"] - a["en"]
        if gap <= 0:
            continue
        gus = gap / 1e3
        if gus > exclude_us:
            continue
        gaps.append((gus, canon_slug(a["name"]), canon_slug(b["name"])))
    gaps.sort(key=lambda g: -g[0])
    return gaps[:top]
