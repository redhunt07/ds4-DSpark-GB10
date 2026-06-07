"""gamut.trace — nsys-sqlite access: slug canon, phase windows, kernel accounting.

Everything that reads an nsys-exported sqlite lives here so the slug, the phase
windows, and the kernel join are defined exactly once. Pure stdlib (sqlite3).

Phase windowing keys on the per-token marker kernel (hw.TOKEN_MARKER) since the
engine isn't NVTX-instrumented: it launches exactly once per decode token.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from dataclasses import dataclass

from . import hw


# ---- kernel-name canonicalization -------------------------------------------

_CUTLASS_RE = re.compile(r"cutlass.*?_(f16|bf16|tf32|f32|s8)_(\d+x\d+)", re.I)
_CAST_RE = re.compile(r"\((?:unsigned\s+|signed\s+)?(?:int|long|short|char)\)")


def canon_slug(full: str) -> str:
    """Canonical join key for a demangled C++ kernel name. NOT truncated —
    truncation is display-only (see disp()), so it can never collide two
    distinct kernels into one row. Keeps template params (`<3>`): different
    instantiations have different registers/occupancy and stay distinct."""
    full = _CAST_RE.sub("", full)        # nsys `<(int)3>` vs c++filt `<3>`
    paren = full.find("(")
    if paren >= 0:
        full = full[:paren]
    full = re.sub(r"^void\s+", "", full)
    full = re.sub(r"ds4::<unnamed>::|ds4::|\(anonymous namespace\)::", "", full)
    return full.strip()


def disp(slug: str, max_len: int = 46) -> str:
    """Display form: cutlass kernels get a compact dtype+tile label; everything
    else keeps head+tail so a distinguishing suffix survives truncation."""
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


def has_table(con: sqlite3.Connection, name: str) -> bool:
    return con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def kernel_span(con: sqlite3.Connection) -> tuple[int, int]:
    if not has_table(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        raise SystemExit(
            "gamut: this sqlite has no CUPTI_ACTIVITY_KIND_KERNEL table — the nsys "
            "capture was truncated or had no CUDA kernels (commonly: a second GPU job "
            "ran concurrently, or `nsys profile` was killed). Re-capture serialized.")
    r = con.execute(
        "SELECT MIN(start) lo, MAX(end) hi FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    if r is None or r["lo"] is None:
        raise SystemExit("gamut: CUPTI_ACTIVITY_KIND_KERNEL is empty (0 kernels traced).")
    return int(r["lo"]), int(r["hi"])


def token_starts(con: sqlite3.Connection) -> list[int]:
    """Sorted start timestamps of the per-token marker kernel."""
    rows = con.execute(
        """SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName)
           WHERE s.value LIKE ? ORDER BY k.start""",
        (hw.TOKEN_MARKER + "%",),
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


def launch_gaps(con: sqlite3.Connection, win: tuple[int, int], top: int = 8,
                exclude_us: float = 50_000.0) -> list[tuple[float, str, str]]:
    """Top inter-kernel idle gaps inside `win` (host/launch stalls).
    Returns (gap_us, prev_slug, next_slug); gaps > exclude_us dropped as
    warmup/inter-rep outliers so the per-token tail surfaces."""
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
