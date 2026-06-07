#!/usr/bin/env python3
"""tools/perf/gamut_diff.py — A/B two gamut report JSON sidecars.

Turns "did the change help?" into one command: compares throughput, the GPU HW
verdict, the time split, and every per-kernel metric (joined by slug), flagging
moves beyond a noise floor. The unit of the perf loop's measure step.

    tools/perf/gamut_diff.py before.json after.json [--noise 2] [--top 14]

Deltas are after−before; for t/s and occupancy up is good, for ms/stall down is
good (annotated ✓ win / ✗ regression / · noise).
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def mark(delta: float, noise: float, lower_is_better: bool) -> str:
    if delta is None or math.isnan(delta):
        return " "
    if abs(delta) < noise:
        return "·"
    good = (delta < 0) if lower_is_better else (delta > 0)
    return "✓" if good else "✗"


def fnum(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def line(label, a, b, noise, lower_is_better=False, nd=1, unit=""):
    if a is None and b is None:
        return None
    d = (b - a) if (a is not None and b is not None) else float("nan")
    m = mark(d, noise, lower_is_better)
    ds = f"{d:+.{nd}f}" if not math.isnan(d) else "—"
    return f"  {m} {label:22s} {fnum(a,nd):>8}{unit} → {fnum(b,nd):>8}{unit}   Δ {ds}{unit}"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--noise", type=float, default=2.0, help="flag floor (abs units)")
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()
    A = json.load(open(args.before))
    B = json.load(open(args.after))

    print(f"# gamut diff\n- before: {A.get('label')}\n- after:  {B.get('label')}\n")

    print("## Throughput / verdict")
    rows = [
        line("decode t/s", g(A, "throughput", "decode_tps"), g(B, "throughput", "decode_tps"), 0.3, nd=2),
        line("trace decode t/s", g(A, "time_split", "trace_decode_tps"), g(B, "time_split", "trace_decode_tps"), 0.3, nd=2),
        line("accept %", g(A, "throughput", "accept_pct"), g(B, "throughput", "accept_pct"), args.noise),
        line("tokens/iter", g(A, "throughput", "tokens_per_iter"), g(B, "throughput", "tokens_per_iter"), 0.02, nd=2),
        line("SM issue %", g(A, "verdict_metrics", "sm_issue"), g(B, "verdict_metrics", "sm_issue"), args.noise),
        line("compute warps %", g(A, "verdict_metrics", "compute_warps"), g(B, "verdict_metrics", "compute_warps"), args.noise),
        line("kernel ms (steady)", g(A, "time_split", "kernel_ms"), g(B, "time_split", "kernel_ms"), 5, lower_is_better=True),
        line("idle %", g(A, "time_split", "idle_pct"), g(B, "time_split", "idle_pct"), args.noise, lower_is_better=True),
    ]
    for r in rows:
        if r:
            print(r)

    # per-kernel join by slug
    ka = {k["kernel"]: k for k in A.get("kernels", [])}
    kb = {k["kernel"]: k for k in B.get("kernels", [])}
    slugs = sorted(set(ka) | set(kb),
                   key=lambda s: -(kb.get(s, ka.get(s, {})).get("pct_time", 0)))
    print("\n## Per-kernel (Δ after−before)")
    print("| kernel | %t a→b | ms a→b | occ% a→b | warps a→b | %peakBW a→b | stall a→b |")
    print("|--------|--------|--------|----------|-----------|-------------|-----------|")

    def occ(k):
        o = k.get("occ_theory")
        return o * 100 if isinstance(o, (int, float)) and not math.isnan(o) else None

    def cell(a, b, nd=1, lower=False):
        if a is None and b is None:
            return "—"
        d = (b - a) if (a is not None and b is not None) else None
        m = mark(d, 0.5, lower) if d is not None else " "
        return f"{fnum(a,nd)}→{fnum(b,nd)} {m}"
    for s in slugs[: args.top]:
        a, b = ka.get(s, {}), kb.get(s, {})
        sa, sb = a.get("stall") or "—", b.get("stall") or "—"
        print(f"| {s[:40]} | {cell(a.get('pct_time'), b.get('pct_time'))} | "
              f"{cell(a.get('ms'), b.get('ms'), lower=True)} | "
              f"{cell(occ(a), occ(b), nd=0)} | "
              f"{cell(a.get('warps'), b.get('warps'), nd=0)} | "
              f"{cell(a.get('pct_peak_bw'), b.get('pct_peak_bw'), nd=0)} | "
              f"{sa}→{sb} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
