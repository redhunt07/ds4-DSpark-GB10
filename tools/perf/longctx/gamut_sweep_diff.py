#!/usr/bin/env python3
"""Sweep diff: N gamut JSONs in, slope + per-kernel growth + flatten/lift/trap.

Fits decode_tps(ctx) ~ a - b*log2(ctx/8192).

  tps@8k             = a
  slope_per_doubling = b   (positive = degrades with ctx)

Compares baseline vs candidate (each a directory containing per-frontier
gamut JSONs named like nomtp-ctx65536.gamut.json or mtp-ctx65536.gamut.json),
emits markdown table + flatten/lift/trap classification.

Usage:
  gamut_sweep_diff.py --baseline DIR --candidate DIR [--mtp-tag nomtp]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

CTX_RX = re.compile(r"ctx(\d+)")


def load_sweep(dir_path: Path, mtp_tag: str) -> dict[int, dict]:
    """Map ctx -> gamut json dict for one mtp variant in a sweep directory."""
    out: dict[int, dict] = {}
    for p in sorted(dir_path.glob(f"{mtp_tag}-ctx*.gamut.json")):
        m = CTX_RX.search(p.name)
        if not m:
            continue
        ctx = int(m.group(1))
        try:
            out[ctx] = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] skip {p}: {e}", file=sys.stderr)
    return out


def decode_tps_of(g: dict) -> float | None:
    """gamut report JSON: throughput.decode_tps (see gamut/report.py build())."""
    v = (g.get("throughput") or {}).get("decode_tps")
    if isinstance(v, (int, float)):
        return float(v)
    # fall back to the trace-derived rate when the bench csv wasn't passed in
    v = (g.get("throughput") or {}).get("trace_decode_tps")
    if isinstance(v, (int, float)) and not math.isnan(v):
        return float(v)
    return None


def fit_slope(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least-squares fit y = a - b*log2(ctx/8192). Returns (tps@8k, slope_per_doubling)."""
    if len(points) < 2:
        if points:
            return points[0][1], 0.0
        return float("nan"), float("nan")
    xs = [math.log2(c / 8192.0) for c, _ in points]
    ys = [y for _, y in points]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    intercept = my - slope * mx
    return intercept, -slope  # b is positive when tps decreases with ctx


def per_kernel_growth(sweep: dict[int, dict]) -> dict[str, dict[int, float]]:
    """Per-kernel avg time (µs) at each ctx. gamut report JSON kernel rows are
    {"kernel": slug, "avg_us": float, "ms": float, ...} (gamut/report.py)."""
    out: dict[str, dict[int, float]] = {}
    for ctx, g in sweep.items():
        for k in g.get("kernels") or []:
            name = k.get("kernel")
            t_us = k.get("avg_us")
            if name and isinstance(t_us, (int, float)):
                out.setdefault(name, {})[ctx] = float(t_us)
    return out


def classify(base_a: float, base_b: float, cand_a: float, cand_b: float,
             tps_eps: float = 0.05, slope_eps: float = 0.15) -> str:
    a_up = (cand_a - base_a) / base_a if base_a else 0.0
    b_down = (base_b - cand_b) / base_b if base_b else 0.0
    lifted = a_up >= tps_eps
    flattened = b_down >= slope_eps
    if lifted and flattened:
        return "PARETO (lift + flatten)"
    if flattened and a_up >= -tps_eps:
        return "FLATTEN"
    if lifted and b_down >= -slope_eps:
        return "LIFT"
    if lifted and b_down < -slope_eps:
        return "TRAP (lift @8k, steeper @long-ctx)"
    if flattened and a_up < -tps_eps:
        return "TRAP (flatten @long-ctx, regress @8k)"
    return "NEUTRAL"


def fmt_row(name: str, a: float, b: float) -> str:
    return f"| {name} | {a:7.2f} | {b:6.2f} |"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--mtp-tag", default="nomtp", choices=("nomtp", "mtp"))
    args = ap.parse_args(argv)

    base = load_sweep(args.baseline, args.mtp_tag)
    cand = load_sweep(args.candidate, args.mtp_tag)
    if not base or not cand:
        print(f"[err] empty sweep (base={len(base)} cand={len(cand)})", file=sys.stderr)
        return 1

    def points(sweep):
        return sorted((c, t) for c, g in sweep.items() if (t := decode_tps_of(g)) is not None)

    base_pts = points(base)
    cand_pts = points(cand)
    base_a, base_b = fit_slope(base_pts)
    cand_a, cand_b = fit_slope(cand_pts)
    verdict = classify(base_a, base_b, cand_a, cand_b)

    print(f"# Sweep diff — {args.mtp_tag}")
    print()
    print(f"**Verdict:** {verdict}")
    print()
    print("| variant | tps@8k | slope/doubling |")
    print("| ------- | ------ | -------------- |")
    print(fmt_row("baseline", base_a, base_b))
    print(fmt_row("candidate", cand_a, cand_b))
    print()

    print("## Per-frontier decode tok/s")
    print()
    all_ctx = sorted(set(c for c, _ in base_pts) | set(c for c, _ in cand_pts))
    print("| ctx | base | cand | Δ |")
    print("| --- | ---- | ---- | - |")
    bd = dict(base_pts); cd = dict(cand_pts)
    for c in all_ctx:
        b = bd.get(c); k = cd.get(c)
        delta = (k - b) if (b is not None and k is not None) else None
        print(f"| {c} | {b if b is None else f'{b:.2f}'} | "
              f"{k if k is None else f'{k:.2f}'} | "
              f"{'' if delta is None else f'{delta:+.2f}'} |")
    print()

    # Per-kernel growth: emit only kernels present in both sweeps where the
    # candidate moves the long-ctx end materially.
    base_k = per_kernel_growth(base)
    cand_k = per_kernel_growth(cand)
    common = sorted(set(base_k) & set(cand_k))
    if common:
        long_ctx = max(all_ctx)
        rows = []
        for name in common:
            bt = base_k[name].get(long_ctx)
            ct = cand_k[name].get(long_ctx)
            if bt is None or ct is None or bt == 0:
                continue
            rows.append((name, bt, ct, (ct - bt) / bt))
        rows.sort(key=lambda r: abs(r[3]), reverse=True)
        if rows:
            print(f"## Per-kernel time @ ctx={long_ctx} (top movers)")
            print()
            print("| kernel | base µs | cand µs | Δ% |")
            print("| ------ | ------- | ------- | -- |")
            for name, bt, ct, d in rows[:15]:
                print(f"| `{name}` | {bt:.1f} | {ct:.1f} | {d*100:+.1f}% |")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
