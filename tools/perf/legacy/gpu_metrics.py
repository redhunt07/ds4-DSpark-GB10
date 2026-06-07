#!/usr/bin/env python3
"""tools/perf/gpu_metrics.py — extract GB10 GPU hardware metrics from a gb20b
nsys capture, phase-windowed and (optionally) per-kernel.

This is the piece the surveyed suites (glint, ds4-spark, llama.cpp, vllm) all
lack: they stop at kernel-time accounting and never touch the GPU_METRICS
table. SM-issue / occupancy / tensor-active are what tell you *whether* a
memory-bound decode is stalled, so the gamut report depends on this.

Capture first:
    nsys profile --gpu-metrics-devices=0 --gpu-metrics-set=gb20b \\
        --gpu-metrics-frequency=20000 -o /tmp/gm ./ds4 -m model --mtp mtp ...
    nsys export --type=sqlite -f true -o /tmp/gm.sqlite /tmp/gm.nsys-rep   # or pass .sqlite

Usage:
    tools/perf/gpu_metrics.py /tmp/gm.sqlite [--phase steady|decode|prefill|all]
        [--per-kernel] [--skip-warmup N] [--top N]
"""

from __future__ import annotations

import argparse
import math
import sys

import perflib as P


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("sqlite")
    ap.add_argument("--phase", choices=["steady", "decode", "prefill", "all"],
                    default="steady")
    ap.add_argument("--per-kernel", action="store_true")
    ap.add_argument("--skip-warmup", type=int, default=8)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--min-samples", type=int, default=4)
    args = ap.parse_args()

    con = P.connect(args.sqlite)
    if not P.has_gpu_metrics(con):
        print("error: no GPU_METRICS table — not a gb20b capture?", file=sys.stderr)
        return 1

    w = P.phases(con, args.skip_warmup)
    win = {"steady": w.steady, "decode": w.decode, "prefill": w.prefill,
           "all": (w.prefill[0], w.decode[1])}[args.phase]

    print(f"# gpu_metrics  {args.sqlite}  phase={args.phase}  "
          f"({w.n_tokens} decode tokens, skipped {w.skip})")
    avgs = P.busy_metric_avgs(con, win, P.VERDICT_METRICS)
    print("\n| metric | busy avg |")
    print("|--------|---------:|")
    for k in P.VERDICT_METRICS:
        print(f"| {k} | {avgs[k]:.1f}% |" if not math.isnan(avgs[k])
              else f"| {k} | — |")

    if args.per_kernel:
        ivals = P.kernel_intervals(con, win)
        issue = P.per_kernel_metric(con, win, ivals, "sm_issue", args.min_samples)
        warps = P.per_kernel_metric(con, win, ivals, "compute_warps", args.min_samples)
        # rank kernels by total time in window
        ks = P.kernels_in(con, win)[: args.top]
        print(f"\n| kernel | SM issue | compute warps |  (≥{args.min_samples} samples)")
        print("|--------|---------:|--------------:|")
        for k in ks:
            si = issue.get(k.slug)
            cw = warps.get(k.slug)
            print(f"| {P.disp(k.slug)} | "
                  f"{si:.1f}% | {cw:.1f}% |" if si is not None and cw is not None
                  else f"| {P.disp(k.slug)} | — | — |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
