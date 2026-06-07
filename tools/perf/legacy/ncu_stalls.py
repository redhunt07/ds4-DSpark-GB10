#!/usr/bin/env python3
"""tools/perf/ncu_stalls.py — per-kernel warp-stall breakdown via Nsight Compute.

The direct latency-vs-bandwidth discriminator the gb20b nsys set can't give:
ncu's issue-stall reasons. `long_scoreboard` ≫ `lg_throttle` means
memory-LATENCY-bound (need occupancy to hide it); the reverse means
bandwidth-THROTTLED. Also grabs achieved occupancy.

OPT-IN AND SLOW. On GB10, ncu's default kernel-replay segfaults on ds4's
80 GB HBM-resident VMM model, so this forces `--replay-mode application`,
which re-runs the whole app once per metric pass (~minutes per run). Not for
the default gamut path — run it when you want the measured stall column.

Writes a JSON keyed by canonical kernel slug, consumable by `gamut.py --ncu`.

Usage (wrap the ds4 command after `--`):
    tools/perf/ncu_stalls.py --out /tmp/ncu.json \\
        --kernels "moe_down_expert_tile8_row32|matmul_q8_0_preq_batch_share_warp" \\
        --launch-skip 150 --launch-count 6 -- \\
        ./ds4 -m ds4flash.gguf --mtp MTP -p knight -n 24 --temp 0 --nothink -sys ""
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import perflib as P

NCU = "/usr/local/cuda/bin/ncu"

# issue-stall reasons, avg warps stalled per issue-active cycle + achieved occ
STALLS = {
    "long_scoreboard": "long_scb",   # global/local load latency  -> LATENCY
    "lg_throttle": "lg_throttle",     # LSU pipe throttle           -> BANDWIDTH
    "mio_throttle": "mio_throttle",   # MIO (smem/special) throttle
    "short_scoreboard": "short_scb",  # shared-memory latency
    "wait": "wait",                   # fixed exec dependency
    "barrier": "barrier",
    "not_selected": "not_selected",   # enough warps, just not picked
}
_METRICS = ([f"smsp__average_warps_issue_stalled_{k}_per_issue_active.ratio"
             for k in STALLS]
            + ["sm__warps_active.avg.pct_of_peak_sustained_active"])


def run_ncu(cmd: list[str], kernels: str, skip: int, count: int) -> Path:
    fd, base = tempfile.mkstemp(prefix="ncu_stalls_")
    os.close(fd)
    rep = Path(base + ".ncu-rep")
    argv = [NCU, "--replay-mode", "application", "-o", base,
            "-f", "--metrics", ",".join(_METRICS),
            "--kernel-name", f"regex:{kernels}",
            "--launch-skip", str(skip), "--launch-count", str(count), *cmd]
    print(f"# running ncu (application replay; slow)…\n#   {' '.join(argv)}",
          file=sys.stderr)
    subprocess.run(argv, check=True)
    return rep


def parse_report(rep: Path) -> dict:
    """Import the .ncu-rep as clean CSV (no app stdout) and aggregate by slug.

    `--page raw` is WIDE: one row per kernel launch, metrics as columns, with a
    units row right after the header (ID empty). We skip the units row and read
    each metric straight from its named column.
    """
    out = subprocess.run([NCU, "--import", str(rep), "--csv", "--page", "raw"],
                         capture_output=True, text=True, check=True)
    rows = list(csv.DictReader(out.stdout.splitlines()))
    if not rows:
        return {}
    kn = next(c for c in rows[0] if c and "Kernel Name" in c)
    acc: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if not (r.get("ID") or "").strip():
            continue  # units row / blank
        slug = P.canon_slug(r[kn])
        bucket = acc.setdefault(slug, {})
        for col, val in r.items():
            if not col or not val:
                continue
            try:
                bucket.setdefault(col, []).append(float(val.replace(",", "")))
            except ValueError:
                pass

    res = {}
    for slug, metrics in acc.items():
        def avg(metric: str) -> float:
            xs = metrics.get(metric, [])
            return sum(xs) / len(xs) if xs else 0.0
        stalls = {short: avg(f"smsp__average_warps_issue_stalled_{long}"
                             f"_per_issue_active.ratio")
                  for long, short in STALLS.items()}
        tot = sum(stalls.values()) or 1.0
        dom = max(stalls, key=lambda k: stalls[k])
        res[slug] = {
            "stalls": stalls,
            "dominant": dom,
            "dominant_pct": 100.0 * stalls[dom] / tot,
            "occupancy_pct": avg("sm__warps_active.avg.pct_of_peak_sustained_active"),
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", required=True, help="JSON output (for gamut --ncu)")
    ap.add_argument("--kernels", required=True, help="regex of kernel names")
    ap.add_argument("--launch-skip", type=int, default=150)
    ap.add_argument("--launch-count", type=int, default=6)
    ap.add_argument("--import-rep", help="skip profiling; parse this .ncu-rep")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- then the ds4 command to profile")
    args = ap.parse_args()

    if args.import_rep:
        res = parse_report(Path(args.import_rep))
    else:
        cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
        if not cmd:
            ap.error("need a command after --")
        rep = run_ncu(cmd, args.kernels, args.launch_skip, args.launch_count)
        res = parse_report(rep)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"# wrote {len(res)} kernels -> {args.out}", file=sys.stderr)
    for slug, d in sorted(res.items(), key=lambda kv: -kv[1]["dominant_pct"]):
        print(f"  {P.disp(slug):46s} {d['dominant']:14s} "
              f"{d['dominant_pct']:5.0f}%  occ {d['occupancy_pct']:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
