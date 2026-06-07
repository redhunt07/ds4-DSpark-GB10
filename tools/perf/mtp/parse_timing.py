#!/usr/bin/env python3
"""Parse DS4_MTP_TIMING=1 stderr into structured stats.

Reads stderr (file or stdin), extracts every `ds4: mtp timing ...` line, and
aggregates:

  * total_steps             — number of spec steps
  * total_drafted           — sum of drafted across steps
  * total_committed         — sum of committed across steps
  * accept_rate             — total_committed / total_drafted
  * tokens_emitted          — total_steps + total_committed (1 base + committed drafts per step)
  * committed_dist          — histogram {0: n, 1: n, 2: n, ...}
  * step_time_ms.{mean,p50,p90,p99}
  * step_kinds              — count per timing variant (combined, sample, decode2, margin-skip, micro)

Usage:
  parse_timing.py path/to/stderr.log       → markdown summary on stdout
  parse_timing.py --json out.json stderr   → also write JSON sidecar
  ... | parse_timing.py -                  → stdin mode
  parse_timing.py --merge run1.json run2.json ... --json merged.json
                                           → mean±std across N per-run JSONs
                                             (same scalar schema as a single run,
                                             plus "std" + "runs" fields)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# Examples:
#   ds4: mtp timing combined drafted=2 committed=1 total=320.764 ms
#   ds4: mtp timing sample drafted=2 committed=2 resampled=0 total=123.671 ms
#   ds4: mtp timing decode2 drafted=2 committed=2 draft=12.0 ms snapshot=1.2 ms verify=80.4 ms total=93.6 ms
#   ds4: mtp timing margin-skip drafted=2 committed=1 margin=1.4 threshold=2.0 ...
#   ds4: mtp timing micro drafted=2 committed=1 ... total=X ms
RX = re.compile(
    r"^ds4: mtp timing (?P<kind>\S+).*?drafted=(?P<d>\d+).*?committed=(?P<c>\d+)"
    r"(?:.*?cap=(?P<cap>\d+) ewma_p1=(?P<ewma>[\d.]+))?"
    r".*?total=(?P<t>[\d.]+)\s*ms"
)


def parse(stream):
    steps = []
    kinds: dict[str, int] = {}
    for line in stream:
        m = RX.search(line)
        if not m:
            continue
        step = {
            "kind": m.group("kind"),
            "drafted": int(m.group("d")),
            "committed": int(m.group("c")),
            "total_ms": float(m.group("t")),
        }
        # Adaptive-cascade fields (DS4_MTP_CASCADE_ADAPTIVE=1 only)
        if m.group("cap") is not None:
            step["cap"] = int(m.group("cap"))
            step["ewma_p1"] = float(m.group("ewma"))
        steps.append(step)
        kinds[m.group("kind")] = kinds.get(m.group("kind"), 0) + 1
    return steps, kinds


def percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def aggregate(steps, kinds):
    if not steps:
        return {"total_steps": 0}
    total_d = sum(s["drafted"] for s in steps)
    total_c = sum(s["committed"] for s in steps)
    dist: dict[int, int] = {}
    for s in steps:
        dist[s["committed"]] = dist.get(s["committed"], 0) + 1
    times = [s["total_ms"] for s in steps]
    # Adaptive-cascade controller state, when present: time share per cap value
    # (how often the controller chose K=1 vs K=2) and the final EWMA.
    cascade = {}
    cap_steps = [s for s in steps if "cap" in s]
    if cap_steps:
        cap_dist: dict[int, int] = {}
        for s in cap_steps:
            cap_dist[s["cap"]] = cap_dist.get(s["cap"], 0) + 1
        cascade = {"cap_dist": cap_dist, "final_ewma_p1": cap_steps[-1]["ewma_p1"]}
    return {
        **({"cascade": cascade} if cascade else {}),
        "total_steps": len(steps),
        "total_drafted": total_d,
        "total_committed": total_c,
        "accept_rate": (total_c / total_d) if total_d else 0.0,
        "tokens_emitted": len(steps) + total_c,
        "committed_dist": dist,
        "step_time_ms": {
            "mean": statistics.mean(times),
            "p50": percentile(times, 50),
            "p90": percentile(times, 90),
            "p99": percentile(times, 99),
        },
        "step_kinds": kinds,
        "implied_decode_tps": (
            (len(steps) + total_c) / (sum(times) / 1000.0) if sum(times) > 0 else 0.0
        ),
    }


MERGE_KEYS = ("accept_rate", "implied_decode_tps", "tokens_emitted", "total_steps",
              "total_drafted", "total_committed")


def merge_runs(paths: list[str]) -> dict:
    """Aggregate N per-run JSONs into one: means in the single-run scalar slots
    (downstream tables keep working), sample-std alongside under "std"."""
    runs = [json.loads(Path(p).read_text()) for p in paths]
    runs = [r for r in runs if r.get("total_steps", 0) > 0]
    if not runs:
        return {"total_steps": 0, "runs": 0}

    def series(key):
        return [float(r[key]) for r in runs if key in r]

    merged: dict = {"runs": len(runs)}
    std: dict = {}
    for key in MERGE_KEYS:
        xs = series(key)
        if not xs:
            continue
        merged[key] = statistics.mean(xs)
        std[key] = statistics.stdev(xs) if len(xs) >= 2 else 0.0
    step_means = [r["step_time_ms"]["mean"] for r in runs if "step_time_ms" in r]
    merged["step_time_ms"] = {
        "mean": statistics.mean(step_means),
        # p50/p90/p99 don't average meaningfully across runs; report the mean of
        # per-run percentiles, clearly keyed as such.
        "p50": statistics.mean([r["step_time_ms"]["p50"] for r in runs]),
        "p90": statistics.mean([r["step_time_ms"]["p90"] for r in runs]),
        "p99": statistics.mean([r["step_time_ms"]["p99"] for r in runs]),
    }
    std["step_time_ms_mean"] = statistics.stdev(step_means) if len(step_means) >= 2 else 0.0
    dist: dict[int, int] = {}
    kinds: dict[str, int] = {}
    for r in runs:
        for k, v in (r.get("committed_dist") or {}).items():
            dist[int(k)] = dist.get(int(k), 0) + int(v)
        for k, v in (r.get("step_kinds") or {}).items():
            kinds[k] = kinds.get(k, 0) + int(v)
    merged["committed_dist"] = dist
    merged["step_kinds"] = kinds
    merged["std"] = std
    return merged


def markdown(label: str, agg: dict) -> str:
    if agg["total_steps"] == 0:
        return f"# {label}\n\n_no mtp timing samples found_\n"
    dist = agg["committed_dist"]
    drafts = sorted(dist.keys())
    std = agg.get("std") or {}

    def pm(key, scale=1.0, prec=1):
        s = std.get(key)
        return f" ± {s * scale:.{prec}f}" if s is not None else ""

    runs_note = f" (mean of {agg['runs']} runs)" if agg.get("runs", 0) > 1 else ""
    if agg.get("runs", 0) > 1:
        accept_detail = ""
    else:
        accept_detail = f" ({agg['total_committed']}/{agg['total_drafted']})"
    out = [
        f"# {label}{runs_note}",
        "",
        f"- spec steps: **{agg['total_steps']:.0f}**",
        f"- accept rate: **{agg['accept_rate'] * 100:.1f}%{pm('accept_rate', 100)}**{accept_detail}",
        f"- tokens emitted: {agg['tokens_emitted']:.0f}",
        f"- implied decode tps: **{agg['implied_decode_tps']:.2f}{pm('implied_decode_tps', prec=2)}**",
        f"- step time ms: mean {agg['step_time_ms']['mean']:.1f}{pm('step_time_ms_mean')}, p50 {agg['step_time_ms']['p50']:.1f}, p90 {agg['step_time_ms']['p90']:.1f}, p99 {agg['step_time_ms']['p99']:.1f}",
        "",
        "| committed | count | fraction |",
        "| ---------:| -----:| --------:|",
    ]
    total = sum(dist.values())
    for k in drafts:
        out.append(f"| {k} | {dist[k]} | {dist[k] / total * 100:.1f}% |")
    out.append("")
    if agg["step_kinds"]:
        out.append("kinds: " + ", ".join(f"{k}={v}" for k, v in agg["step_kinds"].items()))
    return "\n".join(out) + "\n"


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="+",
                   help="stderr log (or - for stdin); with --merge: N per-run JSONs")
    p.add_argument("--label", default="mtp", help="label for the report header")
    p.add_argument("--json", default=None, help="write JSON sidecar here")
    p.add_argument("--merge", action="store_true",
                   help="inputs are per-run JSON sidecars; emit mean±std aggregate")
    args = p.parse_args(argv)

    if args.merge:
        agg = merge_runs(args.input)
    elif args.input[0] == "-":
        steps, kinds = parse(sys.stdin)
        agg = aggregate(steps, kinds)
    else:
        if len(args.input) > 1:
            p.error("multiple inputs only make sense with --merge")
        with open(args.input[0]) as fp:
            steps, kinds = parse(fp)
        agg = aggregate(steps, kinds)

    print(markdown(args.label, agg))
    if args.json:
        Path(args.json).write_text(json.dumps({"label": args.label, **agg}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
