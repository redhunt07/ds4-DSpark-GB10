#!/usr/bin/env python3
"""tools/perf/gamut.py — one joined decode-perf report for ds4 on GB10.

Pulls the whole perf suite onto one page so you don't hand-join six tool
outputs. Ingests existing artifacts (it does not re-profile):

  --plain    nsys-exported sqlite from a `-t cuda` capture  (clean kernel time)
  --metrics  nsys-exported sqlite from a `gb20b` capture     (GPU HW metrics)
  --ptxas    `nvcc -Xptxas=-v` stderr text                   (regs / occupancy)
  --accept   stdout of a DS4_MTP_TIMING=1 run                (accept rate)
  --prefill-tps / --decode-tps / --kvcache-mb                (from ds4-bench)

Join key is the canonical kernel slug (perflib.canon_slug). Timing comes from
--plain; HW metrics come from --metrics (joined by slug, never by timeline,
since the two captures have different timelines and gb20b perturbs timing).
Per-kernel HW metrics are windowed within the gb20b run and gated at >=4
samples (short kernels can't be sampled reliably at 20 kHz).

Emits Markdown to stdout and, with --json PATH, a sidecar for A/B diffing.

Example:
    tools/perf/gamut.py --plain /tmp/p.sqlite --metrics /tmp/gm.sqlite \\
        --ptxas /tmp/ptxas.txt --accept /tmp/accept_run.txt \\
        --prefill-tps 408.9 --decode-tps 16.32 --kvcache-mb 52.2 \\
        --label "gb10-on-upstream @ c5b39429" --json /tmp/gamut.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys

import gamut_html
import perflib as P


def parse_accept(path: str) -> dict:
    """Join MTP `DS4_MTP_TIMING=1` telemetry into accept-rate + per-step timing.

    Lines look like:
      ds4: mtp timing micro    drafted=2 committed=2 draft=5.7 ms snapshot=0.0 ms verify=105.0 ms total=110.7 ms
      ds4: mtp timing combined drafted=2 committed=1 total=143.7 ms
    The `combined` steps are steady-state speculative decode (the throughput-
    relevant cost); `micro` is the warmup probe.  We surface accept-rate AND the
    per-step ms breakdown (combined total drives tok/s; verify is the dominant
    sub-cost) so the verify-forward residual is visible alongside throughput."""
    drafted = committed = iters = 0
    rx_dc = re.compile(r"drafted=(\d+)\s+committed=(\d+)")
    rx_kind = re.compile(r"mtp timing (\w+)")
    def field(k, l):
        m = re.search(rf"{k}=([\d.]+)\s*ms", l)
        return float(m.group(1)) if m else None
    comb_total, comb_acc = [], [0, 0]          # combined step totals; [committed, drafted]
    verify_ms, draft_ms, micro_total = [], [], []
    try:
        with open(path) as f:
            for line in f:
                m = rx_dc.search(line)
                if not m:
                    continue
                drafted += int(m.group(1)); committed += int(m.group(2)); iters += 1
                km = rx_kind.search(line)
                kind = km.group(1) if km else ""
                tot = field("total", line)
                if kind == "combined":
                    if tot is not None: comb_total.append(tot)
                    comb_acc[0] += int(m.group(2)); comb_acc[1] += int(m.group(1))
                elif kind == "micro":
                    if tot is not None: micro_total.append(tot)
                    v = field("verify", line);  d = field("draft", line)
                    if v is not None: verify_ms.append(v)
                    if d is not None: draft_ms.append(d)
    except OSError:
        return {}
    if iters == 0:
        return {}
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    out = {
        "accept_pct": 100.0 * committed / drafted if drafted else float("nan"),
        "tokens_per_iter": 1.0 + committed / iters,
        "iters": iters,
        "drafted": drafted,
        "committed": committed,
    }
    if comb_total:
        out["combined_total_ms"] = mean(comb_total)
        out["combined_steps"] = len(comb_total)
        if comb_acc[1]:
            out["combined_accept_pct"] = 100.0 * comb_acc[0] / comb_acc[1]
            out["combined_tokens_per_iter"] = 1.0 + comb_acc[0] / len(comb_total)
    if verify_ms: out["verify_ms"] = mean(verify_ms)
    if draft_ms:  out["draft_ms"] = mean(draft_ms)
    if micro_total: out["micro_total_ms"] = mean(micro_total)
    return out


def fmt(x: float, nd: int = 1, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}{suffix}"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--plain", required=True, help="nsys -t cuda sqlite")
    ap.add_argument("--metrics", help="nsys gb20b sqlite")
    ap.add_argument("--ptxas", help="nvcc -Xptxas=-v stderr text")
    ap.add_argument("--ncu", help="ncu_stalls.py JSON (per-kernel stall reasons)")
    ap.add_argument("--accept", help="DS4_MTP_TIMING=1 stdout")
    ap.add_argument("--prefill-tps", type=float)
    ap.add_argument("--decode-tps", type=float)
    ap.add_argument("--kvcache-mb", type=float)
    ap.add_argument("--label", default="(unlabeled run)")
    ap.add_argument("--phase", choices=["decode", "prefill"], default="decode",
                    help="which phase to analyze (default decode steady-state)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--skip-warmup", type=int, default=8)
    ap.add_argument("--min-pct", type=float, default=0.5)
    ap.add_argument("--json", help="write JSON sidecar here")
    ap.add_argument("--html", help="write self-contained HTML report here")
    args = ap.parse_args()

    hw = P.HW()
    model = P.Model()

    # --- timing spine from the clean plain trace --------------------------
    con = P.connect(args.plain)
    win = P.phases(con, args.skip_warmup)
    awin = win.prefill if args.phase == "prefill" else win.steady
    kernels = P.kernels_in(con, awin)
    steady_total_ns = sum(k.total_ns for k in kernels) or 1
    wall_ns = awin[1] - awin[0]
    n_steady = 1 if args.phase == "prefill" else max(1, win.n_tokens - win.skip)

    # --- HW metrics from the gb20b run (joined by slug) -------------------
    verdict: dict[str, float] = {}
    pk_issue: dict[str, float] = {}
    pk_warps: dict[str, float] = {}
    if args.metrics:
        mcon = P.connect(args.metrics)
        if P.has_gpu_metrics(mcon):
            mw = P.phases(mcon, args.skip_warmup)
            mwin = mw.prefill if args.phase == "prefill" else mw.steady
            verdict = P.busy_metric_avgs(mcon, mwin, P.VERDICT_METRICS)
            ivals = P.kernel_intervals(mcon, mwin)
            pk_issue = P.per_kernel_metric(mcon, mwin, ivals, "sm_issue")
            pk_warps = P.per_kernel_metric(mcon, mwin, ivals, "compute_warps")

    # --- ptxas regs / occupancy -------------------------------------------
    ptx: dict[str, P.PtxasInfo] = {}
    if args.ptxas:
        with open(args.ptxas) as f:
            ptx = P.parse_ptxas(f.read())

    # --- accept rate -------------------------------------------------------
    acc = parse_accept(args.accept) if args.accept else {}

    # --- ncu stall reasons (opt-in) ---------------------------------------
    ncu: dict = {}
    if args.ncu:
        with open(args.ncu) as f:
            ncu = json.load(f)

    # --- launch gaps (host/scheduling idle in steady decode) --------------
    gaps = P.launch_gaps(con, awin, top=6)

    # --- assemble per-kernel rows -----------------------------------------
    rows = []
    for k in kernels:
        pct = 100.0 * k.total_ns / steady_total_ns
        if pct < args.min_pct:
            continue
        pi = ptx.get(k.slug)
        rf = P.roofline_estimate(k.slug, model)
        bw = P.achieved_gbps(rf.bytes_per_launch, k.launches, k.total_ns)
        occ_t = (P.theoretical_occupancy(pi.regs, pi.smem) if pi else float("nan"))
        nk = ncu.get(k.slug, {})
        rows.append({
            "kernel": k.slug,
            "pct_time": pct,
            "ms": k.total_ns / 1e6,
            "calls": k.launches,
            "avg_us": k.avg_ns / 1e3,
            "regs": pi.regs if pi else None,
            "smem": pi.smem if pi else None,
            "occ_theory": occ_t,
            "sm_issue": pk_issue.get(k.slug, float("nan")),
            "warps": pk_warps.get(k.slug, float("nan")),
            "class": rf.kind,
            "ai": P.arithmetic_intensity(rf),
            "bw_gbps": bw,
            "pct_peak_bw": (100.0 * bw / hw.hbm_read_gbps
                            if not math.isnan(bw) else float("nan")),
            "headroom": P.bw_headroom(rf.bytes_per_launch, k.avg_ns, hw.hbm_read_gbps),
            "stall": (f"{nk['dominant']} {nk['dominant_pct']:.0f}%" if nk else None),
            "stalls": nk.get("stalls") if nk else None,
            "ncu_occ": nk.get("occupancy_pct") if nk else None,
        })
        if len(rows) >= args.top:
            break

    kernel_ns = sum(k.total_ns for k in kernels)
    idle_ns = max(0, wall_ns - kernel_ns)
    trace_decode_tps = n_steady / (wall_ns * 1e-9) if wall_ns else float("nan")

    verdict_line = derive_verdict(verdict, rows)

    report = {
        "label": args.label,
        "hw": hw.name,
        "windows": {"n_tokens": win.n_tokens, "skip": win.skip,
                    "steady_tokens": n_steady, "wall_ms": wall_ns / 1e6},
        "throughput": {"prefill_tps": args.prefill_tps,
                       "decode_tps": args.decode_tps,
                       "kvcache_mb": args.kvcache_mb, **acc},
        "verdict_metrics": verdict,
        "time_split": {"wall_ms": wall_ns / 1e6, "kernel_ms": kernel_ns / 1e6,
                       "idle_ms": idle_ns / 1e6,
                       "kernel_pct": 100.0 * kernel_ns / wall_ns if wall_ns else 0,
                       "idle_pct": 100.0 * idle_ns / wall_ns if wall_ns else 0,
                       "trace_decode_tps": trace_decode_tps},
        "kernels": rows,
        "gaps": [{"us": g, "prev": p, "next": n} for g, p, n in gaps],
        "peak_bw_gbps": hw.hbm_read_gbps,
        "has_ncu": bool(ncu),
        "verdict": verdict_line,
    }

    emit_markdown(report)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n<!-- json sidecar: {args.json} -->", file=sys.stderr)
    if args.html:
        with open(args.html, "w") as f:
            f.write(gamut_html.render(report))
        print(f"<!-- html report: {args.html} -->", file=sys.stderr)
    return 0


def derive_verdict(v: dict, rows: list[dict]) -> str:
    """One-line bottleneck read, derived from the metrics + the kernel join."""
    parts = []
    issue = v.get("sm_issue")
    tensor = v.get("tensor_active")
    if issue is not None and not math.isnan(issue) and issue < 20:
        bound = "memory-latency-bound at moderate occupancy"
        if tensor is not None and not math.isnan(tensor) and tensor < 5:
            bound += " (not compute/tensor-bound)"
        parts.append(bound)
    # Lever = highest-%time kernel that is occupancy-starved (low theory occ).
    lever = None
    for k in rows:
        occ = k.get("occ_theory")
        if occ is not None and not math.isnan(occ) and occ < 0.30 and k["pct_time"] > 5:
            if lever is None or k["pct_time"] > lever["pct_time"]:
                lever = k
    if lever:
        bw = lever.get("pct_peak_bw")
        bwtxt = (f", {bw:.0f}% of peak BW" if bw is not None and not math.isnan(bw)
                 else "")
        stalltxt = f", stall: {lever['stall']}" if lever.get("stall") else ""
        parts.append(
            f"top lever: {lever['kernel']} occupancy-starved "
            f"({lever['regs']} regs → {lever['occ_theory']*100:.0f}% occ{bwtxt}{stalltxt})")
    return ". ".join(parts) if parts else "no clear single bottleneck"


def emit_markdown(r: dict) -> None:
    w, t, v, ts = r["windows"], r["throughput"], r["verdict_metrics"], r["time_split"]
    print(f"# ds4 gamut — {r['label']}")
    print(f"\n`{r['hw']}` · steady decode = {w['steady_tokens']} tokens "
          f"(of {w['n_tokens']}, skipped {w['skip']} warmup) · "
          f"roofline peak {r['peak_bw_gbps']:.0f} GB/s (measured read)\n")

    # throughput + HW verdict, side by side as two small blocks
    print("## Throughput")
    print(f"- prefill   {fmt(t.get('prefill_tps'))} t/s")
    print(f"- decode    {fmt(t.get('decode_tps'),2)} t/s")
    if "accept_pct" in t:
        ca = t.get("combined_accept_pct")
        print(f"- accept    {fmt(t.get('accept_pct'))}%"
              + (f" (combined {fmt(ca)}%)" if ca is not None else "")
              + f"   tokens/iter {fmt(t.get('combined_tokens_per_iter', t.get('tokens_per_iter')),2)}")
    if t.get("combined_total_ms") is not None:
        print(f"- mtp step  combined {fmt(t.get('combined_total_ms'),2)} ms"
              + (f"   verify {fmt(t.get('verify_ms'),2)} ms" if t.get('verify_ms') is not None else "")
              + (f"   draft {fmt(t.get('draft_ms'),2)} ms" if t.get('draft_ms') is not None else ""))
    if t.get("kvcache_mb") is not None:
        print(f"- kvcache   {fmt(t.get('kvcache_mb'))} MB")

    print("\n## GPU HW (decode-windowed, busy samples)")
    for k in P.VERDICT_METRICS:
        if k in v:
            note = "  ← stalled" if k == "sm_issue" and v[k] < 20 else ""
            print(f"- {k:14s} {fmt(v[k])}%{note}")

    print("\n## Time split")
    print(f"- wall {fmt(ts['wall_ms'])} ms · kernel {fmt(ts['kernel_ms'])} ms "
          f"({fmt(ts['kernel_pct'])}%) · idle {fmt(ts['idle_ms'])} ms "
          f"({fmt(ts['idle_pct'])}%)  ·  trace decode {fmt(ts.get('trace_decode_tps'),1)} t/s")
    print("  *(kernel = Σ durations; concurrent kernels not de-overlapped)*")

    print(f"\n## Verdict\n> {r['verdict']}")

    stall_hdr = " stall |" if r.get("has_ncu") else ""
    stall_sep = "------:|" if r.get("has_ncu") else ""
    print("\n## Per-kernel (time ← plain trace · HW ← gb20b windowed · "
          "occ_th = ptxas theoretical · AI = flop/byte · %peakBW = est vs 236 GB/s"
          + (" · stall ← ncu)" if r.get("has_ncu") else ")"))
    print(f"| kernel | %t | ms | calls | regs | occ_th | SMiss | warps | AI | %peakBW |{stall_hdr}")
    print(f"|--------|---:|---:|------:|-----:|-------:|------:|------:|---:|--------:|{stall_sep}")
    for k in r["kernels"]:
        occ = (f"{k['occ_theory']*100:.0f}%" if not math.isnan(k['occ_theory'])
               else "—")
        stall_col = f" {k['stall'] or '—'} |" if r.get("has_ncu") else ""
        print(f"| {P.disp(k['kernel'])} | {fmt(k['pct_time'])} | {fmt(k['ms'])} | "
              f"{k['calls']} | {k['regs'] if k['regs'] else '—'} | {occ} | "
              f"{fmt(k['sm_issue'])} | {fmt(k['warps'])} | {fmt(k['ai'])} | "
              f"{fmt(k['pct_peak_bw'],0,'%')} |{stall_col}")

    if r.get("gaps"):
        print("\n## Top launch gaps (steady decode, host/scheduling idle)")
        print("| gap µs | prev kernel | next kernel |")
        print("|-------:|-------------|-------------|")
        for g in r["gaps"]:
            print(f"| {g['us']:.1f} | {P.disp(g['prev'],28)} | {P.disp(g['next'],28)} |")


if __name__ == "__main__":
    sys.exit(main())
