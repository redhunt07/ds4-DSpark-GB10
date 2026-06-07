"""gamut.report — assemble + render a joined decode-perf report.

build() joins the timing spine (nsys kernels) with gb20b HW metrics, ptxas
regs/occupancy, the roofline, ncu stalls, and MTP accept/verify telemetry into
one report dict. emit_markdown()/render_html() render it; derive_verdict() gives
the one-line bottleneck read.
"""

from __future__ import annotations

import html as _html
import json
import math

from . import hw, metrics as M, trace


def fmt(x, nd: int = 1, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}{suffix}"


def build(plain_sqlite: str, *, metrics_sqlite: str | None = None,
          ptxas_text: str | None = None, ncu: dict | None = None,
          accept: dict | None = None, phase: str = "decode",
          label: str = "(unlabeled run)", prefill_tps=None, decode_tps=None,
          kvcache_mb=None, top: int = 12, skip_warmup: int = 8,
          min_pct: float = 0.5) -> dict:
    h, model = hw.HW(), hw.Model()
    ncu = ncu or {}
    accept = accept or {}

    con = trace.connect(plain_sqlite)
    win = trace.phases(con, skip_warmup)
    awin = win.prefill if phase == "prefill" else win.steady
    kernels = trace.kernels_in(con, awin)
    steady_total_ns = sum(k.total_ns for k in kernels) or 1
    wall_ns = awin[1] - awin[0]
    n_steady = 1 if phase == "prefill" else max(1, win.n_tokens - win.skip)

    verdict: dict = {}
    pk_issue: dict = {}
    pk_warps: dict = {}
    if metrics_sqlite:
        mcon = trace.connect(metrics_sqlite)
        if M.has_gpu_metrics(mcon):
            mw = trace.phases(mcon, skip_warmup)
            mwin = mw.prefill if phase == "prefill" else mw.steady
            verdict = M.busy_metric_avgs(mcon, mwin, hw.VERDICT_METRICS)
            ivals = trace.kernel_intervals(mcon, mwin)
            pk_issue = M.per_kernel_metric(mcon, mwin, ivals, "sm_issue")
            pk_warps = M.per_kernel_metric(mcon, mwin, ivals, "compute_warps")

    ptx = M.parse_ptxas(ptxas_text) if ptxas_text else {}
    gaps = trace.launch_gaps(con, awin, top=6)

    rows = []
    for k in kernels:
        pct = 100.0 * k.total_ns / steady_total_ns
        if pct < min_pct:
            continue
        pi = ptx.get(k.slug)
        rf = M.roofline_estimate(k.slug, model)
        bw = M.achieved_gbps(rf.bytes_per_launch, k.launches, k.total_ns)
        nk = ncu.get(k.slug, {})
        rows.append({
            "kernel": k.slug,
            "pct_time": pct,
            "ms": k.total_ns / 1e6,
            "calls": k.launches,
            "avg_us": k.avg_ns / 1e3,
            "regs": pi.regs if pi else None,
            "smem": pi.smem if pi else None,
            "occ_theory": (M.theoretical_occupancy(pi.regs, pi.smem, h=h) if pi else float("nan")),
            "sm_issue": pk_issue.get(k.slug, float("nan")),
            "warps": pk_warps.get(k.slug, float("nan")),
            "class": rf.kind,
            "ai": M.arithmetic_intensity(rf),
            "bw_gbps": bw,
            "pct_peak_bw": (100.0 * bw / h.hbm_read_gbps if not math.isnan(bw) else float("nan")),
            "headroom": M.bw_headroom(rf.bytes_per_launch, k.avg_ns, h.hbm_read_gbps),
            "stall": (f"{nk['dominant']} {nk['dominant_pct']:.0f}%" if nk else None),
            "stalls": nk.get("stalls") if nk else None,
            "ncu_occ": nk.get("occupancy_pct") if nk else None,
        })
        if len(rows) >= top:
            break

    kernel_ns = sum(k.total_ns for k in kernels)
    idle_ns = max(0, wall_ns - kernel_ns)
    trace_decode_tps = n_steady / (wall_ns * 1e-9) if wall_ns else float("nan")

    report = {
        "label": label, "hw": h.name, "phase": phase,
        "windows": {"n_tokens": win.n_tokens, "skip": win.skip,
                    "steady_tokens": n_steady, "wall_ms": wall_ns / 1e6},
        "throughput": {"prefill_tps": prefill_tps, "decode_tps": decode_tps,
                       "kvcache_mb": kvcache_mb, **accept},
        "verdict_metrics": verdict,
        "time_split": {"wall_ms": wall_ns / 1e6, "kernel_ms": kernel_ns / 1e6,
                       "idle_ms": idle_ns / 1e6,
                       "kernel_pct": 100.0 * kernel_ns / wall_ns if wall_ns else 0,
                       "idle_pct": 100.0 * idle_ns / wall_ns if wall_ns else 0,
                       "trace_decode_tps": trace_decode_tps},
        "kernels": rows,
        "gaps": [{"us": g, "prev": p, "next": n} for g, p, n in gaps],
        "peak_bw_gbps": h.hbm_read_gbps,
        "has_ncu": bool(ncu),
    }
    report["verdict"] = derive_verdict(verdict, rows)
    return report


def derive_verdict(v: dict, rows: list[dict]) -> str:
    parts = []
    issue = v.get("sm_issue")
    tensor = v.get("tensor_active")
    if issue is not None and not math.isnan(issue) and issue < 20:
        bound = "memory-latency-bound at moderate occupancy"
        if tensor is not None and not math.isnan(tensor) and tensor < 5:
            bound += " (not compute/tensor-bound)"
        parts.append(bound)
    lever = None
    for r in rows:
        occ = r.get("occ_theory")
        if r["pct_time"] >= 10 and isinstance(occ, float) and not math.isnan(occ) and occ < 0.4:
            lever = r
            break
    if lever:
        parts.append(f"top lever: {trace.disp(lever['kernel'])} "
                     f"({lever['pct_time']:.0f}% time, occ {lever['occ_theory']*100:.0f}%)")
    return "; ".join(parts) if parts else "no single dominant lever flagged"


def emit_markdown(report: dict) -> str:
    t = report["throughput"]
    w, v, ts = report["windows"], report["verdict_metrics"], report["time_split"]
    L = [f"# gamut · {report['label']}  ({report['hw']}, {report.get('phase','decode')})", ""]
    L.append("## throughput")
    L.append(f"- prefill   {fmt(t.get('prefill_tps'))} t/s")
    L.append(f"- decode    {fmt(t.get('decode_tps'),2)} t/s")
    if "accept_pct" in t:
        ca = t.get("combined_accept_pct")
        L.append(f"- accept    {fmt(t.get('accept_pct'))}%"
                 + (f" (combined {fmt(ca)}%)" if ca is not None else "")
                 + f"   tokens/iter {fmt(t.get('combined_tokens_per_iter', t.get('tokens_per_iter')),2)}")
    if t.get("combined_total_ms") is not None:
        L.append(f"- mtp step  combined {fmt(t.get('combined_total_ms'),2)} ms"
                 + (f"   verify {fmt(t.get('verify_ms'),2)} ms" if t.get('verify_ms') is not None else "")
                 + (f"   draft {fmt(t.get('draft_ms'),2)} ms" if t.get('draft_ms') is not None else ""))
    if t.get("kvcache_mb") is not None:
        L.append(f"- kvcache   {fmt(t.get('kvcache_mb'))} MB")
    L.append("")
    L.append(f"## windows · {w['n_tokens']} tok (skip {w['skip']}), "
             f"wall {fmt(w['wall_ms'],1)} ms · kernel {fmt(ts['kernel_pct'])}% / idle {fmt(ts['idle_pct'])}%")
    if v:
        L.append("")
        L.append("## GPU HW (decode-windowed, busy samples)")
        for k in hw.VERDICT_METRICS:
            if k in v:
                note = "  ← stalled" if k == "sm_issue" and v[k] < 20 else ""
                L.append(f"- {k:14s} {fmt(v[k])}{note}")
    L.append("")
    L.append(f"**verdict:** {report['verdict']}")
    L.append("")
    L.append(f"## top {len(report['kernels'])} kernels ({report.get('phase','decode')})")
    L.append("| kernel | %t | ms | calls | avg µs | regs | occ | issue | class | %peakBW | stall |")
    L.append("| --- | --: | --: | --: | --: | --: | --: | --: | --- | --: | --- |")
    for r in report["kernels"]:
        L.append("| {k} | {pt} | {ms} | {c} | {au} | {rg} | {oc} | {iss} | {cl} | {bw} | {st} |".format(
            k=trace.disp(r["kernel"]), pt=fmt(r["pct_time"]), ms=fmt(r["ms"]),
            c=r["calls"], au=fmt(r["avg_us"]), rg=(r["regs"] if r["regs"] else "—"),
            oc=fmt(r["occ_theory"]*100 if isinstance(r["occ_theory"], float)
                   and not math.isnan(r["occ_theory"]) else None),
            iss=fmt(r["sm_issue"]), cl=r["class"], bw=fmt(r["pct_peak_bw"]),
            st=(r["stall"] or "—")))
    if report["gaps"]:
        L.append("")
        L.append("## top launch gaps (host/sched idle)")
        for g in report["gaps"]:
            L.append(f"- {fmt(g['us'])} µs  {trace.disp(g['prev'])} → {trace.disp(g['next'])}")
    return "\n".join(L)


def render_html(report: dict) -> str:
    """Self-contained single-file HTML report (no external assets)."""
    def esc(x): return _html.escape(str(x))
    t = report["throughput"]
    head = (f"<h1>gamut · {esc(report['label'])}</h1>"
            f"<p class=hw>{esc(report['hw'])} · {esc(report.get('phase','decode'))}</p>")
    tp = [f"<b>decode</b> {fmt(t.get('decode_tps'),2)} t/s",
          f"<b>prefill</b> {fmt(t.get('prefill_tps'))} t/s"]
    if "accept_pct" in t:
        tp.append(f"<b>accept</b> {fmt(t.get('combined_accept_pct', t.get('accept_pct')))}%")
    if t.get("combined_total_ms") is not None:
        tp.append(f"<b>mtp step</b> {fmt(t.get('combined_total_ms'),1)} ms "
                  f"(verify {fmt(t.get('verify_ms'),1)})")
    verdict = f"<p class=verdict>{esc(report['verdict'])}</p>"
    hdr = ["kernel", "%t", "ms", "calls", "avg µs", "regs", "occ%", "issue", "class", "%peakBW", "stall"]
    th = "".join(f"<th>{esc(c)}</th>" for c in hdr)
    trs = []
    for r in report["kernels"]:
        occ = (r["occ_theory"]*100 if isinstance(r["occ_theory"], float)
               and not math.isnan(r["occ_theory"]) else None)
        cells = [trace.disp(r["kernel"]), fmt(r["pct_time"]), fmt(r["ms"]), r["calls"],
                 fmt(r["avg_us"]), r["regs"] or "—", fmt(occ), fmt(r["sm_issue"]),
                 r["class"], fmt(r["pct_peak_bw"]), r["stall"] or "—"]
        trs.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in cells) + "</tr>")
    css = ("body{font:14px/1.5 ui-monospace,monospace;margin:2rem;max-width:60rem}"
           "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;"
           "padding:.25rem .5rem;text-align:right}th:first-child,td:first-child{text-align:left}"
           ".verdict{background:#fffbe6;padding:.5rem;border-left:3px solid #fa0}"
           ".tp{display:flex;gap:1.5rem;flex-wrap:wrap}.hw{color:#888}")
    return (f"<!doctype html><meta charset=utf-8><title>gamut {esc(report['label'])}</title>"
            f"<style>{css}</style>{head}<div class=tp>{''.join(f'<span>{x}</span>' for x in tp)}</div>"
            f"{verdict}<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
            f"<details><summary>raw json</summary><pre>{esc(json.dumps(report, indent=2))}</pre></details>")
