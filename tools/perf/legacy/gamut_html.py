#!/usr/bin/env python3
"""tools/perf/gamut_html.py — render a gamut report JSON as a self-contained
HTML page (inline SVG + CSS, no JS deps, no CDN). Open it directly or serve it:

    tools/perf/gamut_html.py /tmp/gamut.json -o /tmp/gamut.html
    ( cd /tmp && python3 -m http.server 8009 )   # then open :8009/gamut.html

`gamut.py --html PATH` calls render() directly so one command produces both the
Markdown and the HTML.

Charts (drawn only where the data supports them):
  - kernel time distribution (horizontal bars)
  - GPU HW verdict (SM-issue / occupancy / tensor, with the stall threshold)
  - per-kernel occupancy: theoretical vs achieved
  - roofline scatter: arithmetic intensity vs % of peak BW
  - warp stall composition (stacked bars; only with --ncu data)
  - top launch gaps
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys

# palette (dark)
BG, FG, MUTE = "#0d1117", "#c9d1d9", "#8b949e"
ACCENT, BAR, WARN, GOOD, MID = "#58a6ff", "#388bfd", "#f85149", "#3fb950", "#d29922"
GRID = "#21262d"
STALL_COLORS = {
    "long_scb": "#f85149", "lg_throttle": "#d29922", "mio_throttle": "#db6d28",
    "short_scb": "#a371f7", "wait": "#58a6ff", "barrier": "#ff7b72",
    "not_selected": "#3fb950",
}


def _n(x, d=None):
    return d if x is None or (isinstance(x, float) and math.isnan(x)) else x


def esc(s) -> str:
    return html.escape(str(s))


# ---- SVG chart helpers ------------------------------------------------------

def hbars(items: list[tuple[str, float, str]], unit="", width=720, label_w=300,
          row_h=24, vmax=None) -> str:
    """items: (label, value, color). Horizontal bar chart as inline SVG."""
    if not items:
        return ""
    vmax = vmax or max(v for _, v, _ in items) or 1.0
    bar_w = width - label_w - 70
    h = row_h * len(items) + 8
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img">']
    for i, (label, v, color) in enumerate(items):
        y = i * row_h + 4
        w = max(1, bar_w * (v / vmax)) if vmax else 1
        out.append(
            f'<text x="{label_w-6}" y="{y+row_h*0.62}" class="lbl" '
            f'text-anchor="end">{esc(label)}</text>'
            f'<rect x="{label_w}" y="{y+3}" width="{w:.1f}" height="{row_h-8}" '
            f'rx="2" fill="{color}"/>'
            f'<text x="{label_w+w+6:.1f}" y="{y+row_h*0.62}" class="val">'
            f'{v:.1f}{unit}</text>')
    out.append("</svg>")
    return "".join(out)


def grouped_bars(rows: list, lna, lnb,
                 width=720, label_w=300, row_h=30) -> str:
    """rows: (label, a, b) two bars per row (0..100 scale)."""
    if not rows:
        return ""
    bar_w = width - label_w - 70
    h = row_h * len(rows) + 24
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img">']
    out.append(f'<text x="{label_w}" y="12" class="leg"><tspan fill="{MID}">■</tspan> '
               f'{esc(lna)}  <tspan fill="{ACCENT}">■</tspan> {esc(lnb)}</text>')
    for i, (label, a, b) in enumerate(rows):
        y = i * row_h + 20
        wa, wb = bar_w * (_n(a, 0) / 100), bar_w * (_n(b, 0) / 100)
        out.append(
            f'<text x="{label_w-6}" y="{y+row_h*0.55}" class="lbl" '
            f'text-anchor="end">{esc(label)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{wa:.1f}" height="{row_h/2-3}" rx="2" fill="{MID}"/>'
            f'<text x="{label_w+wa+5:.1f}" y="{y+row_h/2-5}" class="val">{_n(a,0):.0f}%</text>'
            f'<rect x="{label_w}" y="{y+row_h/2}" width="{wb:.1f}" height="{row_h/2-3}" rx="2" fill="{ACCENT}"/>'
            f'<text x="{label_w+wb+5:.1f}" y="{y+row_h-3}" class="val">{_n(b,0):.0f}%</text>')
    out.append("</svg>")
    return "".join(out)


def stacked_bars(rows: list[tuple[str, dict]], width=720, label_w=300,
                 row_h=26) -> str:
    """rows: (label, {stall: value}). Stacked horizontal composition (to 100%)."""
    if not rows:
        return ""
    bar_w = width - label_w - 10
    h = row_h * len(rows) + 26
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img">']
    keys = list(STALL_COLORS)
    lx = label_w
    out.append(f'<text x="{lx}" y="12" class="leg">' + "  ".join(
        f'<tspan fill="{STALL_COLORS[k]}">■</tspan>{esc(k)}' for k in keys) + "</text>")
    for i, (label, stalls) in enumerate(rows):
        y = i * row_h + 20
        tot = sum(stalls.values()) or 1.0
        x = label_w
        out.append(f'<text x="{label_w-6}" y="{y+row_h*0.6}" class="lbl" '
                   f'text-anchor="end">{esc(label)}</text>')
        for k in keys:
            v = stalls.get(k, 0.0)
            if v <= 0:
                continue
            w = bar_w * (v / tot)
            out.append(f'<rect x="{x:.1f}" y="{y+3}" width="{w:.1f}" '
                       f'height="{row_h-8}" fill="{STALL_COLORS[k]}"><title>'
                       f'{esc(k)}: {100*v/tot:.0f}%</title></rect>')
            x += w
    out.append("</svg>")
    return "".join(out)


def scatter(points: list, width=720, height=360) -> str:
    """points: (label, ai, pct_peak_bw, color). x=AI, y=%peakBW, ceiling at 100."""
    pts = [(l, ai, bw, c) for (l, ai, bw, c) in points
           if _n(ai) is not None and _n(bw) is not None]
    if not pts:
        return ""
    pad = 48
    xmax = max(ai for _, ai, _, _ in pts) * 1.15
    ymax = max(100.0, max(bw for _, _, bw, _ in pts) * 1.1)
    pw, ph = width - 2 * pad, height - 2 * pad

    def X(ai): return pad + pw * (ai / xmax)
    def Y(bw): return pad + ph * (1 - bw / ymax)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    # axes + ceiling
    out.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad+ph}" stroke="{GRID}"/>'
               f'<line x1="{pad}" y1="{pad+ph}" x2="{pad+pw}" y2="{pad+ph}" stroke="{GRID}"/>')
    yceil = Y(100)
    out.append(f'<line x1="{pad}" y1="{yceil:.1f}" x2="{pad+pw}" y2="{yceil:.1f}" '
               f'stroke="{WARN}" stroke-dasharray="4 3"/>'
               f'<text x="{pad+pw}" y="{yceil-5:.1f}" class="val" text-anchor="end" '
               f'fill="{WARN}">236 GB/s read ceiling</text>')
    out.append(f'<text x="{pad+pw/2}" y="{height-8}" class="axt" text-anchor="middle">'
               f'arithmetic intensity (flop/byte)</text>')
    out.append(f'<text x="14" y="{pad+ph/2}" class="axt" text-anchor="middle" '
               f'transform="rotate(-90 14 {pad+ph/2})">% of peak read BW</text>')
    for l, ai, bw, c in pts:
        cx, cy = X(ai), Y(bw)
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{c}" '
                   f'fill-opacity="0.8"><title>{esc(l)}: AI {ai:.1f}, {bw:.0f}% peak</title></circle>'
                   f'<text x="{cx+9:.1f}" y="{cy+4:.1f}" class="pt">{esc(l.split("<")[0][:22])}</text>')
    out.append("</svg>")
    return "".join(out)


# ---- page assembly ----------------------------------------------------------

def kclass_color(k: dict) -> str:
    bw = _n(k.get("pct_peak_bw"))
    if bw is not None and bw >= 70:
        return WARN          # near BW wall
    occ = _n(k.get("occ_theory"))
    if occ is not None and occ < 0.30:
        return MID           # occupancy-starved
    return BAR


def render(r: dict) -> str:
    w, t, v, ts = r["windows"], r["throughput"], r["verdict_metrics"], r["time_split"]
    ks = r["kernels"]

    def card(title, body):
        return f'<div class="card"><h3>{esc(title)}</h3>{body}</div>'

    # throughput card
    tp = [f'<div class="kv"><span>prefill</span><b>{_n(t.get("prefill_tps"),"—")} t/s</b></div>',
          f'<div class="kv"><span>decode</span><b>{_n(t.get("decode_tps"),"—")} t/s</b></div>']
    if "accept_pct" in t:
        tp.append(f'<div class="kv"><span>accept</span><b>{t["accept_pct"]:.1f}%</b></div>')
        tp.append(f'<div class="kv"><span>tokens/iter</span><b>{t["tokens_per_iter"]:.2f}</b></div>')
    if t.get("kvcache_mb") is not None:
        tp.append(f'<div class="kv"><span>kvcache</span><b>{t["kvcache_mb"]:.1f} MB</b></div>')

    # HW verdict bars
    hw_items = []
    for k in ("sms_active", "sm_issue", "tensor_active", "compute_warps"):
        if k in v:
            col = WARN if k == "sm_issue" and v[k] < 20 else BAR
            hw_items.append((k, v[k], col))
    hw_chart = hbars(hw_items, unit="%", vmax=100, label_w=150, width=560)

    # time split
    tsplit = (f'<div class="kv"><span>wall</span><b>{ts["wall_ms"]:.0f} ms</b></div>'
              f'<div class="kv"><span>kernel</span><b>{ts["kernel_ms"]:.0f} ms ({ts["kernel_pct"]:.0f}%)</b></div>'
              f'<div class="kv"><span>idle</span><b>{ts["idle_ms"]:.0f} ms ({ts["idle_pct"]:.0f}%)</b></div>'
              f'<div class="kv"><span>trace decode</span><b>{_n(ts.get("trace_decode_tps"),0):.1f} t/s</b></div>')

    # kernel time distribution
    timed = [(k["kernel"].split("<")[0][:34] + ("<3>" if "<3>" in k["kernel"] else ""),
              k["pct_time"], kclass_color(k)) for k in ks]
    time_chart = hbars(timed, unit="%", label_w=320)

    # occupancy theoretical vs achieved (compute_warps as achieved proxy)
    occ_rows = [(k["kernel"].split("<")[0][:34], (_n(k.get("occ_theory")) or 0) * 100,
                 _n(k.get("warps")))
                for k in ks if _n(k.get("occ_theory")) is not None]
    occ_chart = grouped_bars(occ_rows, "theoretical occ", "achieved warps", label_w=320)

    # roofline scatter
    rl = scatter([(k["kernel"], _n(k.get("ai")), _n(k.get("pct_peak_bw")), kclass_color(k))
                  for k in ks])

    # stall composition (only with ncu)
    stall_rows = [(k["kernel"].split("<")[0][:34], k["stalls"])
                  for k in ks if k.get("stalls")]
    stall_chart = stacked_bars(stall_rows, label_w=320) if stall_rows else ""

    # per-kernel detail table
    cols = ["kernel", "%t", "ms", "calls", "regs", "occ_th", "SMiss", "warps",
            "AI", "%peakBW"] + (["stall"] if r.get("has_ncu") else [])
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    trs = []
    for k in ks:
        occ = f'{k["occ_theory"]*100:.0f}%' if _n(k.get("occ_theory")) is not None else "—"
        cells = [
            f'<span class="mono">{esc(k["kernel"])}</span>',
            f'{k["pct_time"]:.1f}', f'{k["ms"]:.1f}', str(k["calls"]),
            str(k["regs"]) if k.get("regs") else "—", occ,
            f'{_n(k.get("sm_issue"),0):.1f}', f'{_n(k.get("warps"),0):.1f}',
            f'{_n(k.get("ai")):.1f}' if _n(k.get("ai")) is not None else "—",
            f'{_n(k.get("pct_peak_bw")):.0f}%' if _n(k.get("pct_peak_bw")) is not None else "—",
        ]
        if r.get("has_ncu"):
            cells.append(esc(k.get("stall") or "—"))
        trs.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    table = f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(trs)}</tbody></table>'

    # gaps
    gaps_html = ""
    if r.get("gaps"):
        gi = [(f'{esc(g["prev"][:20])} → {esc(g["next"][:20])}', g["us"], MID)
              for g in r["gaps"]]
        gaps_html = card("Top launch gaps (steady decode, host idle)",
                         hbars(gi, unit=" µs", label_w=360))

    def section(title, body):
        return f'<section><h2>{esc(title)}</h2>{body}</section>' if body else ""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ds4 gamut — {esc(r['label'])}</title>
<style>
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background:{BG}; color:{FG}; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
       margin:0; padding:32px; max-width:1000px; margin:0 auto; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:16px; color:{ACCENT}; border-bottom:1px solid {GRID};
      padding-bottom:6px; margin:34px 0 14px; }} h3 {{ font-size:13px; color:{MUTE}; margin:0 0 10px;
      text-transform:uppercase; letter-spacing:.04em; }}
.sub {{ color:{MUTE}; margin:0 0 8px; }}
.verdict {{ background:#161b22; border-left:3px solid {WARN}; padding:12px 16px; border-radius:6px;
            margin:18px 0; font-size:15px; }}
.cards {{ display:flex; flex-wrap:wrap; gap:14px; }}
.card {{ background:#161b22; border:1px solid {GRID}; border-radius:8px; padding:14px 16px; flex:1 1 240px; }}
.kv {{ display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px solid {GRID}; }}
.kv:last-child {{ border:0; }} .kv span {{ color:{MUTE}; }} .kv b {{ color:{FG}; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
th,td {{ text-align:right; padding:5px 8px; border-bottom:1px solid {GRID}; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:{MUTE}; font-weight:600; }}
.mono {{ font-family:ui-monospace,Menlo,monospace; font-size:11.5px; }}
.chart {{ width:100%; height:auto; margin:4px 0; }}
.chart .lbl {{ fill:{FG}; font:11px ui-monospace,monospace; }}
.chart .val {{ fill:{MUTE}; font:11px ui-monospace,monospace; }}
.chart .leg,.chart .axt {{ fill:{MUTE}; font:11px sans-serif; }}
.chart .pt {{ fill:{FG}; font:10px ui-monospace,monospace; }}
footer {{ color:{MUTE}; font-size:11px; margin-top:40px; border-top:1px solid {GRID}; padding-top:10px; }}
</style></head><body>
<h1>ds4 gamut — {esc(r['label'])}</h1>
<p class="sub"><span class="mono">{esc(r['hw'])}</span> · steady decode {w['steady_tokens']} tokens
 (of {w['n_tokens']}, skipped {w['skip']}) · roofline peak {r['peak_bw_gbps']:.0f} GB/s (measured read)</p>
<div class="verdict">⚑ {esc(r['verdict'])}</div>
<div class="cards">
  {card("Throughput", "".join(tp))}
  {card("Time split", tsplit)}
</div>
{section("GPU hardware (decode-windowed, busy samples)", hw_chart)}
{section("Kernel time distribution", time_chart)}
{section("Per-kernel detail", table)}
{section("Roofline — arithmetic intensity vs % peak BW", rl)}
{section("Occupancy: theoretical vs achieved", occ_chart)}
{section("Warp stall composition (ncu)", stall_chart)}
{gaps_html}
<footer>generated by tools/perf/gamut_html.py · charts are inline SVG (no external deps).
%peakBW &amp; AI are byte-model estimates; SM-issue/occupancy from gb20b; stalls from ncu.</footer>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("json", help="gamut report JSON (from gamut.py --json)")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    with open(args.json) as f:
        report = json.load(f)
    with open(args.out, "w") as f:
        f.write(render(report))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
