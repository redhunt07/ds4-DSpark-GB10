#!/usr/bin/env python3
"""Render the PR#1 correctness-gate primer markdown to a self-contained dark HTML.

No external network deps: Pygments CSS is inlined, all styling embedded.
"""
import sys
import markdown
from pygments.formatters import HtmlFormatter

SRC = sys.argv[1] if len(sys.argv) > 1 else "PR1-correctness-gate-primer.md"
OUT = sys.argv[2] if len(sys.argv) > 2 else "PR1-correctness-gate-primer.html"

with open(SRC, "r") as f:
    md_text = f.read()

md = markdown.Markdown(
    extensions=["fenced_code", "tables", "codehilite", "toc", "sane_lists", "attr_list"],
    extension_configs={
        "codehilite": {"guess_lang": False, "noclasses": False, "pygments_style": "dracula"},
        "toc": {"permalink": False, "title": "Contents"},
    },
)
body = md.convert(md_text)
toc = getattr(md, "toc", "")

pyg_css = HtmlFormatter(style="dracula").get_style_defs(".codehilite")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PR #1 — CUDA Correctness Gate · Primer</title>
<style>
:root {{
  --bg:#0e1116; --panel:#161b22; --fg:#d6deeb; --muted:#8b98b0; --line:#222b38;
  --accent:#7ee787; --accent2:#79c0ff; --code-bg:#11151c; --warn:#f0c674;
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{
  margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ display:grid; grid-template-columns:280px minmax(0,1fr); gap:0; max-width:1400px; margin:0 auto; }}
nav {{
  position:sticky; top:0; align-self:start; height:100vh; overflow-y:auto;
  padding:28px 18px 60px; border-right:1px solid var(--line); background:var(--panel);
  font-size:13.5px;
}}
nav .toctitle {{ display:none; }}
nav ul {{ list-style:none; margin:0; padding-left:12px; }}
nav > ul {{ padding-left:0; }}
nav a {{ color:var(--muted); text-decoration:none; display:block; padding:3px 6px; border-radius:5px; }}
nav a:hover {{ color:var(--fg); background:#1f2733; }}
main {{ padding:40px 56px 120px; min-width:0; }}
main > h1:first-child {{ margin-top:0; }}
h1,h2,h3,h4 {{ line-height:1.25; font-weight:650; }}
h1 {{ font-size:30px; letter-spacing:-.01em; border-bottom:1px solid var(--line); padding-bottom:14px; }}
h2 {{ font-size:23px; margin-top:48px; color:#fff; border-bottom:1px solid var(--line); padding-bottom:8px; }}
h3 {{ font-size:18.5px; margin-top:32px; color:var(--accent2); }}
h4 {{ font-size:16px; margin-top:24px; color:var(--accent); }}
a {{ color:var(--accent2); }}
p, li {{ color:var(--fg); }}
em {{ color:var(--muted); }}
strong {{ color:#fff; font-weight:650; }}
hr {{ border:0; border-top:1px solid var(--line); margin:40px 0; }}
code {{
  font:13.5px/1.5 "JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:var(--code-bg); padding:.12em .4em; border-radius:5px; color:#e6b673;
  border:1px solid #1c2430;
}}
pre {{ margin:18px 0; }}
.codehilite {{
  background:var(--code-bg); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; overflow-x:auto; box-shadow:0 1px 0 #0a0d12 inset;
}}
.codehilite pre {{ margin:0; }}
.codehilite code {{ background:none; border:0; padding:0; color:inherit; font-size:13.5px; line-height:1.55; }}
table {{ border-collapse:collapse; margin:20px 0; width:100%; font-size:14.5px; }}
th,td {{ border:1px solid var(--line); padding:8px 12px; text-align:left; vertical-align:top; }}
th {{ background:#1b2330; color:#fff; font-weight:600; }}
tr:nth-child(even) td {{ background:#12171f; }}
blockquote {{ border-left:3px solid var(--accent); margin:18px 0; padding:2px 18px; color:var(--muted); background:#12171f; border-radius:0 8px 8px 0; }}
.banner {{
  background:linear-gradient(90deg,#11202a,#161b22); border:1px solid var(--line);
  border-radius:10px; padding:14px 18px; margin:0 0 28px; font-size:14px; color:var(--muted);
}}
.banner b {{ color:var(--accent); }}
@media (max-width:900px) {{
  .wrap {{ grid-template-columns:1fr; }}
  nav {{ position:static; height:auto; border-right:0; border-bottom:1px solid var(--line); }}
  main {{ padding:28px 22px 80px; }}
}}
{pyg_css}
</style>
</head>
<body>
<div class="wrap">
<nav>{toc}</nav>
<main>
<div class="banner"><b>Internal primer</b> — not part of the PR commit. Branch <code>prep/correctness-gate</code> (<code>8950efb</code>), fork-only, no upstream PR open.</div>
{body}
</main>
</div>
</body>
</html>
"""

with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({len(html)} bytes)")
