"""gamut.cli — one entry point: capture | bench | report | db.

  gamut capture --label X --fast --rebuild           # nsys-trace + report + store
  gamut bench   --label X --matrix --iter 3 --fast   # throughput matrix + monitor
  gamut report  /tmp/run_p.sqlite --accept acc.txt   # analyze an existing trace
  gamut db list|show <ref>|compare <a> <b>|backfill
"""

from __future__ import annotations

import argparse
import json
import sys

from . import mtp, report as R, store


def _fmt(v, w, p=2):
    if isinstance(v, (int, float)):
        return ("{:>%d.%df}" % (w, p)).format(v)
    return (str(v or "-")).rjust(w)


# ---- capture ----------------------------------------------------------------

def cmd_capture(a) -> int:
    from .capture import CaptureCfg, run, MTP_DEFAULT
    cfg = CaptureCfg(
        label=a.label, model=a.model, mtp=(None if a.no_mtp else (a.mtp or MTP_DEFAULT)),
        prompt=a.prompt, prompt_file=a.prompt_file, ntok=a.ntok, temp=a.temp,
        ctx=a.ctx, think=a.think, warm=a.warm, rebuild=a.rebuild, do_ncu=a.ncu,
        fast_verify=a.fast)
    run(cfg, db=a.db)
    return 0


# ---- bench ------------------------------------------------------------------

def cmd_bench(a) -> int:
    from .bench import BenchCfg, run
    cfg = BenchCfg(
        label=a.label, matrix=a.matrix, use_mtp=not a.no_mtp, use_temp=a.temp,
        iters=a.iter, warmup=a.warmup, ctx_start=a.ctx_start, ctx_max=a.ctx_max,
        gen_tokens=a.gen_tokens, fast_verify=a.fast, prewarm=not a.no_prewarm,
        cooldown=not a.no_cooldown, cooldown_c=a.cooldown_c)
    if a.model:
        cfg.model = a.model
    if a.prompt_file:
        cfg.prompt_file = a.prompt_file
    run(cfg)
    import pathlib
    print((pathlib.Path(__file__).resolve().parents[1] / "runs" / a.label / "summary.txt").read_text())
    return 0


# ---- report -----------------------------------------------------------------

def cmd_report(a) -> int:
    accept = mtp.parse_timing(a.accept) if a.accept else {}
    ptxas_text = open(a.ptxas).read() if a.ptxas else None
    ncu = json.load(open(a.ncu)) if a.ncu else {}
    rep = R.build(a.plain, metrics_sqlite=a.metrics, ptxas_text=ptxas_text, ncu=ncu,
                  accept=accept, phase=a.phase, label=a.label, top=a.top,
                  skip_warmup=a.skip_warmup, min_pct=a.min_pct,
                  prefill_tps=a.prefill_tps, decode_tps=a.decode_tps, kvcache_mb=a.kvcache_mb)
    print(R.emit_markdown(rep))
    if a.json:
        open(a.json, "w").write(json.dumps(rep, indent=2))
    if a.html:
        open(a.html, "w").write(R.render_html(rep))
    if a.ingest:
        con = store.connect(a.db)
        rid = store.ingest(con, rep, source_path=a.json, code_id=a.code_id,
                           binary_sha=a.binary_sha, flags=a.flags, driver=a.driver,
                           model=a.model_name, phase=a.phase)
        print(f"\n# ingested id={rid}" if rid else "\n# already present", file=sys.stderr)
    return 0


# ---- db ---------------------------------------------------------------------

def cmd_db(a) -> int:
    con = store.connect(a.db)
    if a.sub == "backfill":
        n, skip = store.backfill(con, a.dir)
        print(f"backfill: {n} ingested, {skip} already present -> {a.db}")
    elif a.sub == "list":
        q = "SELECT id,ts_iso,label,phase,code_id,flags,prefill_tps,decode_tps,accept_pct FROM runs"
        params = []
        if a.label:
            q += " WHERE label LIKE ?"; params.append(f"%{a.label}%")
        q += " ORDER BY ts DESC, id DESC LIMIT ?"; params.append(a.n)
        print(f"{'id':>4} {'when':19} {'label':22} {'phase':7} {'code':12} "
              f"{'pf':>7} {'dec':>7} {'acc%':>6}  flags")
        for r in con.execute(q, params):
            print(f"{r[0]:>4} {r[1] or '-':19} {(r[2] or '-')[:22]:22} {(r[3] or '-'):7} "
                  f"{(r[4] or '-')[:12]:12} {_fmt(r[6],7)} {_fmt(r[7],7)} {_fmt(r[8],6,1)}  "
                  f"{(r[5] or '')[:36]}")
    elif a.sub == "show":
        rep = store.resolve(con, a.ref)
        if not rep:
            sys.exit(f"no run matching '{a.ref}'")
        print(R.emit_markdown(rep))
    elif a.sub == "compare":
        _compare(con, a.before, a.after)
    return 0


def _compare(con, before: str, after: str) -> None:
    a, b = store.resolve(con, before), store.resolve(con, after)
    if not a or not b:
        sys.exit("could not resolve both runs")
    ta, tb = a.get("throughput") or {}, b.get("throughput") or {}
    print(f"# compare  {a.get('label')}  ->  {b.get('label')}")
    print(f"decode t/s  {ta.get('decode_tps')} -> {tb.get('decode_tps')}")
    for key, lbl, p in (("accept_pct", "accept %", 1),
                        ("combined_accept_pct", "accept % (combined)", 1),
                        ("combined_tokens_per_iter", "tokens/iter", 2),
                        ("combined_total_ms", "mtp step ms", 2),
                        ("verify_ms", "verify ms", 2), ("draft_ms", "draft ms", 2)):
        va, vb = ta.get(key), tb.get(key)
        if va is not None or vb is not None:
            fa = f"{va:.{p}f}" if isinstance(va, (int, float)) else "-"
            fb = f"{vb:.{p}f}" if isinstance(vb, (int, float)) else "-"
            print(f"{lbl:22} {fa:>9} -> {fb:>9}")
    ka = {k["kernel"]: k for k in a.get("kernels", [])}
    kb = {k["kernel"]: k for k in b.get("kernels", [])}
    print(f"\n{'kernel':46} {'ms a':>9} {'ms b':>9} {'Δms':>9}")
    from .trace import disp
    for n in sorted(set(ka) | set(kb), key=lambda n: -((kb.get(n, {}).get("ms") or 0))):
        ma = ka.get(n, {}).get("ms") or 0.0
        mb = kb.get(n, {}).get("ms") or 0.0
        flag = " ✗" if mb - ma > 0.5 else (" ✓" if ma - mb > 0.5 else "")
        print(f"{disp(n):46} {ma:>9.2f} {mb:>9.2f} {mb-ma:>+9.2f}{flag}")


# ---- argparse ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gamut", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=store.DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="nsys-trace a ds4 run -> report + store")
    c.add_argument("--label", required=True)
    c.add_argument("-m", "--model", default="ds4flash.gguf")
    c.add_argument("--mtp"); c.add_argument("--no-mtp", action="store_true")
    c.add_argument("-p", "--prompt", default="knight")
    c.add_argument("--prompt-file")
    c.add_argument("-n", "--ntok", type=int, default=48)
    c.add_argument("--temp", type=float, default=0.0)
    c.add_argument("--ctx", type=int)
    c.add_argument("--think", action="store_true")
    c.add_argument("--warm", action="store_true")
    c.add_argument("--rebuild", action="store_true")
    c.add_argument("--ncu", action="store_true")
    c.add_argument("--fast", action="store_true", help="DS4_CUDA_FAST_VERIFY=1")
    c.set_defaults(fn=cmd_capture)

    b = sub.add_parser("bench", help="throughput matrix + GPU/thermal monitor")
    b.add_argument("--label", required=True)
    b.add_argument("--matrix", action="store_true")
    b.add_argument("--iter", type=int, default=1)
    b.add_argument("--warmup", type=int, default=0,
                   help="leading iters per cell, run but excluded from stats")
    b.add_argument("--no-mtp", action="store_true")
    b.add_argument("--temp", action="store_true", help="sampled (temp=1.0) cell")
    b.add_argument("-m", "--model"); b.add_argument("--prompt-file")
    b.add_argument("--ctx-start", type=int, default=4096)
    b.add_argument("--ctx-max", type=int, default=32768)
    b.add_argument("--gen-tokens", type=int, default=32)
    b.add_argument("--fast", action="store_true", help="DS4_CUDA_FAST_VERIFY=1")
    b.add_argument("--no-prewarm", action="store_true",
                   help="skip reading model+MTP into page cache before the cells")
    b.add_argument("--no-cooldown", action="store_true",
                   help="skip the between-cell GPU cooldown (anti-soak) wait")
    b.add_argument("--cooldown-c", type=int, default=55,
                   help="cool the GPU to <= this °C before each next cell (default 55)")
    b.set_defaults(fn=cmd_bench)

    r = sub.add_parser("report", help="analyze an existing nsys sqlite")
    r.add_argument("plain", help="nsys -t cuda sqlite")
    r.add_argument("--metrics"); r.add_argument("--ptxas"); r.add_argument("--ncu")
    r.add_argument("--accept", help="DS4_MTP_TIMING=1 stdout")
    r.add_argument("--phase", choices=["decode", "prefill"], default="decode")
    r.add_argument("--label", default="(unlabeled run)")
    r.add_argument("--top", type=int, default=12)
    r.add_argument("--skip-warmup", type=int, default=8)
    r.add_argument("--min-pct", type=float, default=0.5)
    r.add_argument("--prefill-tps", type=float); r.add_argument("--decode-tps", type=float)
    r.add_argument("--kvcache-mb", type=float)
    r.add_argument("--json"); r.add_argument("--html")
    r.add_argument("--ingest", action="store_true")
    r.add_argument("--code-id"); r.add_argument("--binary-sha"); r.add_argument("--flags")
    r.add_argument("--driver"); r.add_argument("--model-name")
    r.set_defaults(fn=cmd_report)

    d = sub.add_parser("db", help="run-store: list/show/compare/backfill")
    dsub = d.add_subparsers(dest="sub", required=True)
    p = dsub.add_parser("list"); p.add_argument("-n", type=int, default=25); p.add_argument("--label")
    p = dsub.add_parser("show"); p.add_argument("ref")
    p = dsub.add_parser("compare"); p.add_argument("before"); p.add_argument("after")
    p = dsub.add_parser("backfill"); p.add_argument("dir", nargs="?", default=store.DEFAULT_RUNS)
    d.set_defaults(fn=cmd_db)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
