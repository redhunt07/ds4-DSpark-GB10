#!/usr/bin/env python3
"""tools/perf/gamut_db.py — persistent run-store for gamut perf reports.

Every gamut capture (decode/prefill profile) is recorded as one immutable row in
a SQLite DB (default tools/perf/runs.db) keyed by a code+binary+flags fingerprint
plus rich metadata, so we never lose the ability to review/compare historical
runs even as run/<label>.json sidecars get overwritten.

  # one-time / safe to repeat — pull every existing runs/*.json into the DB
  tools/perf/gamut_db.py backfill

  # record a fresh capture (capture.sh does this automatically)
  tools/perf/gamut_db.py ingest runs/postfast.json \
      --code-id "$(jj log -r @ --no-graph -T change_id.short())" \
      --binary-sha "$(sha256sum ds4-bench | cut -c1-16)" \
      --flags "FAST_VERIFY=1 ctx4096 gen96 plain" --driver 580.159.03 --model ds4flash

  tools/perf/gamut_db.py list                 # recent runs
  tools/perf/gamut_db.py list --label postfast # filter
  tools/perf/gamut_db.py show 42               # one run + top kernels
  tools/perf/gamut_db.py compare prefast postfast   # per-kernel A/B (latest of each label)

Pure stdlib (sqlite3/json/hashlib/argparse). Date.now is fine here (CLI tool).
"""
import argparse, hashlib, json, os, sqlite3, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "runs.db")
DEFAULT_RUNS = os.path.join(HERE, "runs")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  ts          REAL,            -- unix epoch (capture/ingest/file-mtime)
  ts_iso      TEXT,
  label       TEXT,
  phase       TEXT,            -- decode | prefill
  hw          TEXT,
  code_id     TEXT,            -- jj change id or git sha
  binary_sha  TEXT,
  flags       TEXT,            -- e.g. "FAST_VERIFY=1 ctx4096 gen96 plain"
  driver      TEXT,
  model       TEXT,
  fingerprint TEXT,            -- sha256(code_id|binary_sha|flags|driver|model|phase)
  prefill_tps REAL,
  decode_tps  REAL,
  kvcache_mb  REAL,
  accept_pct  REAL,
  peak_bw_gbps REAL,
  digest      TEXT UNIQUE,     -- sha256 of report_json (idempotent ingest)
  kernels_json TEXT,
  report_json  TEXT,
  source_path  TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_label ON runs(label);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts);
CREATE INDEX IF NOT EXISTS idx_runs_fp ON runs(fingerprint);
"""


def connect(db):
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    return con


def fingerprint(code_id, binary_sha, flags, driver, model, phase):
    raw = "|".join(str(x or "") for x in (code_id, binary_sha, flags, driver, model, phase))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def ingest_report(con, report, *, source_path=None, ts=None, code_id=None,
                  binary_sha=None, flags=None, driver=None, model=None, phase=None):
    rj = json.dumps(report, sort_keys=True)
    digest = hashlib.sha256(rj.encode()).hexdigest()[:16]
    tp = report.get("throughput", {}) or {}
    ts = ts if ts is not None else time.time()
    phase = phase or report.get("phase") or _infer_phase(report)
    fp = fingerprint(code_id, binary_sha, flags, driver, model, phase)
    row = (
        ts, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
        report.get("label"), phase, report.get("hw"),
        code_id, binary_sha, flags, driver, model, fp,
        tp.get("prefill_tps"), tp.get("decode_tps"), tp.get("kvcache_mb"),
        tp.get("accept_pct"), report.get("peak_bw_gbps"),
        digest, json.dumps(report.get("kernels", [])), rj, source_path,
    )
    try:
        con.execute(
            "INSERT INTO runs(ts,ts_iso,label,phase,hw,code_id,binary_sha,flags,"
            "driver,model,fingerprint,prefill_tps,decode_tps,kvcache_mb,accept_pct,"
            "peak_bw_gbps,digest,kernels_json,report_json,source_path) "
            "VALUES(" + ",".join("?" * 20) + ")", row)
        con.commit()
        return con.execute("SELECT last_insert_rowid()").fetchone()[0]
    except sqlite3.IntegrityError:
        return None  # already ingested (same digest)


def _infer_phase(report):
    ks = report.get("kernels", [])
    names = " ".join(k.get("kernel", "") for k in ks)
    return "decode" if "decode" in names or "_warp8" in names else "prefill"


def cmd_ingest(args):
    con = connect(args.db)
    report = json.load(open(args.json))
    rid = ingest_report(con, report, source_path=args.json, code_id=args.code_id,
                        binary_sha=args.binary_sha, flags=args.flags,
                        driver=args.driver, model=args.model, phase=args.phase)
    print(f"ingested id={rid}" if rid else "already present (same digest) — skipped")


def cmd_backfill(args):
    con = connect(args.db)
    d = args.dir
    n = skip = 0
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(d, fn)
        try:
            report = json.load(open(path))
        except Exception:
            continue
        if "throughput" not in report and "kernels" not in report:
            continue
        # historical sidecars carry no fingerprint metadata; derive what we can
        rid = ingest_report(con, report, source_path=path,
                            ts=os.path.getmtime(path),
                            code_id="historical", flags=fn[:-5],
                            driver="unknown", model="unknown")
        if rid:
            n += 1
        else:
            skip += 1
    print(f"backfill: {n} ingested, {skip} already present  ->  {args.db}")


def _fmt(v, w, p=2):
    return ("{:>%d.%df}" % (w, p)).format(v) if isinstance(v, (int, float)) else " " * (w - len(str(v or "-"))) + str(v or "-")


def cmd_list(args):
    con = connect(args.db)
    q = "SELECT id,ts_iso,label,phase,code_id,flags,prefill_tps,decode_tps FROM runs"
    params = []
    if args.label:
        q += " WHERE label LIKE ?"
        params.append(f"%{args.label}%")
    q += " ORDER BY ts DESC, id DESC LIMIT ?"
    params.append(args.n)
    rows = con.execute(q, params).fetchall()
    print(f"{'id':>4} {'when':19} {'label':22} {'phase':7} {'code':12} {'pf':>7} {'dec':>7}  flags")
    for r in rows:
        pid, ts, lbl, ph, code, flags, pf, dec = r
        print(f"{pid:>4} {ts or '-':19} {(lbl or '-')[:22]:22} {(ph or '-'):7} {(code or '-')[:12]:12} "
              f"{_fmt(pf,7) } {_fmt(dec,7)}  {(flags or '')[:40]}")


def _resolve(con, ref):
    if ref.isdigit():
        r = con.execute("SELECT report_json FROM runs WHERE id=?", (int(ref),)).fetchone()
    else:
        r = con.execute("SELECT report_json FROM runs WHERE label LIKE ? ORDER BY ts DESC LIMIT 1",
                        (f"%{ref}%",)).fetchone()
    return json.loads(r[0]) if r else None


def cmd_show(args):
    con = connect(args.db)
    rep = _resolve(con, args.ref)
    if not rep:
        sys.exit(f"no run matching '{args.ref}'")
    tp = rep.get("throughput", {})
    print(f"# {rep.get('label')}  ({rep.get('hw')})")
    print(f"prefill={tp.get('prefill_tps')} decode={tp.get('decode_tps')} accept%={tp.get('accept_pct')}")
    print(f"{'kernel':46} {'ms':>9} {'%t':>6} {'regs':>5}")
    for k in sorted(rep.get("kernels", []), key=lambda x: -(x.get("ms") or 0))[:args.top]:
        print(f"{(k.get('kernel') or '')[:46]:46} {_fmt(k.get('ms'),9)} {_fmt(k.get('pct_time'),6,1)} {_fmt(k.get('regs'),5,0)}")


def cmd_compare(args):
    con = connect(args.db)
    a, b = _resolve(con, args.before), _resolve(con, args.after)
    if not a or not b:
        sys.exit("could not resolve both runs")
    ka = {k.get("kernel"): k for k in a.get("kernels", [])}
    kb = {k.get("kernel"): k for k in b.get("kernels", [])}
    tpa = a.get("throughput") or {}
    tpb = b.get("throughput") or {}
    ta, tb = tpa.get("decode_tps"), tpb.get("decode_tps")
    print(f"# compare  {a.get('label')}  ->  {b.get('label')}")
    print(f"decode t/s  {ta} -> {tb}")
    # MTP telemetry (joined from DS4_MTP_TIMING): accept + per-step verify cost
    def g(d, k): return d.get(k)
    for key, lbl, p in (("accept_pct", "accept %", 1), ("combined_accept_pct", "accept % (combined)", 1),
                        ("combined_tokens_per_iter", "tokens/iter", 2), ("combined_total_ms", "mtp step ms", 2),
                        ("verify_ms", "verify ms", 2), ("draft_ms", "draft ms", 2)):
        va, vb = g(tpa, key), g(tpb, key)
        if va is not None or vb is not None:
            fa = f"{va:.{p}f}" if isinstance(va, (int, float)) else "-"
            fb = f"{vb:.{p}f}" if isinstance(vb, (int, float)) else "-"
            print(f"{lbl:22} {fa:>9} -> {fb:>9}")
    print(f"{'kernel':46} {'ms a':>9} {'ms b':>9} {'Δms':>9} {'regs a→b':>10}")
    allk = sorted(set(ka) | set(kb), key=lambda n: -((kb.get(n, {}).get("ms") or 0)))
    for n in allk:
        if not n:
            continue
        ma = (ka.get(n, {}).get("ms") or 0.0)
        mb = (kb.get(n, {}).get("ms") or 0.0)
        ra = ka.get(n, {}).get("regs")
        rb = kb.get(n, {}).get("regs")
        flag = " ✗" if mb - ma > 20 else (" ✓" if ma - mb > 20 else "")
        print(f"{(n or '')[:46]:46} {_fmt(ma,9)} {_fmt(mb,9)} {mb-ma:>+9.1f} {str(ra)+'→'+str(rb):>10}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest"); p.add_argument("json")
    for opt in ("code-id", "binary-sha", "flags", "driver", "model", "phase"):
        p.add_argument("--" + opt, dest=opt.replace("-", "_"))
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("backfill"); p.add_argument("dir", nargs="?", default=DEFAULT_RUNS); p.set_defaults(fn=cmd_backfill)
    p = sub.add_parser("list"); p.add_argument("-n", type=int, default=25); p.add_argument("--label"); p.set_defaults(fn=cmd_list)
    p = sub.add_parser("show"); p.add_argument("ref"); p.add_argument("--top", type=int, default=15); p.set_defaults(fn=cmd_show)
    p = sub.add_parser("compare"); p.add_argument("before"); p.add_argument("after"); p.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
