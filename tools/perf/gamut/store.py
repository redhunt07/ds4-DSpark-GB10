"""gamut.store — persistent SQLite run-store for gamut reports.

Every report (decode/prefill profile, or a bench-matrix cell) is one immutable
row keyed by a code+binary+flags fingerprint plus rich metadata, so historical
runs stay reviewable even as runs/<label>.json sidecars get overwritten. The
full report (incl. MTP accept + verify timing) is kept in report_json; the hot
columns are denormalized for fast list/compare.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.dirname(HERE), "runs.db")
DEFAULT_RUNS = os.path.join(os.path.dirname(HERE), "runs")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  ts          REAL,
  ts_iso      TEXT,
  label       TEXT,
  phase       TEXT,            -- decode | prefill
  hw          TEXT,
  code_id     TEXT,            -- jj change id or git sha
  binary_sha  TEXT,
  flags       TEXT,
  driver      TEXT,
  model       TEXT,
  fingerprint TEXT,            -- sha256(code|bin|flags|driver|model|phase)
  prefill_tps REAL,
  decode_tps  REAL,
  kvcache_mb  REAL,
  accept_pct  REAL,
  combined_total_ms REAL,      -- MTP steady-state per-step cost
  verify_ms   REAL,            -- MTP verify-forward cost
  tokens_per_iter REAL,
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

_COLS = ["ts", "ts_iso", "label", "phase", "hw", "code_id", "binary_sha", "flags",
         "driver", "model", "fingerprint", "prefill_tps", "decode_tps", "kvcache_mb",
         "accept_pct", "combined_total_ms", "verify_ms", "tokens_per_iter",
         "peak_bw_gbps", "digest", "kernels_json", "report_json", "source_path"]


# Columns that may be absent in a runs.db created by an older schema. CREATE
# TABLE IF NOT EXISTS won't add them, so _migrate ALTER-ADDs any that are
# missing (SQLite adds them nullable, no rewrite). Keep in sync with the schema.
_MIGRATE_COLS = {
    "kvcache_mb": "REAL", "accept_pct": "REAL", "combined_total_ms": "REAL",
    "verify_ms": "REAL", "tokens_per_iter": "REAL", "peak_bw_gbps": "REAL",
    "kernels_json": "TEXT", "report_json": "TEXT", "source_path": "TEXT",
}


def _migrate(con: sqlite3.Connection) -> None:
    have = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    for name, typ in _MIGRATE_COLS.items():
        if name not in have:
            con.execute(f"ALTER TABLE runs ADD COLUMN {name} {typ}")
    con.commit()


def connect(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    _migrate(con)
    return con


def fingerprint(code_id, binary_sha, flags, driver, model, phase) -> str:
    raw = "|".join(str(x or "") for x in (code_id, binary_sha, flags, driver, model, phase))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _infer_phase(report: dict) -> str:
    names = " ".join(k.get("kernel", "") for k in report.get("kernels", []))
    return "decode" if "decode" in names or "_warp8" in names else "prefill"


def ingest(con: sqlite3.Connection, report: dict, *, source_path=None, ts=None,
           code_id=None, binary_sha=None, flags=None, driver=None, model=None,
           phase=None) -> int | None:
    rj = json.dumps(report, sort_keys=True)
    digest = hashlib.sha256(rj.encode()).hexdigest()[:16]
    tp = report.get("throughput", {}) or {}
    ts = ts if ts is not None else time.time()
    phase = phase or report.get("phase") or _infer_phase(report)
    fp = fingerprint(code_id, binary_sha, flags, driver, model, phase)
    row = {
        "ts": ts, "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
        "label": report.get("label"), "phase": phase, "hw": report.get("hw"),
        "code_id": code_id, "binary_sha": binary_sha, "flags": flags,
        "driver": driver, "model": model, "fingerprint": fp,
        "prefill_tps": tp.get("prefill_tps"), "decode_tps": tp.get("decode_tps"),
        "kvcache_mb": tp.get("kvcache_mb"), "accept_pct": tp.get("accept_pct"),
        "combined_total_ms": tp.get("combined_total_ms"), "verify_ms": tp.get("verify_ms"),
        "tokens_per_iter": tp.get("combined_tokens_per_iter", tp.get("tokens_per_iter")),
        "peak_bw_gbps": report.get("peak_bw_gbps"), "digest": digest,
        "kernels_json": json.dumps(report.get("kernels", [])), "report_json": rj,
        "source_path": source_path,
    }
    try:
        con.execute(
            f"INSERT INTO runs({','.join(_COLS)}) VALUES({','.join('?' * len(_COLS))})",
            [row[c] for c in _COLS])
        con.commit()
        return con.execute("SELECT last_insert_rowid()").fetchone()[0]
    except sqlite3.IntegrityError:
        return None  # same digest already ingested


def backfill(con: sqlite3.Connection, d: str) -> tuple[int, int]:
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
        rid = ingest(con, report, source_path=path, ts=os.path.getmtime(path),
                     code_id="historical", flags=fn[:-5], driver="unknown", model="unknown")
        n, skip = (n + 1, skip) if rid else (n, skip + 1)
    return n, skip


def resolve(con: sqlite3.Connection, ref: str) -> dict | None:
    if ref.isdigit():
        r = con.execute("SELECT report_json FROM runs WHERE id=?", (int(ref),)).fetchone()
    else:
        r = con.execute("SELECT report_json FROM runs WHERE label LIKE ? ORDER BY ts DESC LIMIT 1",
                        (f"%{ref}%",)).fetchone()
    return json.loads(r[0]) if r else None
