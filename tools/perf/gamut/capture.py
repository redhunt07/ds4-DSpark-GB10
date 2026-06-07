"""gamut.capture — nsys-trace a ds4 run and build a joined report (replaces capture.sh).

One command runs the whole dance and preserves it: nsys plain-cuda trace +
optional gb20b metrics + ptxas regs + MTP accept telemetry (+ optional ncu),
then build a report and ingest it to the run-store with a code+binary+driver
fingerprint. Captures are serialized with an flock so concurrent runs can't
corrupt each other's nsys traces (the failure mode behind the old cryptic
"no such table: CUPTI_ACTIVITY_KIND_KERNEL").
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import metrics as M, mtp, report as R, store

ROOT = Path(__file__).resolve().parents[3]   # repo root (…/ds4)
PERF = Path(__file__).resolve().parents[1]    # …/tools/perf
NVCC = "/usr/local/cuda/bin/nvcc"
ARCH = "-gencode=arch=compute_121a,code=sm_121a"
LOCK = "/tmp/ds4-capture.lock"
MODEL_DEFAULT = "ds4flash.gguf"
MTP_DEFAULT = str(Path.home() / "models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf")


@dataclass
class CaptureCfg:
    label: str
    model: str = MODEL_DEFAULT
    mtp: str | None = MTP_DEFAULT
    prompt: str = "knight"
    prompt_file: str | None = None
    ntok: int = 48
    temp: float = 0.0
    ctx: int | None = None
    think: bool = False
    warm: bool = False
    rebuild: bool = False
    do_ncu: bool = False
    fast_verify: bool = False
    ncu_kernels: str = ("moe_down_expert_tile8_row32|matmul_q8_0_preq_batch_share_warp"
                        "|moe_gate_up_mid_expert_tile8_row32")
    extra_env: dict = field(default_factory=dict)


def _ds4_args(cfg: CaptureCfg) -> list[str]:
    a = ["-m", cfg.model, "-n", str(cfg.ntok), "--temp", str(cfg.temp),
         "--think" if cfg.think else "--nothink", "-sys", ""]
    if cfg.mtp:
        a += ["--mtp", cfg.mtp, "--mtp-draft", "2"]
    if cfg.prompt_file:
        a += ["--prompt-file", cfg.prompt_file]
    else:
        a += ["-p", cfg.prompt]
    if cfg.ctx:
        a += ["--ctx", str(cfg.ctx)]
    if cfg.warm:
        a += ["--warm-weights"]
    return a


def _env(cfg: CaptureCfg) -> dict:
    e = dict(os.environ)
    if cfg.fast_verify:
        e["DS4_CUDA_FAST_VERIFY"] = "1"
    e.update(cfg.extra_env)
    return e


def _nsys_sqlite(out_base: str, ds4: str, args: list[str], env: dict,
                 extra: list[str]) -> str:
    """nsys profile + export to sqlite, with kernel-row validation."""
    rep = out_base + ".nsys-rep"
    sql = out_base + ".sqlite"
    with open(out_base + ".nsyslog", "w") as log:
        subprocess.run(["nsys", "profile", "-o", out_base, "--force-overwrite", "true",
                        *extra, ds4, *args], env=env, stdout=log, stderr=subprocess.STDOUT)
    if not os.path.exists(rep) or os.path.getsize(rep) == 0:
        raise SystemExit(f"gamut.capture: nsys produced no {rep} (see {out_base}.nsyslog)")
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                    "-o", sql, rep], check=True, capture_output=True)
    return sql


def _kernel_rows(sql: str) -> int:
    try:
        c = sqlite3.connect(sql)
        return c.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    except Exception:
        return 0


def _fingerprint() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
        except Exception:
            return ""
    code = run(["jj", "log", "-r", "@", "--no-graph", "-T", "change_id.short()"]) \
        or run(["git", "rev-parse", "--short", "HEAD"]) or "?"
    driver = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).split("\n")[0]
    return {"code_id": code, "driver": driver.strip()}


def run(cfg: CaptureCfg, db: str = store.DEFAULT_DB) -> dict:
    runs = PERF / "runs"
    runs.mkdir(exist_ok=True)
    tmp = f"/tmp/{cfg.label}"
    ds4 = str(ROOT / "ds4")

    lf = open(LOCK, "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"gamut.capture: another capture holds {LOCK}; refusing to start")
    try:
        Path("/tmp/ds4.lock").unlink(missing_ok=True)
        if cfg.rebuild:
            print("## rebuild (make cuda-spark)", file=sys.stderr)
            r = subprocess.run(["make", "cuda-spark"], cwd=ROOT,
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"gamut.capture: build failed\n{r.stderr[-2000:]}")
        if not os.access(ds4, os.X_OK):
            raise SystemExit("gamut.capture: ./ds4 not built; pass rebuild=True")

        args = _ds4_args(cfg)
        env = _env(cfg)

        print("## 1/5 plain CUDA trace", file=sys.stderr)
        plain_sql = _nsys_sqlite(f"{tmp}_p", ds4, args, env, ["-t", "cuda", "--sample", "none"])
        krows = _kernel_rows(plain_sql)
        if krows < 1000:
            raise SystemExit(f"gamut.capture: plain trace has only {krows} kernel rows "
                             "(truncated/empty) — was another GPU job running?")
        print(f"   plain trace OK: {krows} kernel rows", file=sys.stderr)

        print("## 2/5 gb20b GPU metrics", file=sys.stderr)
        metrics_sql = None
        try:
            metrics_sql = _nsys_sqlite(f"{tmp}_gm", ds4, args, env,
                                       ["--gpu-metrics-devices=0", "--gpu-metrics-set=gb20b",
                                        "--gpu-metrics-frequency=20000"])
        except Exception as e:
            print(f"   (gb20b metrics unavailable: {e})", file=sys.stderr)

        print("## 3/5 ptxas registers", file=sys.stderr)
        ptxas_text = None
        try:
            r = subprocess.run([NVCC, "-O3", "--use_fast_math", *ARCH.split(),
                                "-Xptxas=-v", "-c", "-o", "/tmp/_ptxas.o", "ds4_cuda.cu"],
                               cwd=ROOT, capture_output=True, text=True)
            ptxas_text = r.stderr
        except Exception as e:
            print(f"   (ptxas pass failed: {e})", file=sys.stderr)

        print("## 4/5 accept + throughput", file=sys.stderr)
        accept_log = f"{tmp}_accept.txt"
        with open(accept_log, "w") as af:
            subprocess.run([ds4, *args], env={**env, "DS4_MTP_TIMING": "1"},
                           stdout=af, stderr=subprocess.STDOUT)
        accept = mtp.parse_timing(accept_log)
        prefill_tps, decode_tps = _parse_tps(accept_log)

        ncu = {}
        if cfg.do_ncu:
            print("## 4.5 ncu stalls (application replay; slow)", file=sys.stderr)
            try:
                rep = M.run_ncu([ds4, *args], cfg.ncu_kernels, 200, 12, f"{tmp}_ncu")
                ncu = M.parse_ncu(rep)
            except Exception as e:
                print(f"   (ncu failed: {e})", file=sys.stderr)

        print("## 5/5 report", file=sys.stderr)
        rep = R.build(plain_sql, metrics_sqlite=metrics_sql, ptxas_text=ptxas_text,
                      ncu=ncu, accept=accept, label=cfg.label,
                      prefill_tps=prefill_tps, decode_tps=decode_tps)
        (runs / f"{cfg.label}.md").write_text(R.emit_markdown(rep))
        (runs / f"{cfg.label}.json").write_text(json.dumps(rep, indent=2))
        (runs / f"{cfg.label}.html").write_text(R.render_html(rep))

        fp = _fingerprint()
        bin_sha = _sha(ds4)
        flags = (f"mtp={int(bool(cfg.mtp))} ntok={cfg.ntok} temp={cfg.temp} "
                 f"ctx={cfg.ctx or 'def'} warm={int(cfg.warm)}"
                 + (" FAST_VERIFY=1" if cfg.fast_verify else ""))
        con = store.connect(db)
        rid = store.ingest(con, rep, source_path=str(runs / f"{cfg.label}.json"),
                           code_id=fp["code_id"], binary_sha=bin_sha, flags=flags,
                           driver=fp["driver"], model=os.path.basename(cfg.model), phase="decode")
        print(f"done → runs/{cfg.label}.{{md,json,html}} + runs.db "
              f"(id={rid} code={fp['code_id']})", file=sys.stderr)
        print(f"  prefill {prefill_tps} t/s · decode {decode_tps} t/s", file=sys.stderr)
        return rep
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
        lf.close()


def _parse_tps(path: str) -> tuple[float | None, float | None]:
    import re
    pf = dec = None
    try:
        txt = open(path).read()
    except OSError:
        return None, None
    if (m := re.search(r"prefill:\s*([\d.]+)", txt)):
        pf = float(m.group(1))
    if (m := re.search(r"generation:\s*([\d.]+)", txt)):
        dec = float(m.group(1))
    return pf, dec


def _sha(path: str) -> str:
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""
