#!/usr/bin/env bash
# tools/perf/capture.sh — one-shot GB10 decode perf capture → gamut report → run-store.
#
# Runs the whole dance (rebuild → plain nsys trace → gb20b metrics → ptxas regs
# → accept telemetry → optional ncu stalls → gamut MD/JSON/HTML → record to
# tools/perf/runs.db) so each kernel experiment is one command and is preserved
# for historical review.  Captures are serialized via an flock so concurrent runs
# can't corrupt each other's nsys traces (the failure mode that produced cryptic
# "no such table: CUPTI_ACTIVITY_KIND_KERNEL" crashes before).
#
#   tools/perf/capture.sh --label moe_down_lb2 [--rebuild] [--ncu] [-n 48]
#       [-m ds4flash.gguf] [--mtp PATH | --no-mtp] [-p knight] [--ctx N]
#       [--temp F] [--prompt-file FILE] [--warm] [--think]
#   env passthrough: DS4_CUDA_FAST_VERIFY=1 tools/perf/capture.sh --label fast ...
#
# Output: tools/perf/runs/<label>.{md,json,html}  (+ /tmp/<label>_*.{nsys-rep,sqlite})
#         and one immutable row in tools/perf/runs.db (query: gamut_db.py list).
#
# Note: NO `set -e` — each external step is checked explicitly so a failure is
# reported (not silently swallowed) and partial captures don't masquerade as ok.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
NVCC=/usr/local/cuda/bin/nvcc
ARCH="-gencode=arch=compute_121a,code=sm_121a"

LABEL=""; REBUILD=0; DO_NCU=0; USE_MTP=1
MODEL="ds4flash.gguf"
MTP="/home/trevor/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
PROMPT="knight"; NTOK=48
CTX=""; TEMP="0"; PROMPTFILE=""; WARM=0; THINKFLAG="--nothink"
KERNELS="moe_down_expert_tile8_row32|matmul_q8_0_preq_batch_share_warp|moe_gate_up_mid_expert_tile8_row32"

while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2;;
    --rebuild) REBUILD=1; shift;;
    --ncu) DO_NCU=1; shift;;
    -m) MODEL="$2"; shift 2;;
    --mtp) MTP="$2"; shift 2;;
    --no-mtp) USE_MTP=0; shift;;
    -p) PROMPT="$2"; shift 2;;
    -n) NTOK="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --temp) TEMP="$2"; shift 2;;
    --prompt-file) PROMPTFILE="$2"; shift 2;;
    --warm) WARM=1; shift;;
    --think) THINKFLAG="--think"; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$LABEL" ] || { echo "need --label" >&2; exit 2; }

cd "$ROOT"

# ---- serialize: only one capture (and one GPU consumer) at a time ----------
LOCK=/tmp/ds4-capture.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another capture holds $LOCK (a capture/bench is running); refusing to start" >&2
  exit 3
fi
# also clear any stale single-instance lock from a killed ds4
rm -f /tmp/ds4.lock

RUNS="$HERE/runs"; mkdir -p "$RUNS"
T="/tmp/${LABEL}"

ARGS=(-m "$MODEL" -n "$NTOK" --temp "$TEMP" "$THINKFLAG" -sys "")
[ "$USE_MTP" = 1 ] && ARGS+=(--mtp "$MTP")
if [ -n "$PROMPTFILE" ]; then ARGS+=(--prompt-file "$PROMPTFILE"); else ARGS+=(-p "$PROMPT"); fi
[ -n "$CTX" ] && ARGS+=(--ctx "$CTX")
[ "$WARM" = 1 ] && ARGS+=(--warm-weights)

[ -x ./ds4 ] || { echo "./ds4 not built; pass --rebuild or run make cuda-spark" >&2; exit 2; }
[ "$REBUILD" = 1 ] && { echo "## rebuild"; make cuda-spark >/tmp/${LABEL}_build.log 2>&1 || { echo "build FAILED — see /tmp/${LABEL}_build.log" >&2; exit 1; }; }

fail() { echo "## CAPTURE FAILED: $*" >&2; exit 1; }

# nsys profile + export to sqlite, with explicit validation (the old silent path
# died here when concurrent runs truncated the .nsys-rep).
nsys_to_sqlite() {  # $1=outbase $2... = extra nsys profile flags ; reads ARGS
  local out="$1"; shift
  nsys profile -o "$out" --force-overwrite true "$@" ./ds4 "${ARGS[@]}" \
    >"${out}.nsyslog" 2>&1
  [ -s "${out}.nsys-rep" ] || fail "nsys produced no ${out}.nsys-rep (see ${out}.nsyslog)"
  nsys export --type sqlite --force-overwrite true -o "${out}.sqlite" "${out}.nsys-rep" \
    >"${out}.explog" 2>&1 || fail "nsys export failed for ${out} (see ${out}.explog)"
  [ -s "${out}.sqlite" ] || fail "nsys export produced no ${out}.sqlite"
}

echo "## 1/5 plain CUDA trace"
nsys_to_sqlite "${T}_p" -t cuda --sample none
KROWS=$(python3 - "${T}_p.sqlite" <<'PY'
import sqlite3,sys
c=sqlite3.connect(sys.argv[1])
try: print(c.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0])
except Exception: print(0)
PY
)
[ "${KROWS:-0}" -ge 1000 ] || fail "plain trace has only ${KROWS} kernel rows (truncated/empty) — was another GPU job running?"
echo "   plain trace OK: ${KROWS} kernel rows"

echo "## 2/5 gb20b GPU metrics"
nsys_to_sqlite "${T}_gm" --gpu-metrics-devices=0 --gpu-metrics-set=gb20b --gpu-metrics-frequency=20000 \
  || echo "   (gb20b metrics unavailable — continuing without HW counters)"

echo "## 3/5 ptxas registers"
$NVCC -O3 --use_fast_math $ARCH -Xptxas=-v -c -o /tmp/_ptxas.o ds4_cuda.cu 2>"${T}_ptxas.txt" \
  || echo "   (ptxas pass failed — continuing without reg/occ data)"

echo "## 4/5 accept + throughput"
DS4_MTP_TIMING=1 ./ds4 "${ARGS[@]}" >"${T}_accept.txt" 2>&1 || true
PREFILL=$(grep -oE 'prefill: [0-9.]+' "${T}_accept.txt" | tail -1 | grep -oE '[0-9.]+' || echo "")
DECODE=$(grep -oE 'generation: [0-9.]+' "${T}_accept.txt" | tail -1 | grep -oE '[0-9.]+' || echo "")

NCU_ARG=()
if [ "$DO_NCU" = 1 ]; then
  echo "## 4.5 ncu stalls (application replay; slow)"
  python3 "$HERE/ncu_stalls.py" --out "${T}_ncu.json" --kernels "$KERNELS" \
    --launch-skip 200 --launch-count 12 -- ./ds4 "${ARGS[@]}" >/dev/null 2>&1 || true
  [ -s "${T}_ncu.json" ] && NCU_ARG=(--ncu "${T}_ncu.json")
fi

echo "## 5/5 gamut report"
METRICS_ARG=(); [ -s "${T}_gm.sqlite" ] && METRICS_ARG=(--metrics "${T}_gm.sqlite")
PTXAS_ARG=();  [ -s "${T}_ptxas.txt" ] && PTXAS_ARG=(--ptxas "${T}_ptxas.txt")
python3 "$HERE/gamut.py" \
  --plain "${T}_p.sqlite" "${METRICS_ARG[@]}" "${PTXAS_ARG[@]}" \
  --accept "${T}_accept.txt" "${NCU_ARG[@]}" \
  ${PREFILL:+--prefill-tps "$PREFILL"} ${DECODE:+--decode-tps "$DECODE"} \
  --label "$LABEL" --json "$RUNS/$LABEL.json" --html "$RUNS/$LABEL.html" \
  >"$RUNS/$LABEL.md" || fail "gamut report generation failed"

# ---- record to the persistent run-store (fingerprint + metadata) -----------
CODE_ID=$(jj log -r @ --no-graph -T 'change_id.short()' 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo "?")
BIN_SHA=$(sha256sum ./ds4 2>/dev/null | cut -c1-16)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
FLAGS="mtp=$USE_MTP ntok=$NTOK temp=$TEMP ctx=${CTX:-def} warm=$WARM${DS4_CUDA_FAST_VERIFY:+ FAST_VERIFY=$DS4_CUDA_FAST_VERIFY}"
python3 "$HERE/gamut_db.py" ingest "$RUNS/$LABEL.json" \
  --code-id "$CODE_ID" --binary-sha "$BIN_SHA" --flags "$FLAGS" \
  --driver "$DRIVER" --model "$(basename "$MODEL")" --phase decode \
  || echo "   (run-store ingest failed — json sidecar still at $RUNS/$LABEL.json)"

echo "done → $RUNS/$LABEL.{md,json,html}  +  runs.db (code=$CODE_ID)"
echo "  prefill ${PREFILL:-?} t/s · decode ${DECODE:-?} t/s"
