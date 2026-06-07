#!/usr/bin/env bash
# Outer ctx sweep via ds4-bench across 6 frontiers, MTP-on and MTP-off.
# nsys + gb20b gpu-metrics capture per frontier. Cooldown between frontiers.
#
# Usage:
#   sweep.sh [--prompt-file PATH] [--out DIR] [--frontiers "8192,..."]
#
# --prompt-file defaults to prompts/longform.txt, auto-built (stitched copies of
# tests/long_context_story_prompt.txt) if missing — ds4-bench refuses prompts
# shorter than --ctx-max, and the top frontier needs ~131k tokens.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
source "$HERE/thermal_guard.sh"

MODEL="${MODEL:-$ROOT/ds4flash.gguf}"
MTP_GGUF="${MTP_GGUF:-/home/trevor/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"
PROMPT_FILE=""
OUT_DIR=""
FRONTIERS="8192,16384,32768,65536,98304,131072"
GEN_TOKENS=128
NSYS_METRICS="${NSYS_METRICS:-1}"  # gb20b capture is slow; opt-out if needed.

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --frontiers) FRONTIERS="$2"; shift 2 ;;
        --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$PROMPT_FILE" ]]; then
    PROMPT_FILE="$HERE/prompts/longform.txt"
    [[ -f "$PROMPT_FILE" ]] || "$HERE/prompts/ensure_longform.sh"
fi
[[ -f "$PROMPT_FILE" ]] || { echo "need --prompt-file" >&2; exit 2; }
OUT_DIR="${OUT_DIR:-tools/perf/runs/sweep-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

IFS=',' read -ra FRONTIER_LIST <<<"$FRONTIERS"
CTX_MAX=${FRONTIER_LIST[-1]}
CTX_ALLOC=$((CTX_MAX + GEN_TOKENS + 4096))

run_one_frontier() {
    local mtp_tag="$1" ctx="$2"
    local slug="${mtp_tag}-ctx${ctx}"
    local mtp_args=()
    [[ "$mtp_tag" == mtp ]] && mtp_args=(--mtp "$MTP_GGUF" --mtp-draft 2)

    cooldown_wait "$slug"
    thermal_snapshot pre > "$OUT_DIR/$slug.thermal.json"

    # nsys plain CUDA trace (clean kernel time).
    nsys profile -o "$OUT_DIR/$slug.plain" -t cuda --sample none \
                 --force-overwrite true \
        "$ROOT/ds4-bench" -m "$MODEL" \
            --cuda --warm-weights \
            --prompt-file "$PROMPT_FILE" \
            --ctx-start "$ctx" --ctx-max "$ctx" \
            --ctx-alloc "$CTX_ALLOC" \
            --gen-tokens "$GEN_TOKENS" \
            --csv "$OUT_DIR/$slug.csv" \
            "${mtp_args[@]}" \
            >"$OUT_DIR/$slug.stdout" 2>&1 || true

    # gb20b GPU metrics (separate run; the two captures are joined by slug in gamut).
    if [[ "$NSYS_METRICS" == 1 ]]; then
        cooldown_wait "$slug-metrics"
        nsys profile -o "$OUT_DIR/$slug.metrics" \
                     --gpu-metrics-devices=0 --gpu-metrics-set=gb20b \
                     --gpu-metrics-frequency=20000 \
                     --force-overwrite true \
            "$ROOT/ds4-bench" -m "$MODEL" \
                --cuda --warm-weights \
                --prompt-file "$PROMPT_FILE" \
                --ctx-start "$ctx" --ctx-max "$ctx" \
                --ctx-alloc "$CTX_ALLOC" \
                --gen-tokens "$GEN_TOKENS" \
                "${mtp_args[@]}" \
                >"$OUT_DIR/$slug.metrics.stdout" 2>&1 || true
    fi

    thermal_snapshot post >> "$OUT_DIR/$slug.thermal.json"
    echo "[sweep] $slug done" >&2
}

for mtp_tag in nomtp mtp; do
    for ctx in "${FRONTIER_LIST[@]}"; do
        run_one_frontier "$mtp_tag" "$ctx"
    done
done

# Per-frontier gamut report: ingest the captured artifacts, emit JSON sidecar.
# (Roofline/ptxas re-used across frontiers — captured once outside the loop.)
PTXAS="$OUT_DIR/ptxas.txt"
if [[ ! -f "$PTXAS" ]]; then
    /usr/local/cuda/bin/nvcc -O3 --use_fast_math \
        -gencode=arch=compute_121a,code=sm_121a -Xptxas=-v \
        -c -o /tmp/ds4_cuda.ptxas.o "$ROOT/ds4_cuda.cu" 2>"$PTXAS" || true
fi

for mtp_tag in nomtp mtp; do
    for ctx in "${FRONTIER_LIST[@]}"; do
        slug="${mtp_tag}-ctx${ctx}"
        csv="$OUT_DIR/$slug.csv"
        [[ -f "$csv" ]] || continue
        # nsys profile only writes .nsys-rep; the sqlite gamut reads is a
        # separate export step (same dance as gamut/capture.py).
        for kind in plain metrics; do
            rep="$OUT_DIR/$slug.$kind.nsys-rep"
            sql="$OUT_DIR/$slug.$kind.sqlite"
            [[ -f "$rep" && ! -f "$sql" ]] && \
                nsys export --type sqlite --force-overwrite true \
                     -o "$sql" "$rep" >/dev/null 2>&1 || true
        done
        [[ -f "$OUT_DIR/$slug.plain.sqlite" ]] || { echo "[sweep] no sqlite for $slug" >&2; continue; }
        metrics_args=()
        [[ -f "$OUT_DIR/$slug.metrics.sqlite" ]] && \
            metrics_args=(--metrics "$OUT_DIR/$slug.metrics.sqlite")
        # ds4_bench.c csv header — ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes
        read -r PREFILL DECODE KVBYTES < <(awk -F, 'NR==2 {print $3, $5, $6}' "$csv")
        KVMB=$(awk -v b="${KVBYTES:-0}" 'BEGIN {printf "%.1f", b/1048576}')
        "$HERE/../gamut-cli" report "$OUT_DIR/$slug.plain.sqlite" \
            "${metrics_args[@]}" \
            --ptxas "$PTXAS" \
            --prefill-tps "${PREFILL:-0}" --decode-tps "${DECODE:-0}" --kvcache-mb "$KVMB" \
            --label "$slug @ ctx=$ctx" \
            --json "$OUT_DIR/$slug.gamut.json" \
            > "$OUT_DIR/$slug.gamut.md" 2>"$OUT_DIR/$slug.gamut.err" || true
    done
done

echo "[sweep] outputs in $OUT_DIR" >&2
