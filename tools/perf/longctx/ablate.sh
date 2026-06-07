#!/usr/bin/env bash
# Ablation matrix at one anchor ctx using a primed KV.
# For each DS4_CUDA_NO_* toggle: /switch <sha>, decode N tokens, record gen_tps.
# Cooldown between every iter.
#
# Usage:
#   ablate.sh --kv-sha SHA [--gen-tokens 256] [--mtp PATH] [--out DIR]

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
source "$HERE/thermal_guard.sh"

MODEL="${MODEL:-$ROOT/ds4flash.gguf}"
KV_SHA=""
GEN_TOKENS=256
MTP_GGUF=""
OUT_DIR=""
NSYS_ENABLE="${NSYS_ENABLE:-0}"  # nsys is slow; opt-in per toggle.

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kv-sha) KV_SHA="$2"; shift 2 ;;
        --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
        --mtp) MTP_GGUF="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --nsys) NSYS_ENABLE=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$KV_SHA" ]] || { echo "need --kv-sha (see prime_kv.sh)" >&2; exit 2; }
OUT_DIR="${OUT_DIR:-tools/perf/runs/ablate-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

# Each line: "<TOGGLE_VAR>=<VAL>" or "BASELINE=" for the no-toggle baseline.
TOGGLES=(
    "BASELINE="
    "DS4_CUDA_NO_INDEXER_WMMA=1"
    "DS4_CUDA_NO_INDEXER_DIRECT_ONE=1"
    "DS4_CUDA_NO_TOPK_CHUNKED=1"
    "DS4_CUDA_NO_TOPK=1"
    "DS4_CUDA_NO_Q8_DP4A=1"
    "DS4_CUDA_NO_HBM_CACHE=1"
    "DS4_CUDA_NO_TF=1"
)

MTP_ARGS=()
MTP_TAG="nomtp"
if [[ -n "$MTP_GGUF" ]]; then
    MTP_ARGS=(--mtp "$MTP_GGUF" --mtp-draft 2)
    MTP_TAG="mtp"
fi

KV_FILE="$HOME/.ds4/kvcache/$KV_SHA.kv"
[[ -f "$KV_FILE" ]] || { echo "no such KV file: $KV_FILE" >&2; exit 2; }

run_one() {
    local toggle="$1"
    local label="${toggle%%=*}"; label="${label:-baseline}"
    local slug="${MTP_TAG}-${label,,}"
    local csv="$OUT_DIR/$slug.csv"
    local stdout="$OUT_DIR/$slug.stdout"
    local thermal="$OUT_DIR/$slug.thermal.json"

    cooldown_wait "$slug"
    thermal_snapshot pre > "$thermal"

    local env_prefix=()
    [[ "$toggle" == BASELINE=* ]] || env_prefix=(env "$toggle")

    local nsys_cmd=()
    if [[ "$NSYS_ENABLE" == 1 ]]; then
        nsys_cmd=(nsys profile -o "$OUT_DIR/$slug" -t cuda --sample none \
                  --force-overwrite true)
    fi

    # ds4-bench --kv-restore measures the same code path as sweep.sh.
    # CSV row: ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes
    "${env_prefix[@]}" "${nsys_cmd[@]}" \
        "$ROOT/ds4-bench" --cuda --warm-weights \
        -m "$MODEL" \
        "${MTP_ARGS[@]}" \
        --kv-restore "$KV_FILE" \
        --ctx-alloc 200000 \
        --gen-tokens "$GEN_TOKENS" --temp 0 \
        --csv "$csv" > "$stdout" 2>&1 || true

    thermal_snapshot post >> "$thermal"

    awk -F, -v label="$slug" '
        NR == 2 {
            printf "{\"label\":\"%s\",\"ctx\":%s,\"tokens\":%s,\"tps\":%s}\n",
                   label, $1, $4, $5
        }
    ' "$csv" >> "$OUT_DIR/results.jsonl"

    echo "[ablate] $slug done ($(tail -1 "$OUT_DIR/results.jsonl"))" >&2
}

: > "$OUT_DIR/results.jsonl"
for t in "${TOGGLES[@]}"; do run_one "$t"; done

echo "[ablate] results in $OUT_DIR/results.jsonl" >&2
cat "$OUT_DIR/results.jsonl"
