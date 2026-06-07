#!/usr/bin/env bash
# Build a long-context KV checkpoint once, return its sha.
# Reuses ~/.ds4/kvcache via the agent's /save flow.
#
# Usage:
#   prime_kv.sh [--prompt-file PATH] --ctx N [--model GGUF] [--out-sha PATH]
#   (--prompt-file defaults to prompts/longform.txt, auto-built if missing)
#
# Layout of (model, tokenizer, kvcache_version) is hashed into the saved
# session label so a kernel change that touches KV format invalidates the
# cached prime automatically.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
source "$HERE/thermal_guard.sh"

MODEL="${MODEL:-$ROOT/ds4flash.gguf}"
PROMPT_FILE=""
CTX=65536
OUT_SHA=""
LABEL_TAG="${LABEL_TAG:-longctx}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --out-sha) OUT_SHA="$2"; shift 2 ;;
        --label-tag) LABEL_TAG="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$PROMPT_FILE" ]]; then
    PROMPT_FILE="$HERE/prompts/longform.txt"
    [[ -f "$PROMPT_FILE" ]] || "$HERE/prompts/ensure_longform.sh"
fi
[[ -f "$PROMPT_FILE" ]] || { echo "need --prompt-file" >&2; exit 2; }

# Layout fingerprint — any of these changing invalidates the prime.
# GGUF is multi-GB; use (path, size, mtime) instead of content hash.
kv_layout_sha() {
    {
        stat -c '%n %s %Y' "$MODEL" 2>/dev/null
        sha256sum "$ROOT/ds4_kvstore.h" "$ROOT/ds4_kvstore.c" 2>/dev/null
        grep -h 'kv_fp8_store\|compressor_store\|ratio4_shift\|indexer_score' \
             "$ROOT/ds4_cuda.cu" 2>/dev/null | sha256sum
    } | sha256sum | awk '{print substr($1,1,12)}'
}
LAYOUT_SHA="$(kv_layout_sha)"
SESSION_LABEL="${LABEL_TAG}-ctx${CTX}-${LAYOUT_SHA}"

KVDIR="$HOME/.ds4/kvcache"
mkdir -p "$KVDIR"

# Reuse if a <sha>.label sibling already records this label.
# Kvcache layout: $KVDIR/<sha>.kv (payload) + $KVDIR/<sha>.label (our marker).
EXISTING_SHA=""
for lbl in "$KVDIR"/*.label; do
    [[ -f "$lbl" ]] || continue
    if grep -qFx "$SESSION_LABEL" "$lbl"; then
        base="${lbl##*/}"; base="${base%.label}"
        [[ -f "$KVDIR/$base.kv" ]] && EXISTING_SHA="$base" && break
    fi
done

if [[ -n "$EXISTING_SHA" ]]; then
    echo "[prime_kv] reuse $EXISTING_SHA ($SESSION_LABEL)" >&2
    [[ -n "$OUT_SHA" ]] && printf '%s\n' "$EXISTING_SHA" > "$OUT_SHA"
    printf '%s\n' "$EXISTING_SHA"
    exit 0
fi

cooldown_wait "prime"
thermal_snapshot pre >&2

# --save-on-exit prints '+DWARFSTAR_SAVED <sha>'. Stdin path (no -p) is used so
# the multi-hundred-KB prompt doesn't blow ARG_MAX.
TRACE="$(mktemp -t prime-trace.XXXXXX)"
STDOUT_LOG="$(mktemp -t prime-stdout.XXXXXX)"
echo "[prime_kv] priming ${CTX} ctx via $PROMPT_FILE" >&2
# Observed bytes/tok: ~3.0 (Italian) to ~4.5 (English) with the DS4 tokenizer.
# Pass 3.5× CTX bytes and give 32k ctx slack so chat wrapping + tokenizer
# variance can't push us over.
head -c $((CTX * 7 / 2)) "$PROMPT_FILE" | "$ROOT/ds4-agent" --cuda \
    -m "$MODEL" \
    -c $((CTX + 32768)) \
    --warm-weights \
    --non-interactive --save-on-exit \
    --tokens 1 --nothink \
    --trace "$TRACE" \
    > "$STDOUT_LOG" 2>&1

NEW_SHA="$(awk '/^\+DWARFSTAR_SAVED / {print $2; exit}' "$STDOUT_LOG")"
if [[ -z "$NEW_SHA" ]]; then
    echo "[prime_kv] no +DWARFSTAR_SAVED marker (see $STDOUT_LOG)" >&2
    tail -20 "$STDOUT_LOG" >&2
    exit 1
fi
printf '%s\n' "$SESSION_LABEL" > "$KVDIR/$NEW_SHA.label"

thermal_snapshot post >&2
echo "[prime_kv] new $NEW_SHA ($SESSION_LABEL) trace=$TRACE" >&2
[[ -n "$OUT_SHA" ]] && printf '%s\n' "$NEW_SHA" > "$OUT_SHA"
printf '%s\n' "$NEW_SHA"
