#!/usr/bin/env bash
# Build prompts/longform.txt (gitignored) by stitching copies of the in-tree
# story prompt. ds4-bench refuses prompts shorter than --ctx-max, and the top
# sweep frontier (131072) needs ~131k tokens; the story prompt is ~30k tokens
# (140 KB), so 8 copies ≈ 1.1 MB ≈ ~240k tokens — slack for tokenizer variance
# and prime_kv.sh's head -c byte budgeting.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
SRC="$ROOT/tests/long_context_story_prompt.txt"
DST="$HERE/longform.txt"
COPIES="${COPIES:-8}"
[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
: > "$DST"
for ((i = 0; i < COPIES; i++)); do cat "$SRC" >> "$DST"; done
echo "[ensure_longform] $DST ($(wc -c < "$DST") bytes, ${COPIES}x $(basename "$SRC"))" >&2
