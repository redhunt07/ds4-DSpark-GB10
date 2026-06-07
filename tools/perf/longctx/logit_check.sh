#!/usr/bin/env bash
# Logit-equality smoke for the MoE small-batch rowspan path.
# Runs ds4-bench twice against the same primed KV: once with the new
# rowspan<128> default, once with the opt-out fallback to row32. Dumps
# vocab logits via --dump-frontier-logits-dir, then compares.
#
# Pass:  max-abs diff < 1e-3 AND top-1 token matches between paths.
# Fail:  either threshold blown — there's a logit-drift bug in row128.
#
# Usage: logit_check.sh [--kv-sha SHA]
#   default SHA is the 9k anchor we've been using

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

KV_SHA="${1:-b9dbb307b5f4150cf3b1925c92441a015734989c}"
[[ "${1:-}" == "--kv-sha" ]] && KV_SHA="$2"

KV="$HOME/.ds4/kvcache/$KV_SHA.kv"
[[ -f "$KV" ]] || { echo "no KV file: $KV" >&2; exit 2; }

OUT_DIR="$ROOT/tools/perf/runs/logit-check-$(date +%Y%m%d-%H%M%S)"
DUMP_R32="$OUT_DIR/dump-row32"
DUMP_R128="$OUT_DIR/dump-row128"
mkdir -p "$DUMP_R32" "$DUMP_R128"

run_one() {
    local label="$1" dump="$2"; shift 2
    echo "[logit_check] $label" >&2
    "$@" "$ROOT/ds4-bench" --cuda \
        -m "$ROOT/ds4flash.gguf" \
        --kv-restore "$KV" --ctx-alloc 200000 \
        --gen-tokens 1 --temp 0 \
        --dump-frontier-logits-dir "$dump" \
        --csv "$OUT_DIR/$label.csv" \
        > "$OUT_DIR/$label.stdout" 2>&1
}

# Baseline = row32 (opt-out forces fallback)
run_one row32  "$DUMP_R32"  env DS4_CUDA_MOE_NO_SMALL_ROW128=1
# Candidate = row128 (current default)
run_one row128 "$DUMP_R128"

python3 - "$DUMP_R32" "$DUMP_R128" <<'PY'
import glob, json, sys, math
def load_one(d):
    p = sorted(glob.glob(f"{d}/frontier_*.logits.json"))
    if not p:
        sys.exit(f"no logits JSON in {d}")
    with open(p[-1]) as f:
        j = json.load(f)
    return j["logits"], j["argmax_id"], j["argmax_logit"]

a, a_arg, a_lg = load_one(sys.argv[1])
b, b_arg, b_lg = load_one(sys.argv[2])
n = min(len(a), len(b))
diffs = []
for i in range(n):
    if a[i] is None or b[i] is None:
        continue
    diffs.append(abs(a[i] - b[i]))
mx = max(diffs); mn = sum(diffs)/len(diffs)
def topk(L, k):
    return sorted(range(len(L)), key=lambda i: -L[i] if L[i] is not None else float("inf"))[:k]
tk_a = topk(a, 10); tk_b = topk(b, 10)
top1_match = a_arg == b_arg
top10_overlap = len(set(tk_a) & set(tk_b))

print(f"vocab    : {n}")
print(f"max abs  : {mx:.6g}")
print(f"mean abs : {mn:.6g}")
print(f"argmax   : row32={a_arg} ({a_lg:.4g})  row128={b_arg} ({b_lg:.4g})  match={top1_match}")
print(f"top-10   : overlap {top10_overlap}/10")
print()
PASS_MAX = 1e-3
ok = (mx < PASS_MAX) and top1_match and (top10_overlap >= 9)
if ok:
    print(f"PASS  — max-abs < {PASS_MAX} and argmax matches and top-10 overlap >= 9")
    sys.exit(0)
else:
    print(f"FAIL  — drift exceeds threshold or argmax mismatches")
    print("  reasons:")
    if mx >= PASS_MAX: print(f"    max-abs {mx:.6g} >= {PASS_MAX}")
    if not top1_match:  print(f"    argmax differs: {a_arg} vs {b_arg}")
    if top10_overlap < 9: print(f"    top-10 overlap {top10_overlap}/10")
    sys.exit(1)
PY
