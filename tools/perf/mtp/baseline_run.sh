#!/usr/bin/env bash
# Run all prompt classes through ds4-agent with DS4_MTP_TIMING=1, then parse.
# Writes per-prompt JSON to tools/perf/mtp/runs/baseline-<ts>/<class>.json
#
# Usage: baseline_run.sh [--margin F] [--no-cascade] [--label TAG] [--runs N]
#
# --runs N (or RUNS=N env): repeat each class N times with a thermal cooldown
# before every run, then aggregate per-class mean±std (<class>.json carries the
# merged stats; per-run sidecars are <class>.run-K.json). Single-prompt A/Bs are
# ±2-4% binomial noise per run — use N>=3 before trusting per-class deltas.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
source "$HERE/../longctx/thermal_guard.sh"

# MTP gguf is overridable (env or --mtp) so a finetuned head can be A/B'd
# against the baseline without editing this file.
MTP="${MTP:-/home/trevor/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"
DRAFT="${DRAFT:-2}"   # MTP draft depth; finetuned head's gains live at 3-4
CTX="${CTX:-100000}"  # at 100k decode is verify-bound (tps insensitive to accept);
                      # use a small ctx (e.g. 8000) to make decode_tps reflect accept
LABEL_TAG="baseline"
MARGIN=""
NO_CASCADE=""
ADAPTIVE=""
RUNS="${RUNS:-1}"
TEMP="${TEMP:-}"      # empty = agent default (1.0, SAMPLED path). The adaptive
                      # cascade only exists on the greedy/argmax path — cascade
                      # A/Bs must pass --temp 0 or they measure nothing.
CLASSES="${CLASSES:-}"  # comma-separated subset of prompts/ classes; empty = all

while [[ $# -gt 0 ]]; do
    case "$1" in
        --margin) MARGIN="$2"; LABEL_TAG="margin${2}"; shift 2 ;;
        --no-cascade) NO_CASCADE=1; LABEL_TAG="${LABEL_TAG}-nocascade"; shift ;;
        --adaptive) ADAPTIVE=1; LABEL_TAG="${LABEL_TAG}-adaptive"; shift ;;
        --label) LABEL_TAG="$2"; shift 2 ;;
        --mtp) MTP="$2"; shift 2 ;;
        --draft) DRAFT="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --temp) TEMP="$2"; shift 2 ;;
        --classes) CLASSES="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "MTP gguf: $MTP  draft: $DRAFT" >&2

OUT_DIR="$HERE/runs/$LABEL_TAG-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

run_one() {
    local class="$1"
    local tag="$2"   # output basename: "<class>" single-run, "<class>.run-K" multi
    local prompt_file="$HERE/prompts/$class.txt"
    [[ -f "$prompt_file" ]] || { echo "no prompt: $prompt_file" >&2; return 1; }
    local prompt
    prompt="$(cat "$prompt_file")"

    cooldown_wait "$tag" >&2

    # DETERMINISTIC decode: the MoE down-proj uses atomics -> nondeterministic
    # reduction order -> tiny logit diffs compound into divergent greedy paths, so a
    # base-vs-finetuned A/B decodes DIFFERENT sequences (confounds the accept delta).
    # NO_ATOMIC_DOWN makes the base greedy sequence reproducible and head-independent,
    # so base and v2 follow the SAME path -> accept delta is pure head quality.
    local env_args=(DS4_MTP_TIMING=1 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1)
    [[ -n "$MARGIN" ]] && env_args+=("DS4_MTP_MIN_MARGIN=$MARGIN")
    [[ -n "$NO_CASCADE" ]] && env_args+=("DS4_MTP_NO_CASCADE=1")
    [[ -n "$ADAPTIVE" ]] && env_args+=("DS4_MTP_CASCADE_ADAPTIVE=1")

    local extra_args=()
    [[ -n "$TEMP" ]] && extra_args+=(--temp "$TEMP")

    env "${env_args[@]}" "$ROOT/ds4-agent" --cuda \
        -c "$CTX" --warm-weights \
        --mtp "$MTP" --mtp-draft "$DRAFT" --power 85 \
        --non-interactive --nothink \
        --tokens 600 \
        "${extra_args[@]}" \
        -p "$prompt" \
        > "$OUT_DIR/$tag.stdout" 2>"$OUT_DIR/$tag.stderr"

    grep "+DWARFSTAR_METRICS" "$OUT_DIR/$tag.stderr" \
        > "$OUT_DIR/$tag.metrics" || true

    "$HERE/parse_timing.py" "$OUT_DIR/$tag.stderr" \
        --label "$class" \
        --json "$OUT_DIR/$tag.json" \
        > "$OUT_DIR/$tag.md"

    echo "[$tag] $(cat "$OUT_DIR/$tag.metrics" 2>/dev/null | tr -d '\n')" >&2
    grep -E "accept rate|implied decode tps" "$OUT_DIR/$tag.md" 2>/dev/null | head -2 >&2 || true
}

# auto-discover prompt classes (one <class>.txt per category). Drop a new prompt
# file in prompts/ and it's A/B'd automatically — covers the 20 trained categories.
# With RUNS>1, each class runs N times (cooldown before every run, so runs are
# iid samples rather than a thermal staircase) and <class>.json becomes the
# mean±std merge; the per-run sidecars stay alongside for inspection.
for pf in "$HERE"/prompts/*.txt; do
    class="$(basename "$pf" .txt)"
    if [[ -n "$CLASSES" ]] && [[ ",$CLASSES," != *",$class,"* ]]; then
        continue
    fi
    if [[ "$RUNS" -le 1 ]]; then
        run_one "$class" "$class"
        continue
    fi
    run_jsons=()
    for ((r = 1; r <= RUNS; r++)); do
        run_one "$class" "$class.run-$r"
        run_jsons+=("$OUT_DIR/$class.run-$r.json")
    done
    "$HERE/parse_timing.py" --merge "${run_jsons[@]}" \
        --label "$class" \
        --json "$OUT_DIR/$class.json" \
        > "$OUT_DIR/$class.md"
done

echo "===== combined summary ====="
python3 - "$OUT_DIR" <<'PY'
import json, glob, re, sys
out = sys.argv[1]
rows = []
for j in sorted(glob.glob(f"{out}/*.json")):
    if re.search(r"\.run-\d+\.json$", j):   # per-run sidecars; merged file covers them
        continue
    with open(j) as f: d = json.load(f)
    rows.append(d)
def pm(r, key, scale=1.0, prec=1):
    s = (r.get("std") or {}).get(key)
    return f"±{s*scale:.{prec}f}" if s is not None else ""
print(f"| class | accept | step ms | tokens | implied tps |")
print(f"| ----- | ------:| -------:| ------:| -----------:|")
for r in rows:
    if r.get("total_steps", 0) == 0:
        print(f"| {r['label']} | (no data) |")
        continue
    print(f"| {r['label']} | {r['accept_rate']*100:.1f}%{pm(r,'accept_rate',100)} | "
          f"{r['step_time_ms']['mean']:.0f}{pm(r,'step_time_ms_mean',prec=0)} | "
          f"{r['tokens_emitted']:.0f} | "
          f"{r['implied_decode_tps']:.2f}{pm(r,'implied_decode_tps',prec=2)} |")
PY
echo ""
echo "outputs in $OUT_DIR"
