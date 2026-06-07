#!/usr/bin/env bash
# Profile a one-shot ds4-agent invocation under nsys, then aggregate
# kernel timing + MTP acceptance + agent metrics into a single report dir.
#
# Output layout (under tools/perf/runs/agent-profile-<label>-<ts>/):
#   agent.stdout         model output
#   agent.stderr         agent stderr (init + metrics + mtp timing)
#   agent.metrics        the single +DWARFSTAR_METRICS line
#   agent.mtp.txt        ds4: mtp timing lines (input to parse_timing.py)
#   plain.nsys-rep       raw nsys trace (-t cuda)
#   plain.sqlite         auto-exported
#   top-kernels.csv      nsys stats --report cuda_gpu_kern_sum
#   mtp-summary.md       parse_timing.py output
#   gamut.md / .json     optional gamut.py output (if --gamut)
#   report.md            combined headline
#
# Usage:
#   profile_run.sh --prompt-file PROMPT.txt [options]
#   profile_run.sh --prompt "Write a paragraph about X."  [options]
#
# Options:
#   --label TAG          dir slug (default: timestamp)
#   --ctx N              -c (default: 100000)
#   --tokens N           --tokens (default: 400)
#   --temp F             default: 1
#   --power N            default: 85
#   --no-mtp             skip --mtp loading (plain decode)
#   --no-cooldown        skip cooldown gate
#   --gamut              also run gamut.py (needs the gb20b metrics capture too)
#   --gb20b              add gpu-metrics-set=gb20b capture (perturbs timing)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
source "$HERE/../longctx/thermal_guard.sh"

MTP_GGUF="${MTP_GGUF:-/home/trevor/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"
LABEL=""
CTX=100000
N_TOKENS=400
TEMP=1
POWER=85
USE_MTP=1
COOLDOWN=1
RUN_GAMUT=0
RUN_GB20B=0
PROMPT=""
PROMPT_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --tokens) N_TOKENS="$2"; shift 2 ;;
        --temp) TEMP="$2"; shift 2 ;;
        --power) POWER="$2"; shift 2 ;;
        --no-mtp) USE_MTP=0; shift ;;
        --no-cooldown) COOLDOWN=0; shift ;;
        --gamut) RUN_GAMUT=1; RUN_GB20B=1; shift ;;
        --gb20b) RUN_GB20B=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$PROMPT" && -z "$PROMPT_FILE" ]]; then
    echo "need --prompt TEXT or --prompt-file PATH" >&2; exit 2
fi
if [[ -n "$PROMPT_FILE" && -f "$PROMPT_FILE" ]]; then
    PROMPT="$(cat "$PROMPT_FILE")"
fi
if [[ -z "$LABEL" ]]; then
    LABEL="$(date +%Y%m%d-%H%M%S)"
fi

OUT="$ROOT/tools/perf/runs/agent-profile-$LABEL"
mkdir -p "$OUT"

# Build agent argv.
AGENT_ARGS=(--cuda -c "$CTX" --warm-weights --power "$POWER"
            --non-interactive --nothink --tokens "$N_TOKENS" --temp "$TEMP")
if [[ "$USE_MTP" == 1 ]]; then
    AGENT_ARGS+=(--mtp "$MTP_GGUF" --mtp-draft 2)
fi
AGENT_ARGS+=(-p "$PROMPT")

# --- pass 1: clean kernel timing (no gb20b perturbation) ---
if [[ "$COOLDOWN" == 1 ]]; then cooldown_wait "$LABEL-plain" >&2; fi
echo "[profile_run] capturing plain trace..." >&2
DS4_MTP_TIMING=1 nsys profile \
    -o "$OUT/plain" -t cuda --sample none --force-overwrite true \
    "$ROOT/ds4-agent" "${AGENT_ARGS[@]}" \
    > "$OUT/agent.stdout" 2> "$OUT/agent.stderr"

# Pull metrics & mtp timing out.
grep "+DWARFSTAR_METRICS" "$OUT/agent.stderr" > "$OUT/agent.metrics" || true
grep "^ds4: mtp timing" "$OUT/agent.stderr" > "$OUT/agent.mtp.txt" || true

# Export sqlite + top-N kernel CSV.  nsys leaks "Generating SQLite file..."
# and "Processing..." lines onto stdout — strip to CSV-only rows.
nsys stats --report cuda_gpu_kern_sum --format csv --output - "$OUT/plain.nsys-rep" 2>/dev/null \
    | awk 'NR==1 || /^[0-9]/' > "$OUT/top-kernels.csv"

# Parse MTP timing into a summary.
"$ROOT/tools/perf/mtp/parse_timing.py" "$OUT/agent.mtp.txt" \
    --label "$LABEL" --json "$OUT/mtp-summary.json" > "$OUT/mtp-summary.md" \
    2>/dev/null || true

# --- optional pass 2: gb20b GPU metrics ---
if [[ "$RUN_GB20B" == 1 ]]; then
    if [[ "$COOLDOWN" == 1 ]]; then cooldown_wait "$LABEL-gb20b" >&2; fi
    echo "[profile_run] capturing gb20b metrics..." >&2
    DS4_MTP_TIMING=1 nsys profile \
        -o "$OUT/metrics" \
        --gpu-metrics-devices=0 --gpu-metrics-set=gb20b \
        --gpu-metrics-frequency=20000 --force-overwrite true \
        "$ROOT/ds4-agent" "${AGENT_ARGS[@]}" \
        > "$OUT/agent-metrics.stdout" 2> "$OUT/agent-metrics.stderr"
fi

# --- optional pass 3: gamut.py joined report ---
if [[ "$RUN_GAMUT" == 1 ]]; then
    PTXAS="$OUT/ptxas.txt"
    /usr/local/cuda/bin/nvcc -O3 --use_fast_math \
        -gencode=arch=compute_121a,code=sm_121a -Xptxas=-v \
        -c -o /tmp/ds4_cuda.ptxas.o "$ROOT/ds4_cuda.cu" 2>"$PTXAS" || true
    # Pull prefill_tps / decode_tps / kv from agent.metrics.
    read -r PREFILL_TPS DECODE_TPS PREFILL_S < <(awk '
        {
            for (i=1; i<=NF; i++) {
                split($i, kv, "=")
                if (kv[1] == "decode_tps") d = kv[2]
                if (kv[1] == "prefill_s") ps = kv[2]
                if (kv[1] == "prefill_tokens") pt = kv[2]
            }
            print (ps>0 ? pt/ps : 0), d+0, ps+0
        }' "$OUT/agent.metrics")
    "$ROOT/tools/perf/gamut.py" \
        --plain "$OUT/plain.sqlite" --metrics "$OUT/metrics.sqlite" \
        --ptxas "$PTXAS" \
        --accept "$OUT/agent.mtp.txt" \
        --prefill-tps "$PREFILL_TPS" --decode-tps "$DECODE_TPS" --kvcache-mb 0 \
        --label "agent-$LABEL" --json "$OUT/gamut.json" \
        > "$OUT/gamut.md" 2>"$OUT/gamut.err" || true
fi

# --- headline report.md ---
{
    echo "# agent profile: $LABEL"
    echo
    echo "**Metrics:**"
    echo
    if [[ -s "$OUT/agent.metrics" ]]; then
        sed -e 's/^+DWARFSTAR_METRICS //' "$OUT/agent.metrics" | tr ' ' '\n' | sed 's/^/    /'
    else
        echo "    (no metrics — agent did not exit cleanly)"
    fi
    echo
    if [[ -s "$OUT/mtp-summary.md" ]]; then
        echo "## MTP acceptance"
        echo
        sed -n '/^- /p; /^| /p' "$OUT/mtp-summary.md" | head -20
        echo
    fi
    echo "## Top-12 kernels by time"
    echo
    echo "| % | total_ms | instances | avg_us | name |"
    echo "| -:| -------:| ---------:| ------:| ---- |"
    awk -F, 'NR>=2 && NR<=13 {
        gsub(/"/, "", $9)
        n = $9
        sub(/\(.*/, "", n)
        printf "| %s | %.1f | %d | %.0f | `%s` |\n", $1, $2/1e6, $3, $4/1e3, n
    }' "$OUT/top-kernels.csv"
    echo
    echo "## Outputs"
    echo "- \`agent.stdout\` model output"
    echo "- \`agent.stderr\` full stderr including init"
    echo "- \`plain.nsys-rep\` nsys timeline (open in nsys-ui)"
    echo "- \`plain.sqlite\` exported db (input to gamut.py / gpu_metrics.py)"
    echo "- \`top-kernels.csv\` ranked kernel time"
    echo "- \`mtp-summary.md\` accept rate + committed distribution"
    [[ "$RUN_GAMUT" == 1 ]] && echo "- \`gamut.md\` joined kernel + metrics + roofline report"
} > "$OUT/report.md"

echo "[profile_run] report at $OUT/report.md" >&2
echo "$OUT/report.md"
