#!/usr/bin/env bash
# Run ds4-bench, ds4 CLI, ds4-server, ds4-agent at the validated GB10
# chat-perf flag set. Same prompt across all four. Captures each binary's
# tps reporting and aggregates into one summary table for the README.
#
# Flag set (validated 2026-05-28 session, see README "Recommended GB10 settings"):
#   --cuda --warm-weights --power 85 --mtp <gguf> --mtp-draft 2
#   not --quality (-37% decode tps; only useful for cross-backend numerical-drift debug)
#   not --think-max (longer time-to-final-answer on routine prompts)
#
# Output: tools/perf/runs/all-binary-bench-<ts>/

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/longctx/thermal_guard.sh"

MTP=/home/trevor/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
KV9K="$HOME/.ds4/kvcache/b9dbb307b5f4150cf3b1925c92441a015734989c.kv"

OUT="$ROOT/tools/perf/runs/all-binary-bench-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
PROMPT='Write a 300-word essay about lighthouses. Do not use any tools — write the essay directly. Keep going for the full 300 words.'

#### 1. ds4-bench ##############################################################
cooldown_wait "ds4-bench" >&2
echo "[all-bench] ds4-bench (kernel-only, KV-restored)" >&2
"$ROOT/ds4-bench" --cuda --warm-weights \
    -m ds4flash.gguf --mtp "$MTP" --mtp-draft 2 --power 85 \
    --kv-restore "$KV9K" --ctx-alloc 200000 \
    --gen-tokens 256 --temp 1 \
    --csv "$OUT/ds4-bench.csv" > "$OUT/ds4-bench.stdout" 2> "$OUT/ds4-bench.stderr"
BENCH_TPS=$(awk -F, 'NR==2 {print $5}' "$OUT/ds4-bench.csv")
echo "  gen_tps = $BENCH_TPS" >&2

#### 2. ds4 (CLI) — one-shot prompt ############################################
cooldown_wait "ds4-cli" >&2
echo "[all-bench] ds4 (CLI) one-shot" >&2
"$ROOT/ds4" --cuda --warm-weights \
    -m ds4flash.gguf --mtp "$MTP" --mtp-draft 2 --power 85 \
    --nothink --tokens 300 \
    -p "$PROMPT" > "$OUT/ds4-cli.stdout" 2> "$OUT/ds4-cli.stderr"
# CLI prints "ds4: prefill: X t/s, generation: Y t/s" at end
CLI_PREFILL=$(grep -oE "prefill: [0-9.]+" "$OUT/ds4-cli.stderr" | awk '{print $2}' | head -1)
CLI_DECODE=$(grep -oE "generation: [0-9.]+" "$OUT/ds4-cli.stderr" | awk '{print $2}' | head -1)
echo "  prefill=$CLI_PREFILL  decode=$CLI_DECODE" >&2

#### 3. ds4-server — single HTTP request #######################################
cooldown_wait "ds4-server" >&2
echo "[all-bench] ds4-server one HTTP request" >&2
PORT=8765
"$ROOT/ds4-server" --cuda --warm-weights \
    -m ds4flash.gguf --mtp "$MTP" --mtp-draft 2 --power 85 \
    --port "$PORT" > "$OUT/ds4-server.stdout" 2> "$OUT/ds4-server.stderr" &
SERVER_PID=$!
# Wait for server ready (poll on /v1/models or a probe)
for i in {1..120}; do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "200"; then
        break
    fi
    sleep 1
done
echo "  server ready after ~${i}s" >&2

# Single chat-completions request, non-streaming, time the body
START=$(date +%s.%N)
curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"deepseek-chat\",\"max_tokens\":300,\"temperature\":1.0,\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$PROMPT")}]}" \
    > "$OUT/ds4-server.response.json" 2>"$OUT/ds4-server.curl.stderr"
END=$(date +%s.%N)
ELAPSED=$(python3 -c "print($END - $START)")

# Pull avg t/s from server stderr (the server logs per-chunk and avg)
SERVER_AVG=$(grep -oE "avg=[0-9.]+ t/s" "$OUT/ds4-server.stderr" | tail -1 | awk -F= '{print $2}' | awk '{print $1}')
# Also count completion tokens from response and compute tps
COMPLETION_TOKENS=$(python3 -c "import json; d=json.load(open('$OUT/ds4-server.response.json')); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
SERVER_TPS=$(python3 -c "print($COMPLETION_TOKENS / $ELAPSED if $COMPLETION_TOKENS and $ELAPSED > 0 else 0)" 2>/dev/null || echo 0)
echo "  server avg=$SERVER_AVG t/s  measured ${COMPLETION_TOKENS}tok/${ELAPSED}s = $SERVER_TPS t/s" >&2

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

#### 4. ds4-agent — non-interactive one-shot ###################################
cooldown_wait "ds4-agent" >&2
echo "[all-bench] ds4-agent one-shot" >&2
"$ROOT/ds4-agent" --cuda -c 100000 --warm-weights \
    --mtp "$MTP" --mtp-draft 2 --power 85 \
    --non-interactive --nothink --tokens 300 \
    -p "$PROMPT" > "$OUT/ds4-agent.stdout" 2> "$OUT/ds4-agent.stderr"
AGENT_METRICS=$(grep "+DWARFSTAR_METRICS" "$OUT/ds4-agent.stderr")
AGENT_DECODE=$(echo "$AGENT_METRICS" | awk '{for (i=1; i<=NF; i++) if ($i ~ /^decode_tps=/) print substr($i, 12)}')
AGENT_AVG=$(echo "$AGENT_METRICS" | awk '{for (i=1; i<=NF; i++) if ($i ~ /^avg_tps=/) print substr($i, 9)}')
echo "  agent decode_tps=$AGENT_DECODE  avg=$AGENT_AVG" >&2

#### Summary ###################################################################
echo ""
echo "============== summary =============="
{
    echo "| binary | what we measured | tps |"
    echo "| ------ | ---------------- | ---:|"
    echo "| **ds4-bench**  | gen_tps (clean kernel decode, KV-restored, sampled MTP) | $BENCH_TPS |"
    echo "| **ds4 (CLI)**  | generation: ... t/s (chat prefill + sampled MTP decode)  | $CLI_DECODE |"
    echo "| **ds4-server** | avg=... t/s from server logs (HTTP + chat + sampled MTP) | ${SERVER_AVG:-n/a} |"
    echo "| ds4-server (curl wall clock) | completion_tokens / elapsed | $(printf '%.2f' "$SERVER_TPS" 2>/dev/null || echo n/a) |"
    echo "| **ds4-agent**  | decode_tps from +DWARFSTAR_METRICS (one-shot, non-interactive) | $AGENT_DECODE |"
    echo "| ds4-agent (perceived) | avg_tps (decode tokens / wall, includes setup) | $AGENT_AVG |"
} | tee "$OUT/summary.md"
echo "[all-bench] outputs in $OUT" >&2
