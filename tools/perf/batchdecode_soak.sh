#!/usr/bin/env bash
# DS4_BATCH_DECODE promotion gates: full-vocab logit deltas, sampled identity,
# deep-context identity (capped 65k), mtp-correctness regression, agent soak.
#
# Hardened after two box-wedges: each GPU step runs through guarded(), which
#   (1) drops the model from page cache first (posix_fadvise DONTNEED, unpriv)
#       so each fresh model-load + cudaHostRegister starts from a clean cache —
#       back-to-back warm-cache pinning is what wedged the box (model-load OOM
#       at ~121 GiB free, before any KV alloc), and
#   (2) watches MemAvailable and kills the process if it drops below FLOOR_GIB,
#       turning a runaway alloc into a killed process instead of a hard hang.
# The box has no swap, so these guards are load-bearing, not belt-and-suspenders.
set -uo pipefail
cd /home/trevor/Projects/ds4
OUT=/tmp/batchdecode
mkdir -p "$OUT"
MODEL="$HOME/models/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
MTP="$HOME/models/ds4/DeepSeek-V4-Flash-MTP-v2.gguf"
STORY=tests/long_context_story_prompt.txt
LONG=tools/perf/longctx/prompts/longform.txt
DEEP_CTX="${DEEP_CTX:-65536}"   # deep-context gate (131k crashed; 65k is proven safe)
FLOOR_GIB="${FLOOR_GIB:-5}"

free_gib() { awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo; }

drop_cache() {
    # Unprivileged page-cache eviction of the big GGUFs (no sudo / drop_caches).
    python3 - "$MODEL" "$MTP" <<'PY' 2>/dev/null || true
import os, sys
for p in sys.argv[1:]:
    try:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass
PY
}

cool() {
    local t=99
    for _ in $(seq 1 80); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        [ "$t" -le 55 ] && break
        sleep 5
    done
    echo "[cool] ${t}C  free=$(free_gib) GiB"
}

# guarded "<label>" <command...>  — drop cache, cool, run under the mem watchdog.
guarded() {
    local label="$1"; shift
    drop_cache
    cool
    echo "[$(date +%T)] $label start (free=$(free_gib) GiB)"
    "$@" &
    local pid=$! killed=0 fa
    while kill -0 "$pid" 2>/dev/null; do
        fa=$(free_gib)
        if awk -v f="$fa" -v fl="$FLOOR_GIB" 'BEGIN{exit !(f+0 < fl+0)}'; then
            echo "[$(date +%T)] !!! $label free=${fa} GiB < ${FLOOR_GIB} — KILL"
            kill -9 "$pid" 2>/dev/null; killed=1; break
        fi
        sleep 2
    done
    wait "$pid" 2>/dev/null
    local rc=$?
    [ "$killed" = 1 ] && { echo "[$label] WATCHDOG-KILLED"; return 99; }
    return "$rc"
}

BENCH=(./ds4-bench --cuda --warm-weights --power 85)

echo "=== gate 1: full-vocab logit delta probe (det, w=1 + widths 2-5 argmax) ==="
guarded gate1 "${BENCH[@]}" -m "$MODEL" --mtp "$MTP" --mtp-draft 1 \
    --prompt-file "$STORY" --ctx-start 4096 --ctx-max 4096 \
    --batch-check 240 > "$OUT/soak_logits.log" 2>&1
echo "rc=$?"; grep "batch-check" "$OUT/soak_logits.log" || tail -3 "$OUT/soak_logits.log"

echo "=== gate 2: sampled seeded identity @4k (det, temp 1.0 top-p 0.95) ==="
for arm in plain gated; do
    envs=(DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 "DS4_BENCH_TOKEN_DUMP=$OUT/ts_$arm.txt")
    [ "$arm" = gated ] && envs+=(DS4_BATCH_DECODE=1)
    guarded "gate2-$arm" env "${envs[@]}" "${BENCH[@]}" -m "$MODEL" --mtp "$MTP" --mtp-draft 1 \
        --prompt-file "$STORY" --ctx-start 4096 --ctx-max 4096 \
        --gen-tokens 256 --temp 1.0 --top-p 0.95 --seed 1234 \
        --csv "$OUT/ts_$arm.csv" > "$OUT/ts_$arm.log" 2>&1
    echo "  $arm rc=$? $(tail -1 "$OUT/ts_$arm.csv" 2>/dev/null)"
done
echo "== sampled identity:"
diff -q "$OUT/ts_plain.txt" "$OUT/ts_gated.txt" >/dev/null 2>&1 \
    && echo "IDENTICAL ($(wc -l < "$OUT/ts_plain.txt") tokens)" \
    || echo "DIVERGED: $(diff "$OUT/ts_plain.txt" "$OUT/ts_gated.txt" | head -1)"

echo "=== gate 3: ${DEEP_CTX} greedy identity (det) ==="
for arm in plain gated; do
    envs=("DS4_BENCH_TOKEN_DUMP=$OUT/td_$arm.txt")
    [ "$arm" = gated ] && envs+=(DS4_BATCH_DECODE=1)
    guarded "gate3-$arm" env "${envs[@]}" "${BENCH[@]}" -m "$MODEL" --mtp "$MTP" --mtp-draft 1 \
        --prompt-file "$LONG" --ctx-start "$DEEP_CTX" --ctx-max "$DEEP_CTX" \
        --gen-tokens 128 --csv "$OUT/td_$arm.csv" > "$OUT/td_$arm.log" 2>&1
    echo "  $arm rc=$? $(tail -1 "$OUT/td_$arm.csv" 2>/dev/null)"
done
echo "== ${DEEP_CTX} identity:"
diff -q "$OUT/td_plain.txt" "$OUT/td_gated.txt" >/dev/null 2>&1 \
    && echo "IDENTICAL ($(wc -l < "$OUT/td_plain.txt") tokens)" \
    || { echo "DIVERGED:"; diff "$OUT/td_plain.txt" "$OUT/td_gated.txt" | head -4; }

echo "=== gate 4: mtp-correctness + selfconsistency with gate ON ==="
guarded gate4a env DS4_BATCH_DECODE=1 DS4_TEST_MODEL="$MODEL" DS4_TEST_MTP_MODEL="$MTP" \
    ./ds4_test --mtp-correctness > "$OUT/mtpcorr.log" 2>&1
echo "mtp-correctness rc=$?"; tail -3 "$OUT/mtpcorr.log"
guarded gate4b env DS4_BATCH_DECODE=1 DS4_TEST_MODEL="$MODEL" DS4_TEST_MTP_MODEL="$MTP" \
    ./ds4_test --mtp-selfconsistency > "$OUT/mtpself.log" 2>&1
echo "mtp-selfconsistency rc=$?"; tail -3 "$OUT/mtpself.log"

echo "=== gate 5: agent soak (MTP draft 2 + gate, greedy & sampled) ==="
PROMPT="Write a 300-word essay on why memory bandwidth, not compute, limits single-stream LLM inference on unified-memory machines."
for mode in greedy sampled; do
    targs=(); [ "$mode" = greedy ] && targs=(--temp 0)
    guarded "gate5-$mode" env DS4_BATCH_DECODE=1 DS4_MTP_TIMING=1 DS4_CUDA_FAST_VERIFY=1 \
        ./ds4-agent --cuda -m "$MODEL" -c 8192 --warm-weights \
        --mtp "$MTP" --mtp-draft 2 --power 85 \
        --non-interactive --nothink --tokens 400 "${targs[@]}" \
        -p "$PROMPT" > "$OUT/agent_$mode.out" 2>"$OUT/agent_$mode.err"
    echo "  $mode rc=$?"
    ./tools/perf/mtp/parse_timing.py "$OUT/agent_$mode.err" --label "agent-$mode" 2>/dev/null | sed -n '3,6p'
    echo "  BOS-spam: $(grep -c "begin.of.sentence" "$OUT/agent_$mode.out" 2>/dev/null || echo 0) occurrences"
done
echo "=== soak complete ==="
