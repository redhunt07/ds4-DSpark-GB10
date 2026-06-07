#!/usr/bin/env bash
# tools/perf/bench-with-monitor.sh — run ds4-bench under GPU/thermal/kmsg
# monitors so the next hard-trip leaves a forensic trail instead of an empty
# post-mortem. Supports a matrix of decode paths (--matrix) and a soak
# multiplier (--iter N) running back-to-back under one continuous monitor
# stream so per-cell signals and the transitions between them are visible.
#
# Defaults mirror the sweep that hard-rebooted the box on 2026-05-27:
#   frontiers 4k → 8k → 16k → 32k (geometric), --gen-tokens 32, MTP on
#   with Q4K-Q8_0-F32 draft and --mtp-draft 2, --temp 1.0 (spec-sampling).
#
# Usage:
#   tools/perf/bench-with-monitor.sh [--label NAME] [--rebuild]
#       [--prompt-file FILE] [-m MODEL] [--mtp PATH]
#       [--no-mtp] [--no-temp] [--matrix] [--iter N]
#       [-- <pass-through-args-to-ds4-bench>]
#
# Modes:
#   default              one cell (whichever --no-mtp/--no-temp resolve to), 1 iter
#   --matrix             three cells back-to-back: plain → mtp-greedy → mtp-sample
#                        (--no-mtp/--no-temp are ignored when --matrix is set)
#   --iter N             repeat each cell N times before moving on (default 1)
#   --matrix --iter N    full matrix soak: 3*N bench invocations under one
#                        continuous monitor stream
#
# Output layout
#   single cell + iter=1 (backward-compatible):
#     runs/<label>/{bench.log, bench.csv, gpu.csv, gpu-detail.txt, dmesg.log,
#                   stages.tsv, summary.txt, cmd.txt, prompt.txt?}
#   --matrix or --iter>1:
#     runs/<label>/{gpu.csv, gpu-detail.txt, dmesg.log, stages.tsv,
#                   summary.txt, cmd.txt, prompt.txt?}
#     runs/<label>/<cell>/iter-NNN/{bench.log, bench.csv}
#
# gpu.csv has an appended `stage` column (e.g. `mtp-sample#2`) so each
# sample is tied to the cell+iter that was running. stages.tsv carries
# per-cell start/end epoch + ISO timestamps for precise post-hoc bucketing.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUNS="$HERE/runs"

LABEL=""
REBUILD=0
USE_MTP=1
USE_TEMP=1
MATRIX=0
ITER=1
PROMPT_FILE="$ROOT/tests/long_context_story_prompt.txt"
MODEL="$HOME/models/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
MTP="$HOME/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"

EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --label)       LABEL="$2"; shift 2;;
    --rebuild)     REBUILD=1; shift;;
    --prompt-file) PROMPT_FILE="$2"; shift 2;;
    -m|--model)    MODEL="$2"; shift 2;;
    --mtp)         MTP="$2"; shift 2;;
    --no-mtp)      USE_MTP=0; shift;;
    --no-temp)     USE_TEMP=0; shift;;
    --matrix)      MATRIX=1; shift;;
    --iter)        ITER="$2"; shift 2;;
    -h|--help)     sed -n '2,50p' "${BASH_SOURCE[0]}"; exit 0;;
    --)            shift; EXTRA_ARGS+=("$@"); break;;
    *)             EXTRA_ARGS+=("$1"); shift;;
  esac
done

[ -n "$LABEL" ] || LABEL="thermal-$(date +%Y%m%d-%H%M%S)"
[[ "$ITER" =~ ^[0-9]+$ ]] && [ "$ITER" -ge 1 ] || { echo "--iter must be a positive integer" >&2; exit 2; }
OUT="$RUNS/$LABEL"
mkdir -p "$OUT"

cd "$ROOT"

if [ "$REBUILD" = 1 ]; then
  echo "## rebuild (make cuda-spark)" | tee -a "$OUT/summary.txt"
  make cuda-spark 2>&1 | tail -8 | tee -a "$OUT/summary.txt"
fi

[ -x ./ds4-bench ] || { echo "ds4-bench not built; pass --rebuild or run 'make cuda-spark'" >&2; exit 2; }
[ -f "$PROMPT_FILE" ] || { echo "missing --prompt-file: $PROMPT_FILE" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "missing -m model: $MODEL" >&2; exit 2; }
# MTP draft is required for any cell that uses MTP — that's matrix mode or
# single-cell mode with --no-mtp not set.
if { [ "$MATRIX" = 1 ] || [ "$USE_MTP" = 1 ]; } && [ ! -f "$MTP" ]; then
  echo "missing --mtp draft: $MTP" >&2; exit 2
fi

# ds4-bench refuses to start when the prompt tokenizes to fewer tokens than
# --ctx-max ("prompt has N tokens, need at least --ctx-max=M"). Story prompt
# is ~30.5k tokens — short of the 32k frontier — so we auto-stitch into the
# run dir when needed. Pass-through --ctx-max wins (last-wins in ds4-bench).
EFFECTIVE_CTX_MAX=32768
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
  if [ "${EXTRA_ARGS[i]}" = "--ctx-max" ]; then
    EFFECTIVE_CTX_MAX="${EXTRA_ARGS[i+1]}"
  fi
done
NEED_BYTES=$(( EFFECTIVE_CTX_MAX * 5 + 8192 ))
SRC_BYTES=$(stat -c%s "$PROMPT_FILE")
if [ "$SRC_BYTES" -lt "$NEED_BYTES" ]; then
  COPIES=$(( (NEED_BYTES + SRC_BYTES - 1) / SRC_BYTES ))
  STITCHED="$OUT/prompt.txt"
  : > "$STITCHED"
  for ((c=0; c<COPIES; c++)); do cat "$PROMPT_FILE" >> "$STITCHED"; done
  echo "## prompt stitched: ${COPIES}x copies of $(basename "$PROMPT_FILE") → $STITCHED ($(stat -c%s "$STITCHED") bytes, target≈${EFFECTIVE_CTX_MAX} tokens)"
  PROMPT_FILE="$STITCHED"
fi

# --- cell plan -------------------------------------------------------------
# Each cell is "name|use_mtp|use_temp".
declare -a CELLS
if [ "$MATRIX" = 1 ]; then
  CELLS=("plain|0|0" "mtp-greedy|1|0" "mtp-sample|1|1")
else
  name="plain"; [ "$USE_MTP" = 1 ] && name="mtp"
  if [ "$USE_TEMP" = 1 ]; then name="${name}-sample"; else name="${name}-greedy"; fi
  CELLS=("${name}|${USE_MTP}|${USE_TEMP}")
fi
TOTAL_CELLS=${#CELLS[@]}
TOTAL_RUNS=$(( TOTAL_CELLS * ITER ))

# Nest cell outputs under subdirs whenever we have more than one bench
# invocation; otherwise stay at top level for backward compat with the
# original single-cell wrapper.
NEST=0
if [ "$MATRIX" = 1 ] || [ "$ITER" -gt 1 ]; then NEST=1; fi

{
  echo "## label:    $LABEL"
  echo "## started:  $(date -Iseconds)"
  echo "## host:     $(hostname) — uptime: $(uptime -p)"
  echo "## matrix:   $([ "$MATRIX" = 1 ] && echo on || echo off)"
  echo "## iter:     $ITER"
  echo "## cells:    ${CELLS[*]}"
  echo "## plan:     $TOTAL_CELLS cells × $ITER iter = $TOTAL_RUNS bench invocations"
  echo "## prompt:   $PROMPT_FILE"
  echo "## ctx-max:  $EFFECTIVE_CTX_MAX"
} >> "$OUT/cmd.txt"

# --- continuous monitors ---------------------------------------------------
# Per-row stage tag: the driver writes the current "cell#iter" to a small
# file; the awk wrapper around nvidia-smi reads it before printing each row.
# This pins every gpu.csv sample to the run state without any post-hoc
# timestamp arithmetic.
STAGE_FILE="$OUT/.current_stage"
echo "idle" > "$STAGE_FILE"

# 1Hz timeseries with appended stage column. Drop N/A columns (memory clock,
# memory.used) since GB10 unified memory leaves them empty.
stdbuf -oL nvidia-smi \
  --query-gpu=timestamp,temperature.gpu,power.draw,clocks.current.sm,clocks_throttle_reasons.active,utilization.gpu \
  --format=csv,nounits \
  -l 1 \
  2> "$OUT/gpu.err" \
  | awk -v SF="$STAGE_FILE" '
      BEGIN { stage = "idle" }
      NR == 1 { print $0 ", stage"; fflush(); next }
      {
        if ((getline s < SF) > 0) stage = s
        close(SF)
        print $0 ", " stage
        fflush()
      }' \
  > "$OUT/gpu.csv" &
GPU_PID=$!

# T.Limit margin and per-reason throttle breakdown only show up in
# `nvidia-smi -q` text — snapshot every 5s with a stage tag attached.
(
  while :; do
    s="idle"
    [ -f "$STAGE_FILE" ] && s=$(cat "$STAGE_FILE")
    echo "=== $(date -Iseconds)  stage=$s ==="
    nvidia-smi -q -d TEMPERATURE,POWER,CLOCK,PERFORMANCE 2>&1 \
      | grep -E 'GPU Current Temp|GPU T\.Limit|Average Power|Instantaneous Power|Performance State|SM Clock|HW Thermal|HW Power Brake|SW Thermal|SW Power Cap|HW Slowdown'
    sleep 5
  done
) > "$OUT/gpu-detail.txt" 2>&1 &
DETAIL_PID=$!

# Kernel ring buffer follow (unprivileged via `adm` group).
journalctl -kf --since now > "$OUT/dmesg.log" 2>&1 &
KMSG_PID=$!

# CPU monitors. ds4 inference pegs a single thread (likely the launcher /
# scheduler / sampler hot path); on GB10's heterogeneous Cortex-X925 + A725
# topology it matters whether that thread lands on a fast core (capacity
# 1024 @ 3.9 GHz) or a mid core (~720 @ 2.8 GHz). mpstat gives per-core
# utilization; pidstat -t breaks down per-thread %CPU and reports the last
# CPU each thread ran on so we can correlate against the topology snapshot.
# Both mpstat and pidstat get their output epoch-stamped via awk so each row
# can be joined against stages.tsv windows for per-cell aggregation. Killing
# the backgrounded awk SIGPIPEs the upstream sysstat tool cleanly on exit.
mpstat -P ALL 1 2> "$OUT/cpu-mpstat.err" \
  | awk '{ print systime(), $0; fflush() }' \
  > "$OUT/cpu-mpstat.log" &
MPSTAT_PID=$!

pidstat -t -h -C 'ds4-bench' 1 2> "$OUT/cpu-threads.err" \
  | awk '{ print systime(), $0; fflush() }' \
  > "$OUT/cpu-threads.log" &
PIDSTAT_PID=$!

# Snapshot the heterogeneous topology once so the summary can label cores
# as fast / mid based on cpu_capacity (the kernel's normalized perf metric:
# 1024 = strongest core class on this SoC).
{
  echo "cpu  capacity  max_khz  part"
  for c in /sys/devices/system/cpu/cpu[0-9]*; do
    # bash parameter expansion — no fork to sed (which is aliased to a
    # missing gsed in zsh anyway, so this is portable across both shells).
    name=${c##*/}
    n=${name#cpu}
    cap=$(cat "$c/cpu_capacity" 2>/dev/null || echo "?")
    mhz=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo "?")
    part=$(awk -v want="$n" '$1=="processor"{p=$3} $1=="CPU" && $2=="part" && p==want{print $4; exit}' /proc/cpuinfo)
    printf "%-4s %-9s %-8s %s\n" "$n" "$cap" "$mhz" "$part"
  done
} > "$OUT/cpu-topology.txt"

# Stage audit trail — written by the driver between cells.
printf 'epoch\tiso\tcell\titer\tphase\n' > "$OUT/stages.tsv"
mark_stage() {
  # mark_stage <cell> <iter> <phase>
  echo "$1#$2" > "$STAGE_FILE"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$(date -Iseconds)" "$1" "$2" "$3" >> "$OUT/stages.tsv"
}
mark_idle() {
  echo "idle" > "$STAGE_FILE"
}

cleanup() {
  local code=$?
  mark_idle
  kill "$GPU_PID" "$DETAIL_PID" "$KMSG_PID" "$MPSTAT_PID" "$PIDSTAT_PID" 2>/dev/null || true
  wait "$GPU_PID" "$DETAIL_PID" "$KMSG_PID" "$MPSTAT_PID" "$PIDSTAT_PID" 2>/dev/null || true
  {
    echo
    echo "## ended:    $(date -Iseconds)  exit=$code"

    if [ -s "$OUT/gpu.csv" ]; then
      # Global aggregate (all samples, all stages including idle).
      awk -F', *' '
        NR == 1 { next }
        $2 ~ /^[0-9]/ {
          t=$2+0; p=$3+0; sm=$4+0
          if (t > tmax) tmax = t
          if (p > pmax) pmax = p
          if (sm > smmax) smmax = sm
          if ($5 != "" && $5 != "0x0000000000000000") thr[$5]++
          n++
        }
        END {
          if (n == 0) { print "## gpu.csv: no rows captured"; exit }
          printf "## global    samples=%d  peak_temp=%dC  peak_power=%.2fW  peak_sm=%dMHz\n", n, tmax, pmax, smmax
          if (length(thr) > 0) {
            print "## global    throttle masks (non-zero):"
            for (k in thr) printf "    %s  x%d\n", k, thr[k]
          } else {
            print "## global    throttle masks: none (always 0x0)"
          }
        }' "$OUT/gpu.csv"

      # Per-cell aggregate: iterate stage tags in execution order (driven
      # by stages.tsv), bucket gpu.csv rows by the appended stage column.
      if [ -f "$OUT/stages.tsv" ] && [ "$(wc -l < "$OUT/stages.tsv")" -gt 1 ]; then
        echo "## per-cell  (execution order, idle samples between cells excluded):"
        awk -v stages="$OUT/stages.tsv" -F',' '
          BEGIN {
            FS = ","
            n_keys = 0
            getline < stages   # header
            while ((getline line < stages) > 0) {
              n = split(line, sf, "\t")
              if (n < 5) continue
              if (sf[5] != "start") continue
              key = sf[3] "#" sf[4]
              if (!(key in seen)) {
                seen[key] = ++n_keys
                order[n_keys] = key
              }
            }
            close(stages)
          }
          NR == 1 { next }
          $2 ~ /^[ 0-9]+$/ {
            stage = $NF
            sub(/^[ \t]+/, "", stage); sub(/[ \t]+$/, "", stage)
            if (!(stage in seen)) next
            t = $2 + 0; p = $3 + 0; sm = $4 + 0
            if (t > peak_t[stage]) peak_t[stage] = t
            if (p > peak_p[stage]) peak_p[stage] = p
            if (sm > peak_s[stage]) peak_s[stage] = sm
            cnt[stage]++
            mask = $5; sub(/^[ \t]+/, "", mask); sub(/[ \t]+$/, "", mask)
            if (mask != "" && mask != "0x0000000000000000") {
              key = stage SUBSEP mask
              if (!(key in seen_mask)) {
                seen_mask[key] = 1
                masks[stage] = (stage in masks ? masks[stage] "," mask : mask)
              }
            }
          }
          END {
            for (i = 1; i <= n_keys; i++) {
              k = order[i]
              tag = (k in masks ? "  masks=" masks[k] : "")
              printf "    %-22s samples=%-4d peak_temp=%dC peak_power=%.2fW peak_sm=%dMHz%s\n",
                k, cnt[k]+0, peak_t[k]+0, peak_p[k]+0, peak_s[k]+0, tag
            }
          }' "$OUT/gpu.csv"
      fi
    fi

    # CPU per-core peak load + per-thread peaks. mpstat lines for individual
    # CPUs have a numeric $2 (skips the "all" aggregate); pidstat -t thread
    # rows are the ones where TGID == "-". We pipe peaks through sort and
    # join against cpu-topology.txt to label the hottest cores fast/mid.
    if [ -s "$OUT/cpu-mpstat.log" ]; then
      echo "## CPU per-core peak load (top 5, labelled by cpu_capacity):"
      awk '
        $2 ~ /^[0-9]+$/ {
          cpu = $2; idle = $NF + 0; load = 100 - idle
          if (load > peak[cpu]) peak[cpu] = load
        }
        END { for (c in peak) printf "%.1f\t%s\n", peak[c], c }
      ' "$OUT/cpu-mpstat.log" \
        | sort -k1,1 -rn | head -5 \
        | while IFS=$'\t' read -r load cpu; do
            cap=$(awk -v want="$cpu" '$1==want{print $2}' "$OUT/cpu-topology.txt")
            mhz=$(awk -v want="$cpu" '$1==want{print $3}' "$OUT/cpu-topology.txt")
            class="?"
            if [ -n "$cap" ] && [ "$cap" != "?" ]; then
              if [ "$cap" -ge 900 ]; then class="fast"; else class="mid"; fi
            fi
            printf "    cpu%-3s peak_load=%5s%%  capacity=%-5s  max_khz=%-8s  class=%s\n" "$cpu" "$load" "$cap" "$mhz" "$class"
          done
    fi

    # The actual bottleneck question: is a single thread saturated while the
    # GPU has idle time? Join cpu-threads.log (epoch-stamped pidstat -t)
    # against gpu.csv's stage-tagged rows and stages.tsv windows. For each
    # cell-iter we compare the hottest thread's avg %CPU during that window
    # against the GPU utilization avg in the same window.
    if [ -s "$OUT/cpu-threads.log" ] && [ -s "$OUT/gpu.csv" ] && [ -s "$OUT/stages.tsv" ]; then
      echo "## CPU bottleneck check (per-stage hot-thread avg vs GPU util avg):"
      awk '
        # ---- stages.tsv: epoch \t iso \t cell \t iter \t phase ----
        FILENAME ~ /stages\.tsv$/ {
          n = split($0, sf, "\t")
          if (n < 5 || sf[1] == "epoch") next
          key = sf[3] "#" sf[4]
          if (sf[5] == "start") {
            stage_start[key] = sf[1] + 0
            if (!(key in seen_stage)) { seen_stage[key] = 1; stage_order[++n_stages] = key }
          } else if (sf[5] == "end") {
            stage_end[key] = sf[1] + 0
          }
          next
        }
        # ---- gpu.csv: ts, temp, power, sm, throttle, util, stage ----
        FILENAME ~ /gpu\.csv$/ {
          n = split($0, gf, /, */)
          if (n < 7) next
          if (gf[2] !~ /^[ 0-9]+$/) next
          stage = gf[7]; gsub(/^[ \t]+|[ \t]+$/, "", stage)
          if (!(stage in stage_start)) next
          util = gf[6] + 0
          gpu_sum[stage] += util
          gpu_cnt[stage]++
          next
        }
        # ---- cpu-threads.log: <our_epoch> <pidstat_time> UID TGID TID ... %CPU CPU Command ----
        FILENAME ~ /cpu-threads\.log$/ {
          n = split($0, cf, /[[:space:]]+/)
          if (n < 11) next
          if (cf[1] !~ /^[0-9]+$/) next         # need our epoch in col 1
          if (cf[4] != "-") next                # thread rows have TGID == "-"
          ep = cf[1] + 0
          tid = cf[5]
          # Walk right→left: first integer-only field is the CPU id, then the
          # next numeric to its left is %CPU. Robust across sysstat versions.
          cpu_id = "?"; pct = 0; idx = 0
          for (i = n; i >= 1; i--) {
            if (cf[i] ~ /^[0-9]+$/) { cpu_id = cf[i]; idx = i; break }
          }
          for (j = idx - 1; j >= 1; j--) {
            if (cf[j] ~ /^-?[0-9]+(\.[0-9]+)?$/) { pct = cf[j] + 0; break }
          }
          # Track overall-hottest thread (avg %CPU across whole run).
          tid_sum_all[tid] += pct
          tid_cnt_all[tid]++
          tid_cpu_last[tid] = cpu_id
          # Bucket per-stage.
          for (k = 1; k <= n_stages; k++) {
            s = stage_order[k]
            if (ep >= stage_start[s] && ep <= (s in stage_end ? stage_end[s] : ep)) {
              skey = s SUBSEP tid
              tid_sum_stage[skey] += pct
              tid_cnt_stage[skey]++
              if (pct > tid_peak_stage[skey]) tid_peak_stage[skey] = pct
              break
            }
          }
          next
        }
        END {
          # Identify the hottest thread overall (max avg %CPU).
          hot_tid = ""; hot_avg = 0
          for (t in tid_sum_all) {
            avg = tid_sum_all[t] / tid_cnt_all[t]
            if (avg > hot_avg) { hot_avg = avg; hot_tid = t }
          }
          if (hot_tid == "") { print "    (no thread data captured)"; exit }
          printf "##   hot thread overall: tid=%s  avg=%.1f%%  last_cpu=%s\n", hot_tid, hot_avg, tid_cpu_last[hot_tid]
          for (k = 1; k <= n_stages; k++) {
            s = stage_order[k]
            gavg = (s in gpu_cnt && gpu_cnt[s] > 0) ? gpu_sum[s] / gpu_cnt[s] : 0
            skey = s SUBSEP hot_tid
            cavg = (skey in tid_cnt_stage && tid_cnt_stage[skey] > 0) ? tid_sum_stage[skey] / tid_cnt_stage[skey] : 0
            cpeak = (skey in tid_peak_stage) ? tid_peak_stage[skey] : 0
            # Verdict thresholds: GPU >= 95% util = saturated; otherwise hot
            # thread >= 90% avg = CPU bottleneck; else neither pegged.
            if      (gavg >= 95.0)              v = "GPU-saturated"
            else if (cavg >= 90.0)              v = "CPU-BOTTLENECK"
            else                                v = "neither saturated"
            printf "    %-22s  hot_avg=%5.1f%%  hot_peak=%5.1f%%  gpu_util_avg=%5.1f%%  → %s\n",
              s, cavg, cpeak, gavg, v
          }
        }
      ' "$OUT/stages.tsv" "$OUT/gpu.csv" "$OUT/cpu-threads.log"

      # Topology cross-reference for the hot thread's cpu_at_peak.
      hot_tid_line=$(awk '
        $1 !~ /^[0-9]+$/ { next }
        $4 != "-" { next }
        { n = split($0, f, /[[:space:]]+/)
          tid=f[5]; idx=0
          for (i=n; i>=1; i--) if (f[i] ~ /^[0-9]+$/) { cpu=f[i]; idx=i; break }
          for (j=idx-1; j>=1; j--) if (f[j] ~ /^-?[0-9]+(\.[0-9]+)?$/) { pct=f[j]+0; break }
          sum[tid] += pct; cnt[tid]++; lastcpu[tid] = cpu }
        END {
          maxavg=0; bt=""
          for (t in sum) { a=sum[t]/cnt[t]; if (a > maxavg) { maxavg=a; bt=t } }
          if (bt != "") printf "%s\t%s\n", bt, lastcpu[bt]
        }' "$OUT/cpu-threads.log")
      if [ -n "$hot_tid_line" ]; then
        hot_cpu=$(echo "$hot_tid_line" | cut -f2)
        cap=$(awk -v want="$hot_cpu" '$1==want{print $2}' "$OUT/cpu-topology.txt" 2>/dev/null)
        class="?"
        if [ -n "$cap" ] && [ "$cap" != "?" ]; then
          if [ "$cap" -ge 900 ]; then class="fast"; else class="mid"; fi
        fi
        echo "##   hot thread cpu_at_peak=$hot_cpu  capacity=$cap  class=$class  (see cpu-topology.txt)"
      fi
    fi

    # Authoritative throttle signal: nvidia-smi -q PERFORMANCE prints each
    # reason as "Active" or "Not Active". The bitmask in gpu.csv carries
    # benign idle flags too; this scan only fires on the reasons that
    # actually mean trouble (thermal slowdown or power brake).
    if [ -s "$OUT/gpu-detail.txt" ]; then
      active_throttle=$(grep -E '(HW Thermal Slowdown|HW Power Brake Slowdown|SW Thermal Slowdown|HW Slowdown)[[:space:]]*:[[:space:]]*Active' "$OUT/gpu-detail.txt" | wc -l)
      echo "## throttle (Active) snapshots in gpu-detail.txt: $active_throttle"
      if [ "$active_throttle" -gt 0 ]; then
        grep -E '(HW Thermal Slowdown|HW Power Brake Slowdown|SW Thermal Slowdown|HW Slowdown)[[:space:]]*:[[:space:]]*Active' "$OUT/gpu-detail.txt" | head -10
      fi
    fi

    if [ -s "$OUT/dmesg.log" ]; then
      hits=$(grep -ciE 'oom|thermal|throttl|nvidia|nvrm|xid|hw error|aer|hang|reset|fault' "$OUT/dmesg.log" || true)
      echo "## dmesg interesting lines: $hits"
      if [ "$hits" -gt 0 ]; then
        grep -iE 'oom|thermal|throttl|nvrm|xid|hw error|aer|hang|reset|fault' "$OUT/dmesg.log" | head -30
      fi
    fi
    echo "## outputs: $OUT"
  } >> "$OUT/summary.txt"
  echo
  tail -n 60 "$OUT/summary.txt"
}
trap cleanup EXIT INT TERM

# --- common bench args (shared across cells) ------------------------------
COMMON_ARGS=(
  --prompt-file "$PROMPT_FILE"
  -m            "$MODEL"
  --ctx-start   4096
  --ctx-max     32768
  --step-mul    2
  --gen-tokens  32
)

# --- driver ---------------------------------------------------------------
echo "## plan: $TOTAL_CELLS cells × $ITER iter = $TOTAL_RUNS bench invocations (continuous monitor stream)"
echo "## outputs streaming to: $OUT/"

run_idx=0
overall_rc=0
for spec in "${CELLS[@]}"; do
  IFS='|' read -r cell_name cell_mtp cell_temp <<< "$spec"
  for ((it=1; it<=ITER; it++)); do
    run_idx=$((run_idx + 1))
    iter_tag=$(printf "iter-%03d" "$it")

    if [ "$NEST" = 1 ]; then
      cell_out="$OUT/$cell_name/$iter_tag"
    else
      cell_out="$OUT"
    fi
    mkdir -p "$cell_out"

    args=("${COMMON_ARGS[@]}")
    [ "$cell_mtp"  = 1 ] && args+=(--mtp "$MTP" --mtp-draft 2)
    [ "$cell_temp" = 1 ] && args+=(--temp 1.0 --top-p 0.95 --seed 1234)
    args+=(--csv "$cell_out/bench.csv")
    args+=("${EXTRA_ARGS[@]}")

    echo
    echo "## ── [$run_idx/$TOTAL_RUNS] cell=$cell_name iter=$it ── $(date -Iseconds)"

    mark_stage "$cell_name" "$it" "start"
    set +e
    set -o pipefail
    ./ds4-bench "${args[@]}" 2>&1 \
      | awk -v c="$cell_name" -v it="$it" '{ print strftime("[%H:%M:%S]"), "["c"#"it"]", $0; fflush() }' \
      | tee "$cell_out/bench.log"
    rc=${PIPESTATUS[0]}
    set +o pipefail
    set -e
    mark_stage "$cell_name" "$it" "end"
    mark_idle

    if [ "$rc" -ne 0 ]; then
      echo "## ── cell=$cell_name iter=$it FAILED (exit=$rc) — continuing remaining runs"
      overall_rc=$rc
    fi
  done
done

exit "$overall_rc"
