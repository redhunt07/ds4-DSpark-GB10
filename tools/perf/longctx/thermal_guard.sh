#!/usr/bin/env bash
# Source me. Two functions: cooldown_wait and thermal_snapshot.
# Per project_spark-ebf0-thermal-shutoff: keep board <55C between iters.

THERMAL_TARGET_C=${THERMAL_TARGET_C:-55}
THERMAL_MAX_WAIT_S=${THERMAL_MAX_WAIT_S:-600}
THERMAL_POLL_S=${THERMAL_POLL_S:-5}

_gpu_temp_c() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' '
}

_gpu_power_w() {
    nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' '
}

cooldown_wait() {
    local label="${1:-iter}"
    local start_t deadline t now
    start_t=$(date +%s)
    deadline=$((start_t + THERMAL_MAX_WAIT_S))
    while :; do
        t=$(_gpu_temp_c)
        now=$(date +%s)
        if [[ -z "$t" ]]; then
            echo "[thermal_guard] WARN: no nvidia-smi reading; skipping cooldown" >&2
            return 0
        fi
        if (( t <= THERMAL_TARGET_C )); then
            echo "[thermal_guard] $label cooled to ${t}C in $((now - start_t))s" >&2
            return 0
        fi
        if (( now >= deadline )); then
            echo "[thermal_guard] $label TIMEOUT at ${t}C after $((now - start_t))s" >&2
            return 1
        fi
        sleep "$THERMAL_POLL_S"
    done
}

thermal_snapshot() {
    # Emit one-line JSON: {"phase":"pre","temp_c":42,"power_w":18.3,"ts":...}
    local phase="${1:-pre}"
    printf '{"phase":"%s","temp_c":%s,"power_w":%s,"ts":%s}\n' \
        "$phase" "$(_gpu_temp_c)" "$(_gpu_power_w)" "$(date +%s)"
}

# Allow direct invocation for ad-hoc cooldown.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-wait}" in
        wait) cooldown_wait "${2:-cli}" ;;
        pre|post|peak) thermal_snapshot "$1" ;;
        *) echo "usage: $0 {wait [label]|pre|post|peak}" >&2; exit 2 ;;
    esac
fi
