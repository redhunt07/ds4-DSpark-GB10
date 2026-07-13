#!/usr/bin/env bash
set -euo pipefail

# End-to-end OpenCode 1.17.18 harness. OPENCODE_BIN must point at the pinned
# release binary; keeping installation outside this script makes the download
# independently auditable and the soak reproducible offline.
OPENCODE_BIN=${OPENCODE_BIN:-/tmp/opencode-1.17.18/opencode}
DS4_BASE_URL=${DS4_BASE_URL:-http://127.0.0.1:8000/v1}
DS4_MODEL=${DS4_MODEL:-deepseek-v4-flash}
SOAK_CYCLES=${SOAK_CYCLES:-50}
SKIP_INITIAL=${SKIP_INITIAL:-0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d /tmp/ds4-opencode-soak.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
cp -R "$ROOT/tests/fixtures/agent_project/." "$WORK/"

if [[ ! -x "$OPENCODE_BIN" ]]; then
  echo "missing OpenCode 1.17.18 binary: $OPENCODE_BIN" >&2
  exit 2
fi

export OPENCODE_CONFIG_CONTENT
OPENCODE_CONFIG_CONTENT=$(printf '%s' "{
  \"model\": \"ds4/$DS4_MODEL\",
  \"provider\": {
    \"ds4\": {
      \"name\": \"DS4 canary\",
      \"npm\": \"@ai-sdk/openai-compatible\",
      \"env\": [],
      \"models\": {
        \"$DS4_MODEL\": {
          \"name\": \"DS4\",
          \"tool_call\": true,
          \"reasoning\": true,
          \"limit\": {\"context\": 131072, \"output\": 32768}
        }
      },
      \"options\": {\"apiKey\": \"local\", \"baseURL\": \"$DS4_BASE_URL\"}
    }
  }
}")

run_agent() {
  timeout 20m "$OPENCODE_BIN" run --pure --auto --format json \
    --model "ds4/$DS4_MODEL" --dir "$WORK" "$1"
}

if [[ "$SKIP_INITIAL" != 1 ]]; then
  run_agent "Inspect the project, run its tests, add a regression test for negative numbers, implement any required fix, rerun tests, and finish with a concise summary." >"$WORK/initial.jsonl"
  grep -q '"type"' "$WORK/initial.jsonl"
fi

for ((i=1; i<=SOAK_CYCLES; i++)); do
  run_agent "Cycle $i: inspect one source file, run the test suite, and report the result. Do not change correct code." >"$WORK/cycle-$i.jsonl"
  grep -q '"type"' "$WORK/cycle-$i.jsonl"
done

# Persistent-process regression: inherited pipes must not keep the shell tool
# alive. The command exits promptly while the detached child cleans itself up.
run_agent "Use the shell once to start a detached Node process that sleeps for 2 seconds. Redirect all three standard streams, verify the shell call returns immediately, then finish." >"$WORK/detach.jsonl"

echo "OpenCode soak passed: cycles=$SOAK_CYCLES work=$WORK"
