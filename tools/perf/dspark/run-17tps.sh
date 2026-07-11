#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
model=${DS4_DSPARK_MODEL:-$repo_dir/gguf/DeepSeek-V4-Flash-DSpark-Abliterated-COMBINED-Q2.correct.gguf}
prompt=${1:-$repo_dir/tests/test-vectors/prompts/long_code_audit.txt}
shift || true

exec env \
  DS4_CUDA_FAST_VERIFY=1 \
  DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
  DS4_CUDA_NO_INDEXED_HEADS8=1 \
  "$repo_dir/ds4" --cuda \
  --model "$model" --dspark \
  --ctx 131072 --tokens 32768 -t 10 \
  --prefill-chunk 2048 --temp 0 --nothink \
  --prompt-file "$prompt" "$@"
