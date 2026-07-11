#!/bin/sh
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
HF_DIR=${DS4_HF_DIR:-"$ROOT/hf/DeepSeek-V4-Flash-Abliterated-DSpark"}
OUT=${DS4_DSPARK_OUT:-"$ROOT/gguf/DeepSeek-V4-Flash-DSpark-Abliterated-Q2.gguf"}
IMATRIX=${DS4_IMATRIX_FILE:-}

usage() {
    cat <<EOF
Usage: ./quantize_dspark.sh [--hf DIR] [--out FILE] [--imatrix FILE]

Converts the DSpark abliterated Hugging Face checkpoint into a GGUF using the
local deepseek4 quantizer.

Defaults:
  --hf        $HF_DIR
  --out       $OUT
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --hf)
            shift
            HF_DIR=${1:-}
            ;;
        --out)
            shift
            OUT=${1:-}
            ;;
        --imatrix)
            shift
            IMATRIX=${1:-}
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [ ! -d "$HF_DIR" ]; then
    echo "Missing DSpark HF directory: $HF_DIR" >&2
    echo "Run: ./download_model.sh dspark-source" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$OUT")"

cmd="$ROOT/gguf-tools/deepseek4-quantize --hf $HF_DIR --from-config --out $OUT --routed-w1 iq2_xxs --routed-w3 iq2_xxs --routed-w2 q2_k --attention-proj q8_0 --attention q8_0 --shared q8_0 --embedding f16 --output q8_0 --dense q8_0"
if [ -n "$IMATRIX" ]; then
    if [ ! -f "$IMATRIX" ]; then
        echo "Missing imatrix file: $IMATRIX" >&2
        exit 1
    fi
    cmd="$cmd --imatrix $IMATRIX"
fi

echo "Running:"
echo "  $cmd"
eval "$cmd"
