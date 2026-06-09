#!/usr/bin/env bash
# Gemma-4-12B dual-GPU deploy with 200k ctx, parallel=1, ngram-spec
#
# Tested: gpu00 (2× RTX 2060 12 GB), Qwen-style hardware ceiling.
# KV combo: ktq2_1 + vtq3_v8 (v8 lossless tier).
# Expected: ~37 t/s creative TG, boost on repeat via n-gram spec.
#
# Memory at 200k ctx:
#   GPU0: model split + KV 924 MiB + compute 1956 MiB ≈ 6.4 GB
#   GPU1: model split + KV 836 MiB + compute 1355 MiB ≈ 5.7 GB
#
# Run: bash deploy-gemma4-12b-dualgpu-200k.sh
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
LLAMA_BIN="${LLAMA_BIN:-${HOME}/llama-tq-mtp-fusion/build/bin/llama-server}"
SLOTS="${SLOTS:-${HOME}/llama-slots-gemma4/}"

MODEL="$MODELS_DIR/gemma-4-12b-it-Q4_K_M.gguf"
MMPROJ="$MODELS_DIR/gemma-4-12b-mmproj-F16.gguf"

mkdir -p "$SLOTS"

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --host 0.0.0.0 --port 8791 \
    -ngl 99 \
    -c 200000 \
    --parallel 1 \
    --spec-type ngram-cache \
    --draft-max 8 \
    --draft-min 4 \
    --cache-reuse 25000 \
    -fa 1 \
    -ts 1,1 \
    --cache-type-k ktq2 \
    --cache-type-v vtq3 \
    --backend-sampling \
    --slot-save-path "$SLOTS" \
    --jinja \
    --chat-template-kwargs '{"thinking":false}'
