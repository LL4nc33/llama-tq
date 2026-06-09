#!/usr/bin/env bash
# Gemma-4-12B dual-GPU deploy — 256k ctx via YaRN rope-scale 2x, ub=1024
#
# Tested: a local GPU host (2× RTX 2060 12 GB).
# KV combo: ktq2_1 + vtq3_v8 (v8 lossless tier).
# RoPE: YaRN factor=2 lifts ctx 131072 → 262144 (matches official Gemma-4 256k).
# ngram-spec disabled: 0% acceptance on natural-language workloads.
#
# Memory at 256k ctx: KV ≈ 2.4 GB total (sliding window 1024 × 40 layers + 8 global).
#
# Run: bash deploy-gemma4-12b-dualgpu-200k.sh
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
LLAMA_BIN="${LLAMA_BIN:-build/bin/llama-server}"
SLOTS="${SLOTS:-/tmp/llama-slots/}"

MODEL="$MODELS_DIR/gemma-4-12b-it-Q4_K_M.gguf"

mkdir -p "$SLOTS"

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --host 0.0.0.0 --port 8791 \
    -ngl 99 \
    -c 262144 \
    --rope-scaling yarn \
    --rope-scale 2 \
    --yarn-orig-ctx 131072 \
    -ub 1024 \
    --parallel 1 \
    -fa 1 \
    -ts 1,1 \
    --cache-type-k ktq2 \
    --cache-type-v vtq3 \
    --backend-sampling \
    --slot-save-path "$SLOTS" \
    --jinja \
    --chat-template-kwargs '{"thinking":false,"enable_thinking":false}' \
    --reasoning off \
    --reasoning-budget 0
