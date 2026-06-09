#!/usr/bin/env bash
# Gemma-4-12B single-GPU0 deploy — 131k ctx, GPU1 frei für andere services
#
# Tested: a local GPU host (RTX 2060 12 GB, GPU0).
# VRAM: ~11.3 GB / 12 GB (knapp aber stabil dank gemma-4 sliding window + ktq2/vtq3).
# Throughput: ~36 t/s wallclock (gleich auf mit dual-GPU, kein pipeline overhead).
#
# Memory breakdown @ 131072 ctx:
#   model     : ~7.0 GiB
#   KV cache  : ~0.85 GiB (SWA 1024×40 + global 131k×8)
#   compute   : ~2.6 GiB (ub=512)
#   sampling  : ~0.1 GiB
#
# Run: bash deploy-gemma4-12b-singlegpu-131k.sh
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
LLAMA_BIN="${LLAMA_BIN:-build/bin/llama-server}"
SLOTS="${SLOTS:-/tmp/llama-slots/}"

MODEL="$MODELS_DIR/gemma-4-12b-it-Q4_K_M.gguf"

mkdir -p "$SLOTS"

export CUDA_VISIBLE_DEVICES=0

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --host 0.0.0.0 --port 8791 \
    -ngl 99 \
    -c 131072 \
    -ub 512 \
    -b 2048 \
    --parallel 1 \
    -fa 1 \
    --cache-type-k ktq2 \
    --cache-type-v vtq3 \
    --backend-sampling \
    --slot-save-path "$SLOTS" \
    --jinja \
    --chat-template-kwargs '{"thinking":false,"enable_thinking":false}' \
    --reasoning off \
    --reasoning-budget 0
