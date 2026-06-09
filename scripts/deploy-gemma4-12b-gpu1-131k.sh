#!/usr/bin/env bash
# Gemma-4-12B on GPU1 — 131k ctx, text-only (no mmproj yet), port 8792
#
# Use case: CRM / Buchhaltung agent (self-hosted), text-first.
# GPU0 stays free for llama-tq development + smoke tests.
# mmproj (vision) pending the gemma4uv unified-projector port (task #169).
#
# VRAM: ~11.0 GB / 12 GB on GPU1 (f16 K + vtq3 V, gemma-4 SWA keeps KV small).
# Throughput: ~37 t/s wallclock.
#
# Run: bash deploy-gemma4-12b-gpu1-131k.sh
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
LLAMA_BIN="${LLAMA_BIN:-build/bin/llama-server}"
SLOTS="${SLOTS:-/tmp/llama-slots-gpu1/}"

MODEL="$MODELS_DIR/gemma-4-12b-it-Q4_K_M.gguf"

mkdir -p "$SLOTS"

export CUDA_VISIBLE_DEVICES=1

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --host 0.0.0.0 --port 8792 \
    -ngl 99 \
    -c 131072 \
    -ub 512 \
    -b 2048 \
    --parallel 1 \
    -fa 1 \
    --cache-type-k f16 \
    --cache-type-v vtq3 \
    --backend-sampling \
    --slot-save-path "$SLOTS" \
    --jinja \
    --chat-template-kwargs '{"thinking":false,"enable_thinking":false}' \
    --reasoning off \
    --reasoning-budget 0
