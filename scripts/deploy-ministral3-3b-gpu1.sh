#!/usr/bin/env bash
# Ministral-3-3B on GPU1 — text + vision (mmproj), port 8792
#
# Use case: CRM / Buchhaltung agent (self-hosted). Vision works (mmproj
# supported in fork, unlike gemma-4 gemma4uv). Belege/Rechnungen scannbar.
# GPU0 stays free for llama-tq development.
#
# KV: q4_0 K + vtq2_1 V — Ministral-3 produces gibberish with KTQ K
# (see memory project_ministral_ktq_incompatible), so K stays q4_0.
# ~100 t/s decode, fits 262k ctx single-GPU.
#
# Run: bash deploy-ministral3-3b-gpu1.sh
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
LLAMA_BIN="${LLAMA_BIN:-${HOME}/llama-tq/build/bin/llama-server}"
SLOTS="${SLOTS:-${HOME}/llama-slots-ministral3-gpu1/}"

MODEL="$MODELS_DIR/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
MMPROJ="$MODELS_DIR/Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf"

mkdir -p "$SLOTS"

export CUDA_VISIBLE_DEVICES=1

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --host 0.0.0.0 --port 8792 \
    -ngl 99 \
    -c 131072 \
    --parallel 1 \
    -fa 1 \
    --cache-type-k q4_0 \
    --cache-type-v vtq2_1 \
    --backend-sampling \
    --slot-save-path "$SLOTS" \
    --jinja
