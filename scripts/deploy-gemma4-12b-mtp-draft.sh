#!/usr/bin/env bash
# Gemma-4-12B + gemma4-assistant MTP draft — speculative decoding (PR #23398)
#
# Target: gemma-4-12B-it-Q4_K_M (main model)
# Draft:  Janvitos gemma-4-12B-it-qat-assistant-MTP-Q8_0 (443 MB, 4-layer MTP head)
# Spec:   draft-mtp (shared KV cache between target + draft)
#
# Expected: 1.6-2.0x TG boost vs vanilla (per ik_llama.cpp PR #1744: 2.6-2.98x).
# Single-GPU0, 131k ctx, f16/vtq3 KV.
#
# Run: bash deploy-gemma4-12b-mtp-draft.sh
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
LLAMA_BIN="${LLAMA_BIN:-${HOME}/llama-tq/build/bin/llama-server}"
SLOTS="${SLOTS:-${HOME}/llama-slots-gemma4-mtp/}"

MODEL="$MODELS_DIR/gemma-4-12b-it-Q4_K_M.gguf"
DRAFT="$MODELS_DIR/gemma-4-12B-it-qat-MTP-Q8_0.gguf"

mkdir -p "$SLOTS"

export CUDA_VISIBLE_DEVICES=0

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --model-draft "$DRAFT" \
    --spec-type draft-mtp \
    --draft-max 4 \
    --draft-min 1 \
    --host 0.0.0.0 --port 8791 \
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
