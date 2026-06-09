#!/usr/bin/env bash
# Gemma-4-12B + gemma4-assistant MTP draft — speculative decoding (PR #23398)
#
# Target: gemma-4-12b-it-Q4_K_M (main model)
# Draft:  gemma-4-12B-it-qat-assistant-MTP-Q8_0 (443 MB, 4-layer MTP head)
# Spec:   draft-mtp (shared KV cache between target + draft)
#
# Tuning (2026-06-10, RTX 2060, code prompt):
#   --draft-max 2     → 58% acceptance, 38.1 t/s (best; max 3 = 35, max 4 = 32)
#   --draft-p-min 0.0 → REQUIRED. Default 0.75 drops every draft (MTP head
#                       confidence is 0.28-0.51). Rejection-sampling keeps it lossless.
#   f16 K + f16 V     → REQUIRED. The draft reads K/V from the shared cache;
#                       vtq3/ktq2 quantization corrupts draft attention → 0.5% accept.
#                       This trades KV size for draft acceptance.
#
# Speedup is modest (+3% on code, ~0% on prose) because the 4-layer MTP draft
# overhead is relatively expensive on a dense 12B target + RTX 2060. The point
# is that MTP works correctly and losslessly; f16-vs-quantized-KV is the big lever.
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
    --draft-max 2 \
    --draft-min 1 \
    --draft-p-min 0.0 \
    --host 0.0.0.0 --port 8791 \
    -ngl 99 \
    -c 131072 \
    -ub 512 \
    -b 2048 \
    --parallel 1 \
    -fa 1 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --backend-sampling \
    --slot-save-path "$SLOTS" \
    --jinja \
    --chat-template-kwargs '{"thinking":false,"enable_thinking":false}' \
    --reasoning off \
    --reasoning-budget 0
