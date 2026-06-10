#!/usr/bin/env bash
# Gemma-4-12B + gemma4-assistant MTP draft — speculative decoding (PR #23398)
#
# Target: gemma-4-12b-it-Q4_K_M (main model)
# Draft:  gemma-4-12B-it-qat-assistant-MTP-Q8_0 (443 MB, 4-layer MTP head)
# Spec:   draft-mtp (shared KV cache between target + draft)
#
# Tuning sweep (2026-06-10, RTX 2060, BST code prompt, 300 tok):
#   q8_0/q8_0  draft-max 2 min 2 → 63% acc, 41.67 t/s  ← BEST (+13% vs 37 baseline)
#   f16/f16    draft-max 2 min 1 → 58% acc, 39.70 t/s  (+7%)
#   f16/f16    draft-max 3 min 2 → 44% acc, 35.06 t/s  (worse — more draft overhead)
#   ktq2/vtq3  (any, +protect)   → 0.3% acc, 17.5 t/s  ← DEAD (2-3 bit KV too coarse
#                                                         for the MTP draft attention;
#                                                         tq-protect-layers does not
#                                                         cover gemma-4 SWA global layers)
#
# Key settings:
#   --draft-p-min 0.0 → REQUIRED. Default 0.75 drops every draft (MTP head conf 0.28-0.51).
#   --draft-max 2     → sweet spot. 3/4 add draft overhead without enough extra acceptance.
#   q8_0 KV           → best of both: 8-bit is precise enough for the draft AND saves VRAM
#                       vs f16. KTQ/VTQ (2-3 bit) breaks the draft — separate code task to
#                       give the draft its own f16 cache for shared layers (#176).
#
# Real +13% speedup on code (41.67 vs 37), modest on prose (acceptance drops with
# less predictable text). Lossless (rejection-sampled).
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
    --draft-min 2 \
    --draft-p-min 0.0 \
    --host 0.0.0.0 --port 8791 \
    -ngl 99 \
    -c 131072 \
    -ub 512 \
    -b 2048 \
    --parallel 1 \
    -fa 1 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --backend-sampling \
    --slot-save-path "$SLOTS" \
    --jinja \
    --chat-template-kwargs '{"thinking":false,"enable_thinking":false}' \
    --reasoning off \
    --reasoning-budget 0
