#!/usr/bin/env bash
# Gemma-4-12B + gemma4-assistant MTP draft — speculative decoding (PR #23398)
#
# Target: gemma-4-12b-it-Q4_K_M (main model)
# Draft:  gemma-4-12B-it-qat-assistant-MTP-Q8_0 (443 MB, 4-layer MTP head)
# Spec:   draft-mtp (shared KV cache between target + draft)
#
# ⚠️ RE-MEASURED 2026-06-10 (RTX 2060, build @ turboquant 745f4d2ed): MTP is a NET LOSS
# on real content. The earlier "63% acc / 41.67 t/s / +13%" numbers were NOT reproducible
# (they came from a short/repetitive prompt). Honest re-measurement:
#   baseline (no MTP, llama-bench tg128) →           36.75 t/s
#   MTP q8_0/q8_0 on chat prose          → 15.7% acc, 24.12 t/s  (−34% — DRAFT OVERHEAD WINS)
#   MTP q8_0/q8_0 on code/structured     → 42.8% acc, 35.23 t/s  (−4%)
#
# Why: gemma-4-12B at 36.75 t/s (~27ms/step) is too fast and the 4-layer Q8 draft head too
# weak — the per-step draft decode + sync costs more than the few accepted tokens save.
# Iron law: draft-spec only pays off when target_step_time >> draft_overhead. gemma-4 fails it.
# (Spec is NOT useless on Turing — Qwen3.6-27B + ngram-spec hits 80→176 t/s. gemma-4-12B
# is simply the wrong size/head for draft-MTP.)
#
# RECOMMENDATION: run gemma-4 WITHOUT MTP (36.75 t/s coherent > 24 t/s on chat). This script
# is kept for reproducibility of the measurement, not as a deploy default.
# KTQ/VTQ (2-3 bit) KV additionally breaks the draft (0.3% acc) — see #176.
#
# Key settings (if you still want to run MTP):
#   --draft-p-min 0.0 → REQUIRED. Default 0.75 drops every draft (MTP head conf 0.28-0.51).
#   --draft-max 2     → least-bad. 3/4 add draft overhead without enough extra acceptance.
#
# Lossless (rejection-sampled) regardless of acceptance — the cost is speed, never quality.
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
