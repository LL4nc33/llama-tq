#!/usr/bin/env bash
# Gemma-4-12B KV cache combo sweep
# Goal: find best ktq*/vtq* combo for dual-2060 12GB + 200k ctx target
#
# Outputs: TG t/s + PP@4k t/s + alloc-check für ctx=200k
#
# Run: ./sweep-gemma4-12b-kv.sh > sweep-out.txt 2>&1
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
LLAMA_BIN="${LLAMA_BIN:-${HOME}/llama-tq-mtp-fusion/build/bin}"
MODEL="$MODELS_DIR/gemma-4-12b-it-Q4_K_M.gguf"
MMPROJ="$MODELS_DIR/gemma-4-12b-mmproj-F16.gguf"

# Test small bench first (ctx=4k, fits easily), then alloc-only probe at 200k
COMBOS=(
  "f16 f16"
  "ktq2 vtq2"
  "ktq2 vtq3"
  "ktq3 vtq3"
)

echo "=== Gemma-4-12B KV pareto sweep ==="
echo "Model: $MODEL"
echo "Date: $(date)"
echo

# Stage 1 — bench TG/PP at 4k ctx (real run, fits everywhere)
echo "## Stage 1: bench@4k ctx, dual-GPU layer-split (-ts 1,1)"
for combo in "${COMBOS[@]}"; do
  read -r kt vt <<<"$combo"
  echo "--- ktq=$kt vtq=$vt ---"
  "$LLAMA_BIN/llama-bench" \
    -m "$MODEL" \
    --cache-type-k "$kt" --cache-type-v "$vt" \
    -ngl 99 -fa 1 -ts 1,1 \
    -p 4096 -n 128 -r 1 \
    2>&1 | grep -E '^\| ' | tail -3 || echo "  FAILED"
  echo
done

# Stage 2 — alloc-only probe at 200k ctx (load + bail before generation)
echo "## Stage 2: alloc-only probe @200k ctx, dual-GPU layer-split"
for combo in "${COMBOS[@]}"; do
  read -r kt vt <<<"$combo"
  echo "--- ktq=$kt vtq=$vt ---"
  timeout 60 "$LLAMA_BIN/llama-bench" \
    -m "$MODEL" \
    --cache-type-k "$kt" --cache-type-v "$vt" \
    -ngl 99 -fa 1 -ts 1,1 \
    -ctk 200000 -p 0 -n 1 -r 1 \
    2>&1 | grep -E '^\| |buffer size|out of memory|alloc' | tail -5 || echo "  FAILED/OOM"
  echo
done

echo "=== done ==="
