#!/bin/bash
# TG throughput bench for O(1) bit-window decoder validation.
# Baseline: f16 V ~174 tok/s on 0.8B (Turing sm_75).
# Old walking decoder: 7.3 tok/s (26x slower).
# Target: 50-150 tok/s = proves O(1) works.

set -e
MODEL="${MODEL:-$HOME/models/qwen3.5-0.8b-q8_0.gguf}"
BIN="${BIN:-$(pwd)/build-cuda/bin/llama-bench}"
OUT="${OUT:-tests/trellis-phase1/results/run23_tg_o1decoder.csv}"

mkdir -p "$(dirname "$OUT")"

echo "=== O(1) decoder TG bench ==="
echo "Model: $MODEL"
echo "Binary: $BIN"
echo ""

# Configs: f16 baseline + VTQ_2 triplet (all with f16 K)
# -ctk f16 keeps K-cache consistent, isolates V-decoder overhead
"$BIN" -m "$MODEL" \
    -ctk f16 -ctv f16 \
    -ctk f16 -ctv vtq2_2 \
    -ctk f16 -ctv vtq3_2 \
    -ctk f16 -ctv vtq4_2 \
    -fa 1 -ngl 99 -r 3 \
    -p 0 -n 128 \
    -o csv 2>&1 | tee "$OUT"

echo ""
echo "=== Key numbers ==="
grep -E 'tg128|model_filename' "$OUT" | head -10
