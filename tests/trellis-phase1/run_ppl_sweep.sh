#!/bin/bash
# Run PPL sweep on Qwen3.5-0.8B for VTQ2_2/3_2/4_2 + baselines.
# Limited chunks (=10) to bound runtime — Viterbi encoder is ~100ms/block.
#
# Output: tests/trellis-phase1/results/run11_ppl_0.8b.csv

set -e

MODEL="${MODEL:-$HOME/models/qwen3.5-0.8b-q8_0.gguf}"
WIKI="${WIKI:-$(pwd)/wikitext-2-raw/wiki.test.raw}"
CHUNKS="${CHUNKS:-10}"
CTX="${CTX:-512}"
BIN="${BIN:-$(pwd)/build-cpu/bin/llama-perplexity}"
OUT="${OUT:-tests/trellis-phase1/results/run11_ppl_0.8b.csv}"

mkdir -p "$(dirname "$OUT")"
echo "config,ppl,ppl_err,time_s,cache_k,cache_v" > "$OUT"

run_one() {
    local ctk="$1"
    local ctv="$2"
    local label="$3"
    echo "=== $label (ctk=$ctk ctv=$ctv) ==="
    local start=$(date +%s)
    local output
    output=$("$BIN" -m "$MODEL" -f "$WIKI" -c "$CTX" -b "$CTX" -ngl 0 \
        --cache-type-k "$ctk" --cache-type-v "$ctv" \
        --no-warmup --chunks "$CHUNKS" 2>&1 | tail -60)
    local elapsed=$(( $(date +%s) - start ))
    # Parse "Final estimate: PPL = 11.4240 +/- 2.29351"
    local ppl=$(echo "$output" | grep "Final estimate: PPL" | awk '{print $5}')
    local err=$(echo "$output" | grep "Final estimate: PPL" | awk '{print $7}')
    echo "  PPL=$ppl err=$err time=${elapsed}s"
    echo "$label,$ppl,$err,$elapsed,$ctk,$ctv" >> "$OUT"
}

run_one f16 f16   "f16_baseline"
run_one f16 vtq2_1 "vtq2_1"
run_one f16 vtq2_2 "vtq2_2"
run_one f16 vtq3_2 "vtq3_2"
run_one f16 vtq4_2 "vtq4_2"

echo "=== Results ==="
cat "$OUT"
