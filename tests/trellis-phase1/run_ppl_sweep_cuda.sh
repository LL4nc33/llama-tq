#!/bin/bash
# Run PPL sweep using CUDA binary with hybrid CPU-V-cache placement.
# -ngl 99 for weights on GPU, -fa on for FA with CPU V-dequant fallback.

set -e
MODEL="${MODEL:-$HOME/models/qwen3.5-0.8b-q8_0.gguf}"
WIKI="${WIKI:-$(pwd)/wikitext-2-raw/wiki.test.raw}"
CHUNKS="${CHUNKS:-10}"
CTX="${CTX:-512}"
BIN="${BIN:-$(pwd)/build-cuda/bin/llama-perplexity}"
OUT="${OUT:-tests/trellis-phase1/results/run12_ppl_0.8b_cuda.csv}"

mkdir -p "$(dirname "$OUT")"
echo "config,ppl,ppl_err,time_s,cache_k,cache_v,fa,ngl" > "$OUT"

run_one() {
    local ctk="$1" ctv="$2" fa="$3" ngl="$4" label="$5"
    echo "=== $label (ctk=$ctk ctv=$ctv fa=$fa ngl=$ngl) ==="
    local start=$(date +%s)
    local output
    output=$("$BIN" -m "$MODEL" -f "$WIKI" -c "$CTX" -b "$CTX" -ngl "$ngl" -fa "$fa" \
        --cache-type-k "$ctk" --cache-type-v "$ctv" --no-warmup --chunks "$CHUNKS" 2>&1 | tail -30)
    local elapsed=$(( $(date +%s) - start ))
    local ppl=$(echo "$output" | grep "Final estimate: PPL" | awk '{print $5}')
    local err=$(echo "$output" | grep "Final estimate: PPL" | awk '{print $7}')
    echo "  PPL=$ppl err=$err time=${elapsed}s"
    echo "$label,$ppl,$err,$elapsed,$ctk,$ctv,$fa,$ngl" >> "$OUT"
}

# GPU baseline
run_one f16 f16    auto 99 "f16_gpu"
run_one f16 vtq2_1 auto 99 "vtq2_1_gpu"  # existing CUDA FA path
# Hybrid GPU+CPU for new types (no CUDA FA yet)
run_one f16 vtq2_2 on   99 "vtq2_2_hybrid"
run_one f16 vtq3_2 on   99 "vtq3_2_hybrid"
run_one f16 vtq4_2 on   99 "vtq4_2_hybrid"

echo "=== Results ==="
cat "$OUT"
