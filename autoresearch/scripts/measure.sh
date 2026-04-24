#!/usr/bin/env bash
# measure.sh — run PPL + TG for a given K/V cache config on gpu00
# Emits metrics.json with ppl, tg, score.
#
# Usage: measure.sh <K_type> <V_type> <out_dir>
# Env:   MODEL_PATH  (default: /home/lance/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf)
#        WIKI_PATH   (default: wikitext-2-raw/wiki.test.raw)
#        PPL_BASE    (default: 6.7251 — f16/f16 on this model at ctx=2048/5ch)
#        TG_BASE     (default: 72.91 — f16/f16 tg256 tok/s on 2x RTX 2060, FA on)

set -euo pipefail

K="${1:?need K type}"
V="${2:?need V type}"
OUT="${3:?need output dir}"

MODEL_PATH="${MODEL_PATH:-/home/lance/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf}"
WIKI_PATH="${WIKI_PATH:-wikitext-2-raw/wiki.test.raw}"
PPL_BASE="${PPL_BASE:-6.7251}"
TG_BASE="${TG_BASE:-72.91}"

mkdir -p "$OUT"

# PPL measurement
ppl_out=$(./build/bin/llama-perplexity \
    -m "$MODEL_PATH" \
    --cache-type-k "$K" --cache-type-v "$V" \
    -f "$WIKI_PATH" -c 2048 -b 2048 --no-warmup --chunks 5 -ngl 99 2>&1)
ppl=$(echo "$ppl_out" | grep "Final estimate" | awk '{print $5}' || echo "0")

# TG measurement
tg_out=$(./build/bin/llama-bench \
    -m "$MODEL_PATH" \
    -ctk "$K" -ctv "$V" \
    -fa 1 -ngl 99 -n 256 -p 0 -r 2 2>&1)
# llama-bench table row: pick tg256 row, parse tok/s column (second-to-last
# pipe-separated cell, format "NN.NN ± S.SS")
tg=$(echo "$tg_out" | awk -F'|' '/tg256/ {gsub(/[[:space:]]/,"",$(NF-1)); split($(NF-1), a, "±"); print a[1]}' | tail -1)
tg="${tg:-0}"

# Compute deltas + score
ppl_delta=$(python3 -c "print(round(($ppl - $PPL_BASE) / $PPL_BASE * 100, 3))")
tg_slowdown=$(python3 -c "print(round(($TG_BASE - $tg) / $TG_BASE * 100, 3))")
score=$(python3 -c "print(round($ppl_delta + 0.5 * $tg_slowdown, 3))")

cat > "$OUT/metrics.json" <<EOF
{
  "k_type": "$K",
  "v_type": "$V",
  "ppl": $ppl,
  "ppl_baseline": $PPL_BASE,
  "ppl_delta_pct": $ppl_delta,
  "tg_tok_per_sec": $tg,
  "tg_baseline": $TG_BASE,
  "tg_slowdown_pct": $tg_slowdown,
  "score": $score
}
EOF

cat "$OUT/metrics.json"
