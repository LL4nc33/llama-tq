#!/usr/bin/env bash
# TurboQuant Nano throughput bench — single & multi-GPU.
#
# Runs llama-bench across (ctk, ctv) pairs and emits a CSV + markdown table.
# Isolates single-GPU runs via CUDA_VISIBLE_DEVICES=0.
#
# Usage:
#   bench_tq.sh <model.gguf> [out_dir]
#
# Env:
#   PP=1024    prompt-processing length
#   TG=128     token-generation length
#   SKIP_MULTI=1   skip multi-GPU pass (single only)
#   SKIP_SINGLE=1  skip single-GPU pass (multi only)
#   NGL=99     layers offloaded

set -euo pipefail

MODEL="${1:?usage: bench_tq.sh <model.gguf> [out_dir]}"
OUT="${2:-$(pwd)/bench-$(date +%Y%m%d-%H%M)}"
BIN="${BIN:-$(dirname "$0")/../build-cuda/bin/llama-bench}"
PP="${PP:-1024}"
TG="${TG:-128}"
NGL="${NGL:-99}"

if [[ ! -x "$BIN" ]]; then
    echo "llama-bench not found at $BIN — set BIN= or build first" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "model not found: $MODEL" >&2
    exit 1
fi

mkdir -p "$OUT"
MODEL_NAME="$(basename "$MODEL" .gguf)"
CSV="$OUT/${MODEL_NAME}.csv"
MD="$OUT/${MODEL_NAME}.md"

# Config matrix: (ctk, ctv, label)
# Kept short enough to run in a reasonable time; extend as needed.
CONFIGS=(
    "f16 f16 baseline"
    "q8_0 vtq2_1 v1_q8_vtq2_1"
    "q4_0 vtq2_1 v1_q4_vtq2_1"
    "q8_0 vtq3_1 v1_q8_vtq3_1"
    "f16 vtq2_2 v2_f16_vtq2_2"
    "f16 vtq3_2 v2_f16_vtq3_2"
    "f16 vtq4_2 v2_f16_vtq4_2"
)

echo "gpus,ctk,ctv,label,pp_tok_s,tg_tok_s" > "$CSV"

run_config() {
    local gpus="$1" ctk="$2" ctv="$3" label="$4"
    local cuda_env=""
    if [[ "$gpus" == "1" ]]; then cuda_env="CUDA_VISIBLE_DEVICES=0"; fi

    echo "=== [$gpus-gpu] ctk=$ctk ctv=$ctv ($label) ==="
    local log="$OUT/${MODEL_NAME}_${gpus}gpu_${label}.log"
    # llama-bench doesn't accept custom ctk/ctv on some builds; use CLI flags the
    # current binary supports. Fall back to -ctk / -ctv.
    eval "$cuda_env \"$BIN\" -m \"$MODEL\" -ngl $NGL -fa 1 -ctk $ctk -ctv $ctv \
        -p $PP -n $TG -r 2" > "$log" 2>&1 || true

    # Extract "pp1024" and "tg128" lines from llama-bench markdown output.
    local pp tg
    pp=$(grep -E "pp${PP}[[:space:]]" "$log" | awk -F'|' '{print $(NF-1)}' | tr -d ' ' | head -1)
    tg=$(grep -E "tg${TG}[[:space:]]" "$log" | awk -F'|' '{print $(NF-1)}' | tr -d ' ' | head -1)
    pp="${pp:-FAIL}"
    tg="${tg:-FAIL}"
    echo "  PP${PP}=$pp tg/s  TG${TG}=$tg tg/s"
    echo "$gpus,$ctk,$ctv,$label,$pp,$tg" >> "$CSV"
}

if [[ -z "${SKIP_SINGLE:-}" ]]; then
    echo "--- Single-GPU pass (CUDA_VISIBLE_DEVICES=0) ---"
    for cfg in "${CONFIGS[@]}"; do
        read -r ctk ctv label <<< "$cfg"
        run_config 1 "$ctk" "$ctv" "$label"
    done
fi

if [[ -z "${SKIP_MULTI:-}" ]]; then
    echo "--- Multi-GPU pass (all visible) ---"
    for cfg in "${CONFIGS[@]}"; do
        read -r ctk ctv label <<< "$cfg"
        run_config N "$ctk" "$ctv" "$label"
    done
fi

# Generate markdown summary.
{
    echo "# TurboQuant Nano — $MODEL_NAME"
    echo ""
    echo "\`pp$PP\` / \`tg$TG\`, NGL=$NGL, FA=1."
    echo ""
    echo "| GPUs | K | V | Config | PP$PP tok/s | TG$TG tok/s |"
    echo "|---|---|---|---|---:|---:|"
    tail -n +2 "$CSV" | awk -F',' '{
        printf "| %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5, $6
    }'
} > "$MD"

echo ""
echo "=== Done ==="
echo "CSV: $CSV"
echo "MD:  $MD"
cat "$MD"
