#!/bin/bash
# v8 Validation PPL Sweep — runs 8 configs across N models sequentially
# Output: bench/results/v8_ppl_$(date +%Y%m%d_%H%M).tsv
#
# Usage:
#   bash bench/harness/v8_ppl_sweep.sh                # all 8 models
#   bash bench/harness/v8_ppl_sweep.sh 35B-A3B         # single model
#   MODELS="35B-A3B,4B" bash bench/harness/v8_ppl_sweep.sh
#
# Wallclock estimate: ~4-5 min per run × 64 runs ≈ 5-6h total.

set +e

PERP=~/llama-tq/build/bin/llama-perplexity
WIKI=~/llama-tq/wikitext-2-raw/wiki.test.raw
OUT_DIR=~/llama-tq/bench/results
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/v8_ppl_$(date +%Y%m%d_%H%M).tsv"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo -e "model\tk_type\tv_type\tppl\tstatus\twallclock_s\tkv_mib\ttag\tcommit" > "$OUT"

COMMIT_HASH=$(cd ~/llama-tq && git rev-parse --short HEAD)

# Model paths
declare -A MODELS=(
  [35B-A3B]="/home/lance/models/Qwen_Qwen3.6-35B-A3B-IQ2_XXS-bartowski.gguf"
  [4B]="/home/claude/models/Qwen3.5-4B-Q4_K_M.gguf"
  [80B-A3B]="/home/lance/models/Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf"
  [122B]="/home/lance/models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf"
  [27B]="/home/lance/models/Qwen3.6-27B-UD-IQ2_XXS.gguf"
  [Gemma4]="/home/lance/models/gemma-4-26B-A4B-bartowski-IQ2_XXS.gguf"
  [9B]="/home/claude/models/Qwen3.5-9B-Q4_K_M.gguf"
  [Ministral3B]="/home/lance/models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
)

# GPU mapping (GPU1 only for 4B due to TTS-coexist constraint)
declare -A GPU_OVERRIDE=(
  [4B]="1"
)

# Expert-offload for large MoE models
declare -A EXPERT_OFFLOAD=(
  [80B-A3B]="--override-tensor 'blk.1[5-9].ffn_.*_exps.=CPU'"
  [122B]="--override-tensor 'blk.[2-4][0-9].ffn_.*_exps.=CPU'"
)

# Model-specific extra flags
declare -A EXTRA_FLAGS=(
  [Gemma4]="--reasoning none"
)

# Configurations (8 per model)
CONFIGS=(
  "f16:f16:baseline-f16"
  "ktq2:vtq2:v8-default"
  "ktq2:vtq3:v8-quality"
  "ktq3:vtq3:v8-research"
  "ktq4:vtq4:v8-max"
  "ktq2_1:vtq2_2:legacy-2025-prod"
  "ktq2_1:vtq3_1:legacy-aggressive"
  "ktq2_1:vtq4_1:legacy-quality"
)

# Allow user to filter models via $1 or env MODELS
if [ -n "$1" ]; then
  FILTER_MODELS="$1"
elif [ -n "$MODELS" ]; then
  FILTER_MODELS="$MODELS"
else
  FILTER_MODELS="35B-A3B,4B,80B-A3B,122B,27B,Gemma4,9B,Ministral3B"
fi

IFS=',' read -ra MODEL_LIST <<< "$FILTER_MODELS"

for MODEL_NAME in "${MODEL_LIST[@]}"; do
  MODEL_NAME=$(echo "$MODEL_NAME" | xargs) # trim
  MODEL_PATH="${MODELS[$MODEL_NAME]}"
  [ -z "$MODEL_PATH" ] && { echo "Unknown model: $MODEL_NAME (skipping)"; continue; }
  [ ! -f "$MODEL_PATH" ] && { echo "Model file missing: $MODEL_PATH (skipping)"; continue; }

  GPU="${GPU_OVERRIDE[$MODEL_NAME]:-0}"
  OFFLOAD="${EXPERT_OFFLOAD[$MODEL_NAME]:-}"
  EXTRA="${EXTRA_FLAGS[$MODEL_NAME]:-}"

  echo
  echo "######################################################################"
  echo "# Model: $MODEL_NAME (GPU=$GPU, $MODEL_PATH)"
  echo "######################################################################"

  for CFG in "${CONFIGS[@]}"; do
    K=$(echo "$CFG" | cut -d: -f1)
    V=$(echo "$CFG" | cut -d: -f2)
    TAG=$(echo "$CFG" | cut -d: -f3)

    LABEL="${MODEL_NAME}/${K}+${V}"
    echo "=== $LABEL ($TAG) ==="

    START=$(date +%s)
    LOG="$LOG_DIR/v8_${MODEL_NAME}_${K}_${V}.log"

    eval "CUDA_VISIBLE_DEVICES=$GPU OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores \
      \"$PERP\" -m \"$MODEL_PATH\" \
      --cache-type-k \"$K\" --cache-type-v \"$V\" \
      -f \"$WIKI\" -c 512 -b 512 \
      --no-warmup --chunks 3 -ngl 99 \
      --flash-attn on $OFFLOAD $EXTRA \
      > \"$LOG\" 2>&1"
    EXIT=$?
    END=$(date +%s)
    WC=$((END - START))

    PPL=$(grep -oP 'Final estimate: PPL = \K[0-9.]+' "$LOG" | head -1)
    KV_MIB=$(grep -oP 'KV self size\s*=\s*\K[0-9.]+' "$LOG" | head -1)
    [ -z "$PPL" ] && PPL="-"
    [ -z "$KV_MIB" ] && KV_MIB="-"
    STATUS="ok"
    [ "$EXIT" -ne 0 ] && STATUS="error_$EXIT"
    [ "$PPL" = "-" ] && STATUS="${STATUS}_noppl"

    echo -e "${MODEL_NAME}\t${K}\t${V}\t${PPL}\t${STATUS}\t${WC}\t${KV_MIB}\t${TAG}\t${COMMIT_HASH}" | tee -a "$OUT"

    sleep 5  # thermal/VRAM settle
  done
done

echo
echo "=== DONE ==="
echo "Results: $OUT"
cat "$OUT"
