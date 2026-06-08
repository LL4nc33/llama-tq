#!/usr/bin/env bash
# Dual-GPU deploy: Qwen3.6-35B-A3B bartowski + spec + 65k ctx
#
# Phase 37 final config (2026-06-08, 35B-A3B-IQ2 dual-GPU):
#   bartowski quant + ktq2/vtq3 + ngram-cache spec + dual-GPU 12,12
#   = 79 t/s creative, 80 t/s code, 180 t/s repeat (2.53x)
#
# vs prior config (unsloth MTP + dual + ngram dm=8):
#   71 t/s creative, 96 t/s repeat
#
# Improvements:
#   + bartowski quant (5% faster than unsloth UD)
#   + non-MTP (saves 5 t/s of MTP-head overhead)
#   + ngram-cache spec (2.53x repeat boost, lossless)
#   + backend-sampling (+1.5 t/s)
#
# VRAM at 65k ctx: GPU0 6.5GB + GPU1 7.8GB → headroom für mmproj/größere ctx

set -euo pipefail

PORT=${PORT:-8791}
MODEL=${MODEL:-/home/lance/models/Qwen_Qwen3.6-35B-A3B-IQ2_XXS-bartowski.gguf}
LLAMA_BIN=${LLAMA_BIN:-/home/claude/llama-tq-mtp-fusion/build-cuda/bin/llama-server}
SLOTS=${SLOTS:-/home/claude/llama-slots/}
CTX=${CTX:-65536}

mkdir -p "$SLOTS"

if pgrep -f "llama-server.*--port $PORT" > /dev/null; then
  echo "Stopping existing server on port $PORT..."
  pkill -f "llama-server.*--port $PORT" || true
  sleep 5
fi

# Spec-decoding default-on. Bartowski (non-MTP) + ngram-cache.
SPEC_ARGS=()
if [[ "${ENABLE_SPEC:-1}" == "1" ]]; then
  SPEC_ARGS+=(--spec-type ngram-cache --draft-max 8 --draft-min 4)
fi

# MMProj optional: ENABLE_MMPROJ=1 + MMPROJ=...
MMPROJ_ARGS=()
if [[ "${ENABLE_MMPROJ:-0}" == "1" ]] && [[ -n "${MMPROJ:-}" ]] && [[ -r "${MMPROJ}" ]]; then
  MMPROJ_ARGS+=(--mmproj "$MMPROJ")
fi

echo "=== Deploy: Qwen3.6-35B-A3B bartowski dual-GPU 12,12 ${CTX}ctx + spec ==="
echo "Model: $MODEL"
echo "Spec: ${SPEC_ARGS[*]:-(none)}"
echo "Expected: 79-80 t/s creative, 180 t/s repeat, lossless"
echo

CUDA_VISIBLE_DEVICES=0,1 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores \
  nohup "$LLAMA_BIN" \
  -m "$MODEL" \
  "${MMPROJ_ARGS[@]}" \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --flash-attn on \
  -c "$CTX" -ngl 99 -ts 12,12 --parallel 1 \
  --cache-type-k ktq2 --cache-type-v vtq3 \
  --cache-reuse 25000 \
  --reasoning off \
  --moe-pin-experts --backend-sampling \
  "${SPEC_ARGS[@]}" \
  --slot-save-path "$SLOTS" \
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05 --repeat-penalty 1.15 \
  --override-kv general.name=str:OidaNice-GPT-34B \
  > /tmp/llama-server-35b-dualgpu.log 2>&1 &

PID=$!
echo "Server started PID=$PID"

for i in $(seq 1 120); do
  if curl -s -m 2 http://localhost:$PORT/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "Server ready after ${i}s"
    break
  fi
  sleep 1
done

echo
echo "=== Deploy complete ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
curl -s http://localhost:$PORT/health
echo
