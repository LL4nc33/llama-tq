#!/usr/bin/env bash
# Single-GPU0 deploy: Qwen3.6-35B-A3B (non-MTP) UD-IQ2_XXS + mmproj + 100k ctx
#
# Optimale config nach Phase 35-36 messung (2026-06-08):
#   non-MTP base + ktq2/vtq3 single-GPU0 → 82.9 t/s tg128
#
# vs MTP variant (77.2 t/s baseline): MTP-overhead +5.4 t/s kosten, spec-decoding
# auf consumer 12GB IQ2 frisst seine eigenen gewinne wegen p_min filter +
# compute-buffer OOM. Non-MTP ist auf dieser hardware-klasse die schnellere wahl.
#
# Wer MTP-spec testen will: nutze --model …-MTP-… variant + dual-GPU (ts 12,12)
# + ENABLE_SPEC=1 + ngram-cache dm=8 → bringt 3.8x auf repeat-prompts, ~1.0x
# auf creative. Lossless garantiert.

set -euo pipefail

PORT=8791
MODEL=${MODEL:-/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf}
MMPROJ=${MMPROJ:-/models/Qwen3.6-35B-A3B-mmproj-F16.gguf}
LLAMA_BIN=${LLAMA_BIN:-$HOME/llama-tq-mtp-fusion/build-cuda/bin/llama-server}
SLOTS=${SLOTS:-$HOME/llama-slots/}

mkdir -p "$SLOTS"

# Stop existing
if pgrep -f "llama-server.*--port $PORT" > /dev/null; then
  echo "Stopping existing server on port $PORT..."
  pkill -f "llama-server.*--port $PORT" || true
  sleep 5
fi

echo "=== Deploy: Qwen3.6-35B-A3B non-MTP ktq2/vtq3 single-GPU0 100k ==="
echo "Model: $MODEL"
echo "Mmproj: $MMPROJ"
echo "Expected: tg128 ~82.9 t/s, ktq2/vtq3 V-cache 3.56 bpw lossless"
echo

CUDA_VISIBLE_DEVICES=0 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores \
  nohup "$LLAMA_BIN" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --flash-attn on \
  -c 100000 -ngl 99 --no-mmap --parallel 1 \
  --cache-type-k ktq2 --cache-type-v vtq3 \
  --cache-reuse 25000 \
  --predict 16384 -ub 64 --reasoning off \
  --moe-pin-experts --backend-sampling \
  --slot-save-path "$SLOTS" \
  --anthropic-cache 1 \
  --anthropic-cache-ttl-default 300 \
  --anthropic-cache-max-gb 32 \
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05 --repeat-penalty 1.15 \
  --override-kv general.name=str:OidaNice-GPT-34B \
  > /tmp/llama-server-35b-100k.log 2>&1 &

PID=$!
echo "Server started PID=$PID, log: /tmp/llama-server-35b-100k.log"

echo "Waiting for server ready..."
for i in $(seq 1 120); do
  if curl -s -m 2 http://localhost:$PORT/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "Server ready after ${i}s"
    break
  fi
  sleep 1
done

echo
echo "=== Deploy complete ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -1
curl -s http://localhost:$PORT/health
echo
