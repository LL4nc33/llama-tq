#!/usr/bin/env bash
# Single-GPU0 deploy: Qwen3.6-35B-A3B-IQ2_XXS bartowski + mmproj + 100k ctx (v8)
#
# TurboQuant v8 Triple-Goal Winner (2026-05-02 PPL+Speed Sweep):
#   ktq2 (K) + vtq3 (V, = vtq3_v8 NEW) — 3.56 bpw avg KV
#   PPL: -0.03% vs f16/f16 (essentially lossless)
#   TG: 86.61 t/s @ 100k+mmproj (gemessen, +21% vs old prod)
#   PP: 1196 t/s
#   VRAM: ~750 MB für KV @ 100k auf 35B-A3B
#
# Replaces legacy deploy-35b-singlegpu-100k.sh which used ktq2_1/vtq2_1
# (3.0 bpw, +3.85% PPL drift, 85.66 TG t/s).
#
# v8 Improvements vs legacy:
#   Accuracy: -3.88pp PPL drift improvement (lossless)
#   Speed:    +1.1% TG, +8.1% PP
#   VRAM:     +0.56 bpw (acceptable cost for lossless quality)

set -euo pipefail

PORT=8791
MODEL=/home/lance/models/Qwen_Qwen3.6-35B-A3B-IQ2_XXS-bartowski.gguf
MMPROJ=/home/lance/models/Qwen3.6-35B-A3B-mmproj-F16.gguf
LLAMA_BIN=/home/claude/llama-tq/build/bin/llama-server
SLOTS=/home/claude/llama-slots/

mkdir -p "$SLOTS"

# Stop existing
if pgrep -f "llama-server.*--port $PORT" > /dev/null; then
  echo "Stopping existing server on port $PORT..."
  pkill -f "llama-server.*--port $PORT" || true
  sleep 5
fi

echo "=== TurboQuant v8 Deploy: ktq2/vtq3+100k+mmproj ==="
echo "Model: $MODEL"
echo "Mmproj: $MMPROJ"
echo "Single-GPU0, parallel 1, ctx 100k"
echo "v8 quality tier: vtq3_v8 (3.625 bpw trellis-3bit + 2 outliers)"
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
  > /tmp/llama-server-35b-100k-v8.log 2>&1 &

PID=$!
echo "Server started PID=$PID, log: /tmp/llama-server-35b-100k-v8.log"

# Wait for ready
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
