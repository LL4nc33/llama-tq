#!/usr/bin/env bash
# Single-GPU0 deploy: Qwen3.6-35B-A3B-IQ2_XXS bartowski + mmproj + 100k ctx
#
# Triple-Goal Winner (2026-05-02 PPL-Sweep):
#   ktq2_1 (K) + vtq2_1 (V) — 2.78 bpw avg KV
#   PPL: +3.85% vs f16/f16 (BESSER als q5_1+vtq2_1 prod = +4.95%)
#   TG: 71.81 t/s @ 100k+mmproj (gemessen)
#   VRAM: 11699/11833 MiB (134 MB headroom)
#
# Image-OOM fix (2026-05-02 08:08):
#   --no-mmproj-offload: mmproj weights run on CPU (~857 MB freed on GPU0)
#   --image-max-tokens 1024: cap image tokens to prevent 1472x1472 OOM crashes
#   Required because VRAM headroom (134 MB) is too tight for image-encoder
#   compute buffers (~250 MB needed for full-size images)
#
# CPU-spinning fix (2026-05-02 08:38):
#   OMP_WAIT_POLICY=passive: was 'active' (constant 1200% CPU spinning idle).
#   With mmproj on CPU, active-spinning was burning all 12 vCPUs even idle.
#   Passive lets threads sleep when idle, GPU-bound prefill barely loses speed.
#
# Reasoning: KTQ+VTQ kombiniert hatte auf 0.8B +287% PPL drift,
# aber auf 35B-A3B (64 attention layers) ist die drift nur +3.85%
# und damit besser als jede stock-q5_1/q4_1 K + VTQ V Kombination.

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

echo "=== Triple-Goal Deploy: ktq2_1+vtq2_1+100k+mmproj ==="
echo "Model: $MODEL"
echo "Mmproj: $MMPROJ"
echo "Single-GPU0, parallel 1, ctx 100k"
echo

CUDA_VISIBLE_DEVICES=0 OMP_WAIT_POLICY=passive OMP_PROC_BIND=close OMP_PLACES=cores \
  nohup "$LLAMA_BIN" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --no-mmproj-offload \
  --image-max-tokens 1024 \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --flash-attn on \
  -c 100000 -ngl 99 --no-mmap --parallel 1 \
  --cache-type-k ktq2_1 --cache-type-v vtq2_2 \
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
