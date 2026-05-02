#!/usr/bin/env bash
# Single-GPU0 deploy: Qwen3.6-35B-A3B-IQ2_XXS bartowski + mmproj + 100k ctx (v8)
#
# TurboQuant v8 Sweet Spot (2026-05-02 Triple-Goal Sweep):
#   ktq2 (K) + vtq2 (V, = vtq2_2 trellis) — 2.78 bpw avg KV
#   PPL: 7.1807 (-0.33% vs f16/f16)
#   TG: 86.37 t/s @ ctx=2048 measured (+0.66% vs current prod 85.80)
#   PP: 1195.31 t/s (+7.5% vs current prod 1111.59)
#   VRAM: ~360 MB für KV @ 100k auf 35B-A3B
#
# Replaces legacy ktq2_1/vtq2_1 (3.0 bpw, +3.85% PPL drift, 85.80 TG).
#
# v8 Sweet Spot Improvements vs legacy:
#   Accuracy: -4.18pp PPL drift improvement (was +3.85%, now -0.33%)
#   Speed:    +0.66% TG, +7.5% PP
#   VRAM:     -7% bpw (3.0 → 2.78)

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

echo "=== TurboQuant v8 Sweet-Spot Deploy: ktq2/vtq2+100k+mmproj ==="
echo "Model: $MODEL"
echo "Mmproj: $MMPROJ"
echo "Single-GPU0, parallel 1, ctx 100k"
echo "v8 Sweet Spot: 2.78 bpw avg, PPL -0.33%, TG 86.37 t/s"
echo

CUDA_VISIBLE_DEVICES=0 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores \
  nohup "$LLAMA_BIN" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --flash-attn on \
  -c 100000 -ngl 99 --no-mmap --parallel 1 \
  --cache-type-k ktq2 --cache-type-v vtq2 \
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
