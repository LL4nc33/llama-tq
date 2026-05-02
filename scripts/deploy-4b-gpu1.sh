#!/usr/bin/env bash
# Single-GPU1 deploy: Qwen3.5-4B-Q4_K_M (Triple-Goal config)
#
# PPL-Sweep 2026-05-02 winner: ktq2_1 K + vtq4_1 V (~4.0 bpw KV)
# PPL drift: +0.20% vs f16/f16 (essentiell lossless)
# Hardware: GPU1, sharing with Chatterbox + FunctionGemma (~5GB blocked)
# Budget: ~6.9 GB für LLM

set -euo pipefail

PORT=8793
MODEL=/home/claude/models/Qwen3.5-4B-Q4_K_M.gguf
LLAMA_BIN=/home/claude/llama-tq/build/bin/llama-server
CTX=${CTX:-65536}

# Stop existing on this port
if pgrep -f "llama-server.*--port $PORT" > /dev/null; then
  echo "Stopping existing server on port $PORT..."
  pkill -f "llama-server.*--port $PORT" || true
  sleep 5
fi

echo "=== Triple-Goal Deploy 4B: ktq2_1+vtq4_1+ctx${CTX} ==="
echo "GPU1, sharing with Chatterbox/FunctionGemma"
echo

CUDA_VISIBLE_DEVICES=1 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores \
  nohup "$LLAMA_BIN" \
  -m "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --flash-attn on \
  -c "$CTX" -ngl 99 --no-mmap --parallel 1 \
  --cache-type-k ktq2_1 --cache-type-v vtq4_1 \
  --cache-reuse 1024 \
  --predict 8192 -ub 128 --reasoning off \
  --backend-sampling \
  --temp 0.7 --top-p 0.9 --repeat-penalty 1.1 \
  --override-kv general.name=str:OidaNice-GPT-4B \
  > /tmp/llama-server-4b.log 2>&1 &

PID=$!
echo "Server started PID=$PID, log: /tmp/llama-server-4b.log"

echo "Waiting for server ready..."
for i in $(seq 1 60); do
  if curl -s -m 2 http://localhost:$PORT/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "Server ready after ${i}s"
    break
  fi
  sleep 1
done

echo
echo "=== Deploy complete ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | head -2 | tail -1
curl -s http://localhost:$PORT/health
echo
