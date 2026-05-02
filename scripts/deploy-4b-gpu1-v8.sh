#!/usr/bin/env bash
# Single-GPU1 deploy: Qwen3.5-4B-Q4_K_M (TurboQuant v8 Sweet Spot)
#
# Triple-Goal Sweet-Spot Sweep 2026-05-02 winner: ktq2 K + vtq2 V (~2.78 bpw avg KV)
# vs old prod (ktq2_1/vtq4_1, 5.0 bpw):
#   PPL: 8.6643 (-0.35%) vs 8.7117 (+0.20%) — 0.55pp BETTER
#   TG:  79.59 t/s        vs 78.70 t/s     — +1.1% faster
#   VRAM: 2.78 bpw         vs 5.0 bpw       — 44% LESS KV-VRAM
#
# Hardware: GPU1, can coexist with Chatterbox TTS + FunctionGemma
# Budget: ~6 GB für LLM (vs 6.9 GB old) — 0.9 GB MORE for TTS!

set -euo pipefail

PORT=8793
MODEL=/home/claude/models/Qwen3.5-4B-Q4_K_M.gguf
LLAMA_BIN=/home/claude/llama-tq/build/bin/llama-server
CTX=${CTX:-100000}

# Stop existing on this port
if pgrep -f "llama-server.*--port $PORT" > /dev/null; then
  echo "Stopping existing server on port $PORT..."
  pkill -f "llama-server.*--port $PORT" || true
  sleep 5
fi

echo "=== TurboQuant v8 Deploy 4B: ktq2+vtq2+ctx${CTX} ==="
echo "GPU1 — coexists with Chatterbox/FunctionGemma"
echo "v8 Sweet Spot: 2.78 bpw avg, PPL -0.35%, TG 79.59 t/s"
echo

CUDA_VISIBLE_DEVICES=1 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores \
  nohup "$LLAMA_BIN" \
  -m "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --flash-attn on \
  -c "$CTX" -ngl 99 --no-mmap --parallel 1 \
  --cache-type-k ktq2 --cache-type-v vtq2 \
  --cache-reuse 1024 \
  --predict 8192 -ub 128 --reasoning off \
  --backend-sampling \
  --temp 0.7 --top-p 0.9 --repeat-penalty 1.1 \
  --override-kv general.name=str:OidaNice-GPT-4B \
  > /tmp/llama-server-4b-v8.log 2>&1 &

PID=$!
echo "Server started PID=$PID, log: /tmp/llama-server-4b-v8.log"

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
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | grep "^1,"
curl -s http://localhost:$PORT/health
echo
