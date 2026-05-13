#!/usr/bin/env bash
# Optimal deploy: Ministral-3-3B Q4_K_M, dual-GPU + Phase 5 + ub=128
#
# Cumulative gains vs pre-Phase-5 single-GPU baseline:
#   PP@2k: ~160 -> 2168 t/s (13.6x)
#   PP@10k: ~50 -> 773 t/s (15x)
#   TG: ~97 t/s (unchanged, decode is per-token serial)
#
# Stack: KTQ2_1 K + VTQ2_1 V (MMA-KTQ inline kernel) + dual-GPU layer-split + ub=128

set -euo pipefail

PORT=8794
MODEL=/models/models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
LLAMA_BIN=/workspace/llama-tq/build/bin/llama-server
SLOTS=/workspace/ministral-3b-slots/

mkdir -p "$SLOTS"

if pgrep -f "llama-server.*--port $PORT" > /dev/null; then
  echo "Stopping existing server on port $PORT..."
  pkill -f "llama-server.*--port $PORT" || true
  sleep 5
fi

echo "=== Dual-GPU Optimal: Ministral-3-3B Q4_K_M + ub=128 ==="

nohup "$LLAMA_BIN" \
  -m "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --flash-attn on \
  -c 32000 -ngl 99 \
  --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
  --tq-protect-layers 12 \
  -ub 128 -b 2048 \
  --parallel 1 \
  --predict 4096 --reasoning off \
  --backend-sampling \
  --slot-save-path "$SLOTS" \
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05 --repeat-penalty 1.1 \
  --alias Ministral-3-3B --override-kv general.name=str:Ministral-3-3B \
  > /tmp/llama-server-ministral-3b.log 2>&1 &

PID=$!
echo "Server started PID=$PID, log: /tmp/llama-server-ministral-3b.log"

for i in $(seq 1 60); do
  if curl -s -m 2 http://localhost:$PORT/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "Server ready after ${i}s"
    break
  fi
  sleep 1
done

nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
curl -s http://localhost:$PORT/health
echo
