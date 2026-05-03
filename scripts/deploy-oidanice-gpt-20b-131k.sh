#!/usr/bin/env bash
# Deploy: gpt-oss-20b-F16 als OidaNice-GPT-20B + 131k ctx (native)
#
# Model: gpt-oss-20b-F16.gguf (12.85 GB, native 131k ctx)
# KV: ktq2 + vtq4 (v8 latest, lossless PPL)
# Layout: dual-GPU ts 16,8 (mehr GPU0/x16 PCIe für TG)
# Context: 131072 (model native)

set -euo pipefail

PORT=8791
MODEL=/home/lance/models/gpt-oss-20b-F16.gguf
LLAMA_BIN=/home/claude/llama-tq/build/bin/llama-server
SLOTS=/home/claude/llama-slots-20b/

mkdir -p "$SLOTS"

# Stop existing
if pgrep -f "llama-server.*--port $PORT" > /dev/null; then
  echo "Stopping existing server on port $PORT..."
  pkill -f "llama-server.*--port $PORT" || true
  sleep 5
fi

echo "=== Deploy: OidaNice-GPT-20B (gpt-oss-20b-F16) + 131k ctx ==="
echo "Model: $MODEL"
echo "Dual-GPU (ts 16,8), parallel 1, ctx 131072"
echo

CUDA_VISIBLE_DEVICES=0,1 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores \
  GGML_MMAP_HUGEPAGE=1 \
  nohup "$LLAMA_BIN" \
  -m "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --flash-attn on \
  -c 262144 -ngl 99 -ts 16,8 --no-mmap --parallel 2 \
  --cache-type-k ktq2 --cache-type-v vtq4 \
  --cache-reuse 25000 \
  --predict 16384 -ub 512 --reasoning off \
  --moe-pin-experts --backend-sampling \
  --slot-save-path "$SLOTS" \
  --anthropic-cache 1 \
  --anthropic-cache-ttl-default 300 \
  --anthropic-cache-max-gb 16 \
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05 --repeat-penalty 1.15 \
  --alias OidaNice-GPT-20B --override-kv general.name=str:OidaNice-GPT-20B \
  > /tmp/llama-server-oidanice-20b.log 2>&1 &

PID=$!
echo "Server started PID=$PID, log: /tmp/llama-server-oidanice-20b.log"

echo "Waiting for server ready..."
for i in $(seq 1 180); do
  if curl -s -m 2 http://localhost:$PORT/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "Server ready after ${i}s"
    break
  fi
  sleep 1
done

echo
echo "=== Deploy complete ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
curl -s http://localhost:$PORT/health
echo
