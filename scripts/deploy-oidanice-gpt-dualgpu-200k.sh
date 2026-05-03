#!/usr/bin/env bash
# Dual-GPU deploy: OidaNice-GPT-34B-F16 + mmproj + 200k ctx, parallel 1
#
# Model: OidaNice-GPT-34B-F16.gguf (FrankenMoE 35B-A3B base, F16 experts)
# Layout: ts 16,8 (split layers across GPU0 + GPU1)
# Ctx: 200k (was 100k on single-GPU; doubled with dual-GPU VRAM)

set -euo pipefail

PORT=8791
MODEL=/home/lance/models/OidaNice-GPT-34B-F16.gguf
MMPROJ=/home/lance/models/OidaNice-GPT-34B-mmproj-F16.gguf
LLAMA_BIN=/home/claude/llama-tq/build/bin/llama-server
SLOTS=/home/claude/llama-slots/

mkdir -p "$SLOTS"

# Stop existing
if pgrep -f "llama-server.*--port $PORT" > /dev/null; then
  echo "Stopping existing server on port $PORT..."
  pkill -f "llama-server.*--port $PORT" || true
  sleep 5
fi

echo "=== Dual-GPU Deploy: OidaNice-GPT-34B-F16 + 200k ctx ==="
echo "Model: $MODEL"
echo "Mmproj: $MMPROJ"
echo "Dual-GPU (ts 16,8), parallel 1, ctx 200k"
echo

CUDA_VISIBLE_DEVICES=0,1 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores \
  GGML_MMAP_HUGEPAGE=1 \
  nohup "$LLAMA_BIN" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --image-max-tokens 1024 \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --flash-attn on \
  -c 200000 -ngl 99 -ts 16,8 --no-mmap --parallel 1 \
  --cache-type-k ktq2 --cache-type-v vtq4 \
  --cache-reuse 25000 \
  --predict 16384 -ub 512 --reasoning off \
  --moe-pin-experts --backend-sampling \
  --slot-save-path "$SLOTS" \
  --anthropic-cache 1 \
  --anthropic-cache-ttl-default 300 \
  --anthropic-cache-max-gb 16 \
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05 --repeat-penalty 1.15 \
  --override-kv general.name=str:OidaNice-GPT-34B \
  > /tmp/llama-server-oidanice-dualgpu.log 2>&1 &

PID=$!
echo "Server started PID=$PID, log: /tmp/llama-server-oidanice-dualgpu.log"

# Wait for ready (longer timeout for 200k ctx allocation + F16 model load)
echo "Waiting for server ready..."
for i in $(seq 1 240); do
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
