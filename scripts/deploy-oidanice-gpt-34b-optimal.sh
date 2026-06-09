#!/usr/bin/env bash
# Optimal deploy: OidaNice-GPT-34B (Qwen3.6-35B-A3B IQ2_XXS), dual-GPU + ub=1024
#
# Discovery 2026-05-13: ub=1024 beats default ub=512 by +13% PP@2k, +13% PP@4k
# while maintaining TG. ub=1536+ regresses (-19% at ctx=16k).
#
# Cumulative gains vs single-GPU baseline (default ub=512):
#   PP@2k: 962 -> 1443 t/s (+50%)
#   PP@4k: 839 -> 1437 t/s (+71%)
#   PP@8k: ~600 -> 1227 t/s (~+105%)
#   PP@16k: — -> 810 t/s (single-GPU was OOM-bound)
#   TG@64: 81 -> 76 t/s (-6%, inter-GPU comm cost)
#
# Stack: KTQ2_1 K + VTQ2_1 V (split-dequant fallback for D=256) + dual-GPU
#        layer-split + ub=1024 + moe-pin-experts + backend-sampling

set -euo pipefail

PORT=8791
MODEL=/models/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
MMPROJ=/models/models/Qwen3.6-35B-A3B-mmproj-F16.gguf
LLAMA_BIN=/workspace/llama-tq/build/bin/llama-server
SLOTS=/workspace/llama-slots-34b/

mkdir -p "$SLOTS"

if pgrep -f "llama-server.*--port $PORT" > /dev/null; then
  echo "Stopping existing server on port $PORT..."
  pkill -f "llama-server.*--port $PORT" || true
  sleep 5
fi

echo "=== Dual-GPU Optimal: OidaNice-GPT-34B (Qwen3.6-35B-A3B IQ2_XXS) ==="
echo "ub=1024 (35B-A3B MoE sweet-spot, +13% vs default ub=512)"
echo

CUDA_VISIBLE_DEVICES=0,1 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores \
  GGML_MMAP_HUGEPAGE=1 \
  nohup "$LLAMA_BIN" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --image-max-tokens 1024 \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --flash-attn on \
  -c 200000 -ngl 99 --no-mmap --parallel 1 \
  --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
  --tq-protect-layers 16 \
  --cache-reuse 25000 \
  --predict 16384 -ub 1024 -b 2048 --reasoning off \
  --moe-pin-experts --backend-sampling \
  --slot-save-path "$SLOTS" \
  --anthropic-cache 1 \
  --anthropic-cache-ttl-default 300 \
  --anthropic-cache-max-gb 16 \
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05 --repeat-penalty 1.15 \
  --alias OidaNice-GPT-34B --override-kv general.name=str:OidaNice-GPT-34B \
  > /tmp/llama-server-oidanice-34b-optimal.log 2>&1 &

PID=$!
echo "Server started PID=$PID, log: /tmp/llama-server-oidanice-34b-optimal.log"

echo "Waiting for server ready (long ctx allocation)..."
for i in $(seq 1 300); do
  if curl -s -m 2 http://localhost:$PORT/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "Server ready after ${i}s"
    break
  fi
  sleep 1
done

echo
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
curl -s http://localhost:$PORT/health
echo
