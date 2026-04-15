# Tensor Parallelism + TurboQuant KV-Cache (TP+TQ)

> First implementation of Tensor Parallelism with quantized KV-cache on consumer GPUs.

## Overview

llama-tq enables simultaneous use of Tensor Parallelism (TP) and TurboQuant (TQ) KV-cache
quantization. The upstream llama.cpp implementation blocks this combination — we replace
the block with proper validation, making it possible to run large models across multiple
GPUs while keeping the KV-cache compressed to 3.5–5.5 bits per weight.

## Why This Works

AllReduce — the synchronization primitive for TP — operates exclusively on **activations**
(f16/f32 tensors after matrix multiplications). It never touches the KV-cache. Each GPU
stores and dequantizes its own subset of attention heads independently:

```
GPU 0: Q_local × dequant(K_local_TQ) → partial_attn → wo_local → AllReduce ─┐
GPU 1: Q_local × dequant(K_local_TQ) → partial_attn → wo_local → AllReduce ─┘→ full output
```

The KV-cache is partitioned by heads, not by elements within a head. Since TQ operates
on 32-element blocks and head dimensions are always ≥64 (multiples of 32), TQ block
boundaries always align with head boundaries. No TQ block is ever split across GPUs.

## Mathematical Correctness

For weight matrix W split into row-shards W_0, W_1 and input Z split into column-shards Z_0, Z_1:

```
W^T × Z = W_0^T × Z_0 + W_1^T × Z_1 = AllReduce(partial_0, partial_1)
```

This holds exactly in f32. The NCCL AllReduce uses BF16 compression for large tensors
(>32K elements), introducing ≈0.4% rounding — negligible compared to TQ quantization
noise (≈3–10% depending on bit rate).

Each GPU's TQ dequantization is self-contained: the RHT seed, sign bits, codebook lookup,
and norm correction are all per-block operations with no cross-GPU dependencies.

**Note**: TP+TQ results are not bit-identical to single-GPU+TQ because TQ block indices
are local per GPU. Both produce correct results within the expected noise distribution.

## Validation

Three conditions are checked at context creation (`llama-context.cpp`):

1. **Query heads divisible by GPU count**: `n_head(il) % n_gpu == 0` for all layers
2. **KV heads divisible by GPU count**: `n_head_kv(il) % n_gpu == 0` for all layers (important for GQA)
3. **Local KV dimension aligns with TQ block size**: `(head_dim × heads_per_gpu) % 32 == 0`

Error messages tell the user what to do: "try a different model or fewer GPUs."

## Hardware Recommendations

### Symmetric PCIe (x16/x16 or NVLink)
- Use `--split-mode tensor` for full TP
- AllReduce overhead: ≈1.6 ms/token at 32 layers (≈5% of decode time)
- Both GPUs contribute equally

### Asymmetric PCIe (x16/x4, e.g. B450 boards)
- **Use `--split-mode layer` instead** — 30–50× less PCIe overhead than TP
- Layer-split: 2 transfers/token (≈0.06 ms)
- Tensor-parallel: 64+ AllReduces/token (≈2–3 ms through x4 bottleneck)
- Recommended: `-sm layer -ts 1.2,1.0` (slightly more load on faster GPU)

### Memory Savings (TQ2_1, 200K context, 14B model)

| Config | Total KV | Per GPU (TP=2) |
|--------|----------|----------------|
| f16 | 37.5 GB | 18.75 GB |
| TQ2_1 | 8.2 GB | **4.1 GB** |
| TQ4_1 | 12.9 GB | 6.45 GB |

## Usage

```bash
# TP + TQ on 2 GPUs with NCCL
llama-server -m model.gguf \
  --split-mode tensor \
  --cache-type-k tq2_1 --cache-type-v tq2_1 \
  -ngl 99 --flash-attn

# Layer-split + TQ (recommended for asymmetric PCIe)
llama-server -m model.gguf \
  --split-mode layer -ts 1.2,1.0 \
  --cache-type-k tq2_1 --cache-type-v tq2_1 \
  -ngl 99 --flash-attn
```

## Architecture Compatibility

| Architecture | TP+TQ | Notes |
|-------------|-------|-------|
| Llama 3.x | ✅ | D=128, GQA n_kv_heads=8 |
| Qwen 3.x | ✅ | Hybrid (KV+Recurrent), TQ only on attention layers |
| Mistral | ✅ | D=128, GQA |
| Gemma4 (iSWA) | ⚠️ | Known SWA+TQ bug under investigation (workaround: `--tq-protect-layers 999`) |
| DeepSeek2 (MLA) | ❌ | Blocked — MLA incompatible with standard TP |

## Benchmarking

```bash
# Speed comparison: single GPU vs TP vs layer-split
llama-bench -m model.gguf -ngl 99 \
  -sm none,layer,tensor \
  -ctk f16,tq2_1 -ctv f16,tq2_1 \
  -fa 1 -p 512,2048 -n 128 -r 3 -o json

# Accuracy: PPL comparison (delta should be ≈0 for TP overhead)
llama-perplexity -m model.gguf -ngl 99 \
  -sm tensor -ctk tq2_1 -ctv tq2_1 \
  -f wikitext-2-raw/wiki.test.raw --ctx-size 512
```

## Implementation

The change is minimal — a single validation block in `src/llama-context.cpp` (≈55 lines)
replaces the upstream hard block. All TP infrastructure (Meta-Backend, NCCL AllReduce,
weight splitting, graph execution) is reused from upstream llama.cpp.

## Known Issues

- **Gemma4 SWA + TQ**: SWA layers (D=256) produce garbage with any TQ type. Global layers
  (D=512) work correctly. Under active investigation. Workaround: `--tq-protect-layers 999`.
- **PCIe 3.0 heuristic**: AllReduce BF16 compression threshold is calibrated for PCIe 4.0.
  May benefit from lowering for PCIe 3.0 setups.
- **No NCCL fallback**: Without NCCL, the P2P AllReduce fallback has no BF16 compression.
