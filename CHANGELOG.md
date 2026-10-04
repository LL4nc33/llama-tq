# Changelog

## 2026-10-04

- **TurboQuant KV fix:** the CUDA readers applied the KTQ sign bits inverted, so every dequantized K value was negated, and the CUDA quantizers used stochastic rounding. CUDA now writes bytes identical to the CPU reference (Qwen3.8-27B `ktq2_1` K: PPL 16.8 -> 6.09, f16 6.04).
- **TurboQuant KV speed (2x RTX 2060, Qwen3.8-27B Q4_K_M, tensor split, 118k context):** decode 6.5 -> 15.8 t/s with `ktq2_1`/`vtq2_1` (f16 KV: ~18.5 t/s at a 112k maximum context), prefill 18 -> 194 t/s.
  - All KTQ blocks share one RHT sign pattern, so Q is rotated once per query instead of per K block.
  - Batches with TurboQuant V dequantize K/V to f16 and use the tensor-core kernel.
  - New decode kernel for quantized KV with grouped-query attention: one block per GQA group, a whole warp per K/V row, the column dot products reduced together, codebooks in shared memory. It also takes q8_0 and q5_0 K/V (q8_0 decode is now faster than f16) and the sparse attention of Qwen3.8-Flash-Next.
  - Warp-cooperative TurboQuant quantization when writing the KV cache.
  - Tensor-core decode kernel for TurboQuant and q5_0 KV (`GGML_CUDA_TQ_WMMA=0` disables it): attention 12-29 % faster than the GQA kernel.
- KV type guidance from perplexity: `ktq4_1`/`vtq4_1` equals f16 on Qwen3.8-27B and Ternary-Bonsai-2-27B; `ktq2_1`/`vtq2_1` costs about +2.6 % there.
- q5_0 K/V flash attention without `GGML_CUDA_FA_ALL_QUANTS` (tensor cores for batches, the GQA kernel for decode).
- **K2-Horizon-MoVA-36B-A4B** (`k2-horizon`, MoE with routed value experts in attention). Q3_K_M on 2x RTX 2060: 47 t/s decode, 840 t/s prefill. With `ktq4_1`/`vtq4_1` KV plus `--tq-protect-layers 4` the PPL is +0.8 % over f16 at about a third of the KV memory.
- Tensor split with a quantized KV cache (attention rotation, staging cache, views of row-split tensors) and for routed value experts.
- Qwen3.8-Flash-Next: the TurboQuant deferred-staging options now reach its hybrid memory (`--no-tq-deferred-k/v` were ignored).

## 2026-10-03

- Tensor split (`-sm tensor`) across GPUs without P2P or NCCL. Partial sums are reduced through mapped pinned host memory; `GGML_CUDA_HOST_ALLREDUCE_BF16=1` sends bf16 to halve link traffic, `GGML_CUDA_HOST_ALLREDUCE=0` disables the path.
- Ternary weight types `PQ2_0` and `PTQ1_0` with Hadamard-rotated activations and CUDA mat-vec / MMQ kernels (Ternary-Bonsai-2-27B).
- Qwen3.8 and Qwen3.8-Flash-Next (`qwen4exp`) support, including sparse flash attention for the compressed-attention indexer.
- Faster Gated DeltaNet layers for the hybrid Qwen3.5 / 3.8 models (state gather and gate activations inside the kernel).
- DFlash / DFlash2 block-diffusion speculative decoding.
- MoE LoRA fine-tuning on quantised models (`--train-skip-regex`, `MUL_MAT_ID` backward). See [docs/finetune.md](docs/finetune.md).

### Known issues

- TurboQuant KV (`ktq2_1` / `vtq2_1`) produced garbage output on Qwen3-4B-Instruct: this was the inverted KTQ sign convention, fixed on 2026-10-04. Small models with strong outlier channels (Qwen3-4B) stay sensitive to 2-3 bit KV; use `ktq4_1` or q8_0 there.
