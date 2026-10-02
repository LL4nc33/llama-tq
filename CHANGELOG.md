# Changelog

## Unreleased

- Tensor split (`-sm tensor`) across GPUs without P2P or NCCL. Partial sums are reduced through mapped pinned host memory; `GGML_CUDA_HOST_ALLREDUCE_BF16=1` sends bf16 to halve link traffic, `GGML_CUDA_HOST_ALLREDUCE=0` disables the path.
- Ternary weight types `PQ2_0` and `PTQ1_0` with Hadamard-rotated activations and CUDA mat-vec / MMQ kernels (Ternary-Bonsai-2-27B).
- Qwen3.8 and Qwen3.8-Flash-Next (`qwen4exp`) support, including sparse flash attention for the compressed-attention indexer.
- Faster Gated DeltaNet layers for the hybrid Qwen3.5 / 3.8 models (state gather and gate activations inside the kernel).
- DFlash / DFlash2 block-diffusion speculative decoding.
- MoE LoRA fine-tuning on quantised models (`--train-skip-regex`, `MUL_MAT_ID` backward). See [docs/finetune.md](docs/finetune.md).

### Known issues

- TurboQuant KV (`ktq2_1` / `vtq2_1`) produces garbage output on Qwen3-4B-Instruct at a prompt of about 7k tokens; f16 KV is correct. Under investigation.
