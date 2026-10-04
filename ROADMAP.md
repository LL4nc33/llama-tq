# Roadmap

What works, what's in flight, and what's planned. Each area has its own detailed
doc — this file is the index. Feedback and PRs welcome.

## ✅ What works today

| Area | Summary | Details |
|------|---------|---------|
| **TurboQuant KV cache** | KTQ × VTQ; `ktq4_1`/`vtq4_1` matches f16 perplexity at about a third of the memory. Dedicated GQA and tensor-core decode kernels (also for q8_0 / q5_0). CUDA sm_75+. | [turboquant.md](docs/turboquant.md) |
| **Speculation** | MTP (gemma-4 draft) + n-gram hybrid. 2.28× on Qwen3.6-35B-A3B-IQ2_XXS. | [speculative.md](docs/speculative.md) |
| **Fine-tune on quantised** | LoRA on `ffn_*_exps` of Qwen3.6-A35B-IQ2_XXS, single 12 GB GPU. | [finetune.md](docs/finetune.md) |
| **Vulkan backend** | Upstream `ggml-vulkan` + Turing tunings. PP parity reached. | [vulkan.md](docs/vulkan.md) |
| **Tensor split without P2P** | `-sm tensor` across GPUs without peer access or NCCL: partial sums are reduced through mapped pinned host memory. Optional bf16 transfer via `GGML_CUDA_HOST_ALLREDUCE_BF16=1`. | [tp-tq-design.md](docs/tp-tq-design.md) |
| **Ternary weights** | `PQ2_0` / `PTQ1_0` group-128 ternary types with Hadamard-rotated activations (Ternary-Bonsai-2-27B). CUDA mat-vec / MMQ kernels. | [models.md](docs/models.md) |
| **Qwen3.8 family** | Qwen3.8 and Qwen3.8-Flash-Next (`qwen4exp`) incl. sparse flash attention for the compressed-attention indexer. | [models.md](docs/models.md) |
| **Gated DeltaNet speed-ups** | State gather and gate activations inside the kernel for the hybrid Qwen3.5 / 3.8 models. | [README.md](README.md) |
| **K2-Horizon-MoVA-36B-A4B** | MoE with routed value experts in attention; layer and tensor split, boundary-layer KV protection. | [models.md](docs/models.md) |
| **DFlash / DFlash2** | Block-diffusion speculative decoding on top of the MTP / n-gram stack. | [speculative.md](docs/speculative.md) |

## 🚧 In flight

- **Eagle3 draft head** — extraction + GGUF plumbing landed; dormant until a trained
  head loads.
- **Vulkan KTQ/VTQ port** — PP parity reached, TG gap remaining. → [vulkan.md](docs/vulkan.md)
- **Stage-4 QAT wire-up** — `ggml_quantize_dequantize_fake` op landed; CLI flag +
  LoRA-graph integration queued. → [finetune.md](docs/finetune.md)

## 🎯 Planned

- Full-capability fine-tuning (attention backward, dense gradient flow through
  quantised activations, SSM training). → [finetune.md](docs/finetune.md)

## Known issues

- **Gemma 4 perplexity:** the instruction-tuned models score implausibly high on raw wikitext
  with any engine (upstream gives the same values); use other measures for their KV quality.
  → [models.md](docs/models.md)
- **Small models and 2-bit K:** Qwen3-4B and Ministral-3B are sensitive to low-bit K; use
  `ktq4_1` or q8_0 there. → [turboquant.md](docs/turboquant.md)

## Maintenance policy

- **Upstream sync.** Upstream `llama.cpp` fixes are cherry-picked when they apply
  cleanly; larger features are integrated case-by-case. Full record:
  [upstream-integration.md](docs/upstream-integration.md).
- **Stability target.** TurboQuant kernels: CUDA sm_75+, daily-driven on Turing
  RTX 2060. Vulkan and HIP are experimental; macOS / Metal are upstream-stock.
- **Regressions are blockers.** Every merge to the default branch must pass the
  local PPL + speed gates on 0.8B-Q8 and 35B-A3B-IQ2_XXS before landing.
