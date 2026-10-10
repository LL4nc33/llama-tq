# Roadmap

What works, what's in flight, and what's planned. Each area has its own detailed
doc — this file is the index. Feedback and PRs welcome.

## ✅ What works today

| Area | Summary | Details |
|------|---------|---------|
| **TurboQuant KV cache** | KTQ × VTQ; `ktq4_1`/`vtq4_1` reaches `q4_0`-level accuracy at 5 bpw (KL divergence measured per model, 2026-10-10); VTQ V is the stronger half. Dedicated GQA and tensor-core decode kernels (also for q8_0 / q5_0). CUDA sm_75+. | [turboquant.md](docs/turboquant.md) |
| **Speculation** | MTP (gemma-4 draft) + n-gram hybrid. 2.28× on Qwen3.6-35B-A3B-IQ2_XXS. | [speculative.md](docs/speculative.md) |
| **Fine-tune on quantised** | LoRA directly on quantized GGUFs: attention, MoE experts and Gated DeltaNet layers, flash-attention backward, gradient checkpointing, models larger than VRAM. 13 of 13 evaluated models reach 100 % on a held-out task; faster than PyTorch QLoRA on an RTX 5090. | [finetune.md](docs/finetune.md) |
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
- **Vulkan KTQ/VTQ port** — researched, not started: worth it for AMD, Intel Arc and Pascal (where Vulkan
  decodes faster than CUDA), not for Turing. The KV types are CUDA-only today. → [vulkan.md](docs/vulkan.md)
- **Stage-4 QAT wire-up** — `ggml_quantize_dequantize_fake` op landed; CLI flag +
  LoRA-graph integration queued. → [finetune.md](docs/finetune.md)

## 🎯 Planned

- TurboQuant on every RTX generation: speed within 5 % of f16 on Turing to Blackwell, better KTQ K accuracy, a decode
  kernel for `q8_0` K with VTQ V. → [turboquant.md](docs/turboquant.md)
- Fine-tuning: faster CPU path and layer prefetch for experts in RAM, SSM (Mamba) training, fewer small kernels on Turing.
  → [finetune.md](docs/finetune.md)

## Known issues

- **Gemma 4 perplexity:** the instruction-tuned models score implausibly high on raw wikitext
  with any engine (upstream gives the same values); use other measures for their KV quality.
  → [models.md](docs/models.md)
- **KTQ K accuracy:** with K quantized in prefill, `ktq4_1` is only at the `q4_0` level and the 2-bit K types lose a lot;
  gpt-oss and Gemma 4 react strongly to any KV quantization. Use `f16`/`q8_0` when accuracy matters.
  → [turboquant.md](docs/turboquant.md#accuracy)
- **q8_0 K with VTQ V** is accurate but has no dedicated decode kernel yet (slow at long context).

## Maintenance policy

- **Upstream sync.** Upstream `llama.cpp` fixes are cherry-picked when they apply
  cleanly; larger features are integrated case-by-case. Full record:
  [upstream-integration.md](docs/upstream-integration.md).
- **Stability target.** TurboQuant kernels: CUDA sm_75+, daily-driven on Turing
  RTX 2060. Vulkan and HIP are experimental; macOS / Metal are upstream-stock.
- **Regressions are blockers.** Every merge to the default branch must pass the
  local PPL + speed gates on 0.8B-Q8 and 35B-A3B-IQ2_XXS before landing.
