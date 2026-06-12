# Roadmap

What works, what's in flight, and what's planned. Each area has its own detailed
doc — this file is the index. Feedback and PRs welcome.

## ✅ What works today

| Area | Summary | Details |
|------|---------|---------|
| **TurboQuant KV cache** | KTQ × VTQ at 2.78 bpw, f16-equivalent quality. CUDA sm_75+. | [turboquant.md](docs/turboquant.md) |
| **Speculation** | MTP (gemma-4 draft) + n-gram hybrid. 2.28× on Qwen3.6-35B-A3B-IQ2_XXS. | [speculative.md](docs/speculative.md) |
| **DiffusionGemma** | Coherent 2-bit text diffusion (26B-A4B) on a single 12 GB GPU. | [diffusion-gemma.md](docs/diffusion-gemma.md) |
| **Fine-tune on quantised** | LoRA on `ffn_*_exps` of Qwen3.6-A35B-IQ2_XXS, single 12 GB GPU. | [finetune.md](docs/finetune.md) |
| **Vulkan backend** | Upstream `ggml-vulkan` + Turing tunings. PP parity reached. | [vulkan.md](docs/vulkan.md) |

## 🚧 In flight

- **Aggressive low-bit DiffusionGemma** — pushing the weight footprint lower to fit
  full 256k context per single 12 GB GPU. → [diffusion-gemma.md](docs/diffusion-gemma.md)
- **Eagle3 draft head** — extraction + GGUF plumbing landed; dormant until a trained
  head loads. → [speculative.md](docs/speculative.md)
- **Vulkan KTQ/VTQ port** — PP parity reached, TG gap remaining. → [vulkan.md](docs/vulkan.md)
- **Stage-4 QAT wire-up** — `ggml_quantize_dequantize_fake` op landed; CLI flag +
  LoRA-graph integration queued. → [finetune.md](docs/finetune.md)

## 🎯 Planned

- Full-capability fine-tuning (attention backward, dense gradient flow through
  quantised activations, SSM training). → [finetune.md](docs/finetune.md)
- One DiffusionGemma instance per GPU at full context, batch-parallel serving.

## Maintenance policy

- **Upstream sync.** Upstream `llama.cpp` fixes are cherry-picked when they apply
  cleanly; larger features are integrated case-by-case. Full record:
  [upstream-integration.md](docs/upstream-integration.md).
- **Stability target.** TurboQuant kernels: CUDA sm_75+, daily-driven on Turing
  RTX 2060. Vulkan and HIP are experimental; macOS / Metal are upstream-stock.
- **Regressions are blockers.** Every merge to the default branch must pass the
  local PPL + speed gates on 0.8B-Q8 and 35B-A3B-IQ2_XXS before landing.
