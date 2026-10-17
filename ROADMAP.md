# Roadmap

What works, what is in flight and what is planned. Measured numbers are on the [benchmarks page](docs/benchmarks/README.md),
the history in the [changelog](CHANGELOG.md).

## What works today

| Area | Summary | Details |
|------|---------|---------|
| **TurboQuant KV cache** | `ktq*` (K) and `vtq*` (V) cache types; GQA and tensor-core decode kernels for them and for `q8_0` / `q5_0` / `q4_0`. CUDA sm_75+. | [turboquant.md](docs/turboquant.md) |
| **Fine-tuning on quantized models** | LoRA directly on quantized GGUFs: attention, MoE experts, Gated DeltaNet and hyper-connection layers, flash attention backward, gradient checkpointing, models larger than VRAM. | [finetune.md](docs/finetune.md) |
| **Tensor split without P2P** | `-sm tensor` across GPUs without peer access or NCCL, partial sums reduced through pinned host memory. | [tensor-split.md](docs/tensor-split.md) |
| **Speculative decoding** | MTP with n-gram drafts, DFlash / DFlash2 block diffusion, with vision (the drafters are now also in upstream). | [speculative.md](docs/speculative.md) |
| **Ternary weights** | `PQ2_0` / `PTQ1_0` group-128 ternary types with Hadamard-rotated activations, CUDA mat-vec / MMQ kernels. | [models.md](docs/models.md) |
| **More architectures** | Kolibri-1 (not in upstream); K2-Horizon-MoVA-36B-A4B, Qwen3.8 and Qwen3.8-Flash-Next, Ternary-Bonsai-2 (the first three are now also in upstream). | [models.md](docs/models.md) |
| **Vulkan backend** | Upstream `ggml-vulkan` with Turing tunings; TurboQuant types are CUDA-only. | [vulkan.md](docs/vulkan.md) |

## In flight

- **Eagle3 draft head:** extraction and GGUF plumbing landed; dormant until a trained head loads.
- **QAT:** `--qat-target-quant` fake-quantizes the LoRA delta in training; landed, not yet validated on a task.
  → [finetune.md](docs/finetune.md)

## Planned

- TurboQuant on every RTX generation: decode speed close to f16 on Turing to Blackwell, better 2- and 3-bit K accuracy, a
  cheaper encoder scale search, a faster trellis encoder (first prompt and decode of `ktq2`-`ktq4` / `vtq2`-`vtq4`). → [turboquant.md](docs/turboquant.md)
- CUDA release for arm64 (GH200, DGX Spark).
- Vulkan KTQ/VTQ port: planned, after the CUDA work; meant for AMD, Intel Arc and Pascal
  (untested whether Vulkan decodes faster there). → [vulkan.md](docs/vulkan.md)
- Fine-tuning: faster CPU path and layer prefetch for experts in RAM, SSM (Mamba) training, fewer small kernels on Turing.
  → [finetune.md](docs/finetune.md)

## Known issues

- TurboQuant: [turboquant.md#known-issues](docs/turboquant.md#known-issues)
- Model-specific (e.g. Gemma 4 perplexity): [models.md#known-limitations](docs/models.md#known-limitations)

## Maintenance policy

- **Upstream sync:** upstream `llama.cpp` fixes are cherry-picked when they apply cleanly; larger features are
  integrated case by case and noted in the [changelog](CHANGELOG.md).
- **Stability target:** TurboQuant kernels on CUDA sm_75+, used daily on an RTX 2060. Vulkan and HIP are experimental;
  macOS / Metal are upstream stock.
- **Regressions are blockers:** new kernels are checked against the CPU backend (`test-backend-ops`) and by perplexity or
  KL divergence, speed changes by `llama-bench` against the previous build.
