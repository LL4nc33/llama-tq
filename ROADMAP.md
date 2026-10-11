# Roadmap

What works, what is in flight and what is planned. Measured numbers are on the [benchmarks page](docs/benchmarks/README.md),
the history in the [changelog](CHANGELOG.md).

## What works today

| Area | Summary | Details |
|------|---------|---------|
| **TurboQuant KV cache** | `ktq*` (K) and `vtq*` (V) cache types; `ktq4_1`/`vtq4_1` at or above `q4_0` accuracy at 5 bpw, `q8_0` K with `vtq4_1` V close to `q8_0`. GQA and tensor-core decode kernels for the TurboQuant types and for `q8_0` / `q5_0` / `q4_0`. CUDA sm_75+. | [turboquant.md](docs/turboquant.md) |
| **Fine-tuning on quantized models** | LoRA directly on quantized GGUFs: attention, MoE experts, Gated DeltaNet and hyper-connection layers, flash attention backward, gradient checkpointing, models larger than VRAM. | [finetune.md](docs/finetune.md) |
| **Tensor split without P2P** | `-sm tensor` across GPUs without peer access or NCCL, partial sums reduced through pinned host memory. | [tensor-split.md](docs/tensor-split.md) |
| **Speculative decoding** | MTP with n-gram drafts, DFlash / DFlash2 block diffusion, with vision. | [speculative.md](docs/speculative.md) |
| **Ternary weights** | `PQ2_0` / `PTQ1_0` group-128 ternary types with Hadamard-rotated activations, CUDA mat-vec / MMQ kernels. | [models.md](docs/models.md) |
| **More architectures** | Kolibri-1, K2-Horizon-MoVA-36B-A4B, Qwen3.8 and Qwen3.8-Flash-Next (sparse flash attention for the indexer), Ternary-Bonsai-2; faster Gated DeltaNet layers. | [models.md](docs/models.md) |
| **Vulkan backend** | Upstream `ggml-vulkan` with Turing tunings; TurboQuant types are CUDA-only. | [vulkan.md](docs/vulkan.md) |

## In flight

- **Eagle3 draft head:** extraction and GGUF plumbing landed; dormant until a trained head loads.
- **Vulkan KTQ/VTQ port:** researched, not started; worth it for AMD, Intel Arc and Pascal (where Vulkan decodes faster
  than CUDA), not for Turing. → [vulkan.md](docs/vulkan.md)
- **QAT:** `ggml_quantize_dequantize_fake` landed; CLI flag and LoRA graph integration queued. → [finetune.md](docs/finetune.md)

## Planned

- TurboQuant on every RTX generation: speed within 5 % of f16 on Turing to Blackwell, better 2- and 3-bit K accuracy, a
  cheaper encoder scale search. → [turboquant.md](docs/turboquant.md)
- CUDA release for arm64 (GH200, DGX Spark).
- Fine-tuning: faster CPU path and layer prefetch for experts in RAM, SSM (Mamba) training, fewer small kernels on Turing.
  → [finetune.md](docs/finetune.md)

## Known issues

- **Gemma 4 perplexity:** the instruction-tuned models score implausibly high on raw wikitext with any engine (upstream
  gives the same values); their KV accuracy is measured on their own prose. → [models.md](docs/models.md)
- **Low-bit K:** the 2- and 3-bit K types still lose a lot, and gpt-oss loses with every 4-bit K type (`q4_0` as well).
  Use `f16`/`q8_0`, or `q8_0` K with `vtq4_1` V, when accuracy matters. → [turboquant.md](docs/turboquant.md#accuracy)

## Maintenance policy

- **Upstream sync:** upstream `llama.cpp` fixes are cherry-picked when they apply cleanly; larger features are
  integrated case by case and noted in the [changelog](CHANGELOG.md).
- **Stability target:** TurboQuant kernels on CUDA sm_75+, used daily on an RTX 2060. Vulkan and HIP are experimental;
  macOS / Metal are upstream stock.
- **Regressions are blockers:** new kernels are checked against the CPU backend (`test-backend-ops`) and by perplexity or
  KL divergence, speed changes by `llama-bench` against the previous build.
