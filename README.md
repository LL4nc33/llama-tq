# llama-tq

<div align="center">

<b>llama.cpp for consumer NVIDIA GPUs: compact KV caches, fine-tuning on quantized models, multi-GPU without P2P</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/LL4nc33/llama-tq?color=brightgreen)](https://github.com/LL4nc33/llama-tq/releases)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-orange)](https://github.com/ggml-org/llama.cpp)

[benchmarks](docs/benchmarks/README.md) / [TurboQuant KV cache](docs/turboquant.md) / [fine-tuning](docs/finetune.md) /
[models](docs/models.md) / [changelog](CHANGELOG.md) / [roadmap](ROADMAP.md)

</div>

## Quick start

A few options to get `llama-tq`:

- Download a build for your GPU generation from the [releases page](https://github.com/LL4nc33/llama-tq/releases)
  (Linux: CUDA per generation, CPU and Vulkan for x64 and arm64)
- Run the CPU server image with Docker: `ghcr.io/ll4nc33/llama-tq:server` (see [Docker](docs/docker.md))
- Build from source with CUDA, see the [build guide](docs/build.md):
  `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j`
  (75 Turing, 86 Ampere, 89 Ada, 120 Blackwell)

Once installed:

```sh
# OpenAI-compatible server with an accurate 8-bit KV cache
llama-server -m model.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0

# more context on the same GPU: compact TurboQuant KV cache
llama-server -m model.gguf -ngl 99 -fa on -ctk ktq4_1 -ctv vtq4_1

# LoRA fine-tuning directly on the quantized GGUF
llama-finetune -m model.gguf -f train.jsonl -o task.gguf --lora-train-rank 16 -ngl 99 -c 512
```

## Description

`llama-tq` is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that aims to run large models with long
context, and to fine-tune them, on consumer NVIDIA GPUs with 12 to 32 GB. It is developed on 2× RTX 2060 12 GB without
P2P and tested on RTX 30, 40 and 50 cards. Everything upstream offers stays available; the fork adds:

- **TurboQuant KV cache:** `ktq*` (K) and `vtq*` (V) cache types with fast decode kernels; `ktq4_1`/`vtq4_1` stores the
  cache at 5 bits per value with accuracy at or above `q4_0`, `q8_0` K with `vtq4_1` V is close to `q8_0`
- **Fine-tuning on quantized models:** LoRA on k-quant, IQ, MXFP4 and ternary GGUFs without dequantized weights, MoE
  experts, models larger than VRAM, flash attention backward on CPU and CUDA
- **Tensor split without P2P:** `-sm tensor` across GPUs that have no peer access or NCCL
- **Ternary weights:** `PQ2_0` and `PTQ1_0` with Hadamard-rotated activations
- **More architectures:** Kolibri-1, K2-Horizon, Qwen3.8 and Qwen3.8-Flash-Next, Ternary-Bonsai-2
- **Speculative decoding:** MTP with n-gram drafts, DFlash/DFlash2 block diffusion, with vision

Measured speed and accuracy for every GPU and model: [docs/benchmarks](docs/benchmarks/README.md).

## Supported GPUs

| GPU generation | Release archive | Tested on |
| --- | --- | --- |
| Turing (RTX 20, T4) | `cuda-12.8-sm75` (also runs newer GPUs through PTX) | RTX 2060 |
| Ampere (RTX 30, A100) | `cuda-12.8-sm86` | RTX 3060, RTX 3090 |
| Ada (RTX 40, L4) | `cuda-12.8-sm89` | RTX 4060 Ti, RTX 4090 |
| Blackwell (RTX 50, RTX PRO 6000) | `cuda-12.8-sm120` | RTX 5090, RTX PRO 6000 |
| CPU, Vulkan, other backends | upstream backends | |

The TurboQuant KV types need CUDA; the other backends use the upstream KV types.

## Documentation

#### This fork

- [TurboQuant KV cache](docs/turboquant.md): types, accuracy, recommendation
- [Fine-tuning](docs/finetune.md) and the [evaluation task](examples/training/termine/README.md)
- [Models and setups](docs/models.md): commands per model
- [Tensor split without P2P](docs/tensor-split.md)
- [Speculative decoding](docs/speculative.md)
- [Benchmarks](docs/benchmarks/README.md)
- [Claude Code against a llama-tq server](docs/claude-code.md)
- [Vulkan notes](docs/vulkan.md), [calibrating the KTQ codebooks](docs/calibrating-ktq.md)

#### Upstream

- [server](tools/server/README.md), [cli](tools/cli/README.md), [multimodal](docs/multimodal.md),
  [function calling](docs/function-calling.md)
- [build](docs/build.md), [Docker](docs/docker.md), [models](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md)

## Contributing

Issues and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). Upstream fixes are cherry-picked; new
kernels are checked against the CPU backend (`test-backend-ops`) and by perplexity or KL divergence.

## License

MIT, inherited from upstream llama.cpp.
