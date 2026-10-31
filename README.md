# llama-tq

<div align="center">

<b>A llama.cpp fork for consumer NVIDIA GPUs: quantized KV cache types, LoRA fine-tuning on quantized models, tensor split without P2P</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/LL4nc33/llama-tq?color=brightgreen)](https://github.com/LL4nc33/llama-tq/releases)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-orange)](https://github.com/ggml-org/llama.cpp)

</div>

llama-tq runs and fine-tunes models on NVIDIA GPUs with 12 to 32 GB. It is developed on two RTX 2060 12 GB without
P2P and measured on RTX 30, 40 and 50 cards.

## What it adds

- [TurboQuant KV cache](docs/turboquant.md): 2.6 to 6.5 bits per value, more accurate than `q4_0` at the same size
- [Faster than upstream](docs/benchmarks/README.md#against-upstream-two-gpus) with a quantized KV cache at long
  context (up to 2.2 times), at least as accurate
- [MTP speculative decoding](docs/speculative.md#draft_mtp-guidance) for Qwen3.5/3.6/3.8 and Gemma 4
- [Tensor split without P2P](docs/tensor-split.md) for consumer GPUs
- [LoRA fine-tuning](docs/finetune.md) directly on quantized GGUFs, also MoE and larger than VRAM

## Quick start

- **Download** a Linux build (CPU, Vulkan, CUDA per GPU generation) from the
  [releases page](https://github.com/LL4nc33/llama-tq/releases), or **Docker** (CPU): `ghcr.io/ll4nc33/llama-tq:server`
- **Build** with CUDA: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j`
  (75 Turing, 86 Ampere, 89 Ada, 120 Blackwell; [build guide](docs/build.md))

```sh
# accurate 8-bit KV cache
llama-server -m model.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0
# more context: 4.6 bpw, more accurate than q4_0 (ktq2 / vtq2 go down to 2.6 bpw)
llama-server -m model.gguf -ngl 99 -fa on -ctk ktq4 -ctv vtq4
# LoRA directly on the quantized GGUF
llama-finetune -m model.gguf -f train.jsonl -o task.gguf -ngl 99 -c 512 \
  --lora-train-target '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$' --lora-train-rank 16
```

## Supported GPUs

| GPU generation | CUDA architecture | Measured on |
| --- | --- | --- |
| Turing (RTX 20, T4) | 75 | RTX 2060, TITAN RTX |
| Ampere (RTX 30, A100) | 80, 86 | RTX 3060, RTX 3090, RTX A40 |
| Ada (RTX 40, L4) | 89 | RTX 4060 Ti, RTX 4090 |
| Blackwell (RTX 50, RTX PRO 6000) | 120 | RTX 5090, RTX PRO 6000 |

## Limitations

- Kernels are tuned on Turing (RTX 20); newer GPUs are measured, not tuned.
- The trellis types (`ktq2`-`ktq4`, `vtq2`-`vtq4`) are more accurate per bit than the codebook types (`ktq*_1`,
  `vtq*_1`), but decode 12-34 % slower ([benchmarks](docs/benchmarks/README.md#trellis-and-codebook-types-two-gpus)).
- The TurboQuant kernels are CUDA-only; CPU has a slow reference implementation, other backends use the upstream KV
  types.

## Documentation

- [Models and setups](docs/models.md), [benchmarks](docs/benchmarks/README.md), [changelog](CHANGELOG.md),
  [roadmap](ROADMAP.md), [Claude Code against llama-tq](docs/claude-code.md), [Vulkan](docs/vulkan.md),
  [KTQ codebooks](docs/calibrating-ktq.md)
- Upstream: [server](tools/server/README.md), [cli](tools/cli/README.md), [multimodal](docs/multimodal.md),
  [function calling](docs/function-calling.md), [Docker](docs/docker.md)

MIT license, inherited from llama.cpp.
