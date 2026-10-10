# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)

A [llama.cpp](https://github.com/ggml-org/llama.cpp) fork for **long context, multi-GPU and fine-tuning on consumer
GPUs**. Developed on 2× RTX 2060 12 GB (no P2P); tested on RTX 3060, 3090, 4060 Ti, 4090 and 5090.

## Highlights

- **TurboQuant KV cache** — `ktq*`/`vtq*` types, f16 perplexity at ~⅓ of the memory, 256k context on 12 GB cards
  → [docs/turboquant.md](docs/turboquant.md)
- **Fine-tuning on quantized GGUFs** — LoRA without dequantized weights, MoE experts, models larger than VRAM,
  faster than PyTorch QLoRA on an RTX 5090 → [docs/finetune.md](docs/finetune.md)
- **Tensor split without P2P/NCCL** — `-sm tensor` on plain PCIe, +45 % decode on 2× RTX 2060
- **Ternary weights** — `PQ2_0`, `PTQ1_0` with Hadamard-rotated activations
- **Extra models** — Kolibri-1, K2-Horizon, Qwen3.8-Flash-Next, Ternary-Bonsai-2 → [docs/models.md](docs/models.md)
- **Speculation** — MTP + n-gram, DFlash/DFlash2 block diffusion, with vision → [docs/speculative.md](docs/speculative.md)

## Numbers

Inference on 2× RTX 2060 12 GB:

| Model | KV cache | Max context | Decode, short | Decode, long |
|---|---|---|---|---|
| Qwen3.8-27B Q4_K_M (tensor split) | f16 | 72k | 24 t/s | |
| Qwen3.8-27B Q4_K_M (tensor split) | `ktq2_1` / `vtq2_1` | 256k | 24 t/s | 15.8 t/s at 118k |
| Ternary-Bonsai-2-27B PTQ1_0 (tensor split) | f16 | 200k | 40 t/s | 21.6 t/s at 171k |
| Kolibri-1 78B MoE Q3_K_S | `ktq4_1` / `vtq4_1` | 128k | 35 t/s | |
| gpt-oss-20b MXFP4 | `ktq4_1` / `vtq4_1` | 128k | 77 t/s | 40 t/s at 64k |

Inference on one GPU, Qwen3-4B Q4_K_M, 128 tokens:

| GPU | f16 KV | `ktq2_1` / `vtq2_1` KV |
|---|---|---|
| RTX 5090 | 331 t/s | 272 t/s |
| RTX 4090 | 252 t/s | 217 t/s |
| RTX 3090 | 203 t/s | 176 t/s |
| RTX 3060 | 103 t/s | 96 t/s |
| RTX 4060 Ti | 95 t/s | 90 t/s |

LoRA fine-tuning, Qwen3-4B Q4_K_M, 2 epochs, same data and settings, 100 % exact match in all runs:

| GPU | llama-tq | PyTorch QLoRA |
|---|---|---|
| RTX 5090 | 54 s | 102 s |
| RTX 2060 | 321 s | 260 s |

More setups: [docs/models.md](docs/models.md), [docs/benchmarks](docs/benchmarks).

## Quick start

```bash
# prebuilt: releases (Linux x64/arm64, CPU, Vulkan, CUDA 12.8) or Docker
docker run -p 8080:8080 -v /path/to/models:/models ghcr.io/ll4nc33/llama-tq:server -m /models/model.gguf

# from source, CUDA (75 = Turing, 86 = Ampere, 89 = Ada, 120 = Blackwell)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j --target llama-server

# two GPUs without P2P
llama-server -m model.gguf -ngl 99 -fa on -sm tensor -c 65536 -ctk ktq4_1 -ctv vtq4_1
```

TurboQuant KV types are CUDA-only; Vulkan and CPU builds use the upstream types.

## Docs

[CHANGELOG.md](CHANGELOG.md) · [ROADMAP.md](ROADMAP.md) · [turboquant](docs/turboquant.md) ·
[finetune](docs/finetune.md) · [models](docs/models.md) · [speculative](docs/speculative.md) ·
[upstream build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

Upstream fixes are cherry-picked; new kernels are checked against the CPU backend (`test-backend-ops`) and by
perplexity.

## License

MIT — inherited from upstream llama.cpp.
