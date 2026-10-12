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

- **Download** a Linux build from the [releases page](https://github.com/LL4nc33/llama-tq/releases)
  ([requirements](docs/build.md#release-archives); the current release v2026.10.06 predates the changes since 2026-10-07,
  build from source for now)
- **Docker** (CPU): `ghcr.io/ll4nc33/llama-tq:server` ([Docker](docs/docker.md))
- **Build** with CUDA: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j`
  (75 Turing, 86 Ampere, 89 Ada, 120 Blackwell; [build guide](docs/build.md))

Then:

```sh
# OpenAI-compatible server with an accurate 8-bit KV cache
llama-server -m model.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0

# compact and still accurate: 8-bit K, 4-bit TurboQuant V (6.5 bpw)
llama-server -m model.gguf -ngl 99 -fa on -ctk q8_0 -ctv vtq4_1

# most context: 5 bpw, at q4_0 accuracy
llama-server -m model.gguf -ngl 99 -fa on -ctk ktq4_1 -ctv vtq4_1

# LoRA fine-tuning directly on the quantized GGUF
llama-finetune -m model.gguf -f train.jsonl -o task.gguf -ngl 99 -c 512 \
  --lora-train-target '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$' --lora-train-rank 16
```

## Description

`llama-tq` is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that aims to run large models with long
context, and to fine-tune them, on consumer NVIDIA GPUs with 12 to 32 GB. It is developed on 2× RTX 2060 12 GB without
P2P; speed was also measured on RTX 30, 40 and 50 cards, fine-tuning on the RTX 2060, RTX 5090 and RTX PRO 6000. Upstream fixes are cherry-picked rather than
merged continuously, so the fork can lag behind the newest upstream. The fork adds:

- **TurboQuant KV cache:** `ktq*` (K) and `vtq*` (V) cache types from 1.5 to 5.5 bits per value ([types](docs/turboquant.md#available-types)) with dedicated decode
  kernels, for long context on small GPUs ([which type to pick](docs/turboquant.md#recommendation))
- **Fine-tuning on quantized models:** LoRA on k-quant, IQ, MXFP4 and ternary GGUFs without dequantized weights, MoE
  experts, models larger than VRAM, flash attention backward on CPU and CUDA
- **Tensor split without P2P:** `-sm tensor` across GPUs that have no peer access or NCCL
- **Ternary weights:** `PQ2_0` and `PTQ1_0` with Hadamard-rotated activations (upstream's own ternary types `Q2_0` /
  `TQ1_0` use other formats; the GGUF files are not interchangeable)
- **Kolibri-1** (Aleph Alpha, 78B MoE), which upstream does not support

K2-Horizon, Qwen3.8 / Qwen3.8-Flash-Next and the DFlash, DFlash2, Eagle3 and MTP drafters, which this fork added
earlier, are now also in upstream llama.cpp; [models.md](docs/models.md) and [speculative.md](docs/speculative.md) keep
the setups we use.

Measurements: speed per GPU in [docs/benchmarks](docs/benchmarks/README.md), KV accuracy in
[docs/turboquant.md](docs/turboquant.md#accuracy), fine-tuning in [docs/finetune.md](docs/finetune.md#results).

## Supported GPUs

| GPU generation | CUDA architecture | Measured on |
| --- | --- | --- |
| Turing (RTX 20, T4) | 75 | RTX 2060 (development) |
| Ampere (RTX 30, A100) | 80, 86 | RTX 3060, RTX 3090 (decode speed) |
| Ada (RTX 40, L4) | 89 | RTX 4060 Ti, RTX 4090 (decode speed) |
| Blackwell (RTX 50, RTX PRO 6000) | 120 | RTX 5090 (decode speed, fine-tuning), RTX PRO 6000 (fine-tuning) |
| CPU, Vulkan, other backends | upstream backends | |

The fast TurboQuant KV kernels are CUDA-only; the CPU backend has a slow reference implementation (used for tests),
Vulkan and the other backends have none and use the upstream KV types. Kernel tuning was done on Turing; on the
newer generations the code runs and is measured, but not tuned for them yet.

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
- [build](docs/build.md), [Docker](docs/docker.md), [upstream models](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md)

## Contributing

Issues and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). Upstream fixes are cherry-picked; new
kernels are checked against the CPU backend (`test-backend-ops`) and by perplexity or KL divergence.

## License

MIT, inherited from upstream llama.cpp.
