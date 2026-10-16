# llama-tq

<div align="center">

<b>A llama.cpp fork for consumer NVIDIA GPUs: quantized KV cache types, LoRA fine-tuning on quantized models, tensor split without P2P</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/LL4nc33/llama-tq?color=brightgreen)](https://github.com/LL4nc33/llama-tq/releases)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-orange)](https://github.com/ggml-org/llama.cpp)

[benchmarks](docs/benchmarks/README.md) / [TurboQuant KV cache](docs/turboquant.md) / [fine-tuning](docs/finetune.md) /
[models](docs/models.md) / [changelog](CHANGELOG.md) / [roadmap](ROADMAP.md)

</div>

## Quick start

- **Download** a Linux build from the [releases page](https://github.com/LL4nc33/llama-tq/releases): CPU, Vulkan, and
  one CUDA archive per GPU generation ([requirements](docs/build.md#release-archives))
- **Docker** (CPU): `ghcr.io/ll4nc33/llama-tq:server` ([Docker](docs/docker.md))
- **Build** with CUDA: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j`
  (75 Turing, 86 Ampere, 89 Ada, 120 Blackwell; [build guide](docs/build.md))

Then:

```sh
# OpenAI-compatible server with an accurate 8-bit KV cache
llama-server -m model.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0

# compact and still accurate: 8-bit K, 4-bit TurboQuant V (6.5 bpw)
llama-server -m model.gguf -ngl 99 -fa on -ctk q8_0 -ctv vtq4_1

# more context: 4 bits per value (4.6 bpw), more accurate than q4_0; ktq2/vtq2 down to 2.6 bpw
llama-server -m model.gguf -ngl 99 -fa on -ctk ktq4 -ctv vtq4

# LoRA fine-tuning directly on the quantized GGUF
llama-finetune -m model.gguf -f train.jsonl -o task.gguf -ngl 99 -c 512 \
  --lora-train-target '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$' --lora-train-rank 16
```

## Description

`llama-tq` is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for running and fine-tuning models on consumer
NVIDIA GPUs with 12 to 32 GB. It is developed on 2× RTX 2060 12 GB without P2P; speed was also measured on RTX 30, 40
and 50 cards, fine-tuning on the RTX 2060, RTX 5090 and RTX PRO 6000. The fork adds:

- **TurboQuant KV cache:** `ktq1`-`ktq4` (K) and `vtq1`-`vtq4` (V), named by their bits per value, with their own
  decode kernels, to fit more context into the same VRAM ([types](docs/turboquant.md#available-types),
  [which type to pick](docs/turboquant.md#recommendation))
- **Fine-tuning on quantized models:** LoRA directly on k-quant, IQ, MXFP4 and ternary GGUFs, including MoE experts and
  models that do not fit into VRAM; flash attention backward on CPU and CUDA
- **Tensor split without P2P:** `-sm tensor` reduces through pinned host memory on GPUs without peer access (builds
  without NCCL, [details](docs/tensor-split.md))
- **Ternary weights:** `PQ2_0` and `PTQ1_0` with Hadamard-rotated activations (upstream's own ternary types `Q2_0` /
  `TQ1_0` use other formats; the GGUF files are not interchangeable)
- **Kolibri-1:** support for Aleph Alpha's 78B MoE model ([setup](docs/models.md#aleph-alpha-kolibri-1-kolibri1))

Measurements: speed per GPU in [docs/benchmarks](docs/benchmarks/README.md), KV accuracy in
[docs/turboquant.md](docs/turboquant.md#accuracy), fine-tuning in [docs/finetune.md](docs/finetune.md#results).

Limitations:

- Kernels are tuned on Turing (RTX 20); on newer GPUs they run and are measured, but are not tuned.
- Upstream fixes are cherry-picked rather than merged continuously, so the fork lags behind upstream.
- Prefill on Gemma 4 is 1-2 % slower than upstream ([benchmarks](docs/benchmarks/README.md#against-upstream-two-gpus)).
- The TurboQuant short names `ktq2`-`ktq4` / `vtq2`-`vtq4` are trellis types: more accurate per bit, but they decode
  30-40 % slower and take 3.5-4 times as long for the first prompt as the codebook types `ktq*_1` / `vtq*_1`
  ([benchmarks](docs/benchmarks/README.md#trellis-and-codebook-types-two-gpus)).

## Supported GPUs

| GPU generation | CUDA architecture | Measured on |
| --- | --- | --- |
| Turing (RTX 20, T4) | 75 | RTX 2060 (development) |
| Ampere (RTX 30, A100) | 80, 86 | RTX 3060, RTX 3090 (decode speed) |
| Ada (RTX 40, L4) | 89 | RTX 4060 Ti, RTX 4090 (decode speed) |
| Blackwell (RTX 50, RTX PRO 6000) | 120 | RTX 5090 (decode speed, fine-tuning), RTX PRO 6000 (fine-tuning) |
| CPU, Vulkan, other backends | upstream backends | |

The fast TurboQuant KV kernels are CUDA-only; the CPU backend has a slow reference implementation (used for tests),
Vulkan and the other backends have none and use the upstream KV types.

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
