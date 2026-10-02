# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)

A [llama.cpp](https://github.com/ggml-org/llama.cpp) fork tuned for **long context, multi-GPU and speculation on small consumer GPUs**. Lives daily-driven on 2× RTX 2060 12 GB (no P2P, one card on a chipset x4 link).

## Highlights

- **Tensor split without P2P or NCCL** — `-sm tensor` sums the partial results through mapped pinned host memory, GPU to GPU without a host thread (`GGML_CUDA_HOST_ALLREDUCE_BF16=1` halves the traffic). Qwen3.8-27B Q4_K_M on 2× RTX 2060: 24.0 t/s decode instead of 16.5 with layer split; Ternary-Bonsai-2-27B at 171k context: 21.6 instead of 16.0 t/s. Perplexity identical to layer split.
- **Ternary weights (PQ2_0, PTQ1_0)** — group-128 ternary types with Hadamard-rotated activations, CUDA mat-vec / MMQ kernels and a fast Walsh-Hadamard transform. Ternary-Bonsai-2-27B: 200k context with f16 KV and vision on 2× 12 GB, ~40 t/s decode with tensor split.
- **Qwen3.8 family** — Qwen3.8-Flash-Next (qwen4exp: per-layer n-gram embeddings, hyper-connections, compressed-attention indexer with sparse flash attention) and faster Gated DeltaNet layers for the hybrid Qwen3.5/3.8 models (state gather and gate activations inside the kernel).
- **TurboQuant KV cache** — KTQ × VTQ at 2.78 bpw. Drop in: `--cache-type-k ktq2 --cache-type-v vtq2`. Details in [docs/turboquant.md](docs/turboquant.md).
- **Speculation stack** — MTP + n-gram hybrid, mmproj+spec coexistence, and DFlash / DFlash2 block-diffusion drafting. Details in [docs/speculative.md](docs/speculative.md).
- **MoE LoRA on quantised** — fine-tune `ffn_*_exps` on Qwen3.6-A35B-IQ2_XXS in 12 GB. Mechanics in [docs/finetune.md](docs/finetune.md).

## What it does

Measured on 2× RTX 2060 12 GB (Turing, no P2P):

| Setup | Context | Decode |
|---|---|---|
| Ternary-Bonsai-2-27B PTQ1_0, tensor split, f16 KV, vision | 200k | ~40 t/s short, 21.6 t/s at 171k |
| Qwen3.8-27B UD-Q4_K_M, tensor split, f16 KV, vision | 72k | 24 t/s |
| Qwen3.8-27B Q4_K_M + DFlash2 draft (code) | — | 26-28 t/s instead of 15.5 |
| 35B-class MoE (IQ2), single GPU, vision | 100k | — |

Two GPUs without P2P, tensor split:

```bash
GGML_CUDA_HOST_ALLREDUCE_BF16=1 llama-server -m model.gguf -ngl 99 -fa on -sm tensor -c 65536
```

## Deploy

**Prebuilt:** releases carry Linux x64 binaries, including a CUDA 12.8 build (sm_75 plus PTX
for newer GPUs). CPU Docker image:

```bash
docker pull ghcr.io/ll4nc33/llama-tq:server
docker run -p 8080:8080 -v /path/to/models:/models \
  ghcr.io/ll4nc33/llama-tq:server -m /models/your-model.gguf
```

**From source (CUDA):** the TurboQuant template instances make the CUDA build heavy; build
for your architecture only (75 = Turing, 86 = Ampere, 89 = Ada).

```bash
git clone https://github.com/LL4nc33/llama-tq && cd llama-tq
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j"$(nproc)" --target llama-server
```

Vulkan is WIP on the `vulkan` branch. See the [upstream build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for prerequisites.

## Status

Actively maintained and used daily on Turing GPUs. Upstream fixes are cherry-picked; larger
upstream features are integrated case by case. New kernels and model paths are checked
against the CPU backend (`test-backend-ops`) and by perplexity against reference builds.
[ROADMAP.md](ROADMAP.md) lists what is shipped, in flight and known to be broken;
[CHANGELOG.md](CHANGELOG.md) lists recent changes. Detailed benchmarks with settings will be
published separately.

## License

MIT — inherited from upstream llama.cpp.
