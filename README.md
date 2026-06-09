# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)

A [llama.cpp](https://github.com/ggml-org/llama.cpp) fork tuned for **long context and speculation on small consumer GPUs**. Lives daily-driven on 2× RTX 2060 12 GB.

## Highlights

- **TurboQuant KV cache** — KTQ × VTQ at 2.78 bpw, f16-equivalent quality. Drop in: `--cache-type-k ktq2 --cache-type-v vtq2`. Details in [docs/turboquant.md](docs/turboquant.md).
- **Speculation stack** — full MTP integration + n-gram hybrid + mmproj+spec coexistence. Prod: 80 t/s creative / 176 t/s repeat on Qwen3.6-35B-A3B-IQ2_XXS at 200k ctx (2.28× boost).
- **Eagle3 draft-head infra** — hidden-state extraction, GGUF plumbing, head graph fusion, HF converter. Dormant until a trained head loads; single-stream MTP unaffected.
- **MoE LoRA on quantised** — fine-tune `ffn_*_exps` on Qwen3.6-A35B-IQ2_XXS in 12 GB. Adapter saves as `.lora.gguf`. Mechanics in [docs/finetune.md](docs/finetune.md).

## What it does

35B-class MoE with 100k context and vision on a single 12 GB GPU. 200k slots × 2 dual-12 GB. Four 65k slots on a 20B-class MoE single-12 GB. CUDA sm_75+ daily-driven on Turing.

## Deploy

CPU image (no build):

```bash
docker pull ghcr.io/ll4nc33/llama-tq:server
docker run -p 8080:8080 -v /path/to/models:/models \
  ghcr.io/ll4nc33/llama-tq:server -m /models/your-model.gguf
```

CUDA / TurboQuant — build from source (~20-30 min on a multi-core machine; template instances exceed CI budget, so no CUDA image):

```bash
git clone https://github.com/LL4nc33/llama-tq && cd llama-tq
cmake -B build -DGGML_CUDA=ON
cmake --build build -j"$(nproc)" --target llama-server llama-finetune
```

Vulkan is WIP on `vulkan`. [Upstream build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for prerequisites.

## Status

Actively maintained. Upstream fixes cherry-picked; larger features integrated case-by-case. Bench parity verified on 0.8B-Q8 and 35B-A3B-IQ2_XXS at every merge gate. See [ROADMAP.md](ROADMAP.md) for what's working, in flight, and shipped.

## License

MIT — inherited from upstream llama.cpp.
