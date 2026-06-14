# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)

A [llama.cpp](https://github.com/ggml-org/llama.cpp) fork tuned for one thing:
**running modern models at full context on 12 GB Turing GPUs** — the consumer cards
(RTX 2060 / 2070 / 2080) and the datacenter **T4** (sm_75). The KV-cache quantization,
the Gemma-4 sliding-window handling, and the kernel tuning all exist to make that work
where stock llama.cpp runs out of VRAM or context.

## Highlights

- **TurboQuant KV cache** — KTQ × VTQ at 2.78 bpw, f16-equivalent quality. Drop in:
  `--cache-type-k ktq2 --cache-type-v vtq2`. Outlier-robust V (`vtq2_3`) keeps even
  exact code output clean. Full type reference in [docs/turboquant.md](docs/turboquant.md).
- **Gemma-4 first-class** — encoder-free unified vision + audio (`gemma4uv`/`gemma4ua`),
  SWA-aware KV (`--cache-type-{k,v}-swa`) so aggressive KV quant stays coherent on the
  sliding-window layers, and 256k-context recipes that fit a single 12 GB card.
- **Speculation stack** — full MTP integration + n-gram hybrid (`ngram-cache`,
  `ngram-map-k4v`) + mmproj+spec coexistence. Lossless; up to 2.28× on repeat/code.
- **Eagle3 draft-head infra** — hidden-state extraction, GGUF plumbing, head-graph
  fusion, HF converter. Dormant until a trained head loads; single-stream MTP unaffected.
- **MoE LoRA on quantised weights** — fine-tune `ffn_*_exps` on a 35B-class MoE at
  IQ2_XXS inside 12 GB. Adapter saves as `.lora.gguf`. Mechanics in
  [docs/finetune.md](docs/finetune.md).

## What it does

35B-class MoE with 100k context and vision on a single 12 GB GPU. Gemma-4-12B at the
full 256k context on one card (or code-exact at 180k). 256k slots across two 12 GB
cards. CUDA sm_75+, focused on Turing.

**New here?** The [full-context-on-12 GB recipe guide](docs/fullctx-on-12gb.md) is the
fastest way in: verified `llama-server` commands per model, plus the reasoning behind
every KV-cache choice — which combo, why, and what each flag does.

**Every llama-tq flag** — the complete reference for the fork's own options
(`--cache-type-{k,v}` KTQ/VTQ types, `--cache-type-{k,v}-swa`, the TurboQuant tier
table, bit widths, kernels, and benchmarks) lives in
[docs/turboquant.md](docs/turboquant.md).

## Build

CUDA / TurboQuant — build from source (~20-30 min on a multi-core machine; the KV
quant kernels are template-heavy and exceed CI budget, so there is no prebuilt CUDA image):

```bash
git clone https://github.com/LL4nc33/llama-tq && cd llama-tq
cmake -B build -DGGML_CUDA=ON
cmake --build build -j"$(nproc)" --target llama-server llama-finetune
```

CPU image (no build, no TurboQuant):

```bash
docker pull ghcr.io/ll4nc33/llama-tq:server
docker run -p 8080:8080 -v /path/to/models:/models \
  ghcr.io/ll4nc33/llama-tq:server -m /models/your-model.gguf
```

Vulkan is WIP on the `vulkan` branch.
[Upstream build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
cover the prerequisites.

## Quick start (single 12 GB GPU)

```bash
# Gemma-4-12B, full 256k context, chat/prose:
llama-server -m gemma-4-12b-it-Q4_K_M.gguf -ngl 99 -fa on -c 262144 \
  --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
  --cache-type-k-swa f16 --cache-type-v-swa f16
```

For code-exact output, vision, dual-GPU, and other models, see
[docs/fullctx-on-12gb.md](docs/fullctx-on-12gb.md).

## Status

Actively maintained. Upstream fixes are cherry-picked; larger features integrated
case-by-case. Bench parity is verified on a 0.8B-Q8 smoke model and a 35B-A3B-IQ2_XXS
at every merge gate. See [ROADMAP.md](ROADMAP.md) for what's working, in flight, and shipped.

## License

MIT — inherited from upstream llama.cpp.
