# Roadmap

What works, what is in flight and what is planned (as of 2026-10-18). Measured numbers are on the [benchmarks page](docs/benchmarks/README.md),
the history in the [changelog](CHANGELOG.md).

## What works today

| Area | Summary | Details |
|------|---------|---------|
| **TurboQuant KV cache** | `ktq*` (K) and `vtq*` (V) cache types, codebook (`*_1`) and trellis (short names `ktq2`-`ktq4`, `vtq2`-`vtq4`); every K/V pair of them, f16, `q8_0`, `q5_0` and `q4_0` runs on CUDA, with GQA, vector and tensor-core decode kernels. CUDA sm_75+. | [turboquant.md](docs/turboquant.md) |
| **Fine-tuning on quantized models** | LoRA directly on quantized GGUFs: attention, MoE experts, Gated DeltaNet and hyper-connection layers, flash attention backward, gradient checkpointing, chat JSONL with loss mask, resume, models larger than VRAM; 12 evaluated models at 100 % on the test task. | [finetune.md](docs/finetune.md) |
| **Tensor split without P2P** | `-sm tensor` across GPUs without peer access or NCCL, partial sums reduced through pinned host memory. | [tensor-split.md](docs/tensor-split.md) |
| **Speculative decoding** | MTP drafts from the model's own head (Qwen3.5/3.6, several slots), n-gram modes, DFlash / DFlash2 block diffusion; verify batches of 2-16 queries read the quantized KV cache directly. A vision projector can be loaded next to it (requests with images run without speculation). | [speculative.md](docs/speculative.md) |
| **Ternary weights** | `PQ2_0` / `PTQ1_0` group-128 ternary types with Hadamard-rotated activations, CUDA mat-vec / MMQ kernels. | [models.md](docs/models.md) |
| **More architectures** | Kolibri-1 (not in upstream); K2-Horizon-MoVA-36B-A4B, Qwen3.8 and Qwen3.8-Flash-Next, Ternary-Bonsai-2 (the first three are now also in upstream). | [models.md](docs/models.md) |
| **Automatic placement** | `-fit` (on by default) places layers, experts in RAM and context to the free device memory. | [models.md](docs/models.md) |
| **Claude Code** | Anthropic-compatible `/v1/messages` with `cache_control` prompt caching (non-streaming requests), launcher script. | [claude-code.md](docs/claude-code.md) |
| **Releases** | Linux CPU and Vulkan (x64, arm64), one CUDA archive per GPU generation (sm75, sm86, sm89, sm120), CPU Docker images. | [build.md](docs/build.md#release-archives), [docker.md](docs/docker.md) |
| **Vulkan backend** | Upstream `ggml-vulkan` with Turing tunings; TurboQuant types are CUDA-only. | [vulkan.md](docs/vulkan.md) |

## In flight

Nothing at the moment (2026-10-18).

## Planned

- **TurboQuant on every RTX generation:** the kernels are tuned on Turing only; tune them on Ampere, Ada and Blackwell
  (decode close to f16), better 2- and 3-bit K accuracy, a cheaper encoder scale search.
- **Long-context decode:** split the KV length across SMs (Stream-K style) for single-sequence decode at long context.
- **Upstream sync:** the quantized matmul (MMQ) and `rms_norm` kernels, which are behind upstream and cost Gemma 4
  1-2 % prefill; the meta backend of `-sm tensor` (Kolibri-1 abort, see below).
- **Speculative decoding:** an Eagle3 drafter (hidden-state extraction and GGUF plumbing are in; the drafter itself is
  not implemented).
- **Fine-tuning:** validate QAT (`--qat-target-quant`, CPU kernel only) on a task and add a CUDA kernel; backward for
  `SSM_SCAN` (Mamba), plain `GELU` and fused gate+up feed-forward layers; a faster CPU path and layer prefetch for
  experts in RAM; fewer small kernels on Turing. → [finetune.md](docs/finetune.md)
- **Releases:** the Blackwell archive with CUDA 12.8 (builds since the CUDA code no longer uses relocatable device code),
  a CUDA release for arm64 (GH200, DGX Spark).
- **Vulkan KTQ/VTQ port:** after the CUDA work; meant for AMD, Intel Arc and Pascal (untested whether Vulkan decodes
  faster there). → [vulkan.md](docs/vulkan.md)

## Known issues

- **TurboQuant:** the trellis types are slower than the codebook types (first prompt 1.5-1.9 times as long, decode
  12-34 % slower); 2- and 3-bit codebook K lose a lot of
  accuracy, and gpt-oss needs 8-bit K. Details: [turboquant.md#known-issues](docs/turboquant.md#known-issues).
- **Prefill on Gemma 4** is 1-2 % slower than upstream at long context
  ([benchmarks](docs/benchmarks/README.md#against-upstream-two-gpus)).
- **`-sm tensor` with Kolibri-1** aborts in `test-llama-archs` (meta backend); use `-sm layer`
  ([tensor-split.md](docs/tensor-split.md)).
- **MTP with several requests at once** decodes about 5 % slower than without speculation
  ([speculative.md](docs/speculative.md#draft_mtp-guidance)).
- **Fine-tuning:** models with plain GELU, fused gate+up feed-forward layers or Mamba layers cannot be trained yet
  ([finetune.md](docs/finetune.md#what-gets-a-gradient)).
- **Model-specific** (e.g. Gemma 4 perplexity on wikitext): [models.md#known-limitations](docs/models.md#known-limitations)

## Maintenance policy

- **Upstream sync:** upstream `llama.cpp` fixes are cherry-picked when they apply cleanly; larger features are
  integrated case by case and noted in the [changelog](CHANGELOG.md).
- **Stability target:** TurboQuant kernels on CUDA sm_75+, used daily on an RTX 2060. Vulkan and HIP are experimental;
  macOS / Metal are upstream stock.
- **Regressions are blockers:** new kernels are checked against the CPU backend (`test-backend-ops`) and by perplexity or
  KL divergence, speed changes by `llama-bench` against the previous build.
