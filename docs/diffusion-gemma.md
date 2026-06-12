# DiffusionGemma

`llama-tq` runs **DiffusionGemma** — Google's open-weight text-diffusion MoE
(26B total / ~4B active, 30 layers, 256-token denoising canvas, 256k context) —
coherently at aggressive low bit-widths on a single consumer GPU.

Unlike an autoregressive model, DiffusionGemma denoises a whole 256-token canvas
in parallel over a handful of steps, committing finished blocks to the KV cache
and chaining blocks for longer answers. That changes the memory profile in a way
this fork exploits: the denoise compute buffer is **context-independent**, so long
context costs almost only KV — not graph memory.

## What works

- **Coherent 2-bit generation on a single 12 GB GPU.** Mixed-precision quant
  (router + critical attention/FFN tensors kept higher, the bulk at IQ2) plus a
  decoder-path importance matrix and inference-time stabilisers bring a model that
  is garbage at naive 2-bit up to readable output — well below the ~17 GB floor of
  off-the-shelf 4-bit/NVFP4 builds.
- **Position-dependent KV cache.** The active 256-token canvas is kept at f16 for
  coherence; the read-only committed history can use TurboQuant KV (KTQ/VTQ), so
  long context fits where a uniform f16 cache would not. See
  [docs/turboquant.md](turboquant.md).
- **Decoder-path imatrix.** An importance matrix collected from the *real denoise
  loop* (not an autoregressive prefill), so quantisation optimises the distribution
  the model actually runs in.
- **Embedded WebUI.** The diffusion server ships the standard llama.cpp web UI,
  including a live view of the denoising process.

## In flight

- **Aggressive low-bit self-conditioning** to push the weight footprint lower while
  holding coherence, targeting **full 256k context per single 12 GB GPU** so one
  instance can run per card.

## Running it

The model loads through the diffusion CLI / server entry points. Coherent low-bit
deployment combines the mixed-precision quant, the decoder-path imatrix, and the
position-dependent KV cache; the canvas must stay f16. See the build docs for
prerequisites — [docs/build.md](build.md).
