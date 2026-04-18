# TurboQuant Nano

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Ultra-low-bit KV-cache quantization for llama.cpp.**
**2.125 → 4.125 bpw. Lossless at 4-bit (+0.44 % PPL). Multi-GPU ready.**

Built on [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025)
and [QTIP](https://arxiv.org/abs/2406.11235), with a custom group-Viterbi
encoder and Flash-Attention-tuned CUDA kernels.

> Every KV token eats VRAM. `f16` costs 16 bits per value — we compress the
> V-cache to 2–4 bits with trellis-coded quantization. On Qwen3.5-0.8B at
> 4 bits the perplexity shift is within noise (+0.44 %). That's ~4× more
> context in the same VRAM budget, for free.

---

## Two generations, one build

| Family | bpw | Encode | Decode | Best for |
|---|---|---|---|---|
| **v1** — `KTQ*`, `VTQ*_1` | 2.5 · 3.5 · 4.5 · 5.5 | Codebook lookup | `__forceinline__` in FA | Fastest decode, drop-in |
| **v2** — `VTQ*_2` | 2.125 · 3.125 · 4.125 | Group-Viterbi (GPU) | Shift-register + LUT | Best quality/bit |

v1 is a 1-D Lloyd-Max codebook on the post-RHT marginal. v2 is a full
Viterbi DP across 65 536 states per block (N = 256 samples, L = 16
register). Both decode per-sample inside the FA inner loop.

---

## Quick Start

```bash
git clone https://github.com/LL4nc33/llama-tq && cd llama-tq
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda -j --target llama-cli

# 3-bit V-cache, near-lossless (+2.4 % PPL at 5× compression)
./build-cuda/bin/llama-cli -m model.gguf -p "hello" \
    --cache-type-k f16 --cache-type-v vtq3_2 -fa on -ngl 99
```

| Env | Default | Effect |
|---|---|---|
| `GGML_CUDA_VTQ_POOL_SLOTS` | `32` | v2 encoder workspace slots (1 per GB). `64` ≈ 25 % faster. |
| `GGML_TRELLIS_BEAM` | `0` (full Viterbi) | CPU-encoder beam width. `512` is 7–10× faster with small quality cost. |

---

## Benchmarks

All on **2× RTX 2060 12GB** (CC 7.5, PCIe 3.0), FA on, all layers offloaded.

### Quality — v2, Qwen3.5-0.8B

wikitext-2, 10 chunks, ctx 512, multi-GPU.

| V-Cache | PPL | Δ f16 | bpw | V-cache VRAM |
|---|---:|---:|---:|---:|
| `f16`     | 20.22 | — | 16.0  | 100 % |
| `vtq2_2`  | 21.76 | +7.6 % | 2.125 | **13 %** |
| `vtq3_2`  | 20.71 | +2.4 % | 3.125 | **20 %** |
| **`vtq4_2`** | **20.31** | **+0.44 %** | **4.125** | **26 %** |

`vtq4_2` error bars overlap `f16` — statistically indistinguishable at
10 chunks. `vtq3_2` is the sweet spot. `vtq2_2` is aggressive but still
within a few percent; good for VRAM-constrained agentic long-context.

### Throughput — v1 family

<details open>
<summary>Qwen3.6-35B-A3B UD-IQ2_XXS (10.01 GiB) — fits one GPU</summary>

| K | V | PP512 tok/s | TG128 tok/s | PP Δ | TG Δ |
|---|---|---:|---:|---:|---:|
| `f16` | `f16` | **820** | **60.1** | — | — |
| `q8_0` | `vtq2_1` | 757 | 58.7 | −8 % | **−2 %** |
| `q4_0` | `vtq2_1` | 756 | 58.8 | −8 % | **−2 %** |
| `q8_0` | `vtq3_1` | 735 | 58.1 | −10 % | −3 % |

v1's V-dequant is a single codebook lookup — lighter than `q4_0`'s per-lane
shift+scale. Measured: `vtq2_1` at 2.5 bpw beats `q4_0` at 5 bpw on PP512.
</details>

<details>
<summary>Qwen3.5-35B-A3B Q4_K_M (19.92 GiB) — pooled 2× 12 GB</summary>

| K | V | PP512 tok/s | TG128 tok/s | PP Δ | TG Δ |
|---|---|---:|---:|---:|---:|
| `f16` | `f16` | **842** | **61.9** | — | — |
| `q8_0` | `vtq2_1` | 781 | 60.4 | −7 % | **−2 %** |
| `q4_0` | `vtq2_1` | 783 | 60.4 | −7 % | **−2 %** |
</details>

<details>
<summary>Qwen3.5-27B Dense Q4_K_M (15.94 GiB)</summary>

| K | V | PP512 tok/s | TG128 tok/s | PP Δ | TG Δ |
|---|---|---:|---:|---:|---:|
| `f16` | `f16` | **318** | **14.6** | — | — |
| `q8_0` | `vtq2_1` | 297 | 14.5 | −7 % | **−1 %** |
| `q4_0` | `vtq2_1` | 297 | 14.5 | −7 % | **−1 %** |

Dense models give up more on PP512 (no MoE sparsity), same ~1–2 % TG tax.
</details>

### Throughput — v2 family

v2 encodes via CUDA Viterbi DP (pool-allocated workspace, 32 slots default).
Decode is shift-register + LUT lookup — same order of magnitude as v1.
Encode runs once per cache write; amortized per token it's negligible.

_Full v2 throughput table across 35B and 27B pending — rerun_
_[`scripts/bench_tq.sh`](scripts/bench_tq.sh) to regenerate._

### Run your own bench

```bash
scripts/bench_tq.sh ~/models/Qwen3.5-35B-A3B-IQ2_XS.gguf
# CSV + markdown in bench-<timestamp>/
```

Runs the full `(ctk, ctv)` matrix twice: once isolated to GPU 0 (single-GPU),
once with all visible GPUs (multi-GPU pooled). `SKIP_SINGLE=1` /
`SKIP_MULTI=1` to pick one side.

---

## Choose Your Config

| You need… | Use | Cost |
|---|---|---|
| Drop-in, no quality regression | `vtq4_2` | +0.44 % PPL, 3.88× compression |
| Best bpw/quality curve | `vtq3_2` | +2.4 % PPL, 5.1× compression |
| Maximum compression | `vtq2_2` | +7.6 % PPL, 7.5× compression |
| Fastest FA decode | `vtq2_1` (v1) | 2.5 bpw, codebook-only inner loop |
| CPU-only inference | `vtq{K}_2` + `GGML_TRELLIS_BEAM=512` | 7–10× slower encode |

---

## Technical

- **RHT** (Randomized Hadamard Transform) — applied at graph level for V
  (once per forward pass), not per block. Post-rotation marginals are
  Gaussian-like so 1-D codebooks are near-optimal.
- **Group Viterbi** (v2) — N = 256 samples per block, L = 16 state register,
  K bits emitted per sample, full 65 536-state DP. Per-device `__device__`
  LUT (256 KiB), multi-GPU clean.
- **Flash-Attention integration** — v1 V-dequant is `__forceinline__` in the
  FA vec path. v2 uses the same shift-register read but routes via mul-mat
  on V until v2 FA-vec template instances land.
- **CUDA encoder** — 1 CUDA block per trellis-block, 256 threads,
  packed `atomicMin((cost_u << 32) | prev)` on workspace DP rows. Pool of
  32 slots saturates the 60 SMs of dual RTX 2060 at ~1 GB VRAM.

Deep dive: [`docs/turboquant.md`](docs/turboquant.md),
[`tests/trellis-phase1/`](tests/trellis-phase1/),
[`tests/trellis-phase1/results/RUN7_NOTES.md`](tests/trellis-phase1/results/RUN7_NOTES.md).

---

## Status

**Production:**
- v1 family — FA CUDA kernels shipped for all (K, V) pairs.
- v2 family — CUDA encoder + decode, CPU fallback, multi-GPU validated
  (correctness on 2× RTX 2060, PPL matches reference).

**In flight:**
- v2 FA-vec template instances (unlocks native FA for v2 V-type).
- 35B MoE wikitext-2 PPL table (sweep running — [`tests/trellis-phase1/results/`](tests/trellis-phase1/results/)).
- Nsight-compute profiling to reduce encoder atomicMin contention.

---

## References

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Redefining AI Efficiency
  with Extreme Compression.* arXiv:2504.19874 (2025).
- QTIP: Quantization with Trellises and Incoherence Processing.
  arXiv:2406.11235.
- Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT licensed.

## License

MIT (inherited from llama.cpp).
