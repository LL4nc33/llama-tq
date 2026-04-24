# Native V-dequant on the GPU — away from the CPU fallback

**Date:** 2026-04-18
**Branch:** `trellis-v2-phase1`
**Status:** Build in progress, smoke-test pending

## TL;DR

The V-cache (value side of attention) is quantized to ~3 bit instead of 16 bit.
Until now the kernel dropped back to CPU for that dequant — slow. Now the full
path runs on the GPU. Expected speedup: **1.5-2× on long contexts** (where the
V-cache gets large).

## What

Phase-2c of the Trellis v2 project: native Flash-Attention-vec kernel for the
new types `VTQ2_2`, `VTQ3_2`, `VTQ4_2`. They dequantize the trellis-coded
V-cache directly inside the GPU attention kernel, no host round-trip.

## Why

The old setup had a bypass: if V-cache type was `VTQ_2`, flash-attention
fell back to the CPU path. Every attention step then needed GPU → CPU → GPU
data transfer. On long contexts (4k+) that becomes the dominant bottleneck.

## How

Three changes:

1. **LUT refactor** — the 256 KiB trellis lookup table was declared
   `static __device__` per translation unit. With ~120 fattn-vec-instance
   TUs, each got its own uninitialized copy → garbage output. Fix:
   `extern __device__` with one central definition in `trellis.cu`.

2. **Per-element decoder** — `trellis_decode_element<K>(start_state, d, qs, j)`
   replicates the shift register for arbitrary indices, so FA-vec threads
   can read arbitrary V elements.

3. **Dispatch matrix** — 18 template instances per VTQ_2 type
   (K ∈ {F16, Q8_0, KTQ1-4_1} × D ∈ {64, 128, 256, 512}).

## When

- `7b2f94975` — Phase-2c scaffolding (gated bypass active)
- `9c06bdceb` — receiver-side Viterbi encoder (atomic-free)
- `82d35aacf` — attention-sink fp16 protection
- `1520cb117` — **LUT extern refactor — unblocks native path**

## Where

- `ggml/src/ggml-cuda/trellis.cuh` — decoder, LUT declaration
- `ggml/src/ggml-cuda/trellis.cu` — LUT definition, encoder pool
- `ggml/src/ggml-cuda/fattn.cu` — dispatch matrix
- `ggml/src/ggml-cuda/fattn-common.cuh` — `dequantize_V_*` templates

## Credits

- Maintainer: **LL4nc33** (llama-tq fork)
- Paper basis: **TurboQuant** (Google Research, ICLR 2026)
- Trellis design: **QTIP** (arXiv:2406.11235)
- Upstream: **llama.cpp** team (MIT)

## Measured so far

### Qwen3.5-0.8B (dev loop, wikitext-2)

| Config | bpw  | PPL   | Δ f16   |
|--------|------|-------|---------|
| f16    | 16.0 | 15.59 | —       |
| vtq3_2 | 3.06 | 16.02 | +2.76%  |
| vtq2_2 | 2.06 | 16.76 | +7.50%  |

*Source: `tests/trellis-phase1/results/run19_08b_dev_sweep.csv`*

### TQW2 Python validation (weights, separate branch)

| Variant | bpw    | MSE vs IQ2_XXS |
|---------|--------|----------------|
| TQW2    | 2.0625 | **−62-65%**    |
| TQW3    | 3.0625 | **−89%**       |

*Source: `tests/trellis-phase1/results/run20_tqw2_mse.csv`*

### Qwen3.5-35B MoE (deployment baseline, TQ v5)

| K type   | V type   | KV VRAM | PPL Δ f16 |
|----------|----------|---------|-----------|
| f16      | f16      | 100%    | —         |
| TQ2_1    | q4_0     | ~22%    | +1.2%     |

*200K context / slot on the test hardware.*

## Alternatives rejected

- **`__ldg` on the LUT** — Turing's RO cache is 48 KiB, LUT is 256 KiB.
  `__ldg` forced ~20% hit rate with L2 refill stalls on the atomicMin
  critical path. Plain global loads go directly to L2 (~100% hit after
  warmup). **Reverted in `9d526db23`**, 2-3× encoder speedup recovered.

- **KTQ3_1 as weight alias** — KTQ stores data in the Hadamard domain
  and uses the FA-kernel "push FWHT to Q" trick. `mul_mat` with
  physical-domain activations lacks the inverse-FWHT dequant. **Abandoned**;
  moved to a separate branch with its own physical-domain TQW3 format.

- **Per-model RHT seed calibration** — kurtosis sweep showed std 0.0044
  ≈ noise floor 0.0048. Post-RHT marginals are already near-Gaussian;
  per-model salt cal would sit under the measurement resolution.
  **Abandoned** with a documented null result.

## What's next

1. Smoke-test Phase-2c on 0.8B (gate: PPL ±0.05 vs run19 baseline)
2. TG benchmark native vs CPU fallback (real ms/token numbers)
3. Attention-sink fp16 validation in layer 0
4. TP `-sm row` combined with VTQ_2 on 2× RTX 2060
5. 35B MoE full-stack validation
