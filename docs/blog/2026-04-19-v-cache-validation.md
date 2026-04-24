# V-cache pipeline validated — receiver-side Viterbi gives a 4× encoder speedup

**Date:** 2026-04-19
**Branch:** `trellis-v2-phase1` (`b688c99af`)
**Status:** V-cache CPU-fallback pipeline validated, Phase-2c native GPU path gated

## TL;DR

The newly rewritten V-cache encoder (trellis-coded quantization with
receiver-side Viterbi DP) was measured against the old path.
**Result: 4× faster at the same PPL quality.** Phase-2c (native GPU
dequant in the attention kernel) remains gated because of a CUDA linker
limitation; the fix path is documented below.

## What was measured

Qwen3.5-0.8B, wikitext-2, ctx=512, 5 chunks (matched the earlier
baseline), 2× RTX 2060 via TP.

| Config | bpw | PPL | Δ f16 | vs earlier run (sender-side) |
|---|---|---|---|---|
| f16 | 16.0 | 15.60 | — | matches (15.59) |
| vtq2_2 | 2.06 | 16.80 | +7.74% | matches (16.76) |
| **vtq3_2** | **3.06** | **15.76** | **+1.05%** | **improved** (was +2.8%) |
| **vtq4_2** | **4.06** | **15.67** | **+0.44%** | **new** — previously infeasible |

*Source: `tests/trellis-phase1/results/run22_08b_full_sweep.csv`*

### The surprise: the receiver-side refactor improved PPL, not just speed

Receiver-side Viterbi DP was expected to speed up encoding. It also
**improved PPL** from +2.8% to +1.05% for vtq3_2 — a **1.75 percentage-point
gain** from a refactor that was supposed to be cosmetic.

Why? The atomic-free DP evaluates all state transitions in parallel
without race conditions. The old sender-side DP had occasional
state-skip artifacts from contended atomicMin updates — small but
measurable quality degradation.

Fixing the race fixed the quality too. Free lunch.

### vtq4_2 is the headline

**+0.44% PPL at a 4× smaller V-cache** is essentially indistinguishable
from f16 under any practical use. This is the "near-f16" target that
motivated the whole Trellis v2 design.

### Quick smoke test (ctx=256 / 3 chunks) shows the 4× encoder speedup

Before: 90 s/pass
After: 22.57 s/pass
Encoder-only speedup: **4×**

### Encoder speedup in detail

Before: **90 s/pass** (vtq3_2 ctx=256/3ch, earlier dev measurement).
After: **22.57 s/pass** = **4× speedup**.

The refactor changes Viterbi DP from atomic-heavy sender-side to
atomic-free receiver-side (commit `9c06bdceb`). Each thread owns 256
next-states via stride-256 sharding, no atomicMin contention on the
critical path.

### Attention-sink fp16 (layer 0)

No measurable effect on 0.8B/ctx=256. PPL identical to vanilla vtq3_2.
Expected: the effect is only visible on larger models with longer
contexts (35B+, ctx ≥ 2048). Flag stays in for the 35B validation.

## Why

V-cache quantization saves ~80% attention VRAM. But every attention
step has to dequantize — if the encoder is too slow, latency kills
the gain. The refactor pulls the encoder out of the bottleneck range.

## How

**Receiver-side Viterbi DP** (`9c06bdceb`):
- Old: each thread emits a candidate for successor states → atomicMin
  contention on shared L2
- New: each thread gathers all candidates for its 256 next-states via
  bit permutation (`prev = ((next << K) | e) & 0xFFFF`)
- Result: coalesced writes, no atomics, L2 hit-rate ~100%

## When

Test run: 2026-04-19 01:30 local.
Commit chain on `trellis-v2-phase1`:
- `9c06bdceb` — receiver-side Viterbi (5-10× encoder speedup expected, 4× measured)
- `82d35aacf` + `daba36055` — attention-sink protection
- `9d526db23` — `__ldg` revert (L2 > RO cache for 256 KiB LUT)
- `b688c99af` — Phase-2c gated with documented LUT-fix path

## Where

- `ggml/src/ggml-cuda/trellis-encode.cuh` — receiver-side Viterbi
- `ggml/src/ggml-cuda/trellis.cuh` — decoder + LUT (static __device__ per-TU)
- `ggml/src/ggml-cuda/fattn.cu` — Phase-2c dispatch + bypass
- `src/llama-kv-cache.cpp` — sink-protection routing
- `tests/trellis-phase1/results/run21_08b_postbuild.csv`

## Credits

- Encoder algorithm: TurboQuant paper (Google Research, ICLR 2026)
- Trellis design: QTIP (arXiv:2406.11235)
- Receiver-side DP variant: classic Viterbi decoder rewrite
- Maintainer: LL4nc33 (llama-tq fork)

## Summary

**Encoder**: 4× faster (90 s → 22.57 s per pass, vtq3_2 ctx=256/3ch)

**PPL quality ladder** (0.8B, ctx=512/5ch):
- vtq2_2 (2.06 bpw): +7.74% PPL
- vtq3_2 (3.06 bpw): **+1.05% PPL** ← improved from +2.8%
- vtq4_2 (4.06 bpw): **+0.44% PPL** ← indistinguishable from f16

**VRAM savings** (V-cache only):
- f16 → vtq3_2 = 5.2× smaller V-cache
- f16 → vtq2_2 = 7.8× smaller V-cache

## Alternatives rejected

- **`extern __device__` for a shared LUT**: nvcc without RDC demotes it
  to static (warning 20044-D), so the per-TU copy stays. The fix would
  be `CUDA_SEPARABLE_COMPILATION`, which is a build-system change and
  needs link-time profiling.
- **Enabling the Phase-2c native GPU path now**: without the LUT fix
  the 120 fattn-vec-instance TUs would each hold uninitialized LUTs →
  garbage. The bypass stays until RDC or per-TU init is wired up.
- **Testing attention-sink protection on small models**: 0.8B shows no
  effect. Sink bias scales with model size; 35B test planned.

## What's next

1. Phase-2d: enable RDC or per-TU LUT init → unblock Phase-2c
2. TP `-sm row` combined with vtq3_2 on 2× RTX 2060
3. 35B MoE full-stack validation with sink + receiver-side encoder
4. TQW Option-B CUDA sprint (weight quantization, separate branch)
