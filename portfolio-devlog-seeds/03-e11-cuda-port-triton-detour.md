---
slug: "e11-cuda-port-triton-detour"
title: "E11 CUDA Port — was wir aus dem Triton-Strategy-A-Detour gelernt haben"
date: "2026-04-22"
status: "shipped"
tags: ["turboquant", "vtq2", "cuda", "triton", "decision-log"]
hero_number: "60× regression → reverted"
hardware: "RTX 2060 sm_75"
---

# E11 CUDA Port — Triton Detour Decision Log

Phase 3A1 sollte VTQ_2 (Trellis V-cache) per warp-cooperative cached decode beschleunigen. Spec war E11: full-matrix expansion, kollektives load von 16 trellis states pro warp. Strategy A war ein Triton-prototype statt direkt CUDA.

**Spoiler:** Wurde reverted weil 60× slower than baseline. Aber der Detour war wertvoll — deshalb hier ein Decision-Log.

## Hypothesis

VTQ_2 trellis decode hat eine bottleneck: jeder thread decoded 1 state per shift register cycle. Bei head_dim=128 → 128 sequential shifts per thread per token. **Wenn wir warp-cooperative parallelisieren** (32 threads × 4 states each = 128 in 1 cycle), sollten wir 32× faster sein. Conservatively assume 10-20× real-world.

## Strategy A — Triton prototype

**Warum Triton:** Schneller iteration cycle als CUDA (kein nvcc rebuild, run-on-save). Plus Triton's autotuning könnte block-size für sm_75 finden.

**Erste implementation:** `vtq2_decode_warp_e11.py` — 16 trellis states per warp, parallel decode.

**Erste Bench:** 60× slower than baseline.

## Was schiefging

Drei Probleme stacked:

1. **Triton überhead ist real auf sm_75.** Compute-capability 7.5 hat keine native bf16, kein TMA, kein cluster scheduling. Triton's emitted PTX ist fast-1:1 zu hand-written CUDA, aber das `block_ptr` setup + barrier instructions fügen ~µs constant overhead per dispatch hinzu. Bei kleinen kernels (decode = 1 token) ist das katastrophal.

2. **Warp cooperative load was nicht der bottleneck.** Profile zeigte: actual decode time war ~5% der dispatch latency. Speedup-cap war ~1.05× egal wie viele threads warp-cooperate.

3. **Cache thrashing.** Jeder warp lud 16 states aus einem kleinen LUT. 32 warps × 16 states = 512 fetches gegen einen ~64-entry cache. L1 thrashed zwischen warps statt zu sharen.

## Decision: Strategy A reverted, Strategy B (CUDA) ohne warp-cooperative E11

**Reverted commit:** `141 [completed] CRITICAL: VTQ_2 TG regression 15x vs VTQ_1 (4.32 vs 66.5 tok/s)` — die initiale 15× regression war ein anderes E11-bug (block-size 256 too large), und die warp-cooperative wurde im gleichen Sweep entfernt.

**New design:** Strategy B = direct CUDA kernel ohne warp-cooperative. Per-thread shift-register decode bleibt, aber:
- `__forceinline__` für vec_dot_KQ und dequantize_V — vermeidet local memory spills
- Codebook in `__constant__` memory (broadcast-friendly auf Turing)
- Sparse V dequant: skip dequant für positions mit attention_weight < 1e-6 (>90% der positions bei 32K+ ctx)

## Was wir gewonnen haben aus dem Detour

1. **Profile first, optimize second.** Wir hatten den theoretischen 32× speedup im Kopf, nicht die actual Profile-Zahlen. Eine 30-min Nsight-session vor dem Refactor hätte gezeigt: dispatch latency dominiert, nicht decode compute.

2. **Triton ist nicht universell schneller.** Auf data-center GPUs (H100 mit TMA + cluster + bf16) sicher. Auf sm_75 Turing ohne diese Features = ~equivalent zu CUDA, aber mit höherem dispatch overhead.

3. **Sparse V Dequant emerged as actual win.** Wir hatten Sparse V als afterthought im Backlog. Nach dem failed warp-cooperative attempt war klar: **don't optimize the dequant, eliminate it**. +22% real speedup deployed.

## Numbers

| Config | tg128 (t/s) | Note |
|---|---:|---|
| VTQ_2 baseline (Strategy B initial) | 4.32 | broken — block size 256 too large |
| VTQ_1 baseline | 66.5 | reference |
| E11 warp-cooperative (Triton) | <1 | catastrophic, 60× regression |
| Strategy B fix (block 128) + sparse V dequant | **~75** | **deployed** |

## Lessons

1. **Hypothesis-driven optimization braucht profile-driven validation.** Sonst ist es nur hope.
2. **Hardware-specific shortcuts.** Was auf Hopper/Blackwell shines (Triton + WGMMA + TMA) ist auf Turing nur Overhead.
3. **Negative results haben quotable Headlines.** "60× regression" ist memorierbarer als "+1.05× speedup". Documenting failure builds trust.
4. **Backlog matters.** Sparse V Dequant war als "nice to have" markiert. Nach E11-Detour wurde es Phase-3-priority und der eigentliche win.

## Code references

- Failed Triton prototype: `141` task closed, branch `vtq2-e11-triton` deleted post-revert
- Final Strategy B: VTQ_2 + Sparse V Dequant in current `master` of llama-tq fork
- Trellis decode logic: `ggml/src/ggml-cuda/fattn-tq.cuh` (post-fattn-bloat-fix) and `ggml-trellis.{h,c}`

---

**Status:** shipped. VTQ_2 + Sparse V live deployed seit 2026-04-22.
