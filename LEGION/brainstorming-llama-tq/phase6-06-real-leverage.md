# Phase 6 — What the Profiler Data ACTUALLY Tells Us (2026-04-27)

## TL;DR

Adaptive-k via cumulative softmax is dead, BUT the same calibration data
shows **massive expert-selection locality** that opens 3 viable performance
levers — none of which depend on the flat softmax distribution.

## The Real Numbers

### Expert hotness (which experts get top-8 selected?)

| Model | n_expert | Unique touched (top-8) | Top-10 dispatch share | Top-20 dispatch share |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 256 | **103/256 (40%)** | **45.5%** | ~65% |
| Qwen3-Next-80B-A3B | 512 | **118/512 (23%)** | **45.3%** | ~55% |

**Interpretation:** Even though softmax is flat over 256/512 experts,
top-k SELECTION is highly skewed. On 80B, ~45% of all expert dispatches
hit just 10 experts per layer. ~75% of experts are NEVER in top-8 in this sample.

### Per-layer top-1-expert share

| Layer | 80B top-20 share | top-1 share |
|---|---|---|
| 0-2 (early) | 21-35% | 2-5% |
| 3-46 (mid) | 50-65% | 7-9% |
| 47 (late) | 44% | 6% |

Mid-layer concentration is even stronger than the global average.

### Adjacent-layer overlap

```
80B layer-N top-10 ∩ layer-N+1 top-10 = 0.000  (mean)
```

Cross-layer prefetch hint: **dead.** Each layer has its own private hot set.

## Three Viable Levers

### Lever A: Per-layer hot-expert L3 pinning (recommended)

**What:** For each MoE layer, identify the top-20 hot experts offline. At
inference time, prefetch / pin those experts' weight blocks in L3 cache
during the prior layer's compute.

**Math:** 80B-IQ2_XXS expert = ~0.5 MB. 20 experts × 48 layers × 0.5 MB = 480 MB
hot working set, but we only need ONE LAYER pinned at a time = ~10 MB. 80B
host-RAM is DDR4-3200 (~40 GB/s real), Ryzen 7 3700X has 32 MB L3 split across
2 CCXs (16 MB each). **10 MB per layer fits comfortably in one CCX's L3.**

**Hit rate at top-20 pinning:** ~55-65% per layer (from data above). Means
~60% of expert reads avoid DDR4 entirely.

**Bandwidth math:**
- Current: every active expert read goes to DDR4 (~40 GB/s)
- After: 60% of reads stay in L3 (~600 GB/s), 40% still DDR4
- Effective bandwidth: ~150 GB/s — **3.7× current ceiling**
- TG win: model-dependent, but the bandwidth-bound regime would see the win
  proportionally. Expected: **+30-50% TG on 80B/122B CPU-offload deploys.**

**Cost:** software-side prefetcher. Implementation: `__builtin_prefetch` on
top-20 expert weight pointers right before layer N's mmid begins. Simple loop.
No quantization changes, no PPL risk.

**Risk:** L3 thrashing if other ggml ops compete (FA, residual, norm). Need
nsys profile to verify. But 10 MB out of 16 MB is generous headroom.

### Lever B: Static-prune always-cold experts

**What:** 153/256 experts on 35B (60%) and 394/512 on 80B (77%) **never** appear
in top-8 across our 64-token sample. Larger sample needed (32k tokens) but
trend is clear.

**Implementation:** During GGUF conversion, mark these experts. Either:
- Drop them entirely (quality risk if calibration set didn't hit them)
- Quantize them at Q1/IQ1 (vs IQ2_XXS) — saves ~50% storage, accepts
  rare-case quality dip when an unusual prompt activates them

**Win:** ~30% model size reduction on 80B (from 26GB → 18GB). Could move
122B from 35GB to 24GB → fits 2× 12GB VRAM without offload. **Massive.**

**Risk:** quality degradation on out-of-calibration prompts. Mitigated by:
- Run on multiple corpora (wikitext, code, agentic)
- Conservative threshold (prune only "never seen" not "rarely seen")
- Set bias to -inf for pruned experts at runtime, fall back to top-(n-pruned) if all top picks are pruned

### Lever C: Profile-guided per-expert precision (DynaExq-light)

**What:** Hot top-10 experts (45% of dispatch) → Q4_K_M (4.5 bpw).
Mid-tier 11-50 → IQ2_XXS (2 bpw, current). Cold 51+ → IQ1 (1 bpw) or pruned.

**Math:** Weighted avg bpw stays around current 2 bpw, BUT quality-weighted
bpw is much higher because the high-precision experts get the most dispatch.
Effectively: 45% of compute uses Q4_K_M weights = **~+4-6% MMLU likely**
without size growth.

**Cost:** GGUF conversion-side change (per-expert tensor metadata in tensor list).

## Recommendation

**Ship order:**
1. **Lever A first** (~3 days): per-layer hot-expert prefetcher. Pure runtime
   change, no quant changes, no PPL risk. Most-valuable single change.
2. **Lever B with caution** (1 week + extensive eval): static prune. Big win
   on 122B-fits-VRAM scenario but PPL risk needs careful threshold.
3. **Lever C as a Phase 7** (2+ weeks): per-expert precision. Bigger codebase
   change, quality eval matrix needed.

## What killed Phase 6 was over-fitting to the literature

The literature said "use cumulative softmax mass." That works on Mixtral.
On Qwen3 SOFTMAX_WEIGHT, the cumulative mass doesn't carry the signal —
**but the top-k selection still does**. The signal is in *which experts*
get picked, not in *how confident* the router is.

Phase 6a profiler successfully found this distinction in 30 minutes of
calibration data. That's the win.

## Files

- Profiler: `common/router-profile.{h,cpp}`, `tools/profile-router.py`
- Dumps: `gpu00:/tmp/router-35b-v3.bin`, `/tmp/router-80b.bin`
- Analysis: this file
