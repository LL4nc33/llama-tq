# MoE Expert-Selection Locality on Qwen3 Family

**Date:** 2026-04-27 · **Models profiled:** Qwen3.6-35B-A3B-IQ2_XXS, Qwen3-Next-80B-A3B-IQ2_XXS

## Summary

Even though Qwen3-family routers produce a near-uniform softmax distribution over their 256 / 512 experts (`LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT`: softmax runs *after* top-k, on the 8–10 selected weights only), the **top-k selection itself is highly skewed**. A small fraction of experts captures the majority of dispatches per layer. This opens performance levers for CPU-offload deployments without touching the quantization stack.

## Numbers

### Expert hotness

| Model | n_expert | Touched in top-8 | Top-10 dispatch share | Top-20 dispatch share |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 256 | 103 / 256 (40%) | 45.5 % | ~65 % |
| Qwen3-Next-80B-A3B | 512 | 118 / 512 (23%) | 45.3 % | ~55 % |

On 80B, ~75 % of experts never appear in top-8 within a 64-token sample. A larger 256-token / 40-layer run on 35B raised mean top-20 share to **61.1 %**, with mid-layers (8–21) consistently above 70 % and a peak of 83.6 % at layer 20.

### Per-layer concentration (80B, 64 tokens)

| Layer band | Top-20 dispatch share | Top-1 share |
|---|---|---|
| 0–2 (early)   | 21–35 % | 2–5 % |
| 3–46 (mid)    | 50–65 % | 7–9 % |
| 47 (late)     | 44 %    | 6 %   |

### Adjacent-layer overlap

Layer-N top-10 ∩ Layer-(N+1) top-10 ≈ 0 (mean over 48 layers, 80B). **Each layer has its own private hot set.** Cross-layer prefetch hints based on the previous layer's selection do not help.

## What the data implies

Three independent levers, none of which require re-quantization:

### Lever A — Per-layer hot-expert L3 pinning

Identify the top-N hot experts per layer offline. Issue `__builtin_prefetch` on those expert weight pointers right before that layer's `mmid` op begins. Hides DDR4 latency on CPU-offload deployments.

Bandwidth math (Ryzen 7 3700X, 32 MB L3 / 16 MB per CCX, DDR4-3200 ≈ 40 GB/s effective, 80B-IQ2 expert ≈ 0.5 MB):

| Hit rate (top-20) | Effective bandwidth |
|---|---|
| 0.55 (mean, original sample) | ~150 GB/s (3.7× ceiling) |
| 0.61 (mean, 256-tok sample)  | ~382 GB/s (9.5× ceiling) |
| 0.80 (mid-layers)            | ~488 GB/s |

Per-layer hot working set: 20 experts × 0.5 MB = 10 MB, fits in one CCX's L3 (16 MB) with headroom. Expected end-to-end TG win: **+30–50 %** on 80B/122B CPU-offload, model- and bandwidth-bound.

### Lever B — Static prune of always-cold experts

35B: 153/256 experts (60 %) never selected in the sample. 80B: 394/512 (77 %). Conservative threshold (only "never seen" across multiple corpora, not "rarely seen") allows:

- Drop entirely, or
- Quantize at IQ1 (1 bpw) instead of IQ2_XXS (2 bpw)

Storage win on 80B: ~30 % (26 GB → ~18 GB). On 122B: ~30 % (35 GB → ~24 GB) — the latter would fit on 2× 12 GB without offload. Quality risk is real and requires multi-corpus calibration (wikitext + code + agentic) plus a runtime fallback to top-(n − pruned) when all selected are pruned.

### Lever C — Profile-guided per-expert precision

Hot top-10 experts (45 % of dispatch) → Q4_K_M. Mid-tier 11–50 → IQ2_XXS (current). Cold 51+ → IQ1 or pruned. Weighted average bpw stays near current; quality-weighted bpw rises substantially because high-precision experts handle most compute. Expected MMLU lift without size growth.

## Why the literature numbers don't translate

- **Mixtral 8×7B**: 8 experts, full softmax over all 8. Top-2 native, top-1 carries 60–80 %. Adaptive-k via cumulative softmax mass works here.
- **Qwen3 family**: 256 / 512 experts, `SOFTMAX_WEIGHT` (softmax post-top-k). Distribution is intentionally flat — top-k threshold reduction over the full distribution is meaningless on this family.

The signal is in *which* experts get picked, not in *how confident* the router is. Adaptive-k via softmax cumulative mass cannot work on Qwen3; expert-locality and selection-skew can.

## Tooling

- Profiler: `common/router-profile.{h,cpp}`, flag `--log-router-stats` on `llama-perplexity`
- Analyzer: `tools/profile-router.py`
- Output: per-token, per-layer top-k expert IDs and logits, JSON summary

## Status

Lever A is the recommended next step — pure runtime change, no quant or PPL risk. Levers B and C require larger calibration runs and quality-eval matrices before commit.
