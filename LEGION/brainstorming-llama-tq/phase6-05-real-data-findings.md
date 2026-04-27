# Phase 6a — Real-Data Findings (DECISION GATE: ABORT)

**Date:** 2026-04-27
**Status:** Profiler implemented, validated on Qwen3.6-35B-A3B + Qwen3-Next-80B-A3B
**Verdict:** **ABORT adaptive top-k for Qwen3 family. Pivot recommended.**

---

## Numbers

Profiler: `tools/profile-router.py` analyzing `--log-router-stats` dumps from
`llama-perplexity` (CPU-only, prod GPU untouched).

### Qwen3.6-35B-A3B-UD-IQ2_XXS (40 layers × 256 experts, top-8)

```
records         : 2560 (64 tokens × 40 layers)
n_expert        : 256
mean_k @ τ=0.85 : 145.68     (gate threshold: < 5)
p99_k           : 175
max_k           : 181
mean top-1 prob : 0.0590     (uniform-256 = 0.0039, so ~15× uniform — flat)
median top-1 p  : 0.0481
GATE           : ABORT
```

Per-layer mean_k ranges from 124 (layer 0) to 167 (layer 38). NO layer has
mean_k < 100. The "early-layers-uniform, late-layers-peaked" pattern from
Huang et al. does NOT hold here — late layers are MORE flat (top1 drops to 0.03).

### Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS (48 layers × 512 experts, top-10)

```
records         : 3072 (64 tokens × 48 layers)
n_expert        : 512
mean_k @ τ=0.85 : 172.16     (gate threshold: < 5)
p99_k           : 277
max_k           : 317
mean top-1 prob : 0.0861     (uniform-512 = 0.0020, so ~43× uniform — slightly less flat)
median top-1 p  : 0.0722
GATE           : ABORT
```

Both models miss the gate by **~30×**.

## Root cause

Qwen3 (and Qwen3-Next) use `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT`:

```cpp
// src/llama-graph.cpp:1320-1326
case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
    {
        probs = logits;  // pass logits through, NO softmax here
    } break;
```

The softmax runs only on the `n_expert_used = 8` (or 10 for 80B) **selected
weights**, AFTER top-k:

```cpp
// src/llama-graph.cpp:1391-1396
if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
    weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
    weights = ggml_soft_max(ctx0, weights);  // softmax over 8 (or 10), not 256/512
    ...
}
```

This means: the model was **trained** so that meaningful expert ranking happens
at the logit level (top-k selection), not at the probability level. The
post-top-k softmax is a per-token weight-normalization, not a confidence signal.

**Implication:** Cumulative-mass threshold over the full 256/512-expert
distribution is meaningless — the distribution is intentionally flat. Adaptive-k
based on softmax cumulative mass cannot work on this family.

## Why the literature numbers don't translate

- **Mixtral 8x7B**: 8 experts, conventional `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`
  (full softmax over all 8 experts). Top-2 native, top-1 carries 60-80%. Adaptive-K
  literature data is here.
- **DeepSeek-V3**: sigmoid + per-group routing. Different game.
- **Qwen3 family**: 256/512 experts, SOFTMAX_WEIGHT. **Different gating mechanism
  the literature did not analyze.** Expected to be flatter (the auto-research
  predicted "less peaked than Mixtral") but the magnitude — ~145-172 mean_k vs
  expected 3-5 — is far worse than predicted.

## Phase 6 status: ABORT

Per the spec's bidirectional decision gate:
- mean_k > 5 across all profiled models (35B + 80B both miss by ~30×)
- → adaptive top-k via cumulative-mass threshold is not viable on Qwen3 MoE

## What still has value from this work

The profiler infrastructure (`common/router-profile.{h,cpp}`,
`tools/profile-router.py`, `--log-router-stats` flag) is **reusable** for several
follow-ups that share the same calibration data:

### Pivot 1: Expert-locality L3 prefetch (Innovator §2.3)
Most-promising follow-up. Use the profiler-collected expert-id traces (per token,
per layer: which experts got selected) to:
- Build a per-document Markov chain on expert transitions.
- Issue `__builtin_prefetch` on layer N+1's predicted hot experts during layer N's
  mmid execution.
- Hide DDR4 latency on CPU offload — could exceed adaptive-k's projected win
  without any quality cost (pure prefetch, no quantization change).

Requires extending the profiler to also dump `selected_experts-N` (top-k indices)
in addition to logits. ~1 day of work on top of Phase 6a.

### Pivot 2: Profile-guided gating bias (Innovator §2.7)
Identify experts that are **systematically tail** (always rank > 100 in their
layer). Set their gate bias to -infinity at quant time → static prune.
Same profiler infrastructure, no runtime cost.

### Pivot 3: DynaExq-style hot-expert precision boost (Innovator §2.4)
Identify hot experts (frequent in top-8) and re-quantize them to higher bpw
during model conversion. Cold experts stay at IQ2. Storage iso-VRAM, accuracy +.
Uses the same profiler "expert hotness" data.

## Recommendation

1. **Ship the Phase 6a profiler** as a reusable tool — it already demonstrated
   value by killing a multi-day implementation effort with a 30-minute measurement.
2. **Update spec** to mark Phase 6b/6c as ABORTED, document why.
3. **Pivot to expert-locality prefetch** as Phase 6'. Same profiler data, no
   per-token routing changes (zero PPL risk), targets the same DDR4 ceiling on
   CPU-offload deploys.

## Files

- Profiler: `common/router-profile.{h,cpp}`, `tools/profile-router.py`
- Dumps: `/tmp/router-35b-v3.bin`, `/tmp/router-80b.bin` (on dev box)
- Reports: `/tmp/router-35b-v3.json`, `/tmp/router-80b.json`
- Spec to update: `docs/plans/2026-04-27-phase6-adaptive-topk-moe.md`
