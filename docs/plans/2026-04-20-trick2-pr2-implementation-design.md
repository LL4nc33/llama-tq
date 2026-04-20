# Trick 2 PR2 — Per-Head Precision Mixing Implementation Design

**Status:** Proposed
**Date:** 2026-04-20
**Author:** architect (CWE)
**Target branch:** `trick2-pr2-mixed-precision` (branched from `trick2-pr1-profile-heads` @ 71b56f7fb)
**Scope:** V-cache only (K-cache stays uniform — reuse PR3 territory)

## 1. Executive Summary

Phase 1 shipped three V-Trellis types (VTQ2_2 / VTQ3_2 / VTQ4_2 at 2.06 / 3.06 / 4.06 bpw).
PR1 added a profiling hook exposing per-(layer, head) variance + kurtosis.

Profile Qwen3-0.6B (28 × 8, 50 decode samples) shows:

- Median variance ratio (max/min-head per layer) = 2.19x
- 6/28 layers with ratio > 4x, one outlier at 12.28x (L13)
- Kurtosis range 90–7421 → heavy tails in a minority of heads

**Conclusion:** Heterogeneity lives primarily **across layers**, not across heads within a layer.
A few layers (≈20%) carry disproportionate sensitivity.

**Recommendation:** Ship **per-layer mixed precision** first (PR2). Per-head is deferred to PR3.

Expected outcome: ~3.0 bpw average, PPL regression dropping from uniform VTQ3_2's +1.9% to a projected **+0.6 to +1.0%** (hypothesis, see §6).

---

## 2. Granularity Decision: Per-Layer

### Why Per-Layer (not Per-Head)

| Axis | Per-Layer | Per-Head |
|---|---|---|
| Changes to `type_v` | `type_v` → `std::vector<ggml_type>` (one per layer) | type varies within a tensor row — impossible in ggml today |
| FA kernel | Already dispatches per-type at `fattn.cu` call site | Needs head-aware kernel dispatch + striding logic |
| Tensor layout | One tensor per layer, different block-size OK | Would need split V tensor per head-group (8x tensor count) |
| Qdrant-style savings | Captures 80% of heterogeneity per profile data | Extra 10-20% gain, 5x implementation cost |
| Profile evidence | 6 layers with ratio > 4x — layer-level signal is strong | Within-layer max/min ratio is the *same* signal aggregated |

Per-head would require restructuring V as `n_head` separate tensors (or a new ggml "mixed block" type). Out of scope for PR2. File as **PR3 stretch**.

### Per-Layer Data Model

```cpp
// src/llama-kv-cache.h — change (line ~99 ctor signature)
llama_kv_cache(
    const llama_model & model,
             ggml_type   type_k,
    const std::vector<ggml_type> & type_v_per_layer,  // NEW — size n_layer
    // ... rest unchanged
);
```

Fallback: if vector is size 1, broadcast to all layers (backward-compat with existing callers).

---

## 3. Selection Heuristic

### CLI (recommended)

```
--tq-v-profile <path.json>          # PR1 JSON output, enables auto-selection
--tq-v-strategy <strategy>          # how to pick upgrades
--tq-v-base <type>                  # default: vtq3_2
--tq-v-high <type>                  # default: vtq4_2
--tq-v-low  <type>                  # default: vtq2_2
--tq-v-override "LAYERS:TYPE,..."   # manual override, highest priority
```

### Strategies

| Strategy | Syntax | Semantics |
|---|---|---|
| `top-n:N` | `--tq-v-strategy top-n:4` | Pick N layers with highest `max(head_variance)`. Upgrade to HIGH. Bottom N/2 → LOW. Rest → BASE. |
| `ratio:X` | `--tq-v-strategy ratio:4.0` | Upgrade layers where `max/min head-variance ratio >= X`. Default X=4.0. |
| `kurt:Y` | `--tq-v-strategy kurt:500` | Upgrade layers where `max(head_kurtosis) >= Y`. |
| `mixed` | `--tq-v-strategy mixed` | Upgrade if `ratio>=3.0 OR kurt>=500`. Recommended default. |
| `manual` | — | Use `--tq-v-override` only, ignore profile. |

### Override Syntax (for benchmarking)

`--tq-v-override "0-1:vtq4_2,13:vtq4_2,26-27:vtq4_2,*:vtq3_2"`

- Comma-separated rules, evaluated left-to-right, last match wins for a given layer
- Ranges via dash, wildcard `*` for default
- Integrates cleanly with `--tq-v-profile` (manual override wins)

### Why `mixed` as Default

- Variance alone misses distribution shape (uniform wide vs heavy-tailed)
- Kurtosis alone fires on peaked-but-narrow which quantize fine
- OR of both captures both failure modes at the cost of ~5% extra upgrades

### Budget Constraint

Add `--tq-v-budget-bpw <float>` (default disabled). If specified and heuristic result exceeds budget, **downgrade lowest-ranked upgrade candidates to BASE** until under budget. Provides deterministic memory targeting.

---

## 4. Integration Points

### 4.1 Parameter Plumbing

```
common/arg.cpp
    └─> add args, produce std::vector<ggml_type> type_v_per_layer
          ↓
common/common.h  (common_params)
    └─> std::vector<ggml_type> type_v_layers;   // empty = uniform
          ↓
llama_context_params (include/llama.h)
    └─> const ggml_type * type_v_layers;
    └─> int32_t           type_v_layers_count;  // 0 = use type_v uniformly
          ↓
src/llama-context.cpp:277  (params.type_v → params.type_v_layers fallback)
          ↓
src/llama-kv-cache-iswa.cpp:17  (pass vector through)
          ↓
src/llama-kv-cache.cpp:82  ctor (use per-layer type in loop at line 198)
```

### 4.2 KV-Cache Changes (`src/llama-kv-cache.cpp`)

Key change at **line 198**:

```cpp
// Before
ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

// After
const ggml_type t_v = resolve_v_type(il, type_v, type_v_per_layer);
ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, t_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;
```

Store per-layer type in `layers[]`:

```cpp
// src/llama-kv-cache.h (llama_kv_cache::layer struct, around line 156)
struct layer {
    uint32_t il;
    ggml_tensor * k;
    ggml_tensor * v;
    std::vector<ggml_tensor *> k_stream;
    std::vector<ggml_tensor *> v_stream;
    ggml_type type_v_layer;   // NEW
};
```

Change `type_v() const` (line 1112) to `type_v(uint32_t il) const`. Call sites:
- `llama-kv-cache.cpp:2430` → `llama_kv_cache_context::type_v(il)` — propagate il
- `llama-context.cpp:350, 2956, 2967` — these gate on `ggml_is_quantized(type_v)`. Change to "any layer quantized":

```cpp
bool any_v_quantized = std::any_of(type_v_layers.begin(), type_v_layers.end(), ggml_is_quantized);
```

### 4.3 attn_rot_v (line 283-290)

`attn_rot_v` currently gates on `ggml_is_quantized(type_v)`. For mixed precision, RHT must be enabled if **any** layer is VTQ — it's already required for all three VTQ types. Change to OR across layers. No behavioral regression.

### 4.4 FA Dispatch (`ggml/src/ggml-cuda/fattn.cu`)

**No changes needed.** FA already dispatches on the per-tensor `v->type`. Each layer's V tensor carries its own type.

Verify the three VTQ types are all in the dispatch switch (they were added in Phase 1). Regression test: run `llama-bench` with a mixed config and confirm no fallback-to-CPU warnings.

### 4.5 Memory Accounting

Block sizes per 32-sample block:
- VTQ2_2: 34 B
- VTQ3_2: 50 B
- VTQ4_2: 66 B

`size_v_bytes()` already iterates `layers[]` — just needs to sum per-layer instead of using global `type_v`. Check `src/llama-kv-cache.cpp` around the size-reporting code; if it uses `ggml_nbytes(v)` per tensor it's already correct.

### 4.6 Profile JSON Loader

New file: `src/llama-tq-profile.cpp` + header.

```cpp
namespace llama_tq {
    struct head_stat { int head_idx; float variance; float kurtosis; };
    struct layer_stat { int layer_idx; std::vector<head_stat> heads; };
    struct profile { int n_samples; std::vector<layer_stat> layers; };

    profile load_profile(const std::string & path);

    // Given a profile + strategy spec + base/low/high types, return per-layer type vector
    std::vector<ggml_type> resolve_types(
        const profile & p,
        const std::string & strategy,
        ggml_type base, ggml_type low, ggml_type high,
        int n_layer,
        float budget_bpw = 0.0f);
}
```

Keep json parsing dependency-light: use existing `nlohmann::json` if already linked (check `common/`), else a minimal parser. **Do not add new deps.**

---

## 5. Fallback (No Profile Available)

Based on attention-sink literature + PR1 profile evidence (extreme layers cluster at ends and a few middles):

```cpp
// Heuristic "auto" mode when --tq-v-profile not given
std::vector<ggml_type> auto_types(int n_layer) {
    std::vector<ggml_type> v(n_layer, GGML_TYPE_VTQ3_2);
    // First 2 and last 2 layers: upgrade (attention sinks + output-adjacent)
    for (int i = 0; i < 2 && i < n_layer; ++i)       v[i] = GGML_TYPE_VTQ4_2;
    for (int i = n_layer-2; i < n_layer && i >= 0; ++i) v[i] = GGML_TYPE_VTQ4_2;
    return v;
}
```

This coexists with Trick 1 (attention sinks stay F16 for K — unrelated) — the attention-sink rationale is the same: early/late layers concentrate residual information.

Enable via `--tq-v-strategy auto` without requiring a profile.

---

## 6. PPL Validation Plan

**Baseline rows:**

| Config | Avg bpw | Expected PPL delta (wikitext-2, Qwen3-0.6B) |
|---|---|---|
| F16 V | 16 | 0% (reference) |
| uniform VTQ3_2 | 3.06 | +1.9% (measured) |
| uniform VTQ4_2 | 4.06 | +0.3% (measured, Phase 1) |
| uniform VTQ2_2 | 2.06 | +6.5% (measured, Phase 1) |

**PR2 test matrix:**

| Strategy | Avg bpw target | Hypothesis |
|---|---|---|
| `top-n:4` (upgrade 4 hottest, downgrade 4 coldest) | 3.06 | +1.1% PPL |
| `ratio:4.0` (6 upgrades) | ~3.20 | +0.8% PPL |
| `mixed` default | ~3.20 | +0.6% PPL |
| `auto` (no profile, 4 edge layers) | ~3.21 | +1.3% PPL |
| `ratio:4.0` + `--tq-v-budget-bpw 3.0` | 3.00 | +0.9% PPL |

**Success criteria:**
- `mixed` strategy PPL regression ≤ 1.0% at ≤ 3.25 bpw average
- `auto` fallback within 0.5pp of `mixed`
- No measurable decode-speed regression (< 2%)

**Procedure:**
1. Run PR1 profile hook on Qwen3-0.6B with 200 decode samples (upgrade from 50 for stability)
2. Run each config through `llama-perplexity -f wikitext-2-raw/wiki.test.raw -c 2048`
3. Also validate Qwen3.5-35B-A3B on gpu00:8791 (production target) with 100-sample profile
4. Record decode t/s via `llama-bench -m <model> --cache-type-k q8_0 --cache-type-v <spec>`

---

## 7. Implementation Checklist (PR2 scope)

- [ ] `common/arg.cpp`: add 5 flags, parse override syntax, call profile loader
- [ ] `common/common.h`: add `type_v_layers` field
- [ ] `include/llama.h`: extend `llama_context_params` with array + count
- [ ] `src/llama-tq-profile.{h,cpp}`: JSON loader + strategy resolver (~200 LOC)
- [ ] `src/llama-kv-cache.h`: change ctor signature, add `type_v_layer` to layer struct, change `type_v()` → `type_v(il)`
- [ ] `src/llama-kv-cache.cpp`: use per-layer type at line 198, update `type_v()` accessor, update `attn_rot_v` gate
- [ ] `src/llama-kv-cache-iswa.{h,cpp}`: propagate vector
- [ ] `src/llama-context.cpp`: build vector from params, fan out to cache ctor; fix `ggml_is_quantized` checks
- [ ] `src/llama-memory.h` / `llama-memory-hybrid*`: propagate
- [ ] Unit test: profile loader parses PR1 JSON, produces expected vector for each strategy
- [ ] Integration test: load a model with `--tq-v-override` and verify logged per-layer types match
- [ ] PPL runs for test matrix (§6)
- [ ] Deploy to gpu00:8791, compare to current uniform VTQ3_2 deployment

## 8. Explicitly Out of Scope

- Per-head mixed precision (defer to PR3)
- K-cache mixed precision (separate PR, K sensitivity profile not yet collected)
- Dynamic/online precision adjustment based on runtime stats
- Automatic profile collection during model load (manual two-step workflow for PR2)
- Mixed precision for MLA (`is_mla` models — V is absent there)

## 9. Risks

| Risk | Mitigation |
|---|---|
| Profile from small model (0.6B) doesn't generalize to 35B | Re-profile 35B on gpu00 before PR2 merge; collect both profiles into `docs/tq/profiles/` |
| `size_v_bytes()` incorrect after mixed types | Audit + add unit test covering mixed config |
| `llama_kv_cache_context::type_v()` API break cascades to user code | Keep old signature as deprecated, return layer 0's type; add new `type_v(il)` overload |
| FA dispatch falls back to CPU for one of VTQ types in mixed context | Pre-flight check at cache init: `GGML_ASSERT` that all selected types have FA CUDA kernels registered |
| Budget solver produces degenerate all-LOW config | Minimum one-layer-at-HIGH guarantee if any upgrade candidates exist |

## 10. References

- PR1 profile hook: commit `71b56f7fb` on `trick2-pr1-profile-heads`
- Phase 1 VTQ implementation: `docs/vtq-implementation.md` (split K/V, 8 types)
- Profile example: `docs/tq/profiles/qwen3-0.6b-50s.json` (to be committed alongside PR2)
- Related: attention-sink literature (Xiao et al. 2023) — motivates edge-layer upgrade fallback
