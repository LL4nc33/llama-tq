# XQuant Phase 3+4 — FA Dispatch + KV Pairing (joint spec)

**Status:** spec ready, implementation pending.
**Date:** 2026-04-26
**Phase 1+2 done:** GGML_TYPE_XKTQ2_1 type, block_xktq2_1 struct, CPU `dequantize_row_xktq2_1_paired`, CUDA device-side `ktq_fattn_dequant_block_xktq2_1_paired` helper, round-trip test PASS.

## Why joint

Phase 3 (FA dispatch) and Phase 4 (KV-cache pairing logic) cannot be merged
separately:

- Phase 3 alone: FA-vec accepts `K_dom` as auxiliary input but no layer is
  ever marked subordinate → no behaviour change, dead code.
- Phase 4 alone: pairing logic flips a layer's `eff_type_k = XKTQ2_1` but
  FA dispatch sees an unknown type and aborts.

Both must land together, validated together via 0.8B smoke + 35B PPL gate.

## Phase 3: FA-vec sibling-tensor injection

### 3.1 Graph-level change (`src/llama-graph.cpp`)

Add a callback / new function param to `build_attn_with_kv_unified` (or
similar K-dispatch site) that accepts an *optional* sibling K tensor.
Pseudocode:

```
ggml_tensor * Kcur = ...; // own layer K
ggml_tensor * Kdom = nullptr;
if (kv_self.is_xquant_subordinate(il)) {
    Kdom = kv_self.get_dominant_k(il);  // returns layer (l-1)'s cache_k tensor
}
ggml_tensor * fa = ggml_flash_attn_ext_paired(ctx, Q, Kcur, V, Kdom, ...);
```

### 3.2 New ggml op `ggml_flash_attn_ext_paired`

Either:
- (a) Add 5th tensor input slot to existing `ggml_flash_attn_ext` (cleaner,
  but invalidates upstream-compat ABI)
- (b) Wrap as new op `ggml_flash_attn_ext_paired` with extra src[4] (preferred)

### 3.3 CUDA dispatch (`fattn-vec-dispatch-ktq.cu`)

In the dispatch table, add a XKTQ2_1 branch:

```cpp
case GGML_TYPE_XKTQ2_1:
    fattn_vec_paired_ktq2_1<...>(ctx, dst);  // calls _paired helper
    break;
```

### 3.4 Per-block iteration in FA-vec

When K_type == XKTQ2_1, replace `ktq_fattn_dequant_block_ktq2_1(K, ib, buf)`
with `ktq_fattn_dequant_block_xktq2_1_paired(K_sub, K_dom, ib, buf)`.
Already implemented in `fattn-tq.cuh` line 167.

### 3.5 vec_dot variant

For warp-cooperative FA-vec, the inner-product `vec_dot_fattn_vec_KQ_*`
needs an XKTQ-paired version. It's identical to `vec_dot_fattn_vec_KQ_ktq2_1`
except norm comes from `K_sub[bi].d` while `qs/sb` come from `K_dom[bi]`.
~50 LOC.

## Phase 4: KV-cache pairing (`src/llama-kv-cache.cpp`)

### 4.1 New constructor pass `xquant_pair_layers()`

After existing `tq_protect_layers` (line ~327) and before allocation
(line 329), insert pairing pass:

```cpp
void xquant_pair_layers() {
    if (!params.xquant_enabled) return;
    for (int il = 4; il < hparams.n_layer - 1; il += 2) {
        // skip protected sink/boundary layers
        if (is_protected(il) || is_protected(il+1)) continue;
        // dominant: il (keep KTQ2_1)
        // subordinate: il+1 (eff_type = XKTQ2_1, codes shared)
        eff_type_k[il+1] = GGML_TYPE_XKTQ2_1;
        xq_dominant_of[il+1] = il;
    }
}
```

### 4.2 New member `xq_dominant_of`

Map subordinate layer index → dominant layer index. Used by graph builder
(Phase 3.1) to fetch the right sibling K tensor at build time.

### 4.3 Allocation

XKTQ2_1 row size is 8 bytes / 32 elements = 0.25 bpw. Allocator already
honors `ggml_row_size` per type, so changing `eff_type_k[il+1]` is enough
to shrink the cache buffer.

### 4.4 Boundary protection composition

`tq_protect_layers` (sink layers 0-3, boundary layers, MoE-expert-rich
layers) MUST run before `xquant_pair_layers`. A protected layer cannot be
subordinate AND cannot be dominant of a subordinate. The pair-pass skips
both.

### 4.5 Migration path for existing serialized KV

Old KV checkpoints have all layers as KTQ2_1. Loading into XQuant-enabled
runtime: detect `eff_type_k[il] == KTQ2_1` while pairing wants XKTQ2_1
→ fall back to KTQ2_1 for that layer (no XQuant savings, but no break).

## Combined bench gate

After Phase 3+4 land:

1. Smoke 0.8B: f16/f16 baseline must remain ±1% of pre-Phase numbers.
2. Smoke 0.8B with `--xquant 1` flag: sanity check no crash, comparable
   tg128 (XQuant flag forces minimum 6 layers in pair-pass).
3. 35B-A3B-IQ2_XXS:
   - baseline: ktq2_1 + vtq2_2 → record pp512, tg128, PPL_4chunk.
   - with --xquant: same metrics. Iron-Law gate:
     - pp512 ≥ baseline -1%
     - tg128 ≥ baseline -1%
     - PPL ≤ baseline +0.3%
   - Memory: KV size should drop from 13.6 GB → ~9.16 GB (-32% on K side).
4. If gate passes: ship as `--xquant` opt-in flag with default-off until
   eta-calibration is added.
5. If gate fails: dead-end on this model class; merge with --xquant=0
   default; revisit eta calibration.

## Eta calibration (deferred to Phase 5)

Phase 1-4 use `eta = 0` (no relaxation). Per paper Table 11, eta in
{0, 0.045, 0.09} for 2-bit V; for 2-bit K it's {0, 1/6, 1/3}. Per-model
grid search via offline calibration tool (~5min on test-box).

LOC budget Phase 5: ~250 LOC for calibration tool, ~40 LOC for runtime
eta lookup. Defer until 3+4 gate passes.

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Sibling tensor lifetime mismatch in graph | HIGH | Use ggml_view of dominant cache_k; same lifetime as own |
| Dispatch fallback if dominant is also XKTQ (chain) | MEDIUM | Forbid in pair-pass (only KTQ2_1 → XKTQ2_1) |
| FA register pressure with extra K pointer | MEDIUM | profile before merge; may need pointer arith inlining |
| Asymmetric n_layer (odd count, 47) | LOW | Last layer falls outside pair, stays KTQ2_1 |
| MoE expert-routing diverges per layer | LOW | adjacent-layer Hadamard is layer-agnostic; only norm differs |

## Files touched (ordered for git diff readability)

1. `ggml/include/ggml.h` — `ggml_flash_attn_ext_paired` op decl (~5 LOC)
2. `ggml/src/ggml.c` — op compute graph entry (~30 LOC)
3. `ggml/src/ggml-cuda/ggml-cuda.cu` — op dispatch case (~10 LOC)
4. `ggml/src/ggml-cuda/fattn-tq.cuh` — vec_dot_xktq2_1_paired template (~50 LOC, not committed yet)
5. `ggml/src/ggml-cuda/fattn-vec-dispatch-ktq.cu` — dispatch case (~15 LOC)
6. `src/llama-kv-cache.cpp` — `xquant_pair_layers()` + `xq_dominant_of` member (~80 LOC)
7. `src/llama-kv-cache.h` — pairing API + member (~20 LOC)
8. `src/llama-graph.cpp` — sibling-tensor lookup at FA build site (~30 LOC)
9. `common/arg.cpp`, `tools/llama-bench/llama-bench.cpp` — `--xquant` CLI flag (~15 LOC)
10. `tests/test-xktq-roundtrip.cpp` — already PASS, no change

**Total Phase 3+4: ~255 LOC.** Single PR.

## Implementation order

1. (Phase 3.2) Add `ggml_flash_attn_ext_paired` op stub returning unimpl.
2. (Phase 4.1-4.3) Pairing pass + member + alloc → XKTQ blocks allocated.
3. (Phase 3.1) Graph site: pass sibling tensor when subordinate.
4. (Phase 3.3-3.5) CUDA dispatch + vec_dot impl.
5. (Phase 3.5) Smoke 0.8B end-to-end.
6. (combined) 35B Iron-Law gate.
7. Push or revert.

