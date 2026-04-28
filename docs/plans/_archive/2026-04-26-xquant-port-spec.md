# XQuant Port Spec — llama-tq turboquant branch

**Status:** SPEC ONLY — no implementation
**Paper:** XQuant (arXiv:2510.11236, EMNLP 2025), Yang et al.
**Repo:** https://github.com/brinenick511/XQuant (MIT, builds on KIVI)
**Date:** 2026-04-26

## 1. Executive Summary

XQuant adds two orthogonal tricks on top of any per-token KV quantizer:
1. **Data-free calibration** — relax the endpoint mapping with parameter eta in [0, 0.5]; replaces (z, s) with (z_hat, s_hat) where z_hat = z + eta·s·(2^B−1), s_hat = (1−2eta)·s. Pure arithmetic, no calibration data, no training.
2. **Cross-Layer KV Cache Compression (XLC)** — pair adjacent layers (2k, 2k+1). Layer 2k stores quantized integer codes X_Q; layer 2k+1 stores **only** its own (z_hat, s_hat) and reuses the dominant layer's X_Q at dequant time.

Result in paper: 1.38 bpw average (Mistral-7B) at +0.6 LongBench over KIVI-2bit. Group size G=2 with dominant=first is optimal.

For our fork: **paired KTQ K-cache** where the second layer in each pair drops its quantized-code payload entirely and fetches it from the sibling. RHT commutes through the linear de-mapping, so it stays in place.

## 2. Algorithm (paper Algorithm 1, simplified)

### 2.1 Prefill — quantize phase (per layer l, group size G=2)

```
if l mod 2 == 0:                       # dominant layer
    z_l   = min(X_l)
    s_l   = (max(X_l) - min(X_l)) / (2^B - 1)
    X_Q_l = round((X_l - z_l) / s_l)   # integer codes, B bits
    z_hat_l = z_l + eta · s_l · (2^B - 1)
    s_hat_l = (1 - 2·eta) · s_l
    store (X_Q_l, z_hat_l, s_hat_l)
else:                                  # subordinate layer (l == 2k+1)
    z_l, s_l, z_hat_l, s_hat_l (computed but X_Q discarded)
    store (z_hat_l, s_hat_l)           # NO X_Q — reuse from layer l-1
```

### 2.2 Decode / Attention — dequant phase

```
if l mod 2 == 0 OR l < k_m:
    X_hat_l = X_Q_l       · s_hat_l + z_hat_l
else:                                  # XLC subordinate
    X_hat_l = X_Q_{l-1}   · s_hat_l + z_hat_l  # fetch codes from sibling
```

### 2.3 Calibration eta (data-free, table-driven)

Empirical eta values from paper Table 11:
- 1-bit K: eta_1 in {0, 1/6, 1/3}
- 2-bit V: eta_2 in {0, 0.045, 0.09}

For 2/3/4-bit KTQ + VTQ, eta must be re-tuned per bit-width per model family. Search is offline on a 256-token wikitext probe using PPL.

## 3. Layer-Pair Selection for Qwen3.6-35B-A3B (47 layers)

**Paper finding:** adjacency essential. Layer-similarity drops sharply for distance > 3.

**Recommended pairing:**
- Layers 0-3: **no XLC** (boundary protection — already done via `tq_protect_layers`, src/llama-kv-cache.cpp:304-327)
- Layers 4-45: pair as (4,5), (6,7), ..., (44,45) — 21 pairs
- Layer 46: no XLC

## 4. Stackability with KTQ2_1 / VTQ2_2

### 4.1 K-cache (KTQ2_1) — INSERTION POINT

Today's KTQ2_1 dequant: pack read → sign recover → FWHT inverse → Lloyd-Max scalar → per-block d → f16 to FA softmax.

**XLC insertion:** subordinate layer reads packed codes from layer l-1, but uses its own (z_hat_l, s_hat_l). Hadamard is layer-independent (deterministic Philox seed) → mathematically sound.

**Critical:** new struct `block_xktq2_1` carrying only {d (fp16)} — codes absent. Storage drops from ~3.5 bpw to ~2 bpw of metadata only.

### 4.2 V-cache (VTQ2_2 Trellis) — PARTIAL compatibility

XLC requires linear integer codes. Trellis codes are not linear (state indices). Sharing trellis paths plausible but needs extra calibration.

**v1 ships XLC for K-cache only.** V-cache stays per-layer VTQ2_2.
**v2 (separate spec):** cross-layer trellis sharing.

### 4.3 Boundary / sink protection composition

Existing protections at src/llama-kv-cache.cpp:289-327. XLC must AND with these: protected layers cannot be subordinate or dominant. Pairing pass skips and re-pairs.

## 5. Integration Points — file:line

| Concern | File | Line(s) | Change |
|---|---|---|---|
| New ggml type GGML_TYPE_XKTQ{2,3,4}_1 | `ggml/include/ggml.h` | enum near KTQ | +3 enum values |
| Block struct block_xktq{2,3,4}_1 | `ggml/src/ggml-common.h` | next to block_ktq2_1 | new structs ~4 bytes each |
| Type traits | `ggml/src/ggml.c` (`type_traits`) | KTQ2_1 block | +3 entries |
| CLI parser | `tools/llama-bench/llama-bench.cpp`, `common/common.cpp` | KTQ string→type map | +3 strings |
| **Pairing logic** | `src/llama-kv-cache.cpp` constructor | after reuse pass at 386-408 | new `xquant_pair_layers()` ~40 LOC |
| **Allocation** | `src/llama-kv-cache.cpp` | line 329 | eff_type_k = XKTQ for subordinates |
| **Dequant CUDA kernel** | `ggml/src/ggml-cuda/convert.cu` and `fattn-*.cu` | KTQ entries | new `dequantize_block_xktq2_1` taking 2 input tensors |
| **FA dispatch** | `src/llama-graph.cpp` | KV cache references | add cache_k_l{il-1} as auxiliary input |
| Per-layer eta table | new `src/tq-xquant-calib.h` | n/a | 47-entry table |
| Calibration tool | new `tools/tq-calibrate-xquant/main.cpp` | n/a | grid-search eta |

**Key insight:** existing `reuse` callback (`layer_reuse_cb`, lines 386-408 for iSWA) already aliases `map_layer_ids[il] = map_layer_ids[il_reuse]`. XQuant cannot reuse this directly (need separate scales) but it's the right precedent. Implement as second map `map_layer_ids_xq_codes[il] = map_layer_ids[il_dominant]`.

## 6. Memory Savings — Qwen3.6-35B-A3B (47 layers, 200K ctx)

| Config | K bpw | V bpw | Total KV |
|---|---|---|---|
| f16 baseline | 16 | 16 | 78.1 GB |
| **Today: KTQ2_1 + VTQ2_2** | 3.5 | 2.06 | 13.6 GB |
| **+ XLC on K (this spec, v1)** | 1.69 | 2.06 | 9.16 GB (-32%) |
| + XLC on V (v2) | 1.69 | 1.01 | 6.61 GB (-51%) |

**Concrete win:** at ctx=200k slot×2 today ~13.6 GB KV. XLC-K alone frees ~4.4 GB → ctx ~290k slot×2 OR third parallel slot.

## 7. Quality Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| eta values don't transfer Mistral→Qwen3.6 | HIGH | Per-model eta grid search, ~5 min on test-box |
| Adjacent-layer similarity fails on MoE expert layers | MEDIUM | Pre-flight: measure per-pair corr on probe; abort < 0.7 |
| Interaction with PolarQuant Lloyd-Max codebook | MEDIUM | Codebook layer-independent → geometrically sound; validate 64-chunk wikitext PPL within +0.3% |
| Boundary-layer interaction | LOW | Pairing pass skips protected layers |
| FA kernel register pressure (2-tensor input) | MEDIUM | Profile before merge; may need 2-pass dequant |
| Group-Viterbi VTQ_2 incompatibility | known | v1 K-only |

**Hard quality gate:** PPL delta vs ktq2_1+vtq2_2 baseline ≤ +0.3% on wikitext-2 64-chunk.

## 8. LOC Estimate

| Component | LOC |
|---|---|
| 3 new ggml types + traits + structs | ~120 |
| 3 new CUDA dequant kernels (XKTQ2_1, _3_1, _4_1) | ~250 |
| FA-vec dispatch additions | ~80 |
| `xquant_pair_layers()` + bookkeeping | ~60 |
| CLI parsing | ~15 |
| Calibration table + loader | ~40 |
| Calibration tool tools/tq-calibrate-xquant/ | ~250 |
| Tests | ~150 |
| Docs (`docs/xquant.md`) | ~200 |
| **Total** | **~1,165 LOC** |

Comparable to TQ4_1 v4 patch size (~1,400 LOC). One developer, ~2 weeks with calibration runs.

## 9. Bench Gate

Run on test-box (Qwen3.6-35B-A3B IQ2_XXS):

| Metric | Baseline | XLC-K target | Hard fail |
|---|---|---|---|
| wikitext-2 PPL (64-chunk, ctx=2048) | KTQ2_1: 8.07 | ≤ 8.10 (+0.4%) | > 8.15 (+1.0%) |
| TG @ ctx=8k (t/s) | ~22 t/s | ≥ 21 t/s (-5%) | < 20 t/s |
| PP @ ctx=8k (t/s) | baseline | ≥ -10% | > -15% |
| KV memory @ ctx=200k slot×2 | 13.6 GB | ≤ 9.5 GB | > 10.5 GB |
| LongBench MFQA-Zh | baseline | ≥ baseline -1pt | < baseline -2pt |
| Round-trip dequant(quant(X)) MSE for 1k random rows | n/a | < 1.1× MSE of plain KTQ2_1 | > 1.5× |

**Smoke test first:** qwen3.5-0.8b-q8_0.gguf with --cache-type-k xktq2_1 + 47-layer dummy, check tensor shapes + dequant kernel sanity, before any 35B run.

## 10. Open Questions for Phase 2

1. eta per-layer-pair vs per-bit-width? Paper uses per-bit-width.
2. Does randomized Hadamard D·H·D interact with cross-layer sharing? D-signs seed-deterministic globally → safe. Confirm in calibration tool.
3. VTQ trellis cross-layer sharing — separate spec, pending v1 results.
4. MoE expert-layer pairing — does active expert matter? Probe required.

## Sources

- [XQuant arXiv abstract (2510.11236)](https://arxiv.org/abs/2510.11236)
- [XQuant arXiv HTML (full)](https://arxiv.org/html/2510.11236)
- [XQuant ACL Anthology (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.494/)
- [GitHub: brinenick511/XQuant](https://github.com/brinenick511/XQuant)
- [Moonlight Review of XQuant](https://www.themoonlight.io/en/review/xquant-achieving-ultra-low-bit-kv-cache-quantization-with-cross-layer-compression)
