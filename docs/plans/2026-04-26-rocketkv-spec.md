# RocketKV 2-Stage KV Compression — Port Spec

**Paper:** RocketKV — arXiv:2502.14051 (Feb 2025)
**Stack target:** llama-tq fork @ b8303 + KTQ2_1 (K) + VTQ2_2 (V), 200K ctx slot
**Date:** 2026-04-26
**Status:** SPEC ONLY

## 1. Goal

Two-stage KV compression on top of existing TurboQuant K/V. Reach ≥1.3× decode t/s @ 100K ctx with no RULER regression at b=512. Orthogonal to quant: RocketKV reduces *which* KV entries read per decode step; TurboQuant reduces bits per entry.

**Composed:** ~6 bpw stored × √c traffic reduction.

## 2. Algorithm Summary

### Stage 1 — SnapKV++ (prefill-time permanent eviction)

Once at end of prefill. Drops "unimportant" KV permanently.

**Scoring:**
1. Last W=32 query tokens of prompt = observation window
2. Compute `A = softmax(Q_obs · K^T / sqrt(d))` over window
3. Reduce across window dim AND **across GQA group dim** → per-token importance vector
4. Adaptive 1-D pooling kernel `K_pool ∈ {63, 511}` (model-dep; Llama3.1=511 above 48K, else 63)
5. Keep top-`b1 = ceil(seq_len / sqrt(c))` indices **per attention group** (not per head)

**"++" deltas:** per-group reduction (GQA-coherent), adaptive pooling, shared selection across group.

### Stage 2 — Hybrid Top-k Sparse Attention (decode-time, dynamic)

For every decode step on Stage-1-pruned cache:

1. Per-page min/max of K side-channel (computed once at prefill end)
2. Aggregate `|q|` across GQA group → group-level query magnitude
3. Pick **top-r head dims** of `|q|` (head-dim sparsity, r = d / c^(1/4))
4. Using only those r dims of stored min/max → upper-bound approximation of attention scores
5. Pick **top-k** seq positions where k = b1 / sqrt(c)
6. Run real attention on those k pages only

**Budget split:** total c split as sqrt(c) per stage. Example c=64: Stage1 keeps seq/8, Stage2 reads seq/64 per step.

## 3. GQA-awareness for Qwen3.6-35B-A3B

32 Q-heads, 4 KV-heads → group size 8.

- Importance scoring sums across 8 Q-heads of each group → 4 importance vectors per layer
- Stage 1 selects 4 disjoint page-sets per layer (one per KV head)
- Stage 2 top-k reads 4 disjoint page-subsets — **all 8 Q-heads of group hit same KV pages** (1 cache line, 1 dequant, 8 dot-products). Big win for KTQ2_1 K-cache where dequant is non-trivial.

## 4. Integration Map

### 4.1 Stage-1 hook (prefill eviction)

- **File:** `src/llama-kv-cache.cpp`
- **Function:** new `llama_kv_cache_unified::compact_snapkv_pp(seq_id, params)`
- **Caller:** end of prefill in `src/llama-context.cpp` after last `decode()` of prompt batch, gated by `params.kv_compression == LLAMA_KV_COMPRESSION_ROCKETKV`

Operates on already-quantized KTQ2_1/VTQ2_2:
- Score via dequantized Q·K^T over last 32 tokens (reuse FA path with score-dump)
- Pruning = compact slot's logical→physical page map. Keep `std::vector<int32_t> kept_indices` per (layer, kv_head). **Quantized blocks stay in place; rewrite access list only.**

Side-channel min/max for Stage 2: computed during same dequant pass, stored fp16 `[n_layer][n_kv_head][n_pages][2*d_head]`. At d=128, n_pages=200K/64=3125, 4 KV-heads, 64 layers ≈ 400MB.

### 4.2 Stage-2 hook (decode-time mask)

- **File:** `ggml/src/ggml-cuda/fattn.cu` (and `fattn-vec-*.cu`)
- New attention variant `flash_attn_ext_rocketkv` taking extra `int32_t * top_k_indices [n_groups, b2]`

Two sub-paths:
1. Index pre-compute kernel (one warp/group): reads min/max + |q|-topr → top-k indices to scratch buffer
2. Sparse FA kernel: identical to existing FA but K/V loads gathered through top_k_indices. **Need FA dispatch table extended for KTQ2_1+VTQ2_2** — same gap that bit us in `project_on_llama_tq_bugs.md`. Fix that first or this segfaults.

### 4.3 New files

- `src/llama-rocketkv.h` / `.cpp` — Stage-1 driver, side-channel storage, params
- `ggml/src/ggml-cuda/rocketkv-topk.cu` — index pre-compute kernel
- `ggml/src/ggml-cuda/fattn-rocketkv.cu` — sparse-gather FA wrapper

## 5. Stackability with KTQ2_1 + VTQ2_2

| Concern | Verdict |
|---|---|
| Eviction operates on quantized blocks | Yes — page-aligned, no requant. KTQ2_1 block (256 elements) ≤ typical page; align b1/b2 to block boundary |
| Side-channel min/max from quantized K | Compute from dequantized fp16 during W=32 obs-window pass. One-time cost |
| Stage-2 sparse gather + KTQ2_1 dequant | Each gathered page → normal KTQ2_1 dequant (FWHT shfl_xor). VTQ2_2 V-dequant FWHT-free → cheaper |
| Asymmetric K/V (deployed) | Compatible — Stage 1/2 indices shared across K/V (same token positions) |
| Boundary protection (TQ v7) | Round selections up to nearest block |

**Net expected:** ~6 bpw stored, only 1/sqrt(c) of K reads + 1/c of V reads per decode → memory-bandwidth bound decode wins big. Paper claims up to 3× e2e at b=256.

## 6. CLI Flags

```
--kv-compression {none|rocketkv}      default: none
--rocketkv-budget N                   absolute b2 (paper: 256/512/1024/2048/4096)
--rocketkv-stage1-ratio FLOAT         sqrt(c) override; default sqrt(ctx/budget)
--rocketkv-obs-window N               W; default 32
--rocketkv-pool-kernel {auto|63|511}  default auto
--rocketkv-topr-ratio FLOAT           default c^(-1/4)
--rocketkv-debug-scores               dev only
```

## 7. Bench Gate

| Test | Threshold |
|---|---|
| llama-bench decode t/s @ 100K ctx, KTQ2_1+VTQ2_2 + RocketKV(b=512) | **≥ 1.3× baseline** |
| llama-bench decode t/s @ 200K ctx | ≥ 1.5× baseline |
| RULER avg @ 16K, b=512 | ≥ 88.0% (paper 88.1%; baseline Full-KV 91.3%) |
| RULER avg @ 64K, b=1024 | ≥ baseline-Full-KV − 3.5pp |
| LongBench avg, b=512 | within 1pp of Full-KV |
| NIAH 200K (single needle) | 100% recall at depth ≤ 90% |
| Smoke: qwen3.5-0.8b-q8 | numerical equivalence vs Full-KV at b=ctx |
| Memory peak @ 200K decode | ≤ 0.85× baseline |

Sequential runs only (feedback_no_parallel_bench), smallest-model smoke first.

## 8. Risks

1. **FA dispatch table gap (known bug).** project_on_llama_tq_bugs.md TQ FA dispatch missing → reappears in sparse FA. **Fix as prerequisite PR**, add KTQ/VTQ rows to new sparse-FA dispatch with explicit asserts.
2. **Side-channel staleness on context shift.** RoPE re-rotation in place. K min/max along head_dim may survive (positions don't matter). Add invalidation hook in `seq_rm` for safety.
3. **Permanent eviction breaks recall on adversarial long-context.** If question at end needs token Stage 1 dropped → gone. Make `--kv-compression` per-slot, Markov queries opt out.
4. **GQA group-coherence assumption breaks for non-GQA.** Auto-disable when n_head_kv == n_head, log warning.
5. **Page-alignment fragmentation with KTQ2_1 blocks.** KTQ2_1 = 256-element superblocks. b1 picks 137 → dequant whole block anyway OR partial-block dequant (new path, error-prone). Round b1/b2 UP to block boundary.
6. **Memory overhead of side-channel.** 400MB at 200K, 64 layers, 4 KV-heads. Quantize side-channel to int8 → ~100MB.
7. **Parallel slots (--parallel 2 deployed).** Per-slot state. Doubles side-channel cost.

## 9. Phased Implementation

| Phase | Scope | Gate |
|---|---|---|
| R0 | Fix FA dispatch for KTQ/VTQ types (prereq) | smoke passes |
| R1 | Stage-1 SnapKV++ only (eviction at prefill, normal FA after) | RULER no-regression at b=ctx/4 |
| R2 | Side-channel min/max + index pre-compute kernel | unit tests top-k correctness vs Full-KV |
| R3 | Sparse FA gather kernel (fp16, no quant) | matches paper within 1pp on RULER |
| R4 | Sparse FA + KTQ2_1 K + VTQ2_2 V | bench gate §7 |
| R5 | CLI, per-slot opt-out, server props | merge to master |

## 10. Open Questions

- Adaptive pool kernel threshold for Qwen3.6 (paper specs Llama3.1/Mistral only). Empirically tune on RULER
- Does context-shift path mutate K? Verify in llama-context.cpp
- Should Markov Document Store force `--kv-compression none` for residual extraction? Probably yes — residuals from pruned cache ≠ residuals of actual model

## Sources

- [RocketKV paper (arXiv:2502.14051)](https://arxiv.org/html/2502.14051v1)
