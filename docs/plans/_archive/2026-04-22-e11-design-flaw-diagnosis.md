# E11 Kernel Design Flaw — Static Code Analysis

**Datum:** 2026-04-22
**Kernel:** `flash_attn_ext_vec_vtq2_cached` in `ggml/src/ggml-cuda/fattn-vec-vtq2.cuh`
**Measured on:** Qwen3-0.6B Q8_0 (D=128, KTQ2_1 × VTQ3_2, ncols=1)
**Result:** 1.60 tok/s vs ~100 tok/s baseline (60× slowdown)
**Status:** Root cause identified via static analysis — no ncu profile needed

## The Bug

Line 401-416 of `fattn-vec-vtq2.cuh`:

```cpp
for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
    const int k = threadIdx.y*WARP_SIZE + k0 + ...;

    // --- Warm: warp-cooperative decode of 128 samples ---
    const block_t * x_block = (const block_t *) (V + k*nb21);
    vtq2_block_warm<block_t, K_bits>(x_block, smem_V_cache[threadIdx.y]);
    __syncwarp();

    // --- Consume ---
    for (int j = 0; j < ncols; ++j) { ... VKQ[j] += ... * KQ[j][k]; }
}
```

Each iteration of the k0-loop:
- Decodes a DIFFERENT V-row (each V-row is one VTQ block)
- Writes to the SAME `smem_V_cache[warp]` slot (overwriting previous)
- Reads it back exactly `ncols` times before moving on

**For ncols=1 (decode path):** 32 iterations × 128 sample decodes / warp × (1 sample read each) = 4096 decodes, 4096 reads. Amortization factor: **1×**.

Compare legacy path (`fattn-vec-inl.cuh`):
- Each lane decodes its own sample on-demand inline.
- 128 decodes / warp × 1 read each = 128 decodes total.

**E11 ncols=1 is doing 32× the decode work for zero benefit.** Plus:
- `__syncwarp()` × 32 → pipeline stalls
- `vtq_state_at<K>()` is O(K × QS_SIZE) per call → ~192 ops per sample vs 1 LUT load in legacy path
- LUT traffic: legacy hits L2/L1 with temporal locality (same sample position across iterations), E11 hammers different states in random order

Rough cost ratio: `32 (loop iters) × (192 dequant ops / 1 LUT load) / 1 query / sample = ~6000×` worst case. Observed 60× is the attenuated reality (memory latency hides some ALU cost).

## Why Triton autoresearch said 11.8×

Triton's E11 variant was benchmarked at **`num_warps=1`** and **different KV/Q structure** — it amortized the block warm over **multiple query columns** (Triton's query tiling is larger than our `ncols=1` case). The speedup came from **ncols > 1 amortization**, not from caching per se.

In the CUDA port, `ncols=1` is the **decode path** (per-token generation, batch=1). Cached decode is exactly the wrong pattern here. E11's benefit requires `ncols ≥ 4` to amortize the warm step.

## The Fix (if we want to salvage)

Two correct paths:

### Option A: Restrict E11 to ncols ≥ 4

Dispatch hook guards:
```cpp
if (... && Q->ne[1] >= 4) { // only amortize when cols ≥ 4
    ggml_cuda_flash_attn_ext_vec_vtq2_case<...>(...);
    return true;
}
```
Impact: Phase 3A1 kernel becomes PP-only (prefill path), not TG. Zero benefit for decode workloads (TG is what hurts).

### Option B: Abandon cached-decode for ncols=1, use E14 split instead

Phase 3B (E14 from Triton autoresearch):
- Kernel 1: decode once → fp16 persistent buffer (batch write)
- Kernel 2: cuBLAS GEMV (Q × V_fp16) 
- Amortizes decode across all queries trivially.

This is the **correct pattern for per-token decode** and was already noted as "Priority 2" in the Triton autoresearch report.

## Decision

**Do NOT fix E11 for ncols=1.** Architecturally wrong.

**Path forward:**
1. **Keep E11 hook disabled** (already done in commit 790e65c5d)
2. **Accept that QK=128 fix alone does not move TG** (register pressure is primary, as diagnosed — block-count was secondary)
3. **For real TG improvement, the correct next kernel is E14-style** (split decode + fp16 GEMV), which properly addresses single-query decode amortization
4. E14 design is deferred work — not attempting in this session

## Honest Conclusion

E11 kernel was a correctly-implemented version of **the wrong algorithm** for the ncols=1 decode path. The Triton result that motivated it did not translate because Triton's "E11 win" came from a different amortization dimension.

The register pressure reduction (249 → 128) IS still a valid achievement — it proves the launch_bounds approach works. A future E14 kernel inherits this benefit. But the cached-decode geometry was wrong for single-query decoding.

**Time spent:** ~2h (spec + port + build + bench + diagnosis).
**Value extracted:** Confirmed root cause. Eliminated a wrong path. Design doc for E14 when priority warrants.

Not a win. But cleanly archived.
