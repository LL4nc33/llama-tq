# E11 Triton → CUDA Port Spec: Cached V-Dequant for VTQ_2

**Datum:** 2026-04-21
**Agent:** T4 (Architect, read-only)
**Target:** `fattn-vec` kernel, VTQ_2 family only (VTQ2_2, VTQ3_2, VTQ4_2)
**Reference:** E11 kernel in `/home/claude/llama-tq/triton-autoresearch/variants_r3.py` on gpu00
**Status:** Design spec — no code. Implementation follows (T5).

---

## 1. Problem Recap

Current `fattn-vec` path for VTQ_2 on Turing (sm_75):
- `dequantize_V_vtq_2` is **O(1) per sample** (via `vtq_state_at` direct bit-stream read), but
- kernel compiles to **249 regs/thread → 1 block/SM occupancy** → TG = 4.32 tok/s on Qwen3.5-35B-A3B.
- Every thread independently dequantizes 4 elements per V row. LUT (`vtq_trellis_table_storage`) is in constant/global memory. Redundant LUT traffic across lanes within a warp even though many lanes access same row.

Triton E11 beats this at **112 GB/s (≈11.8× vs naive, 40-100% over num_warps=4)** by:
- 1 program per block, `num_warps=1`, low register pressure → 8–16 blocks/SM.
- **Warp-cooperative decode** of one full 128-sample VTQ block into shared memory, **once**.
- Subsequent `N_REP` query iterations read cached fp16 samples from shmem → zero LUT traffic.

**CUDA port goal:** preserve the same cached-decode pattern, exploit sm_75-specific programmer control (explicit `__launch_bounds__`, shared-memory banking, shuffle), achieve TG ≥ 50 tok/s (~12× baseline, >Triton ceiling).

---

## 2. Interface Design — Two Candidates

### Option A — Specialize `dequantize_V_vtq_2` to read from a prewarmed shmem buffer

Signature: unchanged on the outside. Introduce a parallel helper:

```cpp
// fattn-common.cuh (pseudocode)
template <typename block_t, int K, typename T>
__device__ __forceinline__
void vtq2_block_warm(const block_t* x_block, T* smem_out /*[QK_VTQ_TRELLIS]*/,
                     int lane /*0..31*/);

template <typename T, int ne>
__device__ __forceinline__
void dequantize_V_vtq_2_cached(const T* smem_row_base, T* dst, int il);
```

**Pros:** Zero kernel-level change in `fattn-vec.cuh`. Drop-in for VTQ_2 types via template dispatch.
**Cons:** Warm-up call has to be inserted into the V-loop scaffold of `fattn-vec.cuh` anyway (it's a per-row operation, not hidden behind `dequantize_V`). So we touch the main kernel regardless.

### Option B — Separate kernel `flash_attn_ext_vec_vtq2`

New kernel template, invoked only when `type_V ∈ {VTQ2_2, VTQ3_2, VTQ4_2}`. Dispatch in `fattn-vec-dispatch-vtq2.cu` forks to the new kernel.

**Pros:**
- Clean `__launch_bounds__(nthreads, 4)` without wrecking KTQ paths.
- Independent register budget tuning.
- No `#ifdef`/constexpr forest inside the hot path of `flash_attn_ext_vec`.
- Can use `num_warps=1` equivalent (nthreads=32 variant) if empirically better.
**Cons:** Template instantiation cost, code duplication (~300 LOC copy of the main kernel).

### Recommendation: **Option B**

Rationale:
- Triton E11 confirmed *structural* change (warp count, block size, shmem layout) is what unlocks speedup — not just the decoder.
- Keeping VTQ_2 changes hermetic protects KTQ/Q8_0/F16 paths from compile-time regressions.
- `cicc` OOM risk (seen in earlier rounds) is *lower* with a self-contained kernel than with more `constexpr if` branches in the shared kernel.
- Diff is larger but review is easier: T5 copies → prunes → specializes.

Naming: `flash_attn_ext_vec_vtq2<D, ncols, type_K, type_V, use_logit_softcap>` in new file `fattn-vec-vtq2.cuh`, included only from `fattn-vec-dispatch-vtq2.cu`.

---

## 3. Shmem Layout

### Per-block cache

One VTQ block encodes `QK_VTQ_TRELLIS = 128` samples. In fattn-vec, we process `nthreads = 128` V-rows per outer K-loop iteration. Each V-row contributes `D` elements (128 or 256) → **D / 128 = 1 or 2 VTQ blocks per row**.

Two cache strategies:

**Strategy C1 — cache one V-block per iteration (per thread-group)**
Each warp (or sub-warp) cooperatively decodes **one VTQ block of 128 fp16 samples** → `256 B` per cache. Used by the V-row its lanes accumulate into VKQ. 4 warps × 256 B = **1024 B shmem for cache**, plus existing KQ shmem.

**Strategy C2 — cache a row-panel (all 128 V-rows at once)**
Warp-cooperatively decode all 128 blocks (one per V-row) into a big `[128 rows × 128 samples × fp16]` panel = **32 KiB**. All lanes then read from the panel in the inner accumulate loop. Closest to Triton E11 semantics but blows Turing's 64 KiB shmem budget when combined with KQ shmem + 2 blocks/SM.

**Recommendation: C1 + per-warp slot.**

Layout:
```
extern __shared__ half smem[];

half* KQ_smem       = smem;                           // ne_KQ * sizeof(half)
half* V_cache_smem  = KQ_smem + NE_KQ;                // nwarps * 128 * sizeof(half) = 1024 B (4 warps)
// Optional padding +1 per row to avoid bank conflicts:
// [nwarps][129] half, total = 4*258 = 1032 B
```

Bank conflicts: 128 fp16 samples = 64 banks × 2-way. Access pattern in the consume loop is `smem_cache[lane*V_rows_per_thread + i1]` — lanes read adjacent 8-byte pairs. With `V_rows_per_thread = 4` (fp16 half2 path), lanes 0..31 access 8 B each → distinct banks, no conflict. Pad with +1 half only if empirical conflicts seen.

### Shmem budget (Turing sm_75)

| Resource | Size | Blocks/SM |
|---|---|---|
| SM shmem cap | 64 KiB | — |
| KQ (D=128, ncols=1) | ≈ 256 B | — |
| KQ (D=256) | ≈ 512 B | — |
| V-cache (C1, nwarps=4) | 1 KiB | — |
| Combined budget / block | ~1.5 KiB | cap 40+ blocks/SM shmem-wise |

**Shmem is NOT the binding constraint.** Register pressure is. So we're safe expanding shmem to ~4–8 KiB if needed (e.g. for C1+pipelining).

---

## 4. Occupancy Target + Register Budget

| Metric | Current | Target |
|---|---|---|
| Regs/thread | 249 | ≤ 100 |
| Blocks/SM | 1 | 4 |
| Warps/SM | 4 | 16 |

`__launch_bounds__(nthreads, minBlocksPerSM)`:

| D | nthreads | minBlocksPerSM | reg/thread cap |
|---|---|---|---|
| 64 | 128 | 4 | ~127 |
| 128 | 128 | 4 | ~127 |
| 256 | 128 | 2 | ~127 (full-rank Q regs) |
| 512 | 128 | 1 | (keep current) |

Note Turing has 65536 regs/SM. `blocks_per_SM = floor(65536 / (nthreads * reg_cap))`:
- 4 blocks × 128 threads × 128 regs = 65536 ✓
- 2 blocks × 128 threads × 255 regs > 65536 → fails; must cap at 127.

Levers to drop regs 249 → ~127:
1. **Eliminate per-thread `vtq_state_at` replay** — state already materialized in shmem cache.
2. **FP16 `VKQ[ncols][(D/2)/nthreads_V]`** — already conditional on `V_DOT2_F32_F16_AVAILABLE`.
3. **Drop redundant `Q_reg`/`Q_i32`/`Q_ds`/`Q_f32` slots** — keep only active ones via `if constexpr`.
4. **Reduce `ncols` template instantiations** — start with `ncols=1` only, add `ncols=2,4,8` only after the register budget is verified.
5. **`cuda::pipeline` async copy of V-bytes** — Ampere+ only; *not* applicable on Turing. Skip for Phase 3A.

If `minBlocksPerSM=4` cannot be hit due to register spill, fallback is `minBlocksPerSM=2` (still 2× current).

---

## 5. Dispatch Integration

### `fattn-vec-dispatch-vtq2.cu`

The existing `FATTN_VEC_CASES_ALL_D_WITH_512_RET` macro fans out on `(D, ncols, type_K, type_V)`. We do NOT duplicate the macro; instead we introduce:

```cpp
#define FATTN_VEC_VTQ2_CASES_ALL_D_WITH_512_RET(type_K, type_V) \
    /* expands to same cases but invokes flash_attn_ext_vec_vtq2<...> */
```

Kept inside `fattn-vec-dispatch.cuh` under `#ifdef FATTN_VTQ2_CACHED` so Phase 3A can be toggled at compile time (rollback safety).

### Template instance matrix (Phase 3A2)

K types (4) × V_2 types (3) × D values (4: 64/128/256/512) × ncols (1,2,4,8) × use_logit_softcap (2):
= 4 × 3 × 4 × 4 × 2 = **384 instances**.

Mitigation against cicc OOM:
1. Phase 3A1 ships only `(KTQ2_1 × VTQ3_2, D∈{128,256}, ncols=1, softcap=false)` = **2 instances**.
2. Phase 3A2 expands in two PR slices: first `{KTQ2_1,KTQ3_1,F16,Q8_0} × {VTQ3_2}`, then the other two V_2 types.
3. Use existing per-combo dispatch split pattern (separate .cu files like `fattn-vec-dispatch-vtq2-ktq2.cu`, `-vtq2-f16.cu`) — keeps each .cu ≤ 100 instances.

---

## 6. Kernel Pseudocode (cooperative decode)

```cpp
// Per outer K-loop iter, before the V-accumulate inner loop:
//   k_VKQ_0 .. k_VKQ_0+nthreads (=128 V-rows)

// Each warp handles 32 V-rows. Within a warp, 32 lanes cooperatively decode
// ONE VTQ block at a time into smem_cache[warp][0..127].
//
// But Turing FA-vec processes 128 rows/iter. So each warp owns 32 rows, and
// sequentially decodes 32 VTQ blocks into its smem slot (one per row).
// That serializes decode; better pattern:

for (int row_in_warp = 0; row_in_warp < 32; ++row_in_warp) {
    const int k = warp_id*32 + row_in_warp;              // global V-row
    const block_t* x_block = reinterpret_cast<const block_t*>(V + k*nb21);
    // ... additional ib offsets for D=256 (two blocks per row) ...

    // ---- Cooperative decode: 32 lanes × 4 samples = 128 samples ----
    // Each lane decodes 4 samples of this block (il = lane*4 .. lane*4+3)
    // using vtq_state_at<K>(start_state, qs, il+1). Writes to smem.
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int il = lane*4 + s;
        const uint32_t st = vtq_state_at<K>(x_block->start_state, x_block->qs, il + 1);
        smem_cache[warp_id][il] = __float2half(vtq_trellis_table_storage[st] * ds);
    }
    __syncwarp();

    // ---- Consume: existing accumulate loop, but `dequantize_V` replaced by
    // direct read from smem_cache[warp_id][lane*V_rows_per_thread + i1] ----
    half2 KQ_k = __half2half2(KQ[j*nthreads + k]);
    #pragma unroll
    for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
        half2 tmp[V_rows_per_thread/2];
        // Read decoded samples from smem instead of calling dequantize_V:
        #pragma unroll
        for (int i1 = 0; i1 < V_rows_per_thread/2; ++i1) {
            const int base = 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread;
            tmp[i1] = *reinterpret_cast<half2*>(&smem_cache[warp_id][base + 2*i1]);
        }
        #pragma unroll
        for (int i1 = 0; i1 < V_rows_per_thread/2; ++i1) {
            VKQ[j][i_VKQ_0/nthreads_V + i1] += tmp[i1] * KQ_k;
        }
    }
}
```

Notes:
- For `D=256` the row has **two** VTQ blocks. Decode must loop over `ib ∈ {0,1}` and store into shmem slots `[warp][0..127]` then `[warp][128..255]`. Shmem-per-warp doubles to 512 B.
- Decoded values are **fp16** (cast at write). All downstream math is fp16 — no register pressure for scratch float vectors.
- The `vtq_trellis_table_storage` LUT read moves from the inner loop to the warm-up. In steady state the inner loop does **only** shmem reads.

---

## 7. Correctness Strategy

### Unit-level (mandatory before any benchmark)

1. **Round-trip MSE test** — already exists in `tests/test-backend-ops` for VTQ. Confirm new kernel passes with `rel_err < 5e-3` (same bar as Triton correctness gate).
2. **Direct comparison** — add a gtest that runs `fattn_vec` (legacy) and `fattn_vec_vtq2_cached` (new) against same inputs and asserts element-wise `|a-b| < 1e-3` in fp16. Not PPL-level, but catches wiring bugs.

### End-to-end (before merge to phase2)

Model: **Qwen3-1.5B** (small, fast iteration)
- Compute PPL on wiki.test.raw for:
  - `--cache-type-k f16 --cache-type-v f16` (reference)
  - `--cache-type-k ktq2_1 --cache-type-v vtq3_2` (old path)
  - `--cache-type-k ktq2_1 --cache-type-v vtq3_2 --fa-cached-v` (new path)
- Acceptance: new-path PPL deviation from old-path ≤ **0.5%** (PPL on VTQ is already lossy vs f16; we only guard against regressions from old VTQ).

### Test combination matrix (minimum before Phase 3A merge)

| K × V | D=128 | D=256 |
|---|---|---|
| KTQ2_1 × VTQ2_2 | ✓ | ✓ |
| KTQ2_1 × VTQ3_2 | ✓ | ✓ |
| KTQ2_1 × VTQ4_2 | ✓ | ✓ |
| KTQ3_1 × VTQ3_2 | ✓ | ✓ |

Total 8 combos × {ncols=1, ncols=2} = 16 configs. Run `test-backend-ops -o FLASH_ATTN_EXT` on each.

### Edge cases

- **ne=1 tail query** (last query in sequence, odd `ne01`) — existing kernel guards via `ic0 + j < ne01`. Preserve.
- **k_VKQ_max boundary** — last outer-loop iter may process fewer than 128 V-rows. Decode loop must check `k < k_VKQ_max` before warming cache; otherwise we'd read past V buffer.
- **d==0 block** (zeroed activation) — handle in `vtq2_block_warm` by writing zeros to shmem instead of reading LUT.
- **logit_softcap path** — already has early-out for non-{128,256,512} D. Preserve.

---

## 8. Phasing

### Phase 3A1 — Minimal (target 1–2d)

- New file: `ggml/src/ggml-cuda/fattn-vec-vtq2.cuh` (copy+specialize of `fattn-vec.cuh`).
- New file: `ggml/src/ggml-cuda/fattn-vec-dispatch-vtq2-ktq2.cu` (single combo).
- **Single instance**: `KTQ2_1 × VTQ3_2`, D=128, ncols=1, softcap=false.
- `__launch_bounds__(128, 4)`.
- Correctness gate: MSE + direct-compare gtest.
- Benchmark: `llama-bench -m qwen3-35b.gguf -ctk ktq2_1 -ctv vtq3_2 -n 128 -p 0`.
- **Acceptance: TG ≥ 25 tok/s** (≥ 6× baseline 4.32). If only 2× we stop — the design is wrong.

### Phase 3A2 — Full VTQ_2 matrix (target +1d)

- Add remaining K types (F16, Q8_0, KTQ3_1) and V_2 types (VTQ2_2, VTQ4_2).
- Add D∈{256,512} and ncols∈{2,4,8}.
- Split dispatch into ≤4 .cu files to cap cicc instance count/file.
- PPL validation on Qwen3-1.5B for all 16 combos in test matrix.
- **Acceptance: TG ≥ 50 tok/s** on 35B-A3B, PPL deviation ≤ 0.5%.

### Phase 3A3 — Benchmark + merge

- Full `llama-bench` sweep: TG @ (1, 8, 32, 128) batch, PP @ (512, 2048, 8192).
- Side-by-side against Triton E11 on equivalent shapes.
- Documentation: append results to `docs/plans/2026-04-21-e11-cuda-port-spec-results.md`.
- Merge into `phase2` branch.

---

## 9. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Template/cicc OOM on full matrix | M | build red | Split dispatch per-combo into separate .cu files; Phase 3A1 ships 1 instance only. |
| `minBlocksPerSM=4` not hit (regs > 127) | M | perf below target (~10 tok/s) | Fallback to 2 blocks/SM; profile with `cuobjdump --dump-resource-usage`; aggressively prune live ranges. |
| Shmem conflict on half2 reads | L | ~10% perf | Add +1 half pad per warp slot; verify with `ncu --metrics shared_ld_bank_conflict`. |
| Correctness regression on ne=1 tail / k_VKQ_max edge | M | silent wrong output at long ctx | Explicit gtests for `ne01 % ncols != 0` and `n_ctx % 128 != 0`. |
| QK_VTQ_TRELLIS=128 doubles block count → NB21 stride mismatch | L (already committed) | crash | Already validated in existing VTQ_2 path; re-run existing dequant gtest. |
| Turing-only optimization doesn't scale to Ampere/Hopper | L | perf-plateau on newer HW | Keep legacy path reachable; Ampere can use `cuda::pipeline` async shmem load in Phase 3C. |
| LUT read traffic (constant memory) becomes new bottleneck in warm-up | L | limit at ~200 GB/s | LUT is 65536×fp16 = 128 KiB — fits in L1 once, then cached. Benchmark will reveal if this matters. |

---

## 10. Non-Goals (explicit)

- **No** BLOCK_N halving. Confirmed -50% in Triton.
- **No** FP16 accumulator promotion changes. <5% gain.
- **No** persistent kernel. No gain at Triton level.
- **No** E14 (split decode → GEMM) in Phase 3A. That's Phase 3B.
- **No** warp-specialization (producer/consumer). That's Phase 3C.
- **No** changes to KTQ path, VTQ_1 path, or f16/q8_0 V paths.
- **No** changes to dequantize functions in `convert.cu` or the offline dequant path.

---

## 11. Deliverables for T5 (Implementation Agent)

1. `ggml/src/ggml-cuda/fattn-vec-vtq2.cuh` — specialized kernel.
2. `ggml/src/ggml-cuda/fattn-vec-dispatch-vtq2-{combo}.cu` — up to 4 dispatch files.
3. `ggml/src/ggml-cuda/fattn-common.cuh` — add `vtq2_block_warm_*` helpers (NOT modify existing `dequantize_V_vtq_2`).
4. Gtest addition in `tests/test-backend-ops.cpp` for new path under feature flag `FATTN_VTQ2_CACHED`.
5. `docs/plans/2026-04-21-e11-cuda-port-spec-results.md` — bench + PPL results.

**Branch naming:** `feature/e11-cuda-port-phase3a` off `phase2`.

**Blocker checks before T5 starts:**
- Confirm Triton E11 script on gpu00 is still reproducible (`python variants_r3.py --verify`).
- Confirm existing `test-backend-ops FLASH_ATTN_EXT` passes on phase2 HEAD.
- Confirm `cuobjdump --dump-resource-usage` path on phase2 HEAD shows 249 regs baseline (so we can measure the drop).

---

## 12. Success Criteria Summary

| Metric | Baseline | Phase 3A1 | Phase 3A2 |
|---|---|---|---|
| TG on 35B-A3B (tok/s) | 4.32 | ≥ 25 | ≥ 50 |
| Regs/thread | 249 | ≤ 150 | ≤ 127 |
| Blocks/SM | 1 | ≥ 2 | 4 |
| PPL deviation vs legacy VTQ | 0 | ≤ 0.5% | ≤ 0.5% |
| Build time (full fattn recompile) | ~45min | ~50min | ~60min (capped via file split) |

If Phase 3A1 delivers < 25 tok/s, **stop and re-architect**. The Triton result is not aspirational — it's a 11.8× floor.
