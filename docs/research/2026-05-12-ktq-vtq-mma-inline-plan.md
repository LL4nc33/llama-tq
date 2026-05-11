# KTQ K + VTQ V in MMA-inline FA-kernel — Implementation Plan

## Motivation

Current state (turboquant @ cad032077):
- Ministral-3-3B prefill PP rate decays O(N²): 614 t/s @ 5k → 88 t/s @ 26k
- Root cause: `fattn-mma-ktq.cu:13 ggml_cuda_flash_attn_ext_mma_ktq_split`
  - Allocates **full f16 scratch buffer** for K cache (~1.4 GB @ 200k ctx)
  - Calls `to_fp16(K)` on ENTIRE K cache before standard f16 FA
  - Per forward-pass overhead grows with ctx-length

The existing `ggml_cuda_flash_attn_ext_mma_ktq_inline` path (fattn-mma-ktq.cu:65)
dequants K **in shared memory** per tile (much faster), but requires **V to be f16**.

## Goal

Add VTQ V dequantization to MMA-inline path so Ministral-3-3B can run with
**KTQ K + VTQ V** at full PP speed.

Expected gain: **PP @ 20k ctx from 122 t/s → 400-600 t/s** (per agent roofline)

## Architecture

### Current K-dequant pattern (fattn-mma-ktq-inline.cuh:392)

```cpp
template<int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_ktq_load_tile_K_ktq2_1(
        const block_ktq2_1 * const K_blocks,
        half2 * const tile_K,
        const int D2, const int stride_K_blocks, const int i_sup) {
    // per thread: 1 block per row chunk
    //   - read 32-element block (10 bytes norm + qs + sb)
    //   - 2-bit codebook lookup → float
    //   - apply 32-pt FWHT (warp-parallel via __shfl_xor_sync)
    //   - sign flip
    //   - scale
    //   - write half to tile
}
```

### Proposed V-dequant pattern

VTQ is MUCH simpler than KTQ:
- No FWHT (V is pre-rotated graph-level via `self_v_rot`)
- No sign bits
- Just: codebook lookup × scale

```cpp
// block_vtq2_1: { half d; uint8_t qs[8]; }  -- 10 bytes, 32 elements
template<int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_vtq_load_tile_V_vtq2_1(
        const block_vtq2_1 * const V_blocks,
        half2 * const tile_V,
        const int D2,            // head_dim/2 in half2
        const int stride_V_blocks,
        const int i_sup) {
    const int tid = threadIdx.x;
    const int warp = threadIdx.y;
    const int D = 2 * D2;
    const int nblk_per_row = D / QK_VTQ;  // QK_VTQ = 32

    #pragma unroll
    for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps) {
        const int i = i0 + warp;
        if (i >= nbatch_fa) break;

        const bool oob = oob_check && (i >= i_sup);
        const block_vtq2_1 * row = V_blocks + int64_t(i) * stride_V_blocks;
        half * tile_row_h = reinterpret_cast<half *>(tile_V + i * stride_tile);

        #pragma unroll 4
        for (int ib = 0; ib < nblk_per_row; ++ib) {
            float val = 0.0f;
            if (!oob) {
                const float norm = (float) row[ib].d;
                if (norm > 1e-30f) {
                    // 2-bit lookup (4 elements per byte, tid 0-31 covers 32 elements)
                    const int idx = (row[ib].qs[tid >> 2] >> ((tid & 3) * 2)) & 0x3;
                    val = VTQ_CUDA_CB_2BIT[idx] * PQ_CUDA_CB_SCALE * norm;
                }
            }
            tile_row_h[ib * QK_VTQ + tid] = __float2half(val);
        }
    }
}
```

## Wiring changes needed

1. **Template parameter** add for V type: currently `half2 * V_h2`, needs to become
   conditional on V quant type (compile-time template arg).

2. **Stride conversion**: f16-path uses `stride_V` in half2 units. VTQ needs
   block-unit stride: `stride_V_blocks = stride_V / sizeof(block_vtq2_1)`.

3. **Dispatch in fattn-mma-ktq.cu**: add V-type check
   ```cpp
   if (K->type == GGML_TYPE_KTQ2_1 && V->type == GGML_TYPE_VTQ2_1 &&
       Q->ne[0] == 128 && V->ne[0] == 128) { ... }
   ```

4. **Template instantiation**: existing inline_case is `<128, 128, 8, 4>` —
   need to add V-type as template parameter or duplicate kernel.

## Risks

1. **Numerical equivalence** — VTQ codebook quality vs f16 may show in long-context
   accuracy. Need PPL bench comparable to existing vtq2_1 split-path.

2. **Shared memory pressure** — V tile in shmem may conflict with K/Q tiles.
   Need to verify shmem budget for D=128 GQA=4 on Turing.

3. **Test coverage** — only ncols∈{4,8} compiled. Decode (ncols=1) still uses VEC path.

## Implementation plan

### Phase 1: Scaffolding (this commit)
- [x] Plan document
- [x] Branch `feature/ktq-vtq-mma-inline`
- [ ] Add `flash_attn_ext_f16_vtq_load_tile_V_vtq2_1` standalone (no caller yet, dead code)
- [ ] Bench-gate: verify build doesn't regress

### Phase 2: Kernel template extension

**Approach decision (2026-05-12):** Use `bool V_is_vtq2_1` template param instead
of full `typename V_type`. Keeps changes surgical, all type-conditional logic
guarded by `if constexpr (V_is_vtq2_1)`.

#### 2.1: Extend `flash_attn_ext_f16_ktq_iter` (line 558)
- Add `bool V_is_vtq2_1` template param
- Add `using V_ptr_t = std::conditional_t<V_is_vtq2_1, const block_vtq2_1*, const half2*>;`
- Change `const half2 * V_h2` to `V_ptr_t V_data`
- Stride conversion at call site: f16 stride / sizeof(block_vtq2_1) / 2

#### 2.2: V load conditional (lines 620, 977)
Replace:
```cpp
flash_attn_ext_f16_ktq_load_tile<stride_tile_V, nwarps, nbatch_fa, use_cp_async, oob_check>
    (V_h2 + int64_t(k_VKQ_0)*stride_V, tile_V, nbatch_V2, stride_V, k_VKQ_sup);
```
With:
```cpp
if constexpr (V_is_vtq2_1) {
    // VTQ V dequant in shmem
    flash_attn_ext_f16_vtq_load_tile_V_vtq2_1<stride_tile_V, nwarps, nbatch_fa, oob_check>
        (V_data + int64_t(k_VKQ_0)*stride_V_blocks, tile_V, nbatch_V2, stride_V_blocks, k_VKQ_sup);
} else {
    // existing f16 path with cp_async
    flash_attn_ext_f16_ktq_load_tile<stride_tile_V, nwarps, nbatch_fa, use_cp_async, oob_check>
        (V_data + int64_t(k_VKQ_0)*stride_V, tile_V, nbatch_V2, stride_V, k_VKQ_sup);
}
```

**Note:** VTQ load does NOT use cp_async (it's synchronous dequant with compute).
This may reduce throughput vs f16+cp_async on Ampere+, but Turing has no cp_async
anyway so this is neutral on RTX 2060.

#### 2.3: Extend kernel + inner_loop + main_kernel
Cascade `bool V_is_vtq2_1` through:
- `flash_attn_ext_f16_ktq_inner_loop` (line 1101)
- `flash_attn_ext_f16_ktq_kernel` (line 1645)
- Template instantiation at `flash_attn_ext_mma_ktq_inline_case` (line 1822)

#### 2.4: V_data cast at kernel entry (lines 1748, 1794)
```cpp
const auto V_data = (V_is_vtq2_1
    ? reinterpret_cast<V_ptr_t>(V + nb23*sequence + nb22*z_KV)
    : reinterpret_cast<V_ptr_t>(V + nb23*sequence + nb22*z_KV));
```

#### 2.5: Stride conversion
`stride_V` is in `half2` units for f16. For VTQ, blocks are 10 bytes for 32 elements.
- f16 stride_V (in half2): `nb21 / sizeof(half2)`
- VTQ stride_V_blocks: `nb21 / sizeof(block_vtq2_1)` — but nb21 is in bytes!

Need to verify: when V is `vtq2_1` type, what does `nb21` mean? It should be the row
stride in bytes (n_embd_v_gqa × bytes_per_element = head_dim × 10/32 = 40 bytes for D=128).

Block-stride: `nb21 / sizeof(block_vtq2_1)`.

### Phase 3: Dispatch
- [ ] Update `ggml_cuda_flash_attn_ext_mma_ktq:65` for ktq2_1+vtq2_1 path
- [ ] Add fallback if V is not VTQ2_1

### Phase 4: Validate
- [ ] PPL bench: ktq2_1+vtq2_1 split vs inline (must match within 0.1%)
- [ ] PP bench: Ministral-3B @ 20k ctx, expect 400-600 t/s
- [ ] TG bench: must not regress

## Acceptance criteria

- Build passes
- No regression in existing test suite
- PPL within 0.1% of split-path baseline
- PP rate at 20k ctx improves by ≥2× (122 → 244+ t/s)

## References

- `fattn-mma-ktq-inline.cuh:392` — K-load template (model for V version)
- `turboquant.cuh:1104` — vtq_decode_2bit (codebook lookup logic)
- `ggml-common.h:374-379` — block_vtq2_1 layout
- `fattn-mma-ktq.cu:13` — current split fallback (to bypass)
