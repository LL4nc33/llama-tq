# TurboQuant CUDA → Vulkan Port Spec: KTQ2_1 + VTQ2_2

*Source: Layer-0 research-agent a825dbe5, completed 2026-05-09 11:18.*

All citations are `path:line`. Source repo: `/mnt/d/repos/llama-tq`.

---

## 1. Type identity & enum values

- `GGML_TYPE_KTQ2_1 = 42` — K-cache, 2-bit Lloyd-Max codebook + per-block RHT, **3.5 bpw**, 14 B/block. (`ggml/include/ggml.h:432`)
- `GGML_TYPE_VTQ2_2 = 50` — V-cache, Trellis-v2 group-Viterbi 2-bit, **2.25 bpw**, 36 B/block. (`ggml/include/ggml.h:440`)

Block sizes: `QK_KTQ = 32`, `QK_VTQ_TRELLIS = 128`. (`ggml/src/ggml-common.h:306`, `ggml/src/ggml-trellis.h:26`)

## 2. Block-struct byte layouts

**`block_ktq2_1` — 14 bytes total** (`ggml/src/ggml-common.h:317-322`):
```
offset 0  : ggml_half d              // 2 B — fp16 norm, with v5 norm-correction baked in
offset 2  : uint8_t   qs[8]          // 8 B — 2-bit indices, 4 elements per byte (LSB first)
offset 10 : uint8_t   sb[4]          // 4 B — RHT sign bits, 1 bit/element (LSB first)
```
Bit pack of `qs`: element `j ∈ [0,32)` → `qs[j/4]` bits `[2*(j%4), 2*(j%4)+1]`.
Bit pack of `sb`: element `j` → `sb[j/8]` bit `j%8`. Stored value 0 means +1, 1 means −1.

**`block_vtq2_2` — 36 bytes total** (`ggml/src/ggml-common.h:419-424`):
```
offset 0  : ggml_half d              // 2 B — encoder-output scale
offset 2  : uint16_t  start_state    // 2 B — L=16 open-start trellis state
offset 4  : uint8_t   qs[32]         // 32 B — 128 samples × 2 bits
```

## 3. KTQ2_1 dequant algorithm — exact arithmetic

Source: `ggml/src/ggml-cuda/turboquant.cuh:352-382`; `fattn-tq.cuh:101-116`, `:176-194`.

For each block `ib`, with `norm = fp16→fp32(x[ib].d)`:

1. **Codebook lookup:**
   ```
   idx = (qs[lane/4] >> (2*(lane%4))) & 0x3
   val = PQ_CB_2BIT[idx] * (1.0 / sqrt(32))
   ```
   `PQ_CB_2BIT[4] = {-1.489560, -0.451428, +0.451428, +1.489560}` (`turboquant.cuh:114-116`)
   `PQ_CB_SCALE = 0.17677669529663689f = 1/sqrt(32)` (`turboquant.cuh:130`)

2. **Inverse RHT, part 1 — normalized 32-point FWHT:** `val = fwht32_normalized(val)` across the warp/subgroup.

3. **Inverse RHT, part 2 — sign flip + scale:**
   ```
   sb_bit = (sb[lane/8] >> (lane%8)) & 1
   val   *= (1.0 - 2.0 * sb_bit) * norm
   ```

4. **Norm-correction is NOT applied at dequant** — already folded into `d` at quantize time.

> CRITICAL: do **not** early-return on `norm < 1e-30` in the warp-cooperative path; every lane must reach the FWHT shuffles to keep the subgroup mask full.

## 4. FWHT details — warp-cooperative butterfly

Source: `turboquant.cuh:255-263`.

```cuda
float ktq_cuda_fwht_warp(float val) {
    for (int step = 1; step < 32; step <<= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        float sum   = val + other;
        float diff  = other - val;
        val = (threadIdx.x & step) ? diff : sum;
    }
    return val * 0.17677669529663689f;
}
```

- 5 stages, `step ∈ {1, 2, 4, 8, 16}`
- Pure ±1 butterflies; no multiplies inside the loop
- Self-inverse because `H_n · H_n = I` after the `1/sqrt(n)` scaling

**GLSL/Vulkan port:**
```glsl
float v = val;
[[unroll]] for (uint step = 1u; step < 32u; step <<= 1) {
    float other = subgroupShuffleXor(v, step);
    float s = v + other;
    float d = other - v;
    v = ((gl_SubgroupInvocationID & step) != 0u) ? d : s;
}
v *= 0.17677669529663689;
```

Subgroup size **must be 32**. On AMD wave64 / Intel wave16: `VK_EXT_subgroup_size_control` to force size 32, or shared-memory butterfly fallback.

## 5. Philox-6r — encode-time only

> **For decode-only Vulkan port, Philox is NOT needed.** `sb[]` is precomputed (v5 design, `turboquant.cuh:73-77`). Only relevant if also porting the quantize path.

Source: `turboquant.cuh:179-189` (6-round). Seed: FNV-1a of block_index (`:199-207`).

Constants: `M0 = 0xD2511F53`, `W = 0x9E3779B9`. Uses `__umulhi` → GLSL `umulExtended`.

## 6. Lloyd-Max codebook

Source: `turboquant.cuh:114-116` (KTQ 2-bit), `:131-148` (VTQ).

For KTQ2_1:
```
PQ_CB_2BIT = { -1.489560, -0.451428, +0.451428, +1.489560 }
```
4 entries × fp32. Multiplied by `PQ_CB_SCALE = 1/sqrt(32)` (cancels FWHT normalization).

**VTQ2_2 does NOT use this codebook** — uses trellis LUT (§7).

## 7. VTQ2_2 trellis decode — exact arithmetic

Source: `ggml/src/ggml-cuda/trellis.cuh:52-88`, `fattn-tq.cuh:776-848`, `ggml/src/ggml-trellis.c:51-60`.

**Constants:** `L = 16`, `K = 2`, `N = QK_VTQ_TRELLIS = 128`, `Lmask = 0xFFFF`, `Kmask = 0x3`.

**LUT** (`vtq_trellis_table_storage[65536]`, fp32):
```c
for s in 0..65535:
    h = (uint32) s * 0x9E3779B1 + 0x7F4A7C15
    p = ((double)(h >> 1) + 0.5) / 2^31
    p = clamp(p, 1e-12, 1.0 - 1e-12)
    table[s] = (float) inv_norm_cdf(p)
```
`inv_norm_cdf` is Beasley-Springer-Moro (`ggml-trellis.c:15-43`). LUT is **256 KiB** — must be SSBO.

**O(1) random-access decode** (FA hot path):
```glsl
uint stream_bit = (il + l + 1) * 2;
uint state;
if (stream_bit < 16) {
    uint from_ss = 16 - stream_bit;
    uint lo = (uint(start_state) >> stream_bit) & ((1u << from_ss) - 1u);
    uint qs_word = uint(qs[0]) | (uint(qs[1]) << 8) | (uint(qs[2]) << 16);
    uint hi = qs_word & ((1u << stream_bit) - 1u);
    state = lo | (hi << from_ss);
} else {
    uint qs_bit = stream_bit - 16;
    uint byte   = qs_bit >> 3;
    uint shift  = qs_bit & 7;
    uint w = uint(qs[byte]) | (uint(qs[byte+1]) << 8) | (uint(qs[byte+2]) << 16);
    state = (w >> shift) & 0xFFFFu;
}
float val = table[state] * ds;   // ds = d * (1/sqrt(128))
```

**Sparse-V early-out:** `if (d == 0.0f) emit zeros and return` — preserves +22% decode win.

**OOB caveat:** `qs[byte+2]` at `i=127` reads beyond `qs[31]`. For K=2 the high byte is dead but Vulkan SSBO bounds checking traps. Pad allocations with +4 B sentinel or add explicit bounds clamp.

## 8. Norm-correction (encode-side only)

Source: `turboquant.cuh:497-511`, `:560-574`.

**Decode side: nothing to do.** Encoder reconstructs the block, recomputes L2, stores `d ← norm_input / norm_recon`. ~1.2% PPL recovery at zero runtime cost.

For Vulkan **decode-only port**, this is invisible — `d` already encodes the correction.

## 9. FA integration points

CUDA FA-vec dispatch in `fattn-vec.cuh`. Two function pointers per (type_K, type_V):

- `vec_dot_KQ_t` → `vec_dot_fattn_vec_KQ_ktq2_1<D, nthreads>` (`fattn-tq.cuh:360-433`)
- `dequantize_V_t` → `dequantize_V_vtq2_2<T, ne>` (`fattn-tq.cuh:850-853`)

Wired via `get_vec_dot_KQ()` (`fattn-tq.cuh:953-980`) and `get_dequantize_V()` (`:983+`).

**`vec_dot_KQ_ktq2_1` (warp path, D ≥ 128):**
1. Each lane holds `D/32` Q-scalars striped: lane `t` owns `Q[bi*32 + t]`
2. Per K-block: sign-flip Q with `sb`, FWHT(Q), read 2-bit code from `qs`, accumulate `PQ_CB_2BIT[idx] * PQ_CB_SCALE * Q_rot * norm`
3. Returns lane's partial sum; caller does `warp_reduce_sum`

This is the **v7 Hadamard-domain trick**: K stays as codebook index, Q gets FWHT'd. Cost: 5 FWHT shuffles per K-block instead of per element.

**`dequantize_V_vtq2_2`:** `ib = i0 / 128`, `il = i0 % 128`, walks `ne` consecutive elements with O(1) `vtq_state_at<2>` formula. `ne = V_rows_per_thread/2` (typically 4).

## 10. Open / unclear items

- **UNCLEAR:** Whether existing upstream `flash_attn_vec.comp` is parametric enough to reuse with custom `vec_dot_KQ` block, or whether KTQ needs fully forked `.comp` per (D, type_K, type_V) tuple.
- **UNCLEAR:** Storage of 256 KiB trellis LUT — read-only SSBO bound at descriptor-set 0; host-uploaded once at first FA dispatch.
- **UNCLEAR:** `qs[byte+2]` OOB at `i=127` — CUDA tolerates, Vulkan bounds-check may trap.
- **NOT NEEDED FOR DECODE-ONLY PORT:** Philox-6r, FNV-1a seed, encode kernels.

## Relevant absolute file paths

- `ggml/src/ggml-cuda/turboquant.cuh` — KTQ codebooks, FWHT, Philox, bulk dequant
- `ggml/src/ggml-cuda/fattn-tq.cuh` — FA-vec entry points
- `ggml/src/ggml-cuda/trellis.cuh` — VTQ decoder, LUT
- `ggml/src/ggml-trellis.c` — host-side LUT generation
- `ggml/src/ggml-common.h` — block structs (lines 317-322, 419-424)
- `ggml/include/ggml.h` — type enums (lines 432, 440)
