# VTQ_MIXED Block-Spec (v6 Implementation)

**Datum:** 2026-04-23
**Goal:** Block type der 25% samples @ 3-bit + 75% samples @ 2-bit pro block hat.

## Block Layout

```c
#define QK_VTQ 32
#define VTQ_MIXED_N_HI 8   // 8 samples at 3-bit (25% of block)
#define VTQ_MIXED_N_LO (QK_VTQ - VTQ_MIXED_N_HI)  // 24 samples at 2-bit

typedef struct {
    ggml_half d;              // block scale (L2 norm, same as other VTQ_1 types)
    uint8_t   qs_hi[3];       // 8 samples × 3 bit = 24 bits = 3 bytes
    uint8_t   qs_lo[6];       // 24 samples × 2 bit = 48 bits = 6 bytes
} block_vtq_mixed;  // = 11 bytes

static_assert(sizeof(block_vtq_mixed) == 11, "wrong vtq_mixed block size");
// bpw = 11·8 / 32 = 2.75
```

## Sample-Mapping

Welche Samples sind "hi" (3-bit) und welche "lo" (2-bit)?

**Option 1 — First-N pattern** (simplest):
- Samples 0-7 @ 3-bit (hi)
- Samples 8-31 @ 2-bit (lo)

**Option 2 — Stride-4 pattern** (better spatial coverage):
- Samples 0, 4, 8, 12, 16, 20, 24, 28 @ 3-bit (hi, every 4th)
- Rest @ 2-bit

**Choose Option 2** — distributes high-precision evenly across the block. For head_dim=128, the block tiles 4× so every 4th coord in every tile is high-precision. Good coverage.

## Encode Algorithm

```c
void quantize_row_vtq_mixed_ref(const float *x, block_vtq_mixed *y, int64_t k) {
    for (block in blocks) {
        // Compute L2 norm (same as other VTQ_1)
        float norm = ||xb||
        y->d = f32_to_f16(norm);
        inv_norm = 1/norm

        // Clear output
        memset(y->qs_hi, 0, 3);
        memset(y->qs_lo, 0, 6);

        // Iterate 32 samples
        hi_count = 0; lo_count = 0;
        for j in 0..QK_VTQ-1:
            val = xb[j] * inv_norm
            if j % 4 == 0:  // 3-bit slot
                idx = nearest_centroid(val, VTQ_CODEBOOK_3BIT)
                pack 3 bits at position hi_count * 3 into qs_hi[]
                hi_count++
            else:           // 2-bit slot
                idx = nearest_centroid(val, VTQ_CODEBOOK_2BIT)
                pack 2 bits at position lo_count * 2 into qs_lo[]
                lo_count++

        // Norm correction (same as VTQ_1)
        compute recon, update y->d
    }
}
```

## Decode Algorithm

```c
void dequantize_row_vtq_mixed(const block_vtq_mixed *x, float *y, int64_t k) {
    for block in blocks:
        norm = f16_to_f32(x->d)
        hi_count = 0; lo_count = 0
        for j in 0..QK_VTQ-1:
            if j % 4 == 0:  // 3-bit slot
                idx = extract_3bit_at(x->qs_hi, hi_count)
                y[j] = VTQ_CODEBOOK_3BIT[idx] * cb_scale_3 * norm
                hi_count++
            else:
                idx = (x->qs_lo[lo_count / 4] >> (2 * (lo_count % 4))) & 0x3
                y[j] = VTQ_CODEBOOK_2BIT[idx] * cb_scale_2 * norm
                lo_count++
}
```

## Codebook-Scale Note

**Potentieller Bug:** Die codebook-scales sind `1/sqrt(QK_VTQ)` für alle drei types. Das ist korrekt wenn alle samples im gleichen "Unit-Sphere" skaliert sind. Nach unserem Encode-Schema gilt das — alle 32 samples werden durch `norm` normalisiert, landen also auf der `S^{QK_VTQ-1}` Sphere. Ein codebook-slot auf 3-bit-Präzision vs 2-bit-Präzision nutzt **dieselbe scale** aber unterschiedliche Centroids.

Der Scaling-Factor im Roundtrip ist identisch für hi und lo:
```c
cb_scale = 1.0f / sqrtf((float)QK_VTQ);
// VTQ_CODEBOOK_3BIT[idx] * cb_scale  → für hi slots
// VTQ_CODEBOOK_2BIT[idx] * cb_scale  → für lo slots
```

## Expected Performance

**Accuracy:**
- Per-sample contribution: hi slots get 3.07% rel MSE, lo slots get 11.1% rel MSE
- Blended (8×3.07 + 24×11.1) / 32 = **9.1% rel MSE**
- **18% improvement** vs uniform VTQ2_1 (11.1% → 9.1%)

**bpw:**
- VTQ_MIXED: 2.75 bpw (11 bytes / 32 samples)
- VTQ2_1: 2.50 bpw (10 bytes / 32 samples)
- Cost: +0.25 bpw (+10%)

**Net value:** 18% MSE reduction at 10% memory cost. Probably better than VTQ3_1 for memory-constrained deployments.

## Implementation Files

Modified:
- `ggml/include/ggml.h` — add `GGML_TYPE_VTQ_MIXED = 53` (next available)
- `ggml/src/ggml-common.h` — add `block_vtq_mixed` struct
- `ggml/src/ggml-quants.h` — declare quantize/dequantize
- `ggml/src/ggml-quants.c` — impl encode + decode
- `ggml/src/ggml.c` — register type_traits entry
- `ggml/src/ggml-cpu/ggml-cpu.c` — register CPU traits

New:
- `ggml/src/ggml-cuda/vtq-mixed.cuh` — CUDA dequant kernel  
- `ggml/src/ggml-cuda/convert.cu` — register convert entry
- `ggml/src/ggml-cuda/fattn-vec-dispatch-vtq-mixed.cu` — FA dispatch

CLI:
- `common/arg.cpp` — add `vtq_mixed` to cache-type-v allowed types
- `src/llama-kv-cache.cpp` — add to `is_vtq_v` detection

Tests:
- `tests/test-backend-ops.cpp` — roundtrip test entry
- `tests/trellis-phase1/vtq_roundtrip_bench.py` — extend with VTQ_MIXED

## Phasing

**Phase 1 (today):** CPU reference implementation only.
1. Enum + struct
2. quantize_row_vtq_mixed_ref + dequantize_row_vtq_mixed
3. type_traits entry
4. Roundtrip test passes + matches Python prediction (~9% MSE)

**Phase 2 (tomorrow):** CUDA + FA.
5. CUDA dequant kernel
6. convert.cu entry (for FA vec path)
7. FA dispatch (VTQ_MIXED reuses VTQ2_1 kernel with mixed codebook table)

**Phase 3 (day 3):** Integration + validation.
8. KV-cache detection
9. CLI arg
10. Real PPL sweep on 27B Qwen (if GPU available)
11. TG benchmark on 35B
