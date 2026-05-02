# v8 Unified KV-Cache Algorithm Spec

**Status:** Research complete (2026-05-02 03:00 CEST). Spec draft based on existing v7 implementations in `llama-tq` master.

**Goal:** Consolidate 15 KTQ/VTQ types (ktq{1..4}_1, vtq{1..4}_1, vtq{2..4}_2, vtq{2..4}_3, vtq_mixed, xktq2_1) down to **8 unified types**: `ktq{1,2,3,4}` (K-cache) + `vtq{1,2,3,4}` (V-cache). Each new type combines algorithms from prior families: trellis (vtq*_2) + outlier-sidecar (vtq*_3) + Lloyd-Max codebook (vtq*_1) + RHT (ktq*_1).

## a) Block Layout per New Type

### Design principles

1. **All types ≤ 1 cacheline** (64 B) for KTQ; ≤ 80 B for VTQ trellis variants.
2. **fp16 scale `d` first** (alignment + matches every existing block).
3. **Outliers stored as `(pos:u8, val:fp16)` pairs**, ascending-sorted positions, popcount-friendly.
4. **K-cache always uses RHT + Lloyd-Max**; V-cache uses **trellis** for bpw ≤ 3.5, **Lloyd-Max codebook** for bpw 4.5 (both with optional outlier sidecar).
5. **bpw counted as `(sizeof(block) * 8) / QK_block`** — structural, not index-rate.

### KTQ family (K-cache, RHT+codebook+sb, QK=32, no outliers in v8 phase 1)

KTQ stays unchanged from v7 (already optimal — 2× warp shuffles per dot, no gather). **Migration is alias-only** (`ktq1` → enum `block_ktq1_1`, etc).

```c
// ktq1 alias → block_ktq1_1, enum 45 (2.5 bpw)
// ktq2 alias → block_ktq2_1, enum 42 (3.5 bpw, prod default)
// ktq3 alias → block_ktq3_1, enum 43 (4.5 bpw)
// ktq4 alias → block_ktq4_1, enum 44 (5.5 bpw)
```

### VTQ family (V-cache, unified — codebook OR trellis + optional outliers)

**Key design decision:** the "best" V-encoder differs by bpw. From sweep data:
- **vtq1 (≤2 bpw):** trellis is too lossy at K=1; codebook is best. **0 outliers** (cost > benefit at 1.5 bpw).
- **vtq2 (~2.5 bpw):** trellis (vtq2_2) wins on all quality metrics vs codebook (vtq2_1).
- **vtq3 (~3.5 bpw):** trellis backbone + 2 outliers (downsized from v7 vtq3_3's 4 outliers).
- **vtq4 (~4.5 bpw):** Lloyd-Max codebook (vtq4_1 winner on 4B-Q4_K_M) at 4-bit codebook saturation makes trellis pointless.

```c
// vtq1 alias → block_vtq1_1, enum 46 (1.5 bpw codebook, no outliers)
// vtq2 alias → block_vtq2_2, enum 50 (2.25 bpw trellis, no outliers, prod default since 2026-04-25)
// vtq3 NEW   → block_vtq3_v8, enum 58 or new enum (3.625 bpw trellis + 2 outliers)
// vtq4 alias → block_vtq4_1, enum 49 (4.5 bpw codebook, no outliers)
```

```c
// vtq3 v8 — REDESIGN von vtq3_3 mit 4 outliers → 2 outliers
typedef struct {
    ggml_half d;
    uint16_t  start_state;
    uint8_t   qs[48];         // 3-bit emit stream
    uint8_t   outlier_pos[2]; // ascending positions
    ggml_half outlier_val[2];
} block_vtq3;  // 58 B → 3.625 bpw
```

## b) Quantize Algorithm per Type

### Shared primitives (already implemented, ggml-quants.c:5516-5760)

```c
// Philox 2×32 PRNG (6 rounds, ggml-quants.c:5527)
static inline uint32_t ktq_philox_6r(uint32_t counter, uint32_t key);

// FWHT-32 scaled by 1/√32 (ggml-quants.c:5544)
static void kktq_fwht(float *data, int n);

// Block-index → 16-bit FNV-1a derived seed (ggml-quants.c:5753)
static inline uint16_t kktq_derive_seed(int64_t block_index);

// Lloyd-Max codebooks (Beta(15.5,15.5) optimal for unit-vector projection)
static const float PQ_CODEBOOK_1BIT[2]  = {-0.797885f, +0.797885f};
static const float PQ_CODEBOOK_2BIT[4]  = {-1.489560f, -0.451428f, +0.451428f, +1.489560f};
static const float PQ_CODEBOOK_3BIT[8]  = {-2.071926f, ..., +2.071926f};
static const float PQ_CODEBOOK_4BIT[16] = {-2.732590f, ..., +2.732590f};
// VTQ uses Laplace-optimal at K=2 (sharper peak):
static const float VTQ_CODEBOOK_2BIT[4] = {-1.810000f, -0.395000f, +0.395000f, +1.810000f};
```

### KTQ {1,2,3,4} quantize — RHT + Lloyd-Max + sb precompute (UNCHANGED from v7)

Identical to v7 quantize_row_ktq{K}_1_ref (ggml-quants.c:5832-5891).

### VTQ1, VTQ2, VTQ4 quantize — UNCHANGED from v7

- vtq1 = vtq1_1 (codebook 1-bit + norm-correction)
- vtq2 = vtq2_2 (trellis 2-bit + start_state)
- vtq4 = vtq4_1 (Lloyd-Max codebook 4-bit + norm-correction)

### VTQ3 quantize — NEU (Trellis 3-bit + 2 outliers)

```c
void quantize_row_vtq3_v8_ref(const float *x, block_vtq3 *y, int64_t k) {
    const int nb = k / 128;
    float x_masked[128];
    float out_val_fp32[2];
    for (int i = 0; i < nb; i++) {
        // 1. Pick top-2 |x[i]| → store (pos, fp32 val), zero those slots.
        ggml_trellis_outliers_pick(x + i*128, /*n_out=*/2,
                                   y[i].outlier_pos, out_val_fp32, x_masked);

        // 2. Convert outlier values to fp16 storage
        for (int j = 0; j < 2; j++) {
            y[i].outlier_val[j] = fp32_to_fp16(out_val_fp32[j]);
        }

        // 3. Trellis-encode the masked block (outlier slots = 0).
        float d;
        ggml_trellis_encode_group(x_masked, /*K=*/3,
                                  &y[i].start_state, &d, y[i].qs);
        y[i].d = fp32_to_fp16(d);
    }
}
```

## c) Dequantize Algorithm per Type (FA Hot Loop Optimized)

### KTQ — Hadamard-domain dot product (UNCHANGED from v7)

Math identity: `<H · sign · cb, Q> = <cb, sign · Hᵀ · Q> = <cb, sign · H · Q>`

### VTQ3 v8 dequant — Trellis + outlier overwrite

```c
void dequantize_row_vtq3_v8(const block_vtq3 *x, float *y, int64_t k) {
    const int nb = k / 128;
    for (int i = 0; i < nb; i++) {
        float *yi = y + i*128;
        // 1. Trellis-decode all 128 samples
        ggml_trellis_decode_group(x[i].start_state, /*K=*/3,
                                  fp16_to_fp32(x[i].d), x[i].qs, yi);
        // 2. Overwrite outlier positions with stored fp16 values
        // Only 2 outliers vs v7's 4 → tighter inner loop, less branch divergence
        for (int j = 0; j < 2; j++) {
            uint8_t pos = x[i].outlier_pos[j];
            yi[pos] = fp16_to_fp32(x[i].outlier_val[j]);
        }
    }
}
```

## d) Per-bpw Backend Switch Logic

| Type | bpw | Engine | Outliers | Rationale |
|------|----:|---|:---:|---|
| `vtq1` | 1.5 | Lloyd-Max codebook (1-bit) | **0** | Trellis at K=1 collapses. Outlier sidecar costs +66% block size. |
| `vtq2` | 2.25 | Trellis (K=2, L=16) | 0 | vtq2_2 wins all metrics vs vtq2_1. Block 36 B vs 10 B is acceptable for 4× smaller PPL drift. |
| `vtq3` | 3.625 | Trellis (K=3) + 2 outliers | **2** | Reduce 4→2 outliers to drop bpw 4.0 → 3.625, saves 12% storage. |
| `vtq4` | 4.5 | Lloyd-Max codebook (4-bit, 16 centroids) | 0 | At 4-bit the codebook saturates. Trellis-K4 PPL identical. Codebook is faster (no state machine). |

## e) Migration Path — Backwards Compatibility

### Strategy: keep all v7 enums alive, add v8 aliases

```c
// In ggml.h — add v8 names as enum aliases:
GGML_TYPE_KTQ1 = GGML_TYPE_KTQ1_1,    // 45
GGML_TYPE_KTQ2 = GGML_TYPE_KTQ2_1,    // 42
GGML_TYPE_KTQ3 = GGML_TYPE_KTQ3_1,    // 43
GGML_TYPE_KTQ4 = GGML_TYPE_KTQ4_1,    // 44
GGML_TYPE_VTQ1 = GGML_TYPE_VTQ1_1,    // 46
GGML_TYPE_VTQ2 = GGML_TYPE_VTQ2_2,    // 50  ← new default
GGML_TYPE_VTQ3 = NEW_ENUM_58,         // (2 outliers, downsize from vtq3_3)
GGML_TYPE_VTQ4 = GGML_TYPE_VTQ4_1,    // 49
```

### CLI alias mapping (common/arg.cpp)

```c
if (str == "ktq1" || str == "ktq1_1") return GGML_TYPE_KTQ1_1;
if (str == "ktq2" || str == "ktq2_1") return GGML_TYPE_KTQ2_1;
if (str == "ktq3" || str == "ktq3_1") return GGML_TYPE_KTQ3_1;
if (str == "ktq4" || str == "ktq4_1") return GGML_TYPE_KTQ4_1;
if (str == "vtq1" || str == "vtq1_1") return GGML_TYPE_VTQ1_1;
if (str == "vtq2" || str == "vtq2_2") return GGML_TYPE_VTQ2_2;  // v8 NEW
if (str == "vtq3" || str == "vtq3_v8") return GGML_TYPE_VTQ3_V8;  // new struct (2 outliers)
if (str == "vtq4" || str == "vtq4_1") return GGML_TYPE_VTQ4_1;
// Legacy v7 names continue to work
```

## Recommended Implementation Order

1. **Add v8 alias enums** in `ggml.h` (zero-cost, immediate CLI ergonomic win).
2. **Add CLI short names** in `common/arg.cpp` (`ktq1..ktq4`, `vtq1..vtq4`).
3. **Document v8 surface** in `docs/turboquant.md` — declare v7 names "legacy".
4. **Implement vtq3 v8 (2 outliers)** as new enum 58 — full quantize/dequant + CUDA dispatch + tests.
5. **Sweep matrix run** (4B + 35B + 80B × 16 combos) to validate Pareto curves.
6. **Update prod default** if sweep shows ktq2/vtq3-v8 > ktq2/vtq2 on Pareto.
7. **Drop unused enums** (vtq_mixed=53, xktq2_1=57) only after 2 release cycles.
