# Research: Adding a New K/V Quant Type to the Vulkan Backend

*Source: Layer-0 research-agent ad51482c, completed 2026-05-09 11:13.*

## TL;DR Pattern

Upstream uses **PR #21029** ("vulkan: add FA dequant for q4_1, q5_0, q5_1, iq4_nl") as the canonical minimal template. Treat KTQ2/VTQ2 like a new entry in the existing table-driven type system: add a `tname` string, a GLSL block struct in `types.glsl`, GLSL `dequantize`/`dequantize4`/`get_dm` functions in `dequant_funcs.glsl` (scalar/cm1) and `dequantFuncTQ2` in `dequant_funcs_cm2.glsl` (coopmat2), one shader file `dequant_ktq2.comp`, register the type in `vulkan-shaders-gen.cpp` and `ggml-vulkan.cpp` pipeline tables, then whitelist it in `supports_op`.

There is **no separate K/V quant API** in Vulkan — KV cache types reuse the same dequant infra as weight quants. K and V dispatch through the same FA shader; the shader's `DATA_A_*` define drives both.

## Step-by-Step Touchpoints

### 1. Block struct & `DATA_A_*` macros

**File:** `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl` (see Q4_1 at lines 72-101)

```glsl
#define QUANT_K_KTQ2_1 256
#define QUANT_R_KTQ2_1 1

struct block_ktq2_1 {
    float16_t d;
    uint8_t   qs[/* bpw-derived */];
};
struct block_ktq2_1_packed16 { float16_t d; uint16_t qs[/* /2 */]; };

#if defined(DATA_A_KTQ2_1)
#define QUANT_K   QUANT_K_KTQ2_1
#define QUANT_R   QUANT_R_KTQ2_1
#define QUANT_AUXF 1
#define A_TYPE          block_ktq2_1
#define A_TYPE_PACKED16 block_ktq2_1_packed16
#endif
```

**std430 caveat:** `buffer_reference_align` (used in `dequant_funcs_cm2.glsl`) must equal the C++ struct's natural alignment. Mis-aligned reads silently return garbage on NV drivers.

### 2. Type-name registration

**File:** `vulkan-shaders/vulkan-shaders-gen.cpp`

- Line 45: add `"ktq2_1"` to `type_names[]`
- Line 558-560: `load_vec_quant` group
- Lines 662, 673: FA shader generation whitelist
- Lines 770, 775: copy/set_rows lists

### 3. Block-dequant compute shader

**File:** `vulkan-shaders/dequant_ktq2_1.comp`

Mirror `dequant_iq2_xxs.comp` lines 1-49 or `dequant_q4_1.comp`. Drives `pipeline_dequant[GGML_TYPE_KTQ2_1]`.

### 4. Inline dequant for FA & MMV — `dequant_funcs.glsl`

**File:** `vulkan-shaders/dequant_funcs.glsl` (Q4_1 lives at lines 36-45; `get_dm` at 512-517)

Add three functions guarded by `#if defined(DATA_A_KTQ2_1)`:
- `vec2  dequantize(uint ib, uint iqs, uint a_offset)`
- `vec4  dequantize4(uint ib, uint iqs, uint a_offset)`
- `vec2  get_dm(uint ib, uint a_offset)`

This is the **symbol contract** that `flash_attn.comp` (scalar) and `flash_attn_cm1.comp` (coopmat1) consume.

### 5. Coopmat2 dequant — `dequant_funcs_cm2.glsl`

```glsl
#if defined(DATA_A_KTQ2_1)
layout(buffer_reference, std430, buffer_reference_align = 2)
buffer decodeBufKTQ2_1 { block_ktq2_1 block; };
float16_t dequantFuncKTQ2_1(const in decodeBufKTQ2_1 bl,
                            const in uint blockCoords[2],
                            const in uint coordInBlock[2]) {
    /* return one fp16 element */
}
#endif
```

Add to dispatch chain at line 718+:
```glsl
#elif defined(DATA_A_KTQ2_1)
#define dequantFuncA dequantFuncKTQ2_1
```

Coopmat2 calls `dequantFuncA` once per matrix element via NV's `tensorViewNV` decode hook — **single-element granularity, not block**.

### 6. Optional: integer-dot MMQ path — `flash_attn_mmq_funcs.glsl`

Skip for v1 — fall back to the float path; cost is ~10-15% TG on Turing+.

### 7. Pipeline registration — C++ side

**File:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

`GGML_TYPE_KTQ2_1`, `GGML_TYPE_VTQ2_1` constants must already exist in `ggml.h` (they do in our TQ branch).

**Block-dequant pipeline (~line 4308):**
```cpp
ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_KTQ2_1],
    "dequant_ktq2_1", dequant_ktq2_1_len, dequant_ktq2_1_data,
    "main", 2, 5*sizeof(uint32_t), {256*16, 1, 1}, {}, 1);
```

**FA pipeline registration (~lines 3515-3583):**

For each path, add a `CREATE_FA(GGML_TYPE_KTQ2_1, ktq2_1, FA_SCALAR, )` macro call:
- `FA_SCALAR` × `{fp16-acc, fp16-acc-int8, fp32, fp32-int8}` — 4 variants
- `FA_COOPMAT1` × `_cm1` — 1 variant
- `FA_COOPMAT2` × `_cm2` — 1 variant

The `CREATE_FA` macro auto-expands all sub-permutations.

### 8. Shmem accounting

**File:** `ggml-vulkan.cpp` lines 8848-8900 (`ggml_vk_flash_attn_scalar_shmem_support`)

For float path, no change needed.

### 9. `supports_op` whitelist

**File:** `ggml-vulkan.cpp` line 15469-15497 (`GGML_OP_FLASH_ATTN_EXT` switch)

Add `case GGML_TYPE_KTQ2_1: case GGML_TYPE_VTQ2_1: break;`. **CRITICAL CONSTRAINT** at line 15466: `op->src[1]->type != op->src[2]->type` returns false — **K and V must use the same type in Vulkan FA**. Asymmetric KTQ/VTQ requires either:
- (a) lift this restriction and 2× the FA pipeline matrix, or
- (b) ship both as the same enum value internally.

Also extend `GGML_OP_GET_ROWS`, `GGML_OP_SET_ROWS`, `GGML_OP_CPY` switches (lines 15506-15594).

### 10. `pipeline_dequant_mul_mat_vec_*` slots (optional, low priority for KV-only)

## Build System

Shaders are compiled by `vulkan-shaders-gen` at build time. **No CMakeLists.txt edits required** — they're discovered by `string_to_spv()` calls in `vulkan-shaders-gen.cpp`.

## Permutation Matrix (full FA coverage)

| Path | Suffix | Generator branch | Pipelines |
|---|---|---|---|
| Scalar | (none) | line 671-676 | 8 (2×2×2) |
| Scalar+int8 | `_int8` | line 678-682 | 8 |
| Scalar fp32 | `_fp32` | line 671-676 fp16=false | 4 |
| Coopmat1 | `_cm1` | line 660-666 | 4 |
| Coopmat2 | `_cm2` | line 650-656 | 4 |

Total: ~28 SPV blobs per K/V quant type.

## Caveats & Conventions

- **Rule:** `QUANT_K` must be a power of 2 ≥ 32 — FA shaders assume `hsk % QUANT_K == 0` for tile alignment.
- **Rule:** `dequantize4()` must read 4 contiguous logical elements via `_packed16` or `_packed32` view; don't issue 4 byte-loads.
- **Caveat:** Coopmat2's `decodeBufXxx` `buffer_reference_align` mismatch with the C++ struct's `alignof` causes silent wrong results — **not** a validation error.
- **Caveat:** PR #18450 (iq1_s/iq1_m mmvq) shows lookup-table types need `init_iq_shmem()` at workgroup start; if KTQ2 has trellis/Lloyd-Max codebooks, port them to `dequant_head.glsl`.
- **Caveat:** PR #20657 (iq4_xs "4 at a time") — always implement both `dequantize` and `dequantize4`. FA hot loop uses `dequantize4` exclusively.
- **Asymmetric K≠V is NOT supported** without lifting the line 15466 check.

## Recommended Port Order

1. `types.glsl` block struct + `DATA_A_KTQ2_1` macros.
2. `dequant_ktq2_1.comp` (block dequant) + register type name — verify it builds.
3. `dequant_funcs.glsl` `dequantize/4/get_dm` — covers scalar FA + MMV.
4. C++ `pipeline_dequant[]` slot + scalar `CREATE_FA` lines + `supports_op` — first runnable test.
5. Add `_cm1` (coopmat1) — Turing/Ampere acceleration.
6. Add `dequant_funcs_cm2.glsl` + `_cm2` — Ada/Blackwell.
7. Optional: int8/MMQ path.
8. `cpy_f32_ktq2_1` / `set_rows_ktq2_1` for actual KV cache writes.
