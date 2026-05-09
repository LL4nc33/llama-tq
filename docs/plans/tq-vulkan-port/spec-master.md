# spec-master.md — TurboQuant KTQ2_1 + VTQ2_2 Vulkan Port (V1)

*Authoritative master spec. Synthesised from 4 Layer-0 research docs in this directory.
Subsequent agents (Layers 3–6) treat this as canonical.*

*Source: Layer-1 spec-architect a47830286a0c6562f, completed 2026-05-09 11:33.*

---

## 1. Goals & Non-Goals

**V1 (this spec)**
- Decode-only Vulkan path for `GGML_TYPE_KTQ2_1` (K-cache, 3.5 bpw, 14 B/block, QK=32) and `GGML_TYPE_VTQ2_2` (V-cache, 2.25 bpw, 36 B/block, QK=128).
- Symmetric K==V usage. Both types whitelisted in `supports_op`, but the FA dispatch line at `ggml-vulkan.cpp:15466` (which rejects `src[1]->type != src[2]->type`) is **not lifted**. Models must be loaded with matching K/V quant.
- Three FA paths: scalar (mandatory), coopmat1 (Turing+), coopmat2 (Ada+). All gated behind `subgroupSize == 32`.
- Acceptance gate: PPL drift ≤ 0.3 % vs CUDA reference; ≥ 95 % deterministic prefix-match over 5×500 tokens.
- Hardware tier-1: RTX 2060 (sm_75) + NV proprietary 580.x. Tier-1 RADV: RX 7900 mesa-25+. Tier-2 advisory: Intel Arc A770, MoltenVK.

**V2 (deferred)**
- Asymmetric K/V (lift the `src[1]->type != src[2]->type` check, double FA pipeline matrix).
- Quantize-side (encode) shaders. Philox-6r in GLSL. Norm-correction encode.

**V3 (research)**
- KTQ1_x / VTQ3_x / VTQ4_x ports. MMQ (integer-dot) FA path. Variable subgroup size.

## 2. File Manifest

| File | New/Mod | Purpose |
|---|---|---|
| `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl` | Mod | Add `block_ktq2_1`, `block_vtq2_2`, packed16 views, `DATA_A_*` macros |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_head.glsl` | Mod | Codebook constants, FWHT macro, trellis-LUT SSBO binding |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl` | Mod | Inline `dequantize`/`dequantize4`/`get_dm` for both types (scalar + cm1) |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs_cm2.glsl` | Mod | `dequantFuncKTQ2_1` / `dequantFuncVTQ2_2` + dispatch chain |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_ktq2_1.comp` | New | Block-dequant compute shader |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_vtq2_2.comp` | New | Block-dequant compute shader (V trellis decode) |
| `ggml/src/ggml-vulkan/vulkan-shaders/cpy_f32_ktq2_1.comp` | New | Set_rows / cpy path (KV-cache writes) |
| `ggml/src/ggml-vulkan/vulkan-shaders/cpy_f32_vtq2_2.comp` | New | Same for V |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | Mod | type_names, FA whitelist, copy/set_rows lists |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Mod | Pipeline registration, CREATE_FA, supports_op, trellis-LUT init |
| `ggml/src/ggml-vulkan/ggml-vulkan-trellis.cpp` | New | Host-side 256 KiB trellis LUT generator |
| `ggml/src/ggml-vulkan/ggml-vulkan-trellis.h` | New | Header for trellis LUT init |
| `tests/test-vulkan-tq-dequant.cpp` | New | Per-block bit-diff harness against CUDA |
| `tests/test-vulkan-tq-fa.cpp` | New | FA scalar/cm1/cm2 PPL parity harness |
| `tests/fixtures/tq-golden-blocks.bin` | New | 100 random KTQ2_1 + 100 random VTQ2_2 blocks + CUDA-ref outputs |

Total: 7 new shader/source files, 5 modified, 2 new test files, 1 fixture binary.

## 3. Implementation Order (topological, with gates)

**Stage 0 — Scaffolding** (no GPU work)
- **S0.1** `type_names[]` + shader-gen whitelists in `vulkan-shaders-gen.cpp`. Build green = pass. `[]`
- **S0.2** Block structs + `DATA_A_*` macros in `types.glsl`. `[S0.1]`
- **S0.3** Host-side trellis-LUT generator. Static-asserted byte-for-byte match against CUDA. `[]`

**Stage 1 — KTQ2_1 scalar dequant path** (gates everything else)
- **S1.1** `dequant_funcs.glsl` — KTQ2_1 `dequantize` / `dequantize4` / `get_dm`. fp32 accumulators, FWHT inline-unrolled. `[S0.2]`
- **S1.2** `dequant_ktq2_1.comp` block-dequant shader. `[S1.1]`
- **S1.3** Wire `pipeline_dequant[GGML_TYPE_KTQ2_1]` in `ggml-vulkan.cpp`. `[S1.2]`
- **S1.4** Test harness with golden-block fixture. **Gate G1: max_abs_err ≤ 1e-3, mean_abs_err ≤ 1e-5.** `[S1.3]`

**Stage 2 — VTQ2_2 scalar dequant path**
- **S2.1** Trellis-LUT SSBO binding + upload at first FA dispatch. `[S0.3]`
- **S2.2** `dequant_funcs.glsl` — VTQ2_2 with O(1) `vtq_state_at<2>` formula and sparse-V early-out. `[S0.2, S2.1]`
- **S2.3** `dequant_vtq2_2.comp`. `[S2.2]`
- **S2.4** Pipeline registration. **Gate G2: same numeric envelope as G1.** `[S2.3]`

**Stage 3 — Scalar FA wiring**
- **S3.1** `CREATE_FA(GGML_TYPE_KTQ2_1, ktq2_1, FA_SCALAR, …)` — 4 variants. `[S1.4, S2.4]`
- **S3.2** Same for VTQ2_2. `[S3.1]`
- **S3.3** Extend `supports_op` for FLASH_ATTN_EXT, GET_ROWS, SET_ROWS, CPY. **Don't lift L15466 K==V check (V2 work).** `[S3.2]`
- **S3.4** PPL smoke on Qwen3.5-0.8B-Q8_0. **Gate G3: drift ≤ 0.5% scalar-only path.** `[S3.3]`

**Stage 4 — Coopmat1 path** (Turing accel)
- **S4.1** Verify `dequant_funcs.glsl` symbols picked up by `flash_attn_cm1.comp`. `[S3.4]`
- **S4.2** `CREATE_FA(_, _, FA_COOPMAT1, _cm1)`. `[S4.1]`
- **S4.3** PPL smoke with cm1 forced. **Gate G4: drift ≤ 0.3%.** `[S4.2]`

**Stage 5 — Coopmat2 path** (Ada+)
- **S5.1** `dequantFuncKTQ2_1` / `dequantFuncVTQ2_2` in `dequant_funcs_cm2.glsl`. **Single-element granularity.** `[S3.4]`
- **S5.2** Dispatch chain edits. `[S5.1]`
- **S5.3** `CREATE_FA(_, _, FA_COOPMAT2, _cm2)`. `[S5.2]`
- **S5.4** Run on Ada/Blackwell or skip. **Gate G5: drift ≤ 0.3% OR documented disable.** `[S5.3]`

**Stage 6 — Acceptance** (gates V1 release)
- **S6.1** Full PPL sweep on Qwen3.6-35B-A3B-IQ2_XXS, wikitext-2 ctx=4096, 287 chunks.
- **S6.2** Generation prefix-match: 5×500 tokens deterministic seed. **≥95%.**
- **S6.3** Tier-1 hardware: RTX 2060 + RX 7900. Intel Arc advisory.
- **S6.4** **Final gate G6: §10 acceptance criteria all green.**

## 4. GLSL Shader Spec

### 4.1 KTQ2_1 dequant

**Required extensions:**
```glsl
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_scalar_block_layout : require
```

**Pipeline must request:**
- `VK_KHR_shader_subgroup_uniform_control_flow` (mandatory)
- `VK_EXT_subgroup_size_control` with `requiredSubgroupSize = 32`
- `VK_KHR_shader_maximal_reconvergence` (optional, gated)

**Layout:**
```glsl
layout(scalar, binding = 0) readonly buffer A { block_ktq2_1 data_a[]; };
layout(scalar, binding = 1) writeonly buffer D { float       data_d[]; };
layout(constant_id = 0) const float PQ_CB_2BIT_0 = -1.489560;
layout(constant_id = 1) const float PQ_CB_2BIT_1 = -0.451428;
layout(constant_id = 2) const float PQ_CB_2BIT_2 = +0.451428;
layout(constant_id = 3) const float PQ_CB_2BIT_3 = +1.489560;
layout(local_size_x = 32) in;
```

**Algorithm:**
```
lane := gl_SubgroupInvocationID
ib   := gl_WorkGroupID.x
norm := f16tof32(data_a[ib].d)

// 1. Codebook lookup (turboquant.cuh:114-116)
qbyte := data_a[ib].qs[lane >> 2]
idx   := (qbyte >> ((lane & 3) << 1)) & 0x3
val   := PQ_CB_2BIT[idx]

// 2. FWHT — 5 stages — turboquant.cuh:255-263
[[unroll]] for (uint step = 1u; step < 32u; step <<= 1) {
    subgroupBarrier();                 // MANDATORY on driver 580.x
    float other = subgroupShuffleXor(val, step);
    val = ((lane & step) != 0u) ? (other - val) : (val + other);
}
val *= 0.17677669529663689;  // 1/sqrt(32) ONCE at end

// 3. Sign flip + norm
sb_byte := data_a[ib].sb[lane >> 3]
sb_bit  := (sb_byte >> (lane & 7u)) & 1u
val     *= (1.0 - 2.0 * float(sb_bit)) * norm

// 4. Store
data_d[ib*32 + lane] := val
```

**CRITICAL:** every lane must enter and exit the FWHT loop together. No early-return on `norm == 0` in the shuffle path.

### 4.2 VTQ2_2 dequant

**Layout adds:**
```glsl
layout(scalar, binding = 2) readonly buffer T { float trellis_lut[65536]; };  // 256 KiB
```

**Algorithm:**
```
ib := i0 / 128;  il := i0 % 128
d  := f16tof32(data_a[ib].d)

if (d == 0.0) { for k in 0..ne-1: out[k] = 0; return; }   // sparse-V early-out

ds := d * 0.0883883476;   // 1/sqrt(128)

for (l = 0; l < ne; ++l) {
    stream_bit := (il + l + 1) * 2
    if (stream_bit < 16) {
        from_ss := 16 - stream_bit
        lo := (uint(start_state) >> stream_bit) & ((1u << from_ss) - 1u)
        qs_word := uint(qs[0]) | (uint(qs[1]) << 8) | (uint(qs[2]) << 16)
        hi := qs_word & ((1u << stream_bit) - 1u)
        state := lo | (hi << from_ss)
    } else {
        qs_bit := stream_bit - 16
        byte_  := qs_bit >> 3
        shift_ := qs_bit & 7
        // OOB caveat at il=127, byte+2=33 — pad block alloc with +4B sentinel
        w := uint(qs[byte_]) | (uint(qs[byte_+1]) << 8) | (uint(qs[byte_+2]) << 16)
        state := (w >> shift_) & 0xFFFFu
    }
    out[l] := trellis_lut[state] * ds
}
```

### 4.3 Coopmat2 dequant (per-element)

**V1 strategy:** cm2 path returns single element via per-element decoded block-cache stored in shared memory at workgroup init. If shmem budget exceeds 16 KiB per WG, cm2 is disabled in `supports_op` and we ship cm1+scalar only.

## 5. C++ Wiring Spec — `ggml-vulkan.cpp` Diff

| Region (line range) | Action |
|---|---|
| `pipeline_dequant[]` slot ~L4308 | Register `dequant_ktq2_1` + `dequant_vtq2_2` block pipelines |
| FA pipeline registration L3515-3583 | Add `CREATE_FA(GGML_TYPE_KTQ2_1, ...)` triplets — ~28 SPV blobs per type |
| `supports_op` FLASH_ATTN_EXT L15469-15497 | Add `case GGML_TYPE_KTQ2_1: case GGML_TYPE_VTQ2_2: break;`. **Don't lift L15466 K==V check.** |
| `supports_op` GET_ROWS / SET_ROWS / CPY L15506-15594 | Add same cases |
| Backend init | Call `ggml_vk_init_trellis_lut(device)`; create + upload SSBO |

**Pipeline log assertion:** on first FA dispatch with KTQ/VTQ types, `GGML_LOG_INFO` selected pipeline name; `GGML_ASSERT` it's not a fallback.

## 6. Trellis LUT Host-Side Init

**File:** `ggml/src/ggml-vulkan/ggml-vulkan-trellis.cpp::ggml_vk_init_trellis_lut(vk_device& device)`

Mirror `GGML_CUDA_INIT_TRELLIS_TABLE_IMPL` byte-for-byte:
```c
for (uint32_t s = 0; s < 65536; ++s) {
    uint32_t h = s * 0x9E3779B1u + 0x7F4A7C15u;
    double   p = ((double)(h >> 1) + 0.5) / 2147483648.0;
    p = clamp(p, 1e-12, 1.0 - 1e-12);
    table[s] = (float) inv_norm_cdf_beasley_springer_moro(p);
}
```
- Built once at `ggml_backend_vk_init` (or lazily on first KTQ/VTQ dispatch).
- Read-only SSBO bound to descriptor-set 0, binding 2.
- Static assert against pre-baked CUDA dump (16 sample states) at unit test time.
- **Padding:** allocate `65536 + 2` floats; pad each block buffer with +4 bytes sentinel.

## 7. Test Harness

**`tests/test-vulkan-tq-dequant.cpp`** — extends test-backend-ops:
- 100 random blocks per type → fixture against CUDA reference.
- Acceptance: `max_abs_err ≤ 1e-3, mean_abs_err ≤ 1e-5`.
- Bit-pattern golden test for `sb[]`: input `0x01020408…` → CUDA-identical sign sequence.

**`tests/test-vulkan-tq-fa.cpp`** — 4-layer toy attention:
- Cross-check Vulkan vs CUDA across all three FA paths via env-var force:
  - `GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1` → scalar
  - `GGML_VK_DISABLE_COOPMAT2=1` → cm1
  - default → cm2

**End-to-end PPL** via `tools/test-tq-vulkan-ppl.sh`; outputs JSON for CI.

## 8. Build-System Changes

**`vulkan-shaders-gen.cpp`:**
- L45: add `"ktq2_1"`, `"vtq2_2"` to `type_names[]`
- L558-560: `load_vec_quant` group includes both
- L662, L673: FA shader generation whitelist includes both
- L770, L775: copy/set_rows lists include both

**Compile flags:** `-DGL_EXT_scalar_block_layout=1`. Initially compile with `-O0` for parity debugging; switch to `-O` after G1 passes.

## 9. Worktree Assignment for Layer 4 (parallel impl)

| Agent | Owns | Subtasks |
|---|---|---|
| **A-shaders-K** | KTQ shader files | S0.1 (KTQ rows), S0.2 (KTQ struct), S1.1, S1.2 |
| **A-shaders-V** | VTQ shader + trellis LUT host code | S0.1 (VTQ rows), S0.2 (VTQ struct), S0.3, S2.2, S2.3 |
| **A-cpp-wiring** | All `ggml-vulkan.cpp` and `vulkan-shaders-gen.cpp` edits | S0.1 (merge integrator), S1.3, S2.1, S2.4, S3.1, S3.2, S3.3, S4.2, S5.2, S5.3 |
| **A-cm-paths** | Coopmat fork files | S4.1, S5.1 |
| **A-tests** | Test harness | S1.4, G1, G2, G3, G4, G5, S6.1–S6.3 |

A-shaders-K and A-shaders-V rebase onto A-cpp-wiring's S0.1 branch before modifying their respective files. A-cm-paths blocks on A-shaders-K/V completing.

## 10. Validation Gates (Layer 5/6)

- **G1 (KTQ dequant):** max_abs_err ≤ 1e-3, mean_abs_err ≤ 1e-5. `sb[]` bit-pattern green.
- **G2 (VTQ dequant):** Same. Sparse-V early-out emits exact zeros.
- **G3 (scalar FA PPL):** drift ≤ 0.5% on Qwen3.5-0.8B-Q8_0 wikitext-2 ctx=512.
- **G4 (cm1 FA PPL):** drift ≤ 0.3% on same model.
- **G5 (cm2 FA PPL):** drift ≤ 0.3% OR documented disable.
- **G6 (Layer 6 final acceptance):**
  - PPL drift ≤ 0.3% on Qwen3.6-35B-A3B-IQ2_XXS wikitext-2 ctx=4096 287 chunks
  - Prefix-match ≥ 95% over 5×500 deterministic-seed completions
  - Green on tier-1 RTX 2060 + RX 7900
  - Pipeline-selection log: no fallback path silently chosen
  - All three FA paths force-tested via env vars
  - No regressions in existing Vulkan backend tests
  - Sparse-V early-out preserved (decode benchmark within 5% of theoretical +22%)

## 11. Known Unknowns (Layer 3 POC must answer)

**Q1 (CONTRADICTION).** Asymmetric KTQ K + VTQ V is the prod CUDA path; Vulkan FA blocks K≠V. **POC must verify**: when both K and V are loaded as the *same* type (e.g. both KTQ2_1, both VTQ2_2), does CUDA FA-vec dispatcher even compile/run? The prod CUDA path may *only* exist for asymmetric pairing. → POC: build synthetic 2-layer model with `(KTQ2_1, KTQ2_1)` and `(VTQ2_2, VTQ2_2)` and confirm CUDA reference produces sane outputs.

**Q2.** Does `subgroupShuffleXor` in tight FWHT loop work correctly on driver 580.x Turing **without** `subgroupBarrier()` between stages? POC: micro-bench FWHT with/without barrier; bit-diff against CUDA. **Default: keep barrier.**

**Q3.** Does spec-constant array `PQ_CB_2BIT[4]` indexed by per-thread variable lower to constant-bank loads on NVIDIA 580.x and RADV RDNA3? POC: SASS / RGA dump.

**Q4.** Is upstream `flash_attn_vec.comp` parametric enough for KTQ's custom warp-cooperative `vec_dot_KQ` via `dequant_funcs.glsl`, or does the v7 Hadamard-domain trick require a forked `flash_attn_ktq.comp`? POC: prototype scalar-FA call site.

**Q5.** Coopmat2 single-element granularity vs KTQ block-cooperative FWHT: viable, or ship V1 with cm2 disabled? POC: time cm2 path on Ada device if available.

**Q6.** Does `qs[byte+2]` OOB read at `il=127` actually trap on Vulkan with `robustBufferAccess2` disabled? POC: stress test with validation layers.

**POC exit criteria:** Q1 & Q2 must be resolved (block Layer 4). Q3–Q6 may defer to Layer 4 if defensive defaults (barrier in, sentinel padding, cm2 disabled) keep G3 green.
