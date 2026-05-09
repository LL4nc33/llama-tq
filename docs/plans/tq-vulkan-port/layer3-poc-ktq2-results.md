# Layer 3 POC Results — KTQ2_1 Vulkan Dequant

*Source: Layer-3 builder POC, completed 2026-05-09. Empirical bit-exact
validation of the KTQ2_1 dequant compute shader against a CUDA-arithmetic
reference on the tier-1 target (RTX 2060 sm_75 + NVIDIA proprietary 580.x).*

**Branch note.** Mission text named "turboquant" as the working branch.
The actual TQ Vulkan port work-stream lives on `vulkan-turing-rmkq` (which
already carries the spec-master docs and the Layer 2 driver probe report);
that is where this POC was committed. `turboquant` is a sibling branch
that does not carry the Vulkan-port docs.

---

## Test environment

| Field | Value |
|---|---|
| Host | `gpu00.node` |
| GPU | NVIDIA GeForce RTX 2060 (Turing TU106, sm_75, 12 GB) |
| Driver | NVIDIA proprietary `580.126.09` |
| Vulkan | LunarG SDK 1.4.313, device API 1.4.312 |
| Shader compiler | `glslc` shaderc v2023.8 |
| `subgroupSize` | 32 (locked via `VK_EXT_subgroup_size_control` + `requiredSubgroupSize=32`) |

---

## Result — TL;DR

```
== KTQ2_1 dequant POC ==
device: NVIDIA GeForce RTX 2060
blocks: 100  elements/block: 32  total: 3200
max_abs_err  = 0.000e+00
mean_abs_err = 0.000e+00
wrong_lanes (>1e-3)  = 0
wrong_blocks         = 0
ACCEPTANCE: max_abs_err <= 1e-3 (0.000e+00) AND mean_abs_err <= 1e-5 (0.000e+00) -> PASS
```

**Bit-exact against the CUDA-arithmetic reference for all 3200 lanes across
100 random blocks.** Acceptance gate G1 met with margin.

---

## Method

### Shader

`ggml/src/ggml-vulkan/vulkan-shaders/dequant_ktq2_1.comp` (new, 96 LOC).
One workgroup per block, 32 lanes, `requiredSubgroupSize=32`.

Layout:

- `binding=0`: scalar SSBO of `block_ktq2_1[]` (14 B per block, no padding).
  Uses `VK_EXT_scalar_block_layout` + `GL_EXT_shader_16bit_storage` +
  `GL_EXT_shader_explicit_arithmetic_types_int8/16`.
- `binding=1`: scalar SSBO of `float[]` (32 elements per block).
- Inline `const float[4]` codebook (Layer-2 Probe 3 recommendation, matches
  in-tree IQ-quant prior art).

Algorithm — mirrors CUDA `dequantize_block_ktq2_1_v2`
(`turboquant.cuh:352-382`) operation-for-operation:

1. fp16→fp32 norm read.
2. 2-bit code unpack: `idx = (qs[lane>>2] >> ((lane & 3) << 1)) & 0x3`.
   Codebook lookup `val = PQ_CB_2BIT[idx] * (1/sqrt(32))`.
3. 5-stage subgroupShuffleXor FWHT butterfly with defensive
   `subgroupBarrier()` between stages, trailing `* (1/sqrt(32))`.
4. Sign flip from `sb[]` using the **CUDA** convention `(1.0 - 2.0 * sb_bit)`
   (bit 0 → +1, bit 1 → −1). Multiply by `norm`.
5. Store into `data_d[ib*32 + lane]`.

No early-return on `norm == 0`: every lane reaches the shuffle loop.

### Reference

Plain-C tool (`docs/plans/tq-vulkan-port/poc/gen_fixture.c`, 230 LOC) that:

1. Generates 100 random fp32 blocks of 32 elements each, mixing four
   distributions (uniform [-1,1], gaussian-ish via 8-fold uniform,
   small-magnitude `±1e-3`, large-magnitude `±100`).
2. Encodes each block via a verbatim port of `quantize_row_ktq2_1_ref`
   (`ggml-quants.c:5832`), producing 100 × 14 B blocks.
3. Computes the reference dequant by **CUDA-arithmetic** in plain C —
   codebook×scale → serial FWHT (mathematically equivalent to the warp
   shuffle butterfly per stage and bit-exact under fp32 because both
   visit pairs in the same order with the same operand placement) →
   `(1−2·sb)·norm`.
4. Writes a self-describing fixture binary (`fixture.bin`, 14 KiB total).

The shader is then validated against this exact reference.

### Host harness

`docs/plans/tq-vulkan-port/poc/poc_host.cpp` (290 LOC, raw `vulkan.h`).
Reuses the scaffolding from the Layer 2 probe-host (single queue, host-
visible buffers, descriptor pool, `requiredSubgroupSize=32` +
`REQUIRE_FULL_SUBGROUPS_BIT`, `VK_KHR_shader_subgroup_uniform_control_flow`).

### Why CUDA-arithmetic, not the in-tree CPU dequant

`dequantize_row_ktq2_1` (`ggml-quants.c:5986`) uses the **opposite** sign
convention from the CUDA `dequantize_block_ktq2_1_v2`:

| | bit 0 | bit 1 |
|---|---|---|
| CUDA  `(1 − 2·sb)`            | +1 | −1 |
| CPU   `((sb>>i)&1) ? +1 : -1` | −1 | +1 |

Both paths *work* in production because the CUDA FA-vec hot path uses the
**Hadamard-domain trick** (research-cuda-impl §9): the same sb pattern is
applied to Q before its own FWHT, so the overall sign on K cancels in
`Q·K`. An overall scalar sign on K does not affect the attention output.

For the row-level dequant (which is what this shader implements and what
the V1 spec calls out at §4.1), we mirror **CUDA** — that is the path the
upstream Vulkan FA dispatcher will be wired to in Stage 3 of the spec.
The reference must therefore use the same convention, which is why we
hand-roll it in the fixture generator rather than calling
`dequantize_row_ktq2_1`. This is documented in source comments in both
the shader and the generator.

The "opposite-sign" question is **not** a bug in either CPU or CUDA code:
it is a documented inconsistency that is invisible to the only consumer
(FA-vec dot-product). The Vulkan port locks down the CUDA convention to
keep parity with the FA path; if the upstream `to_float` code path ever
exercises CPU-vs-Vulkan parity for KTQ2_1 row dequant, it will need to be
reconciled there. This is V2-deferred.

---

## Resolved Questions (spec-master §11)

### Q1 — Symmetric K==V == KTQ2_1 (CUDA viability for fixture)

**Status: not exercised by this POC; deferred to Layer 4.**

This POC validates **block-level dequant only** — it does not invoke the
CUDA FA-vec path. The fixture is produced from the CPU encode primitives
(which are well-defined for any block layout), and the reference is
hand-rolled CUDA arithmetic. So the question of whether
`(KTQ2_1, KTQ2_1)` symmetric K==V even compiles in the CUDA FA dispatcher
is **not on the critical path for the POC gate**. Layer 4 will hit this
when it wires up the FA-vec call site (Stage 3 of the spec): if symmetric
KTQ2_1 K==V doesn't dispatch on the CUDA side, we either (a) accept that
parity testing must use asymmetric K/V (in which case the V1 acceptance
gate's "matching K/V quant" constraint relaxes to "any combination CUDA
also supports"), or (b) lift the L15466 restriction earlier than V2 plans.

Recommendation for Layer 4: build the symmetric-K/V smoke model as the
**first** task and confirm CUDA produces sane outputs before sinking time
into the Vulkan FA wiring. If CUDA bombs, the V2 work of allowing K≠V
becomes a V1 task.

### Q2 — `subgroupShuffleXor` correctness on driver 580.x

**Resolved (Layer 2 + Layer 3).** The Layer 2 probe demonstrated bit-exact
correctness across 65,536 workgroups with and without `subgroupBarrier`.
This POC reproduces the same with the additional codebook + sign-flip +
norm multiply chain wrapped around the FWHT, on 100 distinct input
distributions. No regression. The defensive barrier remains in the
canonical shader as a portability hedge for non-NVIDIA tier-1 targets
(MoltenVK / RADV) and future driver regressions.

### Q3 — Spec-constant vs inline `const float[4]` codebook

**Resolved (Layer 2 + Layer 3).** This POC uses inline `const float[4]`
per the Layer 2 Probe 3 recommendation. SPIR-V emission via `glslc -O`
produces an `OpConstantComposite` materialised once per shader, indexed
via `OpAccessChain`+`OpLoad` on a Function-storage local. Performance
ground-truth (SASS) deferred to Layer 4 when Nsight Compute is wired up.

### Q4 — `flash_attn_vec.comp` parametricity for KTQ vec_dot

**Not addressable at the POC layer.** The POC only exercises the
block-dequant compute shader, not the FA call site. Layer 4 must
prototype this.

**Recommendation:** when Layer 4 starts on Stage 3 (scalar FA wiring), the
**first** subtask should be a `dequant_funcs.glsl` patch that adds a
KTQ2_1 `dequantize` / `dequantize4` / `get_dm` set with the
warp-cooperative `vec_dot_KQ` Hadamard-domain trick. If the upstream
`flash_attn_vec.comp` requires more invasive changes than the existing
parametric `#define`-based dispatch covers, fork to `flash_attn_ktq.comp`
and accept a separate FA shader family — the spec already budgets for
`~28 SPV blobs per type` (§5).

### Q5 — Coopmat2 single-element granularity vs warp-cooperative FWHT

**Not addressable at the POC layer (no Ada-class device on the gpu00
target).** Layer 4 spec §4.3 explicitly contemplates V1 shipping with
cm2 disabled if the per-WG shmem block-cache strategy exceeds 16 KiB.
Recommendation: keep cm2 disabled in `supports_op` for V1 unless an Ada
device materialises in the test fleet.

### Q6 — `qs[byte+2]` OOB read with `robustBufferAccess2` disabled

**Resolved by inspection: N/A for KTQ2_1.** This question only applies
to **VTQ2_2** decode, where `vtq_state_at<2>` reads `qs[byte+2]` at
`il=127` past the 32-byte `qs[]` end. KTQ2_1 has 8-byte `qs[]` and the
shader indexes via `qs[lane >> 2]` for `lane ∈ [0,31]` → indices `[0,7]`.
No OOB.

For VTQ2_2 the spec §6 already mandates `+4 B sentinel padding`. The Q6
stress-test will need to live in the Layer 3 VTQ POC (out of scope here).

---

## Driver / validation observations

- The shader compiles with `glslc -O --target-env=vulkan1.3` after one
  fix to the spec — the `[[unroll]]` attribute requires
  `GL_EXT_control_flow_attributes` (not listed in spec §4.1 required
  extensions). Recommendation: add this extension to spec §4.1 required
  list. (Trivial, one line.)
- Run with `VK_LAYER_KHRONOS_validation` engaged: zero validation errors
  or warnings.
- Compile output: 2.3 KB SPIR-V, codebook materialised as
  `OpConstantComposite` (matches Probe 3 expectation).

---

## Go / No-Go for Layer 4 (parallel implementation)

**GO.**

Layer 3 acceptance gate G1 is met with bit-exact margin (max_abs_err =
mean_abs_err = 0.000e+00). The shader is correct on the tier-1 target,
the SSBO layout matches the C struct via `scalar_block_layout`, the
required-subgroup-size pipeline plumbing works, and the Hadamard
butterfly returns identical bits to the serial reference. No
contradictions surfaced against the spec; one minor spec patch (add
`GL_EXT_control_flow_attributes` to §4.1 required-extensions list).

Layer 4 may proceed with the §3 stage layout unchanged. The remaining
unanswered questions (Q1, Q4, Q5) are all FA-wiring concerns that Layer 4
must address as part of Stage 3 — they are not blockers for Stage 0–2
work (scaffolding, KTQ shader integration, VTQ shader). I recommend Layer
4 sequence Q1's symmetric-K/V CUDA viability check as its first task,
since a negative answer there reshuffles V1 vs V2 scope.

---

## Spec deltas (low-cost)

1. **§4.1 Required extensions.** Add
   `#extension GL_EXT_control_flow_attributes : require` to the list.
   Needed for `[[unroll]]` on `glslc 2023.8`.

2. **§4.1 Algorithm pseudocode.** The CUDA reference applies
   `* PQ_CB_SCALE` *before* the FWHT (codebook scaling) and the FWHT
   itself trails with another `* PQ_CB_SCALE`, so the overall pre-sign
   value is `cb[idx] * (1/sqrt(32))² * fwht_butterflies`. The current
   spec text reads "1/sqrt(32) ONCE at end" — this is correct in spirit
   for an unscaled codebook lookup, but does not match the CUDA bit
   pattern. Replace with:

   ```
   val = PQ_CB_2BIT[idx] * PQ_CB_SCALE   // 1/sqrt(32) — codebook scale
   for step in {1,2,4,8,16}:
       subgroupBarrier()
       other = subgroupShuffleXor(val, step)
       val = (lane & step) ? (other - val) : (val + other)
   val *= PQ_CB_SCALE                    // 1/sqrt(32) — FWHT normalization
   ```

   This matches CUDA `ktq_cuda_fwht_warp` + caller in
   `turboquant.cuh:255-263, 367-368`.

3. **§4.1 Sign convention comment.** Add an explicit note that the V1
   spec mirrors **CUDA** (`(1 − 2·sb)`: bit 0 → +1, bit 1 → −1) and that
   this is the **opposite** of the in-tree CPU `dequantize_row_ktq2_1`.
   The two give bit-equivalent attention outputs because the FA-vec
   Hadamard-domain trick applies the same sb to Q, so an overall K-sign
   cancels — but this is non-obvious and worth recording so a future
   review-agent doesn't "fix" the shader to match the CPU side.

---

## Artefacts

**On gpu00 (`/tmp/poc-ktq2/`):**
```
gen_fixture.c        — fixture generator source
gen_fixture          — built tool (`gcc -O2 -Wall ... -lm`)
fixture.bin          — 14216 bytes: 100 blocks × 14 B + 100 × 32 fp32 ref
poc_host.cpp         — Vulkan harness source
poc_host             — built harness (`g++ -std=c++17 -O2 ... -lvulkan`)
dequant_ktq2_1.comp  — copy of the in-repo shader for compilation
dequant_ktq2_1.spv   — 2340-byte SPIR-V (`glslc -O --target-env=vulkan1.3`)
```

**In-repo (committed):**
```
ggml/src/ggml-vulkan/vulkan-shaders/dequant_ktq2_1.comp
docs/plans/tq-vulkan-port/poc/gen_fixture.c
docs/plans/tq-vulkan-port/poc/poc_host.cpp
docs/plans/tq-vulkan-port/layer3-poc-ktq2-results.md  (this file)
```

---

## Reproduction recipe

```bash
# On gpu00:
cd /tmp/poc-ktq2

# 1) Build fixture tool
gcc -O2 -Wall gen_fixture.c -o gen_fixture -lm

# 2) Generate fixture
./gen_fixture                # -> fixture.bin

# 3) Compile shader
glslc -O --target-env=vulkan1.3 dequant_ktq2_1.comp -o dequant_ktq2_1.spv

# 4) Build host
g++ -std=c++17 -O2 poc_host.cpp -o poc_host -lvulkan

# 5) Run
./poc_host dequant_ktq2_1.spv fixture.bin
# expected: max_abs_err = 0.000e+00, mean_abs_err = 0.000e+00, PASS

# Optional: validation layers
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ./poc_host dequant_ktq2_1.spv fixture.bin
```
