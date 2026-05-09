# GLSL/SPIR-V Primitives for TurboQuant Vulkan Port

*Source: Layer-0 research-agent a8d2982f, completed 2026-05-09 11:21.*

**Target:** RTX 2060 (Turing sm_75) on NVIDIA proprietary 580.x; portability to Mesa NVK, RDNA3 (sg=64), Intel Arc, MoltenVK.

## Q1 — `subgroupShuffleXor` under divergent control flow on Turing

**Recommendation:** Keep the FWHT butterfly **subgroup-uniform** (every lane in the warp executes every stage). Wrap the entire FWHT in a region you can prove uniform, request `VK_KHR_shader_subgroup_uniform_control_flow` + `SPV_KHR_subgroup_uniform_control_flow`. Do **not** rely on `subgroupShuffleXor` from divergent branches.

### Why

The base Vulkan/SPIR-V spec gives weak reconvergence guarantees. `OpGroupNonUniformShuffleXor` returns *defined* values only for invocations that are part of the current "tangle".

Turing/Volta+ uses Independent Thread Scheduling — there is no implicit warp lockstep. There is a known **NVIDIA SPIR-V compiler bug** where shuffle-in-loop produces wrong results on Volta+ unless an explicit `subgroupBarrier()` is inserted before the shuffle; NVIDIA shipped a fix only in **driver 591.86 (July 2025)** ([graphics-programming.org](https://graphics-programming.org/blog/subgroup-shuffle-execution-dependency-on-nvidia)). On 580.x — our stated target — this fix is **not yet present**. **Treat shuffle-in-loop as broken on 580.x and insert `subgroupBarrier()` between butterfly stages defensively.**

### Extensions

- **`VK_KHR_shader_maximal_reconvergence`**: Roadmap-2024. NVIDIA proprietary advertises it from Win 538.09 / Linux 535.43.22 onward, so 580.x **does** advertise it on Turing. Note: advertising ≠ bug-free; the 591.86 fix is independent. **Strong** option but **MoltenVK does NOT support it** ([MoltenVK#2478](https://github.com/KhronosGroup/MoltenVK/issues/2478)).
- **`VK_KHR_shader_subgroup_uniform_control_flow`**: Older, weaker, far more widely supported. **Use as baseline requirement.**
- **`subgroupBarrier()` between stages**: Cheap on Turing. Use this **always** in the FWHT inner loop on 580.x.

### Pattern to use

```glsl
#extension GL_KHR_shader_subgroup_shuffle : require

[[unroll]] for (int s = 0; s < LOG2_N; ++s) {
    uint mask = 1u << s;
    subgroupBarrier();                       // belt-and-suspenders for 580.x
    float partner = subgroupShuffleXor(v, mask);
    v = (gl_SubgroupInvocationID & mask) == 0u ? v + partner : partner - v;
}
```

### Plan B

If `subgroupShuffleXor` proves flaky on 580.x even with barriers: fall back to **shared-memory butterflies**. 32 floats × workgroup ≈ 128 B/wg of LDS, write/read with `barrier(); memoryBarrierShared();` between stages. ~10–20% slower than shuffle path, but rock-solid portable (works on MoltenVK, Mali, everything).

## Q2 — Philox-6r in GLSL

**Recommendation:** Hand-port Random123 Philox4x32 with `umulExtended` for the 32×32→64 multiply.

### Details

- `umulExtended(uint a, uint b, out uint msb, out uint lsb)` is **GLSL 4.00 core**, in SPIR-V `GLSL.std.450` as `UMulExtended`. Universally supported. No extension needed.
- Produces **bit-identical** results to CUDA's `__umulhi`.
- Constants (verified against [OpenRAND](https://github.com/msu-sparta/OpenRAND/blob/main/include/openrand/philox.h)): `M0 = 0xD2511F53`, `M1 = 0xCD9E8D57`, `W0 = 0x9E3779B9`, `W1 = 0xBB67AE85`. We use 6 rounds.
- **Endianness is not an issue** inside a shader. Add CRC32 / single test-vector at SSBO upload as cheap insurance.

### Note for KTQ port

**Decode-only port doesn't need Philox** — `sb[]` is precomputed in v5 design. Only relevant if porting the encoder.

### Plan B

If `umulExtended` codegen is slow (it shouldn't be — Turing has native `IMAD.WIDE`): swap to **Threefry-2x32**, which uses only ADD/XOR/ROT.

## Q3 — Lloyd-Max codebook (16× FP16) storage

**Recommendation:** **Specialization constants**, declared as `layout(constant_id = 0..15) const float lloyd_lut[16] = ...`. Bake codebook at pipeline-create time.

### Why

- Spec constants are folded by driver compiler at pipeline bake — same register-file behavior as `__constant__` on CUDA, possibly better.
- 16 entries trivially fits on every driver.
- **Caveat:** indexing a spec-constant array by a *runtime-dynamic* index may not fully constant-fold on Mali/Adreno. Empirical probe needed on RDNA3 and Adreno.

### Alternatives ranked

1. **Spec constants** (recommended) — fastest pipeline-bake-once / dispatch-many.
2. **`layout(push_constant)` block** — for per-tensor codebook variation without recompiling pipelines.
3. **UBO** — stable, universally supported, one extra L1 hit. Good fallback.
4. **Inline literal `const float[16]` in shader source** — equivalent to (1) for static codebook.

### Prior art

llama.cpp's IQ-quant Vulkan shaders use **inline `const` arrays** (e.g. `dequant_iq2_xs.comp`) — codebook embedded in shader source, glslc folds it.

## Cross-cutting decisions

- **Required extensions**: `GL_KHR_shader_subgroup_basic`, `GL_KHR_shader_subgroup_shuffle`, `GL_KHR_shader_subgroup_arithmetic`, `VK_KHR_shader_subgroup_uniform_control_flow` (mandatory), `VK_KHR_shader_maximal_reconvergence` (optional, gated).
- **Subgroup size**: Turing=32, RDNA3=64, Intel Arc=8/16/32, Apple=32. Use `VK_EXT_subgroup_size_control` to **force size=32** — works on RDNA3 (Wave32 mode) and Intel, simplifies FWHT logic.
- **Driver-580.x-on-Turing**: Insert `subgroupBarrier()` between every FWHT butterfly stage as defense.

## Open empirical questions

1. Whether `subgroupShuffleXor` in tight FWHT loop is correct on 580.x Turing **with** `subgroup_uniform_control_flow` requested but **without** `subgroupBarrier()`. **Test required during POC.**
2. Whether spec-constant array indexed by per-thread variable lowers to constant-bank loads or materialized array on NVIDIA 580.x and RADV RDNA3. **SASS/ISA dump required.**
3. Performance of `umulExtended` vs manually-unpacked u16×u16 path on Intel Arc Xe-LPG.

## Sources

- [Re-converging control flow on NVIDIA GPUs (Collabora, 2024)](https://www.collabora.com/news-and-blog/blog/2024/04/25/re-converging-control-flow-on-nvidia-gpus/)
- [NVIDIA SPIR-V Subgroup Shuffle bug (fix in 591.86)](https://graphics-programming.org/blog/subgroup-shuffle-execution-dependency-on-nvidia)
- [VK_KHR_shader_subgroup_uniform_control_flow](https://docs.vulkan.org/guide/latest/extensions/VK_KHR_shader_subgroup_uniform_control_flow.html)
- [VK_KHR_shader_maximal_reconvergence](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_shader_maximal_reconvergence.html)
- [MoltenVK#2478 — no maximal_reconvergence on Apple](https://github.com/KhronosGroup/MoltenVK/issues/2478)
- [OpenRAND philox.h](https://github.com/msu-sparta/OpenRAND/blob/main/include/openrand/philox.h)
- [Random123 paper (Salmon et al., SC11)](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
- [Igalia — specialization constants](https://blogs.igalia.com/itoral/2018/03/20/improving-shader-performance-with-vulkans-specialization-constants/)
