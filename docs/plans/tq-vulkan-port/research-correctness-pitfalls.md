# Correctness Pitfalls for TurboQuant CUDA→Vulkan Port

*Source: Layer-0 research-agent a5376082, completed 2026-05-09 11:23.*

Checklist for review-agents auditing KTQ2/VTQ2 Vulkan shaders against CUDA reference. Target: **PPL drift ≤ +0.5%** vs CUDA on identical GGUF + identical prompts.

## 1. Floating-point precision pitfalls

### 1a. FMA fusion non-deterministic across CUDA and Vulkan
**Description.** CUDA's `__fmaf_rn` is single IEEE-754 correctly-rounded fused multiply-add. GLSL `a*b+c` may be split or fused at driver discretion. The `precise` qualifier *prevents* fusion on NVIDIA's GLSL compiler — opposite of what we want for parity. There is no Vulkan primitive guaranteeing hardware FMA matching `__fmaf_rn` bit-exactly ([Vulkan-Docs#1973](https://github.com/KhronosGroup/Vulkan-Docs/issues/1973)).

**Detection.** Per-block bit-diff of dequant output vs CUDA reference for 100 random KV blocks. Any non-zero ULP delta in FWHT path = fusion mismatch.

**Mitigation.** (1) Compile shaders with `-O0` initially, enable `SPV_KHR_float_controls` denorm/rounding mode hints. (2) FWHT butterfly: write each stage as explicit `t = a + b; b = a - b; a = t;` with no MAD-shaped expressions. (3) Use `precise` only on accumulators aggregating >32 terms. (4) Document expected ULP envelope: ≤14 ULP for N=128 across FWHT.

### 1b. Transcendentals (sqrt, exp, rsqrt)
**Description.** GLSL `sqrt`/`inversesqrt`/`exp` are *not* required to be correctly-rounded; spec allows ~3 ULP. CUDA `sqrtf` is correctly rounded. Lloyd-Max codebook distance and norm correction will diverge.

**Mitigation.** Norm correction: do `dot/sqrt(norm2)` not `dot * inversesqrt(norm2)`. Avoid `exp` entirely — codebook lookup is integer-indexed.

### 1c. fp16 vs fp32 accumulation defaults
**Description.** GLSL doesn't auto-promote like nvcc.

**Mitigation.** All FWHT and norm-correction accumulators must be `float` (fp32). Only final write to KV cache uses fp16. Add shader-level `#if defined(KTQ_DEBUG)` assertion.

### 1d. FWHT associativity under finite precision
**Description.** Hadamard butterflies are exactly associative in real arithmetic but **not** under fp32 with rounding. A warp-shuffle butterfly visits pairs in a different order than a CUDA shared-memory butterfly.

**Mitigation.** Match traversal order exactly — same butterfly stages, same lane-pairing, same scaling factor placement (apply 1/√N once at end).

## 2. Subgroup determinism across vendors

### 2a. NVIDIA Volta+ Independent Thread Scheduling
**Description.** Since Volta, NVIDIA hardware schedules invocations independently. `subgroupShuffleXor` on lanes that diverged earlier is **undefined** unless reconvergence is forced.

**Mitigation.** (1) Enable `VK_KHR_shader_maximal_reconvergence` and decorate FWHT entry point with `MaximallyReconvergesKHR`. (2) Insert `subgroupBarrier()` before any `subgroupShuffleXor`. (3) Keep all FWHT lanes fully active (no `if` inside butterfly).

### 2b. MoltenVK / Apple Silicon
**Description.** Apple Silicon reports subgroupSize=32. MoltenVK has historical bugs with `gl_SubgroupInvocationID` ([MoltenVK#1553](https://github.com/KhronosGroup/MoltenVK/issues/1553)).

**Mitigation.** Gate KTQ Vulkan path on `subgroupSize == 32` with explicit fallback. **MoltenVK = Tier-2.**

### 2c. Intel Arc variable subgroupSize
**Description.** Intel uses 8/16/32 chosen heuristically. Arc 140T reports minSubgroupSize=8 ([llama.cpp#20776](https://github.com/ggml-org/llama.cpp/issues/20776)). Also Arc 140V GPU TDR with `VK_KHR_cooperative_matrix` ([#20554](https://github.com/ggml-org/llama.cpp/issues/20554)).

**Mitigation.** Require `VK_EXT_subgroup_size_control`, lock subgroup size to 32. Reject device if 32 isn't in `requiredSubgroupSizeStages`.

## 3. Bit-packing & std430 layout

### 3a. Storage layout for KTQ block
**Description.** KTQ block has `uint16_t r_norm`, `uint32_t qs[N/4]`, `uint8_t sb[4]`. std430 packs scalars at natural alignment with array stride 16-byte rounded for arrays-of-vec4. C++ `struct{ uint16; uint32[]; uint8[4] }` is **not** ABI-compatible with GLSL struct.

**Mitigation.** **Strongly recommend `VK_EXT_scalar_block_layout`** (`scalar_block_layout`) which matches C/CUDA layout exactly. Or declare flat `uint32_t data[]` SSBO and index manually.

### 3b. u8 array indexing
**Description.** GLSL has no `uint8_t` without `VK_KHR_shader_float16_int8` + `shaderInt8`. Without int8, `qs[idx]` is read via `(data_a[ibi].qs_u32[idx>>2] >> ((idx&3)<<3)) & 0xff`.

**Mitigation.** Prefer `int8` extension when available. Validate with known-pattern golden-block test.

### 3c. Sign-bit array `sb[4]`
**Description.** v5 precomputes 32 sign bits as `sb[4]`. Easy to get wrong by 1 in shift direction.

**Detection.** Unit test with bit pattern `0x01020408...` to verify shader recovers same signs as CUDA at every position.

## 4. FA integration risks

### 4a. Three Vulkan FA paths
**Description.** Vulkan dispatches FA via scalar / coopmat1 / coopmat2 paths; each calls a different dequant entry point. Adding a new K type to one and forgetting another silently produces garbage when cm2-capable driver picks cm2.

**Detection.** Force-run with `GGML_VK_DISABLE_COOPMAT=1` and `GGML_VK_DISABLE_COOPMAT2=1` separately and compare PPL across all three paths against CUDA.

**Mitigation.** Implement scalar first; gate FA on cm-paths off until scalar is bit-stable. Then port cm2. Add `GGML_ASSERT` in dispatch logic that refuses unsupported paths.

### 4b. Turing coopmat2 reachability
**Description.** Turing reports `KHR_coopmat` but not all driver builds expose `NV_coopmat2`. FA-vec dispatch on Turing typically falls back to scalar.

**Mitigation.** Log `vk_pipeline_flash_attn_*` selection at startup; assert path matches expected.

## 5. Validation methodology

### 5a. Per-block bit-exact dequant test
Use `tests/test-backend-ops.cpp` — already guarantees backend-cross result equivalence. Register synthetic dequant op for KTQ2/VTQ2. **Acceptance: max_abs_err ≤ 1e-3 fp32, mean_abs_err ≤ 1e-5.** Note [#20249](https://github.com/ggml-org/llama.cpp/issues/20249) — test-backend-ops on RADV has known issues; cross-validate with proprietary driver.

### 5b. PPL sweep
- Model: Qwen3.5-0.8B-Q8_0 (smoke), then Qwen3.6-35B-A3B-IQ2_XXS (gate, our prod model)
- Dataset: wikitext-2 raw, ctx=4096, all chunks (~287)
- Compare: CUDA-KTQ2 vs Vulkan-KTQ2; CUDA-F16 baseline
- **Acceptance gate: |Vulkan_PPL − CUDA_PPL| / CUDA_PPL ≤ 0.3%**
- **Beware PPL-as-only-metric**: add generation-quality check — 5×500-token completions with deterministic seed, diff against CUDA reference for >95% token-prefix match

### 5c. Required hardware matrix
- Tier-1 gate: RTX 2060/Turing (proprietary 550+) + RX 7900 (RADV mesa-25+)
- Tier-2 gate: Intel Arc A770 (force subgroupSize=32)
- Tier-3 (advisory): Apple M-series via MoltenVK

## 6. Driver footguns 2024–2026

| Issue | Status | Impact |
|---|---|---|
| [#11268](https://github.com/ggml-org/llama.cpp/issues/11268) coopmat2 FA incoherent | Open/intermittent | KTQ via cm2 may inherit |
| [#20554](https://github.com/ggml-org/llama.cpp/issues/20554) Intel Arc 140V cm TDR | Open | Disable cm on Intel |
| [#20776](https://github.com/ggml-org/llama.cpp/issues/20776) Arc 140T not detected as XE2 | Open | cm disabled silently |
| [#19420](https://github.com/ggml-org/llama.cpp/issues/19420) Qwen3-Coder Vulkan Intel ARL crash | Open | Validate dispatcher |
| [#16272](https://github.com/ggml-org/llama.cpp/issues/16272) FA on old NV (P620) | Closed/wontfix | Hard gate Turing+ |
| [#18527](https://github.com/ggml-org/llama.cpp/issues/18527) load fail with `-fa off` | Open | Test both modes |
| [#16767](https://github.com/ggml-org/llama.cpp/issues/16767) Vulkan multi-GPU slowdown | Open | Test single-GPU first |

## Review-agent checklist (TL;DR)

- [ ] Scalar-block-layout SSBO for KTQ block (don't trust std430)
- [ ] `SPV_KHR_maximal_reconvergence` + `subgroupBarrier()` before every shuffle
- [ ] `VK_EXT_subgroup_size_control` lock to 32; reject otherwise
- [ ] `precise` only on FWHT final norm-correction accumulator
- [ ] FWHT traversal order matches CUDA exactly
- [ ] fp32 accumulators throughout; fp16 only at storage I/O
- [ ] Per-block dequant test in test-backend-ops, max_abs_err ≤ 1e-3
- [ ] PPL parity ≤ 0.3% drift on Qwen3.6-35B-A3B-IQ2_XXS wikitext-2
- [ ] Generation prefix-match ≥ 95% over 5×500 tokens, deterministic seed
- [ ] Test all three FA paths (scalar / cm1 / cm2) by force-disable env vars
- [ ] Validate on RADV + NVIDIA proprietary + (Tier-2) Intel Arc
- [ ] Log selected pipeline at runtime; assert KTQ never dispatches to unsupported path

## Sources

See full list in raw research output. Key references:
- [Vulkan-Docs #1973 — fused FMA](https://github.com/KhronosGroup/Vulkan-Docs/issues/1973)
- [Graphics Programming — NVIDIA SPIR-V subgroup shuffle bug](https://graphics-programming.org/blog/subgroup-shuffle-execution-dependency-on-nvidia)
- [Collabora — re-converging control flow on NVIDIA](https://www.collabora.com/news-and-blog/blog/2024/04/25/re-converging-control-flow-on-nvidia-gpus/)
- [llama.cpp Discussion #20969 — TurboQuant Extreme KV Quant](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [llama.cpp DeepWiki — Vulkan Backend](https://deepwiki.com/ggml-org/llama.cpp/5.3-vulkan-backend-(cross-platform))
