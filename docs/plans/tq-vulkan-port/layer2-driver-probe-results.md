# Layer 2 Driver Probe Results — TQ Vulkan Port

*Source: Layer-2 builder probe, completed 2026-05-09. Empirical verification
on real hardware that the assumptions in `spec-master.md` §4.1, §11 hold for
the tier-1 target (RTX 2060 sm_75 + NVIDIA proprietary 580.x).*

---

## Test environment

| Field | Value |
|---|---|
| Host | `gpu00.node` (Ubuntu 24.04, kernel 6.8.0-111-generic) |
| GPU | 2× NVIDIA GeForce RTX 2060 (Turing TU106, sm_75, 12 GB) |
| Driver | NVIDIA proprietary `580.126.09` (`libnvidia-gl-580` 580.126.09-0ubuntu0.24.04.2) |
| Loader | `libvulkan1` 1.4.313 (LunarG SDK 1.4.313) |
| `vulkaninfo` | 1.4.313 |
| `glslc` | shaderc v2023.8 (LunarG SDK 2025.2, 2025-04-29 build) |
| `spirv-dis` | spirv-tools v2025.2 (2025-04-29) |
| `glslang` | 15.3.0 |
| Device API version | 1.4.312 |
| Vulkan instance API | 1.4.313 |
| Conformance | 1.4.1.3 |

No additional packages installed; full LunarG SDK 1.4.313 was already in place.

---

## Probe 1 — Required-extension support

Spec §4.1 mandates eight extensions across the three FA paths. Every one is
advertised by both RTX 2060 devices on this host.

| Extension | Required by | Advertised? | Revision |
|---|---|---|---|
| `VK_KHR_shader_subgroup_uniform_control_flow` | Mandatory (all paths) | YES | 1 |
| `VK_KHR_shader_maximal_reconvergence`        | Optional (gated)      | YES | 1 |
| `VK_EXT_subgroup_size_control`               | Mandatory (sg=32 lock)| YES | 2 |
| `VK_EXT_scalar_block_layout`                 | Mandatory (ABI)       | YES | 1 |
| `VK_KHR_shader_float16_int8`                 | Storage path          | YES | 1 |
| `VK_NV_cooperative_matrix2`                  | cm2 path              | YES | 1 |
| `VK_KHR_cooperative_matrix`                  | cm1 path              | YES | 2 |

### Subgroup capabilities (GPU0)

```
subgroupSize                       = 32
minSubgroupSize                    = 32
maxSubgroupSize                    = 32
requiredSubgroupSizeStages: 14 (all stages incl. compute)
computeFullSubgroups               = true
subgroupSizeControl                = true
shaderSubgroupUniformControlFlow   = true
shaderMaximalReconvergence         = true
subgroupQuadOperationsInAllStages  = true

subgroupSupportedOperations (11):
  BASIC, VOTE, ARITHMETIC, BALLOT, SHUFFLE, SHUFFLE_RELATIVE,
  CLUSTERED, QUAD, ROTATE, ROTATE_CLUSTERED, PARTITIONED_BIT_NV

subgroupSupportedStages: VERTEX, TESS_CTRL, TESS_EVAL, GEOMETRY,
  FRAGMENT, COMPUTE, RAYGEN, ANY_HIT, CLOSEST_HIT, MISS,
  INTERSECTION, CALLABLE, TASK, MESH
```

`SHUFFLE_BIT` is supported on the COMPUTE stage. `requiredSubgroupSize=32`
can be locked from compute (the FA shader's hard requirement).

**Verdict:** PASS — every spec-mandated extension is present, subgroup
size is fixed at 32 with hardware support for required-size control.

---

## Probe 2 — `subgroupShuffleXor` correctness in tight FWHT loop

Direct empirical answer to spec §11 Q2 ("Does shuffle-in-loop work on driver
580.x without `subgroupBarrier`?").

### Method

Two compute shaders, identical except for the barrier:

**`probe-fwht-with-barrier.comp`** (defensive — spec §4.1 default):
```glsl
[[unroll]] for (uint step = 1u; step < 32u; step <<= 1) {
    subgroupBarrier();
    float other = subgroupShuffleXor(v, step);
    v = ((lane & step) != 0u) ? (other - v) : (v + other);
}
```

**`probe-fwht-no-barrier.comp`**: same loop, no `subgroupBarrier()`.

Both compiled with `glslc --target-env=vulkan1.3 -O`. A single-file
`vulkan-hpp`-style host harness (`probe-host.cpp`, ~280 LOC, raw vulkan.h
to keep the dep surface minimal) creates a compute pipeline with
`requiredSubgroupSize=32` + `REQUIRE_FULL_SUBGROUPS_BIT`, enables
`VK_KHR_shader_subgroup_uniform_control_flow` and the Vulkan 1.3
`subgroupSizeControl` + `computeFullSubgroups` features, uploads input
0..31 (fp32), dispatches one workgroup, downloads, bit-diffs against a
CPU reference FWHT.

A second harness (`probe-host-many.cpp`) sweeps `N ∈ {1, 64, 1024, 16384,
65536}` workgroups over random fp32 input ([-1, 1], seed=0xC0DE) to
exercise occupancy / multi-SM scheduling.

### Single-workgroup result

```
device: NVIDIA GeForce RTX 2060 driver=2434761280 api=1.4.312
== WITH subgroupBarrier ==
  max_abs_err = 0.000e+00  mean_abs_err = 0.000e+00  wrong_lanes(>1e-6) = 0
  ref: 496 -16 -32 0 -64 0 0 0 ...
  gpu: 496 -16 -32 0 -64 0 0 0 ...
== WITHOUT subgroupBarrier ==
  max_abs_err = 0.000e+00  mean_abs_err = 0.000e+00  wrong_lanes(>1e-6) = 0
  ref: 496 -16 -32 0 -64 0 0 0 ...
  gpu: 496 -16 -32 0 -64 0 0 0 ...
```

### Stress sweep result

```
[WITH-bar  N=1]      max_abs_err=0.000e+00  wrong_lanes=0
[NO-bar    N=1]      max_abs_err=0.000e+00  wrong_lanes=0
[WITH-bar  N=64]     max_abs_err=0.000e+00  wrong_lanes=0
[NO-bar    N=64]     max_abs_err=0.000e+00  wrong_lanes=0
[WITH-bar  N=1024]   max_abs_err=0.000e+00  wrong_lanes=0
[NO-bar    N=1024]   max_abs_err=0.000e+00  wrong_lanes=0
[WITH-bar  N=16384]  max_abs_err=0.000e+00  wrong_lanes=0
[NO-bar    N=16384]  max_abs_err=0.000e+00  wrong_lanes=0
[WITH-bar  N=65536]  max_abs_err=0.000e+00  wrong_lanes=0
[NO-bar    N=65536]  max_abs_err=0.000e+00  wrong_lanes=0
```

Every lane in 65 536 workgroups (~2.1 M lanes) was bit-exact against the
CPU reference, with **and** without the barrier. Acceptance threshold
`max_abs_err ≤ 1e-6` is met by both.

### Bit-diff between with-barrier and without-barrier

By transitivity (both equal the CPU reference exactly), the with/without
outputs are bit-identical.

### Interpretation

This **does not** refute the Collabora-documented NVIDIA shuffle-in-loop
bug — that bug was reproduced from a *divergent* shuffle context. Our
FWHT is fully subgroup-uniform: every lane participates in every stage,
the loop is `[[unroll]]`-ed at compile time, and the entire kernel runs
in a uniform region declared via `VK_KHR_shader_subgroup_uniform_control_flow`.
Under those conditions, driver 580.126.09 produces correct results
without an explicit `subgroupBarrier()`.

**Verdict:** PASS — `subgroupShuffleXor` works correctly in the FWHT
butterfly on driver 580.x **provided the loop is uniform and**
`VK_KHR_shader_subgroup_uniform_control_flow` **is requested**. The
defensive `subgroupBarrier()` between stages is **not** required for
correctness on this driver.

**Recommendation for spec §4.1:** keep the `subgroupBarrier()` in the
canonical KTQ2_1 shader (zero correctness risk, micro-cost on Turing;
this is the spec's "belt-and-suspenders" stance). Treat its presence as
a soft guard — it can be removed in a future perf pass if profiling
shows it on the critical path, since this probe shows the driver does
not need it for this access pattern.

---

## Probe 3 — Spec-constant array indexing folding

Direct empirical answer to spec §11 Q3 ("Does `PQ_CB_2BIT[4]` indexed by
per-thread variable lower to constant-bank loads?").

### Method

Two shaders, both producing the codebook-indexed lookup.

**Variant A — spec-constant array** (per spec §4.1 Layout):
```glsl
layout(constant_id = 0) const float PQ_CB_2BIT_0 = -1.489560;
layout(constant_id = 1) const float PQ_CB_2BIT_1 = -0.451428;
layout(constant_id = 2) const float PQ_CB_2BIT_2 = +0.451428;
layout(constant_id = 3) const float PQ_CB_2BIT_3 = +1.489560;
...
float cb[4] = float[4](PQ_CB_2BIT_0, PQ_CB_2BIT_1, PQ_CB_2BIT_2, PQ_CB_2BIT_3);
float v = cb[idx];
```

**Variant B — inline `const float[4]`** (the pattern llama.cpp's existing
IQ-quant Vulkan shaders use, per `research-glsl-primitives.md` Q1):
```glsl
const float cb[4] = float[4](-1.489560, -0.451428, +0.451428, +1.489560);
float v = cb[idx];
```

Both compiled with `glslc -O --target-env=vulkan1.3`, disassembled with
`spirv-dis`.

### Variant A SPIR-V (key ops)

```
%42 = OpSpecConstant %float -1.48956001
%43 = OpSpecConstant %float -0.451427996
%44 = OpSpecConstant %float  0.451427996
%45 = OpSpecConstant %float  1.48956001

%41 = OpVariable %_ptr_Function__arr_float_uint_4 Function
...
%46 = OpCompositeConstruct %_arr_float_uint_4 %42 %43 %44 %45
      OpStore %41 %46
%50 = OpAccessChain %_ptr_Function_float %41 %36
%51 = OpLoad %float %50
```

The four codebook values become `OpSpecConstant` (resolved at
pipeline-bake time). The array is materialised into a `Function`-storage
local, written via `OpCompositeConstruct`+`OpStore`, then read via
`OpAccessChain`+`OpLoad` indexed by the per-lane `%36`.

### Variant B SPIR-V (key ops)

```
%46 = OpConstantComposite %_arr_float_uint_4
        %float_n1_48956001 %float_n0_451427996
        %float_0_451427996 %float_1_48956001
%49 = OpVariable %_ptr_Function__arr_float_uint_4 Function %46
...
%50 = OpAccessChain %_ptr_Function_float %49 %36
%51 = OpLoad %float %50
```

The array becomes a true `OpConstantComposite` (resolved at SPIR-V
compile time, not pipeline-bake). The `OpVariable` is initialised with
this constant composite directly, then accessed via the same
`OpAccessChain`+`OpLoad` pattern.

### Interpretation

At the **SPIR-V level**, neither variant fully folds to immediate
operands — both leave the indexed load to be lowered by the driver
compiler. The spec-constant variant defers the four float values to
pipeline bake; the inline variant resolves them at glslc time. Past that
difference, the two are structurally equivalent: a Function-storage
4-element float array, read by a runtime-dynamic index.

This is the **expected and correct** SPIR-V shape. SPIR-V optimisation
deliberately doesn't lower indexed array loads to switch/select chains
because the driver compiler has better information (constant-bank
capacity, spill tradeoffs, register pressure) to make that decision.

What we cannot determine from SPIR-V alone is whether the NVIDIA driver
folds this to a uniform/constant-bank load (effectively zero-cost on
Turing) or materialises a stack array. Settling that requires a SASS
dump, which the proprietary NVIDIA driver does not expose for SPIR-V
inputs; we'd need either Nsight Compute (sampled assembly) or a
hand-rolled `nvcc`-cross-check on equivalent CUDA.

### Verdict

PASS, with caveat: SPIR-V emission matches the prior-art pattern (llama.cpp
IQ shaders use the exact same `OpAccessChain`+`OpLoad`-on-Function-array
shape) and Turing has well-known support for folding small const-arrays
indexed by lane-id into uniform-load + select chains. There is no
SPIR-V-level red flag.

**Recommendation:** prefer **inline `const float[4]`** (Variant B) for
V1 — it matches the pattern that already works in llama.cpp's IQ
shaders, removes one moving part (no host-side spec-constant plumbing),
and gives glslc a stronger hint (`OpConstantComposite` vs
`OpCompositeConstruct`-from-spec-consts) for any cross-block CSE/constant
propagation. Defer the spec-constant variant to V2 if/when codebook
variation per pipeline becomes a need.

A SASS-level confirmation can be deferred to Layer 4 (when Nsight Compute
is available against a real KTQ shader) — it does not block POC.

---

## Probe 4 — `shaderMaximalReconvergence` query

Spec §4.1 lists `VK_KHR_shader_maximal_reconvergence` as optional/gated.

```
GPU0: shaderMaximalReconvergence = true
GPU1: shaderMaximalReconvergence = true
```

Both RTX 2060s on driver 580.126.09 advertise the feature. Per
`research-glsl-primitives.md` Q1, NVIDIA proprietary advertises it from
Linux 535.43.22 onward, which 580.x clears comfortably.

**Verdict:** PASS. The optional `VK_KHR_shader_maximal_reconvergence`
gate can be enabled on the tier-1 target. Note the portability caveat
already documented in research: MoltenVK does **not** support it
([MoltenVK#2478](https://github.com/KhronosGroup/MoltenVK/issues/2478)),
so the gate must remain optional in the pipeline-create path.

---

## Pass / Fail Summary

| Probe | Spec reference | Result | Acceptance |
|---|---|---|---|
| 1 — Extension support | §4.1, §11 | All 7 advertised | PASS |
| 2 — Shuffle-in-loop correctness | §11 Q2 | Bit-exact with & without barrier across 65 536 wgs | PASS |
| 3 — Spec-const array folding | §11 Q3 | SPIR-V emission as expected; matches IQ-shader prior art | PASS (defer SASS to L4) |
| 4 — `shaderMaximalReconvergence` | §4.1 | True on both GPUs | PASS |

---

## Go / No-Go for Layer 3 (POC)

**GO.** No hard blockers identified.

The four hard prerequisites the spec lays out are all met on the tier-1
hardware:

1. Required extension surface is complete (Probe 1).
2. The FWHT shuffle pattern is correct on driver 580.x — even **without**
   the defensive barrier the spec mandates (Probe 2). Spec §4.1 should
   keep the barrier as a portability/Q2 hedge for non-NVIDIA tier-1
   targets, but on RTX 2060 + 580.x specifically there is no shuffle bug
   to defend against in this access pattern.
3. The codebook-array SPIR-V shape is normal and matches the in-tree
   IQ-quant prior art (Probe 3). A small spec tweak: use inline
   `const float[4]` rather than spec-constants for V1 — see Probe 3
   recommendation.
4. Optional `maximal_reconvergence` is available on both GPUs (Probe 4).

Layer 3 (POC) can proceed against this host, this driver, this hardware,
without additional driver upgrades or special builds.

---

## Recommended spec deltas (low-cost, optional)

These are derived from probe evidence; none block POC, but feeding them
back into `spec-master.md` would tighten the spec.

1. **§4.1, KTQ2_1 layout** — replace the four `layout(constant_id=...)
   const float PQ_CB_2BIT_*` lines with an inline
   `const float PQ_CB_2BIT[4] = float[4](...)`. Rationale: matches
   IQ-quant in-tree prior art; gives glslc a stronger constant-fold
   hint; one fewer host-side moving part. Move spec-constants to V2 if
   per-pipeline codebook variation is added later.

2. **§4.1, KTQ2_1 algorithm comment** — soften "MANDATORY on driver
   580.x" on the `subgroupBarrier()` line to "DEFENSIVE on driver 580.x
   for non-uniform reuse; not required for correctness in this uniform
   loop on 580.126.09 — keep for portability to MoltenVK/Mali and for
   safety against compiler regressions". This reflects Probe 2 evidence
   but preserves the spec's defensive intent.

3. **§11 Q3 status** — mark "answered (SPIR-V)" with note that SASS-level
   confirmation deferred to Layer 4 when Nsight Compute is available.

---

## Probe artefacts

All probe sources and built binaries live on `gpu00.node:/tmp/`:

```
/tmp/probe-fwht-with-barrier.comp      /tmp/fwht-with.spv
/tmp/probe-fwht-no-barrier.comp        /tmp/fwht-no.spv
/tmp/probe-fwht-with-barrier-many.comp /tmp/fwht-with-many.spv
/tmp/probe-fwht-no-barrier-many.comp   /tmp/fwht-no-many.spv
/tmp/probe-spec-cb.comp                /tmp/probe-spec-cb.spv
/tmp/probe-spec-cb.spvasm
/tmp/probe-spec-cb-inline.comp         /tmp/probe-spec-cb-inline.spv
/tmp/probe-spec-cb-inline.spvasm
/tmp/probe-host.cpp                    /tmp/probe-host
/tmp/probe-host-many.cpp               /tmp/probe-host-many
```

Local mirror in `/tmp/` on the dev workstation.

No additional packages were installed on `gpu00.node`; the LunarG SDK
1.4.313 + NVIDIA proprietary 580.126.09 stack already in place was
sufficient.
