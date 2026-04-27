# FA Kernel TU-Bloat Profile — Verifying the Bisect Agent's Hypothesis

**Date:** 2026-04-26
**Author:** Debugging investigation (Leichenbestatter run)
**Repo:** `/mnt/d/repos/llama-tq` @ HEAD `559dc7809` (turboquant branch)
**Reference:** `/mnt/d/repos/llama-tq/docs/plans/2026-04-26-moe-pp-regression-bisect.md`

## TL;DR — Verdict: **TU-BLOAT CONFIRMED**

Including `turboquant.cuh` (1376 LOC) and `trellis.cuh` (311 LOC) from `fattn-common.cuh` causes ptxas to allocate **+23 to +79 more registers per thread** on the hot `flash_attn_ext_f16` MMA prefill kernels for Turing (sm_75), even though those kernels never call any TQ symbol. The register pressure increase causes occupancy collapse on the kernels handling `head_dim ∈ {112, 128}` with `ncols1 ∈ {4, 8}` and `ncols2 ∈ {4, 8}` — which is exactly the prefill batch tile used by Qwen3.6-35B-A3B (head_dim=128) at batch 512.

Removing `-rdc=true` (CUDA_SEPARABLE_COMPILATION) from the llama-tq build does **not** reduce the regression (verified by isolation experiment), so RDC is not the primary cause — the TQ headers themselves are.

Runtime confirmation on test-box (2× RTX 2060):

| Build | Model | Test | t/s | Δ |
|---|---|---|---|---|
| upstream `0c6ee1cad` | Qwen3.6-35B-A3B-UD-IQ2_XXS | pp512 | **1174.15 ± 2.27** | baseline |
| llama-tq `6e50fc701` | Qwen3.6-35B-A3B-UD-IQ2_XXS | pp512 | **1014.40 ± 0.31** | **−13.6 %** |

Matches the bisect agent's −14 % observation.

---

## Methodology

1. **Static register-pressure analysis (nvcc -Xptxas -v --resource-usage)**

   Both `~/llama-cpp-upstream` (commit `0c6ee1cad`) and `~/llama-tq` (commit `6e50fc701`) had identical CMake configurations on test-box except for:
   - llama-tq has `CUDA_SEPARABLE_COMPILATION ON` (-rdc=true) and `-DTQ_FATTN_DEBUG`
   - llama-tq has additional FA template instances (KTQ/VTQ pairs)

   Per build, `compile_commands.json` was used to extract the exact nvcc invocation for 7 representative TUs covering all FA kernel families:
   - `fattn-mma-f16-instance-ncols1_{4,8,16,64}-ncols2_*.cu` — MMA prefill kernels
   - `fattn-vec-instance-f16-f16.cu` — vec decode kernels
   - `fattn-tile-instance-dkq{128,256}-dv{128,256}.cu` — tile fallback kernels

   Each was recompiled with `-Xptxas=-v --resource-usage -fatbin`, ptxas output parsed, and the same kernel template instantiations matched between the two trees by mangled name. The llama-tq names carried a static-isolation prefix from `-rdc=true` (`__nv_static_NN__hash__N_filename_..._Z...`); the prefix was stripped to align names.

2. **RDC-isolation experiment**

   To rule out `-rdc=true` as the cause, `fattn-mma-f16-instance-ncols1_8-ncols2_4.cu` from llama-tq was recompiled three ways:
   a. Upstream as-is.
   b. llama-tq as-is (with `-rdc=true` and TQ includes).
   c. llama-tq with `-rdc=true` removed (TQ includes still present).

   If RDC were the cause, (c) should match (a). It did not — (c) matches (b).

3. **Runtime confirmation**

   Sequential `llama-bench -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf -ngl 99 -p 512 -n 0 -r 2` on both trees, GPU0+GPU1 (`CUDA_VISIBLE_DEVICES=0,1`).

---

## Static analysis results

### Hot prefill kernel register pressure (relevant subset)

Kernels with the largest deltas — all are `flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, logit_softcap, has_sinks>` instantiations from `fattn-mma-f16-instance-*.cu` TUs. **`b0b0`** = no logit softcap, no sinks (most common path). **`b1b0`** = with logit softcap.

| TU | Kernel (D, ncols1, ncols2, b_softcap, b_sinks) | Upstream regs | llama-tq regs | Δ regs | Spill stores Δ |
|---|---|---:|---:|---:|---:|
| ncols1_8-ncols2_4 | f16<112,112,8,4,**0**,0> | 176 | **254** | **+78** | 0 |
| ncols1_4-ncols2_8 | f16<112,112,4,8,**0**,0> | 179 | **254** | **+75** | 0 |
| ncols1_4-ncols2_8 | f16<128,128,4,8,**0**,0> | 178 | **213** | **+35** | 0 |
| ncols1_8-ncols2_4 | f16<128,128,8,4,**0**,0> | 184 | **207** | **+23** | 0 |
| ncols1_4-ncols2_8 | f16<96,96,4,8,0,0>     | 167 | 177 | +10 | 0 |
| ncols1_8-ncols2_4 | f16<96,96,8,4,0,0>     | 167 | 174 | +7  | 0 |
| ncols1_8-ncols2_4 | f16<80,80,8,4,0,0>     | 167 | 173 | +6  | 0 |
| ncols1_8-ncols2_4 | f16<64,64,8,4,0,0>     | 158 | 166 | +8  | 0 |
| ncols1_4-ncols2_8 | f16<64,64,4,8,0,0>     | 159 | 162 | +3  | 0 |
| ncols1_8-ncols2_4 | f16<128,128,8,4,**1**,0> | 181 | **168** | **−13** | 0 |
| ncols1_4-ncols2_8 | f16<128,128,4,8,**1**,0> | 176 | **168** | **−8**  | 0 |

Note the asymmetry: TUs with **logit_softcap=false (b0)** regress sharply, but **logit_softcap=true (b1)** sometimes *improves*. The `__noinline__` `dequantize_V_*` and `vec_dot_fattn_vec_KQ_*ktq*` definitions in `fattn-common.cuh` perturb ptxas's allocator differently depending on what other live ranges exist.

Kernels at D ∈ {256, 512, 576} were already register-capped at 255 in **both** branches (line 159+ of the parser output). At-cap, register count is meaningless, but spill bytes vary unpredictably — for the `b1b0`/`b0b1` configs llama-tq usually has higher spill, but for some `b0b0`/`b1b0` configs it has lower. These are noise compared to the clear D=112/128 regression.

### Healthy TUs

`fattn-mma-f16-instance-ncols1_16-ncols2_4.cu` and `ncols1_64-ncols2_1.cu` show **no** register-count differences in the actual hot kernels (both 24 regs in launcher stubs, 255 at-cap in real kernels with mixed spill deltas). These tile geometries are not used for batch-512 prefill on Qwen3.6-35B-A3B.

`fattn-vec-instance-f16-f16.cu` shows mixed small deltas (±5 regs) — vec is the decode kernel, not prefill, so irrelevant for pp512 regression.

`fattn-tile-instance-dkq{128,256}.cu` shows ±1-8 reg deltas in fallback kernels — small impact, but this path isn't selected for f16/f16 KV on supported geometries.

### Non-hot deltas (auxiliary kernels)

A consistent **+5 reg** shift on `flash_attn_stream_k_fixup_general<...>` (30→35 across all TUs) and **+18-19 reg** shift on `flash_attn_combine_results<DV={80,96,112}>` (44/45→63). These are launched once per FA call but are not in the per-token critical path — small absolute time contribution.

### Aggregate

- 322 kernel pairs compared across 7 TUs
- 149 kernels identical (registers + spills)
- 173 kernels differ
- Total registers (sum across all kernels): upstream 22 318 → llama-tq 23 136 (Δ +818, +3.7 %)

The 3.7 % aggregate doesn't tell the runtime story because regression concentrates in the hot D=128 ncols1/2 ∈ {4,8} kernels.

---

## RDC-isolation experiment

Three-way recompile of `fattn-mma-f16-instance-ncols1_8-ncols2_4.cu`:

| Configuration | f16<128,128,8,4,**0**,0> regs | f16<112,112,8,4,**0**,0> regs |
|---|---:|---:|
| Upstream                                  | 184 | 176 |
| llama-tq default (`-rdc=true`)            | 207 | 254 |
| llama-tq with `-rdc=true` **removed**     | 212 | 255 |

Removing `-rdc=true` from the llama-tq compile changes results by ≤ +5 regs vs the rdc-enabled compile, but **does not** restore upstream's allocation. **The TQ header includes are the dominant cause, not the build flag.**

(Side observation: rdc actually *helps* slightly here, presumably because it lets ptxas see fewer side-effects and inline more aggressively within the TU.)

---

## Why this happens

Both `turboquant.cuh` (1376 LOC) and `trellis.cuh` (311 LOC) are pulled into every TU that includes `fattn-common.cuh`, which is every FA TU. They contain:

1. ~60 `static __device__ __forceinline__` functions (`ktq_fattn_dequant_*`, `vec_dot_fattn_vec_KQ_ktq*`)
2. Four `static __device__ __noinline__` functions (`dequantize_V_ktq{1,2,3,4}_1`)
3. Trellis decoder state machines for VTQ{2,3,4}_2
4. PolarQuant Lloyd-Max codebook constants (`__constant__`)
5. Philox 6r-round PRNG + FWHT decoder helpers

Even though no f16-typed kernel calls these, ptxas must:
- Parse all device declarations to build the call-graph
- Reserve register footprint for the `__noinline__` functions (they get emitted, even if uncalled, when their name is taken — and `-rdc=true` makes them externally addressable)
- Process the `__constant__` tables (each takes a `cmem` slot)

The end effect is that ptxas's register allocator runs against a more cluttered TU, picks a different live-range solution, and ends up using more registers per thread for the f16 kernel. On Turing's 64K-register/SM budget at 256 threads/block, **going from 184 regs to 207 regs reduces theoretical occupancy** from `floor(64*1024 / (184*256)) = 1.39` blocks/SM to `floor(64*1024 / (207*256)) = 1.20` blocks/SM. With the kernel's `__launch_bounds__(256, 2)` requesting 2 blocks/SM, neither hits the hint, but the higher-pressure version has less slack for warp scheduling, hurting memory-latency hiding during prefill (which is FLOP-heavy but still bandwidth-sensitive on RTX 2060's ~336 GB/s).

The 254-reg case (D=112) is a step worse: it's literally at the soft cap, suggesting spill avoidance came at the cost of pushing arithmetic into long-latency dependent chains. This config isn't used by 35B-A3B (head_dim=128) but documents that the bloat is severe enough to reach the cap.

---

## Proposed fix (sketch — not implemented in this run)

**Goal:** Don't include `turboquant.cuh` / `trellis.cuh` in TUs that don't need TQ.

**Approach 1 — Move TQ helpers out of `fattn-common.cuh`** (preferred):

Create `fattn-tq.cuh` containing the TQ-specific dequant + KQ-vec-dot helpers currently in `fattn-common.cuh` lines 311-870 (~560 LOC). Include only from:
- `fattn-vec.cuh` (it does need them — has the `is_vtq*_family` constexpr branches)
- `fattn-vec-dispatch-ktq.cu`, `fattn-vec-dispatch-vtq*.cu`
- `fattn-mma-ktq*.cu*`, `fattn-mma-ktq-inline*.cuh`

**Do not** include from:
- `fattn-vec-dispatch-f16.cu` (already separated as the "non-TQ slice")
- `fattn-mma-f16.cuh` and its instance TUs
- `fattn-tile.cu*`, `fattn-wmma-f16.cu*`
- `fattn-common.cuh` itself (move includes upward)

The TQ-template-dispatch in `fattn-common.cuh` lines 1362-1418 (the constexpr `if (type_K == GGML_TYPE_KTQ*)` chain) needs to stay where it is, but it only references `dequantize_V_*` by name, so a forward declaration block in `fattn-common.cuh` plus including `fattn-tq.cuh` only in the TUs that instantiate KTQ/VTQ paths should suffice.

**LOC delta estimate:**
- `fattn-tq.cuh` new file: ~570 LOC (extracted from fattn-common.cuh)
- `fattn-common.cuh`: ~570 LOC removed, ~30 LOC of forward decls added → net −540 LOC
- 4-5 ktq/vtq TUs gain `#include "fattn-tq.cuh"`: +5 lines
- Risk: if the constexpr type_K dispatch in fattn-common ever gets called in a TU without the include, link error. Acceptable — it's a static_assert away.

**Expected impact:** Restores `flash_attn_ext_f16` register count on f16 TUs to upstream levels. Predicted pp512 recovery: ~10-12 % of the 14 % regression (some regression remnant from the auxiliary kernels and `-rdc=true` will persist).

**Approach 2 — Splitting `fattn-common.cuh` further** (more invasive):

Also extract the `flash_attn_combine_results` + `flash_attn_stream_k_fixup_general` register growth (which is independent of TQ — appears to come from the `-rdc=true` static prefix interacting with the templated dispatch). Lower priority — those are auxiliary kernels.

**Approach 3 — Mark `__noinline__` TQ functions as `static`** (incomplete):

The four `dequantize_V_ktq{1,2,3,4}_1` functions are already `static __device__ __noinline__`. With `-rdc=true`, `static` only restricts external linkage; they're still emitted in the TU. Approach 1 is correct.

---

## Action items (in priority order)

1. **Implement Approach 1** — extract TQ-specific helpers from `fattn-common.cuh` into a separate header included only by TQ-aware TUs. Estimated 1-2 hours of focused work + rebuild + bench validation.
2. **Re-run nvcc -Xptxas -v** on a representative ncols1_8/ncols2_4 TU to confirm registers drop back to upstream baseline (≤ 184 for D=128, ≤ 176 for D=112).
3. **Re-run `llama-bench -m Qwen3.6-35B-A3B-UD-IQ2_XXS -p 512 -n 0`** to measure pp512 recovery. Target: ≥ 1140 t/s (within 3 % of upstream's 1174 t/s).
4. **Decode-side check:** also re-run `-p 0 -n 128` to confirm decode (TG) is not impacted negatively by the include reorganization.
5. **Defer:** investigate the `-rdc=true` necessity. Comment in CMakeLists says it's required for Trellis cross-TU `extern __device__` LUT. Phase-2c-only requirement; could in principle be replaced with a `__constant__` per TU at the cost of memory duplication. Not in scope for this fix.

---

## Files / artefacts

- `/tmp/profile_fa_tu.sh` (test-box) — nvcc -Xptxas -v driver script
- `/tmp/parse_fa.py` (test-box) — comparison-table generator
- `/tmp/fa-profile/results.txt` (test-box) — raw ptxas output (2653 lines)
- `/tmp/fa-rdc-test/{UPSTREAM,LLAMA_TQ_DEFAULT,LLAMA_TQ_NO_RDC}.log` (test-box) — RDC isolation logs
- `/mnt/d/repos/llama-tq/ggml/src/ggml-cuda/fattn-common.cuh` (lines 6-7: TQ includes)
- `/mnt/d/repos/llama-tq/ggml/src/ggml-cuda/turboquant.cuh` (1376 LOC)
- `/mnt/d/repos/llama-tq/ggml/src/ggml-cuda/trellis.cuh` (311 LOC)
- `/mnt/d/repos/llama-tq/ggml/src/ggml-cuda/CMakeLists.txt` (lines 152-158: rdc setup)

---

## Caveats / things this analysis does NOT prove

- nsys profiling was not run (deferred — nvcc evidence + runtime bench were sufficient and cheaper). If anyone wants per-kernel µs deltas to confirm which specific instance(s) eat the wall-clock budget, run `nsys profile -t cuda --stats=true llama-bench -p 512 -n 0`.
- The 3.6 % regression on dense 27B (head_dim 128 too) is likely the same mechanism but with smaller absolute impact because dense models do fewer FA calls per pp token (1 attention block per layer, not 1 per expert × layer). The MoE −14 % vs dense −2.3 % gap is consistent with FA-call-count scaling.
- This analysis does not address the TG (decode) regression separately. Decode uses `flash_attn_ext_vec`, where the deltas were small (±2-11 regs); decode regression on this codebase was already <3 % per the project's bench history.
- Other minor sources of −1 to −3 % could exist (e.g. extra `cudaMemcpyAsync` from the FORK_CHANGES added subsystems). Those would need separate bisection if the proposed fix doesn't fully recover the gap.
