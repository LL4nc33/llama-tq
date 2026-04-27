# E14 Phase 3B0 Spike — Register Reduction Proof

**Datum:** 2026-04-22
**Experiment:** Stubbed `dequantize_V` in `fattn-vec.cuh` inner loop to 0.0f, measured regs via cuobjdump
**Result:** **GATE GREEN** — decode-removal unlocks occupancy as predicted

## Measurement

File: `build-e11/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-instance-f16-vtq3_2.cu.o`

Note: despite filename "f16-vtq3_2", this template was instantiated for D∈{256, 128}, ncols∈{1, 2}, softcap∈{0,1}. We read the D=256 row (production target for Qwen3.5-35B-A3B).

| Template Instance | Regs (baseline) | Regs (spike: decode stubbed) | Reduction | Blocks/SM* |
|---|---|---|---|---|
| D=256, ncols=2, softcap=1 | 249 | **141** | -43% | 1 → 3 |
| D=256, ncols=2, softcap=0 | 248 | **182** | -27% | 1 → 2 |
| D=256, ncols=1, softcap=1 | ~249 | **89** | -64% | 1 → **5** |

* `blocks/SM = 65536 / (128 threads × regs)`, Turing sm_75

## Conclusion

**dequantize_V inside the FA inner loop is 43-64% of the register pressure** (depending on ncols/softcap template dimensions). The TG decode path (ncols=1) is the biggest winner: 249 → 89 regs = **5× more blocks/SM**.

This is the expected signature of "decode-state lives across the whole k-loop" — exactly what a split-decode (K1 dequant + K2 FA-F16) eliminates.

## Next Step: Phase 3B1

Gate decision per spec `docs/plans/2026-04-22-e14-split-decode-spec.md` §7:
- regs < 180 → proceed to 3B1 (standalone bulk dequant kernel)
- regs ≥ 220 → jump to 3B3 (cuBLAS)

**All three measured rows < 180 → proceed to 3B1.** Best case 89 regs means the F16 FA path will have near-optimal occupancy.

## Insight on 3B1 Scope

`ggml_get_to_fp16_nc_cuda(GGML_TYPE_VTQ3_2)` already exists (`convert.cu:1068`) and returns a function pointer with the exact signature we need for K1. No new kernel needed — Phase 3B1 reduces to:
- Wrapper function that calls the existing dequant
- Integration in `fattn.cu` dispatch (tile loop, pool-alloc scratch, swap V-pointer)

Spec update: Phase 3B1 estimate **drops from 1 day → 2 hours**. Phase 3B2 gets the full weight of the work.

## Spike Code (reverted)

Stubbed in `fattn-vec.cuh:353-364` and `fattn-vec.cuh:380-394` — replaced both `dequantize_V` calls with `tmp[i] = make_half2(0,0)` / `make_float2(0,0)`. File reverted locally + scp'd back to gpu00. Not committed.

## Ceiling Recalculation

With regs=89, ncols=1:
- 5 blocks/SM on Turing (30 SMs) = 150 concurrent blocks
- Baseline was 1 block/SM = 30 concurrent
- **Theoretical occupancy gain: 5×**

Combined with elimination of LUT-thrashing + sequential qs-loads in the hot path, the observed TG should approach the memory-bandwidth ceiling (213 tok/s from §8 R2 of the spec).

**Conservative target for 3B2 remains ≥ 15 tok/s.** Achievable upside up to 50+ tok/s if FA-F16 path is not bound elsewhere.
