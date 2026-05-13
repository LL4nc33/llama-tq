# MAJOR FIND: VTQ-V is the bottleneck, not KTQ-K — 2026-05-13

## TL;DR

KTQ2_1 K-quant ist auf 35B-A3B (D=256 GQA=8) **praktisch kostenlos** (-0.6-0.8% vs f16).
VTQ2_1 V-quant kostet **-16 bis -25%**. KTQ+VTQ kombiniert = -16% PP.
Asymmetric K=KTQ, V=f16 hits **f16-niveau** bei half VRAM für K.

## Bench-matrix (Qwen3.6-35B-A3B IQ2_XXS, dual-GPU, ub=1024)

| K type | V type | PP@2k | PP@4k | TG@32 | Δ vs f16 (PP@4k) |
|--------|--------|-------|-------|-------|------------------|
| f16    | f16    | 1765  | 1985  | 79    | baseline |
| **ktq2_1** | **f16** | **1755** | **1969** | **78** | **-0.8%** ✓ |
| q8_0   | q8_0   | 1738  | 1941  | 76    | -2.2% |
| q4_0   | q4_0   | 1733  | 1936  | 76    | -2.5% |
| ktq2_1 | vtq2_1 | 1471  | 1465  | 77    | -26% |
| q8_0   | vtq2_1 | 1330  | 1259  | 76    | -37% |
| ktq2_1 | q4_0   | 315   | 194   | 57    | -90% (falls in VEC) |

## Interpretation

1. **KTQ K-dequant ist gut implementiert** — der convert kernel mit FWHT + sign-decode
   ist nicht relevant teurer als q4_0/q8_0 weil:
   - K wird einmal per layer dequantiert in scratch
   - FWHT ist warp-cooperative, hides latency
   - K-tensor strides sind groß (D × heads) → fully coalesced

2. **VTQ V-dequant ist der bottleneck**:
   - V wird auch einmal per layer dequantiert in scratch
   - VTQ-decode = codebook lookup + scale (eigentlich BILLIGER als KTQ FWHT)
   - ABER: V hat eine andere shape (`head_dim_v` × `n_kv_heads_v`) und stride-pattern
   - Vermutung: VTQ-dequant kernel ist NICHT optimal coalesced für V-shape

3. **Asymmetric K=KTQ V=f16 ist der praktische sweet-spot** für PP-heavy workloads:
   - Half VRAM für K saved (KTQ2_1 = 2.5 bpw vs f16 = 16 bpw → 84% reduction)
   - Full VRAM cost für V
   - Total KV-cache size ~58% of f16/f16 = significant memory saving
   - Speed-overhead ~0.8% (negligible)

## Action items

### Tier 1: Production deploy decision
- Current `deploy-oidanice-gpt-dualgpu-200k.sh` uses `ktq2 + vtq4`. Consider switching
  to `ktq2_1 + f16` for **better speed**, marginally less VRAM saving:
  - 200k ctx: KTQ2_1+VTQ4_1 = ~3GB total KV, KTQ2_1+f16 = ~14GB total KV
  - At 200k ctx, fits dual-GPU only with VTQ V → KTQ+f16 only for <50k ctx
  - **Practical recommendation:** if ctx <50k, use ktq2_1/f16; if ctx >50k, use ktq2_1/vtq2_1

### Tier 2: Optimize VTQ-V dequant kernel
The dequant kernel for VTQ2_1 in `convert.cu` should be profiled and optimized.
Likely targets:
- Memory access pattern not coalesced for V-tensor shape
- Block-per-warp launch config could be tuned
- Pre-scaled codebook (saves 1 FMUL) — already considered but reverted earlier

### Tier 3: Direct VTQ-V inline in mma-f16 kernel
Skip the scratch-buffer altogether for V — read VTQ blocks directly in the
tile-load function, dequant in shmem. Same pattern as Phase 5 inline path
but for the standard mma-f16 (non-inline) kernel.

## Conclusion

**Wir hatten die ganze zeit auf falsches optimization target geschaut.**
KTQ-K ist NICHT das problem. VTQ-V ist der bottleneck. Eine
einfache config-änderung (ktq2_1/f16 statt ktq2_1/vtq2_1) bringt 35B-A3B
auf **f16-niveau bei 58% KV-cache memory**.
