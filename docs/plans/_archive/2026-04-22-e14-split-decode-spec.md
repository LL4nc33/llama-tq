# E14 Split-Decode Kernel — Spec für VTQ_2 Regression Recovery

**Datum:** 2026-04-22
**Status:** Proposed — ready-to-execute
**Target:** TG ≥ 50 tok/s auf Qwen3.5-35B-A3B IQ2_XS / KTQ2_1 × VTQ3_2 / RTX 2060 sm_75

---

## 1. Problem Recap

Baseline TG mit VTQ_2 (Trellis v2, QK_VTQ_TRELLIS=128): **4.35 tok/s** vs. ~60 tok/s Pre-VTQ. Root Cause (cuobjdump + Triton autoresearch):

- `flash_attn_ext_vec` in `ggml/src/ggml-cuda/fattn-vec.cuh:21` hat **249 regs/thread** → 1 block/SM Occupancy auf Turing (sm_75, 64K reg file/SM).
- Inner Loop (Z. 340-394) ruft `dequantize_V(V + k*nb21, ...)` **pro Sample pro Iteration** (Z. 355/362/382). VTQ3_2 = Trellis-Walk mit ~192 ALU + LUT-Gather pro Sample.
- GQA=8: 16 Q-heads teilen 2 KV-heads → decode wird 8× multipliziert.

**E11 (abandoned):** `fattn-vec-vtq2.cuh` cached-decode war für `ncols=1` algorithmisch falsch (Amortisation 1×, 60× slowdown). Doku: `docs/plans/2026-04-22-e11-design-flaw-diagnosis.md`.

**Warum E14 passt:** Decode einmal global, amortisiert über alle Q-heads × ctx (statt ncols). GQA=8 spielt uns in die Hand: 16 Q-heads reusen denselben dequantisierten V-Block.

**Triton autoresearch** (`docs/plans/2026-04-21-triton-autoresearch-results.md:24`) zeigt E14 = 102 GB/s (11.8× über naive), Pattern `Split + FP16 GEMM + num_warps=1`. Direkt in CUDA portierbar, llama.cpp MMQ nutzt bereits denselben dequant→GEMM-Pattern.

---

## 2. High-Level Design — 2-Kernel-Pipeline

```
VTQ3_2 V-Cache (2.1 GB @ 400k)  ─[K1: bulk dequant tile]→  fp16 V-Scratch (875 MB/tile)
                                                                    ↓
              Q fp16 (1×16×256)  ─────[K2: existing F16 FA-vec or cuBLAS]→  VKQ fp16
```

**K1 — `bulk_dequantize_vtq3_2_to_fp16`:**
- Input: VTQ3_2 V-slice `(tile_ctx, n_kv_heads=2, D=256)`.
- Output: dense fp16 gleicher Shape.
- Adapter auf existing `dequantize_block_vtq3_2_nc_cuda` (siehe `convert.cu:1068` + row-kernel `convert.cu:786`).
- `__launch_bounds__(128, 4)` — low reg pressure.

**K2:**
- **K2a (Option A):** Existing `flash_attn_ext_vec<D, ncols, type_K=KTQ2_1, type_V=F16, ...>` Pfad. Der F16-V-Fall ist bereits kompiliert, `dequantize_V` wird zu memcpy.
- **K2b (Option C):** `cublasHgemmStridedBatched` + custom softmax.

K2 existiert als battle-tested F16 Pfad. Nur K1 davorschalten, `type_V` auf `GGML_TYPE_F16` umbiegen wenn V=scratch.

---

## 3. Integration Points — 3 Optionen

| Option | Was | VRAM | Change-Footprint | Decode-Overhead |
|---|---|---|---|---|
| **A: Throwaway Pre-Pass** | Vor FA-Launch: `bulk_dequant(V_vtq, V_scratch_fp16)`; dann FA mit `type_V=F16` | +875 MB transient/tile | ~150 LOC | ~4.7ms/step (memory-bound) |
| **B: Persistent fp16 Mirror** | fp16-Spiegel beim KV-cache write | +5.4 GB permanent | tief, `llama-kv-cache.cpp` | 0 |
| **C: cuBLAS GEMV statt FA** | K1 wie A; K2 = `cublasHgemmStridedBatched` | +875 MB transient/tile | +200 LOC; verliert FA-Softmax-Fusion | gleich + extra softmax-pass |

**Empfehlung: Option A als 3B2, Option C als Fallback in 3B3.**

- **B fällt aus** wegen VRAM-Budget (§4) — 12 GiB VRAM, Modell+KV ≈ 9.5 GiB.
- **A minimiert Risiko**: FA-Kernel unverändert, softmax/sinks-Fusion bleibt. Testet zuerst ob reg-pressure durch F16-Pfad allein verschwindet — `dequantize_V` für F16 ist trivial, wahrscheinlich DER Hebel der 249 → ≤150 Regs bringt.
- **C nur falls A nicht reicht** — cuBLAS GEMM hat bei batch=1 bekannt schlechte Latenz + kein softmax-fusion.

**Hook-Point:**
- `ggml/src/ggml-cuda/fattn.cu` `ggml_cuda_flash_attn_ext`: branch `if (V->type in {VTQ2_2, VTQ3_2, VTQ4_2} && ncols==1)` → `ggml_cuda_pool_alloc<half>` scratch + swap V-pointer + fall-through F16 dispatch.
- `fattn-vec-dispatch.cuh` bleibt unverändert, wir treffen nur den F16-V-Case.

---

## 4. Memory Budget — WICHTIG

**Prod-Config:** Qwen3.5-35B-A3B @ 400k ctx, GQA=8, D=256, n_kv_heads=2.

- V-Cache VTQ3_2 full: `400_000 × 2 × 256 × 3.5bit / 8` = **2.10 GB** ✓
- V-Scratch full-context (Option A naiv): `400_000 × 2 × 256 × 2B` = **5.36 GB** ✗ zu groß

**⚠️ Korrektur zum Auftrag:** Nicht "400 MB/step" — full-attention braucht komplette V-Matrix dequantisiert.

**Mitigation: Tiled decode.**
- TILE = 65_536 Tokens → 875 MB Scratch/Tile
- 6 Tiles × (dequant 0.78ms + FA-step ~1ms) × online-softmax-merge über `dst_meta` (FA supports nativ)
- Overhead negligible bei TG

**3B2 MUSS tiling from day one** — naiver full-dequant scheitert OOM.

---

## 5. cuBLAS GEMM Path (Option C Details)

Bei batch-gen: `Q.shape=(1, 16, 256)`, `V.shape=(tile_ctx, 2, 256)`.

GQA=8 → pro KV-head 8 Q-heads. Reshape Q zu `(2, 8, D)`, pro KV-head:
- `cublasHgemmStridedBatched`: `C[8, tile_ctx] = Q_kv[8, D] × V_kv[D, tile_ctx]^T`
- Batch=2 (KV-heads), strides entsprechend

Call (Pattern aus `ggml-cuda.cu:2075`):
```cpp
cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    tile_ctx, 8, D,
    &alpha_h, V_fp16, CUDA_R_16F, D, stride_V,
              Q_kv,   CUDA_R_16F, D, stride_Q,
    &beta_h, KQ_logits, CUDA_R_16F, tile_ctx, stride_logits,
    2, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

Dann: custom softmax (~50 LOC) + zweiter cublasHgemm für `VKQ = softmax(KQ) × V_fp16`.

**Turing Tensor Cores** für fp16×fp16→fp16 verfügbar. **Break-even vs. FA** erst wenn FA weiter reg-limited ist.

---

## 6. Fallback — Direct Mod zu fattn-vec (Phase 3B0)

Kleinerer Patch: `dequantize_V` aus inner loop (`fattn-vec.cuh:341-394`) rausziehen in Pre-Loop über alle `k`. Output `half V_cache[WARP_SIZE][D]` in shmem.

Risiken:
- WARP_SIZE × D × 2B = 32 × 256 × 2 = 16 KB/warp. 4 warps = 64 KB = **exakt sm_75 shmem limit**. Zero margin.
- VKQ[ncols][D/2] accumulator ist vermutlich der Reg-hog, nicht decode-state.

**Verdikt:** Phase-3B0 Spike (0.5d) um zu validieren ob reg-pressure durch decode-removal signifikant sinkt. Falls ja → Option A wird billig. Falls nein (VKQ confirmed hog) → direkt Option C.

---

## 7. Phase Gates

| Gate | Scope | Success Criterion | Fail → |
|---|---|---|---|
| **3B0** (0.5d) | Spike: reg-count nach decode-removal aus FA inner loop (broken correctness, regs only) | Regs < 180 → 3B1. Regs ≥ 220 → 3B3 direkt | 3B3 direkt |
| **3B1** (1d) | Standalone `bulk_dequantize_vtq3_2_to_fp16` + tiling harness + `test-backend-ops` | MSE < 1e-4 vs reference, ≥ 300 GB/s | convert.cu adapter revidieren |
| **3B2** (2d) | Option A Integration in fattn dispatch, tiled (TILE=64k), benchmark TG | **TG ≥ 15 tok/s** (3.4× baseline) | 3B3 oder design revidieren |
| **3B3** (2d) | Option C cuBLAS GEMM + custom softmax | **TG ≥ 50 tok/s** (11.5× baseline, matches Triton) | post-mortem |

**Win-Condition:** 3B3 ≥ 50 tok/s → E11 post-mortem + E14 win-doc + upstream-PR.

---

## 8. Risiken

| # | Risiko | Mitigation |
|---|---|---|
| R1 | cuBLAS GEMM latency bei Tile=65k schlechter als FA custom | Tile-size sweep in 3B3; Tensor Core fp16 dominiert ab 8k×256 |
| R2 | Dequant throughput floor: 2.1 GB / 448 GB/s = 4.7ms/step → **213 tok/s theoretical ceiling** | Hohes Ziel, Target 50 ist 4× darunter, ok |
| R3 | fp16 persistent buffer VRAM (Option B) | Ausgeschlossen, Tiling |
| R4 | Throwaway alloc overhead | ggml cuda pool reused buffers, ~1µs |
| R5 | Tile-boundary softmax merge correctness | FA `dst_meta` online-softmax production-tested. `llama-perplexity` gate |
| R6 | KTQ2_1 K-path könnte alleiniger Bottleneck werden nach V-Fix | Profile nach 3B2. Analoger K-split als Phase 3C (out-of-scope) |
| R7 | VTQ2_2/VTQ4_2 parallel pipeline needed | 3B1 template über `type_V`, alle 3 billig |

---

## 9. LOC Estimate

| Component | New LOC | Modified | Files |
|---|---|---|---|
| `bulk_dequantize_vtq_to_fp16.cu` | ~180 | 0 | 1 new |
| dispatch hook + pool-alloc + tile loop | ~120 | ~40 | `fattn.cu`, `fattn-vec-dispatch.cuh` |
| softmax kernel (C only) | ~60 | 0 | 1 new |
| cuBLAS wrapper (C only) | ~80 | ~20 | `ggml-cuda.cu` |
| test-backend-ops cases | ~50 | ~10 | `tests/test-backend-ops.cpp` |
| **Total Option A** | **~350** | **~50** | ~400 LOC |
| **Total A+C** | **~490** | **~70** | ~560 LOC |

**Wiederverwendung:** `dequantize_block_vtq3_2_nc_cuda` (`convert.cu:1068`) deckt 90% K1 ab — nur stride-aware launcher wrappen. Spart ~200 LOC.

---

## 10. tl;dr

1. **Decode-overhead ist der bottleneck**, nicht attention-compute.
2. **Split-decode pattern** (E14) amortisiert decode über alle Q-heads × ctx.
3. **Option A (throwaway + tiled)** = primärer Pfad. **Option C (cuBLAS)** = Fallback.
4. Gate 3B2 TG ≥ 15 conservative, 3B3 TG ≥ 50 = Win.
5. Falls 3B3 erreicht → **10-12× speedup**, E11 post-mortem + E14 win-doc + upstream.

---

## Ready-to-Execute Checklist

- [ ] **3B0 Spike:** Patch `fattn-vec.cuh:355-383` — inline decode durch `half tmp[4] = {0,0,0,0}` ersetzen (broken correctness, measure regs only). Build, `cuobjdump --dump-resource-usage build/.../libggml-cuda.so | grep -A2 flash_attn_ext_vec` → read reg count. Decision gate.
- [ ] **3B1:** Create `ggml/src/ggml-cuda/bulk-dequant-vtq.cu`, `bulk-dequant-vtq.cuh`. Template über `type_V ∈ {VTQ2_2, VTQ3_2, VTQ4_2}`. Adapt `dequantize_block_vtq3_2_nc_cuda` launcher from `convert.cu:1068`. `test-backend-ops` case `BULK_DEQUANT_VTQ` gegen `ggml_get_to_fp16_cuda`.
- [ ] **3B2:** `ggml_cuda_flash_attn_ext()` top-level dispatch: check `V->type == GGML_TYPE_VTQ3_2 && dst->ne[1] == 1`. If true:
  - `ggml_cuda_pool_alloc<half> v_scratch(pool, TILE × n_kv × D)`
  - Tile loop `TILE=65536`
  - Per tile: launch `bulk_dequant_vtq3_2_to_fp16` → swap V-ptr → `ggml_cuda_flash_attn_ext_vec_case<D, KTQ2_1, F16>` → accumulate via `dst_meta` online-softmax.
- [ ] Bench: `llama-bench -m Qwen3.5-35B-A3B-IQ2_XS.gguf -ctk ktq2_1 -ctv vtq3_2 -p 0 -n 128`. Gate: ≥ 15 tok/s.
- [ ] Regression: `llama-perplexity -c 2048` vs baseline — max drift 0.5%.
- [ ] **3B3 (bei Bedarf):** cuBLAS Path wenn 3B2 < 50 tok/s.
- [ ] Deploy zu gpu00:8791, A/B test vs Prod 4.35 baseline.

**Relevante Dateien:**
- `ggml/src/ggml-cuda/fattn-vec.cuh` (Z. 19-21 kernel entry, Z. 340-394 inner loop mit dequant calls)
- `ggml/src/ggml-cuda/fattn-vec-dispatch.cuh` (dispatch macro)
- `ggml/src/ggml-cuda/convert.cu` (Z. 786 row-dequant, Z. 1067-1070 nc-dequant — reuse targets)
- `ggml/src/ggml-cuda/fattn.cu` (top-level dispatch — hook point Option A)
- `ggml/src/ggml-cuda/ggml-cuda.cu:2075` (cuBLAS GEMM batched pattern — copy für Option C)
- `docs/plans/2026-04-21-triton-autoresearch-results.md` (E14 reference)
- `docs/plans/2026-04-22-e11-design-flaw-diagnosis.md` (warum nicht E11)
- `docs/plans/2026-04-21-vtq2-regression-analysis.md` (baseline regression analysis)
