# MMA-KTQ Kernel Integration — Spec

**Datum:** 2026-04-22
**Priorität:** HIGH (einziger bekannter Path zu f16-nahem PP auf KTQ)
**Target-Hardware:** 2× RTX 2060 (sm_75, Turing)
**Target-Model:** Qwen3.5-35B-A3B-IQ2_XS + 400k ctx (production config)

## Motivation

Autoresearch-Session 2026-04-22 identifizierte Root Cause für KTQ PP-Regression:
`fattn.cu:359-372` dispatcht **alle TQ-Typen unconditional zum VEC kernel**.
VEC kernel supports nur `cols_per_block ∈ {1, 2}` → keine tensor-core parallelism
bei PP. f16 und q4_0 laufen durch MMA_F16 bei PP (tensor-core accelerated).

**Resultat:** KTQ2_1 PP = 92 tok/s (9× langsamer als f16 @ 876). TG ist ok (55.9 tok/s
= 78% von f16) weil VEC kernel für TG workloads angemessen ist.

## Ziel

Füge MMA-Path-Support für KTQ-Typen hinzu. Dispatcher wählt:
- PP (`Q->ne[1] >= 8`): MMA-KTQ Kernel (tensor cores)
- TG (`Q->ne[1] <= 2`): VEC-KTQ Kernel (unverändert)

## Target-Metriken

Auf Qwen3.5-35B-A3B-IQ2_XS, 2× RTX 2060, ktq2_1 K + f16 V:

| Metrik | Aktuell | Target | Stop-Ship falls |
|---|---|---|---|
| PP512 tok/s | 92 | **≥ 500** | < 300 |
| TG128 tok/s | 55.9 | **≥ 50** (keine Regression) | < 50 |
| PPL Δ | baseline | **< 0.1%** drift | > 1% |
| Build-Zeit | ~35min | **< 90min** | > 90min |

## Phasing

### Phase 0 — Setup & Reading (0.5 Tag)

1. Read `ggml/src/ggml-cuda/fattn.cu:359-372` (dispatcher)
2. Read `ggml/src/ggml-cuda/fattn-mma-f16.cu` (existing MMA path)
3. Read existing template-instances: `fattn-mma-f16-instance-ncols1_X-ncols2_Y.cu`
4. Read `ggml/src/ggml-cuda/fattn-vec.cuh` (current KTQ path, for dequant reference)
5. Document:
   - MMA kernel entry point signature
   - Template parameters signature (D, ncols1, ncols2, type_K, type_V)
   - Where KTQ dequant would inject into MMA tile-load
   - Register budget estimate (`ptxas -v` on existing f16 MMA)
6. Define correctness baseline:
   - Run `llama-bench -m Qwen3.5-35B ... -ctk ktq2_1 -ctv f16 -n 128 -p 512` → save output
   - Run `llama-perplexity -m ... -ctk ktq2_1 -ctv f16 -f wiki-test.raw` → record PPL

**Deliverable:** Updated spec-doc mit konkreten Line-numbers + reg-budget.

### Phase 1 — Dispatcher Hook (0.5 Tag)

Ziel: Dispatcher route KTQ zu MMA für PP, zu VEC für TG. MMA Kernel ist aber
noch stub der VEC fallback macht.

1. `fattn.cu`:
   ```cpp
   // Existing:
   if (Q->src[0]->type == GGML_TYPE_F16 || ...) {
       if (Q->ne[1] >= 8) { ggml_cuda_flash_attn_ext_mma_f16(ctx, dst); return; }
   }

   // New:
   bool is_ktq = (K->type == GGML_TYPE_KTQ1_1 || ... || K->type == GGML_TYPE_KTQ4_1);
   if (is_ktq && V->type == GGML_TYPE_F16 && Q->ne[1] >= 8) {
       ggml_cuda_flash_attn_ext_mma_ktq(ctx, dst);  // NEW entry point
       return;
   }
   // Else fall through to existing VEC path
   ```

2. Neue Datei `fattn-mma-ktq.cuh` mit stub:
   ```cpp
   void ggml_cuda_flash_attn_ext_mma_ktq(ctx, dst) {
       // Phase 1: just call VEC path as placeholder
       ggml_cuda_flash_attn_ext_vec(ctx, dst);
   }
   ```

3. Build + test: existing TG/PP numbers unchanged → dispatcher hook ok

**Commit:** "feat(fattn): dispatcher hook for MMA-KTQ path (stub)"

### Phase 2 — MMA-KTQ Kernel Instance (2-3 Tage)

**Critical Phase.** Hier entstehen die neuen Kernels.

1. `fattn-mma-ktq.cu` + Template-Instances:
   - Template params matching MMA-F16: `D, ncols1, ncols2, KQ_stride, type_K, type_V`
   - Modification point: **K tile load**
   - Current f16-MMA loads K via `ldmatrix.sync.aligned.m8n8.x4` (f16 direct)
   - KTQ-MMA needs: 1) load quantized K, 2) dequant in warp-cooperative mode, 3) stage in shmem as f16, 4) MMA from shmem

2. Dequant shim:
   ```cuda
   // Warp-cooperative dequant of one K-block into shmem
   __device__ void dequant_ktq_to_shmem(
       const block_ktq2_1 * K_quant_block,
       __half * K_dequant_shmem,
       const int lane_id)
   {
       // 32 threads = 1 block (QK_VTQ = 32)
       // Read 32 sample codes + FWHT reconstruct + Lloyd-Max lookup
       // Write to shmem K_dequant_shmem[lane_id]
       // Syncthreads before MMA loads from shmem
   }
   ```

3. Template instantiations (je KTQ type × f16 V × ncols combos):
   - `fattn-mma-ktq-instance-ncols1_8-ncols2_1-ktq2_1-f16.cu`
   - Similar für ktq3_1, ktq4_1
   - Initial limit: nur `ncols1_8-ncols2_1` (minimal parallelism)
   - Extend zu weiteren ncols combos nach ersten Benchmarks

4. Build-Check: incrementally, teste eine template-instance ← vollständig, dann expand

**Commits:**
- "feat(fattn-mma-ktq): warp-cooperative dequant shim"
- "feat(fattn-mma-ktq): ktq2_1 + f16 V template instance"
- "feat(fattn-mma-ktq): ktq3_1 + ktq4_1 template instances"

### Phase 3 — Correctness (0.5 Tag)

Vor Performance-Bench: Output must be bit-identisch (oder quasi-identisch) zu VEC path.

1. Smoke test:
   ```bash
   llama-cli -m Qwen3.5-35B ... -ctk ktq2_1 -ctv f16 -p "Wien ist" -n 50
   ```
   Compare output text vs VEC-path output.

2. PPL test:
   ```bash
   llama-perplexity -m ... -ctk ktq2_1 -ctv f16 -f wiki-test.raw -c 2048
   ```
   PPL must match within 0.1% of VEC baseline.

3. NaN check: first-token attention with sinks protection.

4. Unit test: produce attention output for known (Q, K, V) tensors,
   compare VEC vs MMA path output max-diff < 1e-3.

**Abort-Condition:** PPL drift > 1% → rewind, math bug in dequant shim.

### Phase 4 — Benchmark (0.5 Tag)

Full comparison sweep:

```bash
./build/bin/llama-bench \
    -m Qwen3.5-35B-A3B-IQ2_XS.gguf \
    -ctk f16,q4_0,ktq2_1,ktq3_1,ktq4_1 \
    -ctv f16 \
    -fa 1 -ngl 99 -ts 12,12 \
    -p 512,2048 -n 128 \
    -r 3
```

Record: PP512, PP2048, TG128 for all configs. Compare against pre-MMA baseline.

**Long-context test:**
```bash
./build/bin/llama-bench ... -ctk ktq2_1 -ctv f16 -p 16384 -n 128
```
Stress-test at realistic production context length.

### Phase 5 — Commit + Deploy (0.5 Tag)

1. README update: neue benchmark table mit echten Zahlen
2. Commit als series: each phase = separate commit
3. Push phase2
4. Deploy-Test auf prod gpu00:8791 (after stopping existing prod)
5. Measure 30min steady-state TG auf prod workload

## Risiken & Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|---|---|---|---|
| MMA register-budget > 255 regs/thread | MEDIUM | HIGH | Start mit `__launch_bounds__(128, 1)` wie f16-MMA, messe mit `ptxas -v`. Falls zu hoch: reduce ncols2 target oder accept 1 block/SM. |
| KTQ dequant pre-MMA verliert coalescing | MEDIUM | MEDIUM | Shmem staging + explicit `cp.async` für K-block load. Bench memory-bound vs compute-bound. |
| PPL drift > 1% | LOW | HIGH | Bit-identical dequant math (copy from VEC path). Unit-test output match. |
| Template instantiations explodieren build-time | HIGH | LOW | Nur 3 KTQ types × 1 V type = 3 new instances. Limit ncols combos zu {8,1}, {16,1} initial. |
| MMA-KTQ dequant slower than VEC für große ncols | MEDIUM | HIGH | Threshold tuning: MMA nur wenn `ne[1] >= 8`. Below → VEC. Measure, don't guess. |
| Neue build-time > 90min | LOW | MEDIUM | Parallel build jobs, skip unused types. |

## Abort-Plan

Falls nach Phase 2 + 3 kein messbarer Gain:

1. Revert commits, reset zu pre-MMA-KTQ state
2. Dokumentiere Learning im README (limitations section)
3. Ship current state als "KTQ = decode-first quality format, use q4_0 for balanced workloads"
4. Close task #153 mit "structural limitation, see docs"

## Success-Gate (shipping only wenn alle drei)

1. **PP512 KTQ2_1 ≥ 500 tok/s** (5× über current 92)
2. **TG128 KTQ2_1 ≥ 50 tok/s** (no regression vs 55.9)
3. **PPL KTQ2_1 Δ < 0.1%** vs current baseline

## Out-of-Scope

- VTQ V-cache Optimizations (separate topic, V ist already f16 in prod)
- Other hardware targets (Ampere/Hopper — sm_75 focus)
- Other attention variants (GQA-only, no MHA)
- FP8 Tensor cores (Turing doesn't have)

## Timeline

- Phase 0: 0.5 Tag (morgen Vormittag)
- Phase 1: 0.5 Tag (morgen Nachmittag)
- Phase 2: 2-3 Tage (neue Kernels, am kritischsten)
- Phase 3: 0.5 Tag (Correctness)
- Phase 4: 0.5 Tag (Benchmark)
- Phase 5: 0.5 Tag (Deploy + Docs)

**Total: 4-5 Tage fokussierte Arbeit.**

## References

- Autoresearch doc: `docs/plans/2026-04-22-ktq-autoresearch.md` (local)
- Sweep results: `docs/plans/2026-04-23-fa-profile-results.md` (local)
- Reality check: `docs/plans/2026-04-23-reality-check-session.md` (local)

---

## Session 2026-04-22 — Phase 1+2-Split Findings

Phase 1 Stub: dispatcher hook wired + verified. KTQ2_1 PP512/TG128 unchanged
(97 / 56 t/s, matching baseline) — confirms hook routes to VEC fallback when
stub active.

Phase 2 Split-Dequant prototype (bulk K→fp16 scratch + existing MMA-F16
kernel, blueprint from `fattn-vec-dispatch-vtq2-split.cu`):

| test | KTQ2_1 t/s | f16 t/s | notes |
|---|---|---|---|
| PP64  | 264 | — | +164% vs VEC baseline ~100 |
| PP128 | 219 | — | +119% |
| PP256 | 165 | — | +65% |
| PP512 | 97  | 877 | tie — dequant cost eats MMA win |
| PP1024| 54  | — | **-45% regression** |
| PP2048| 29  | 871 | **-70% regression** |
| TG128 | 56  | 72  | unchanged (gated to VEC) |

**Root cause of regression:** split-dequant bulk-rewrites K->ne[1] × head_dim
fp16 values per FA call. K-cache grows with prefill, so the dequant cost
scales O(ctx·hidden) per call while VEC's on-the-fly dequant stays O(batch·hidden).
For ctx > ~256 the dequant memory pass dominates.

**Ship decision:** Phase 1 stub stays wired (enum + entry point + switch).
Dispatcher gate disabled — KTQ keeps routing to VEC until an inline
warp-cooperative dequant lands in the MMA tile-load path. That's the
original Phase 2 "variant A" in this spec (fattn-mma-f16.cuh:547 injection
point) and it's the only path that can win at PP >= 512.

The split-dequant prototype is kept in `fattn-mma-ktq.cu` as reference — it
is wired but unreachable unless the dispatcher gate is re-enabled. Future
inline kernel can replace the function body.
