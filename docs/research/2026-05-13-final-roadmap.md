# Final Roadmap — 2026-05-13 morning research session

Sechs spezialisierte agents haben parallel optimization-möglichkeiten analysiert.
Konsolidierte erkenntnisse + konkrete priorität für nächste schritte.

## Was wir HEUTE schon erreicht haben

| Optimization | Status | Win |
|--------------|--------|-----|
| V_rows=8 OOB fix (D=128 safety) | ✅ committed | safety only |
| Multi-GPU layer-split discovery | ✅ documented | +40-74% PP |
| ub=128 für 3B/14B (öffis-pattern) | ✅ deployed | +18% PP on top |
| **ub=1024 für 35B-A3B (auto-pattern)** | ✅ deployed | +13% PP |
| MMA-KTQ split V-dequant extension | ✅ committed | code unification |
| Direct MMA-F16 route (skip wrapper) | ✅ committed | +2% PP |
| Multi-warp KTQ/VTQ dequant | ✅ committed | flat (not bottleneck) |
| Pre-scaled VTQ read codebook | ✅ committed | building/pending bench |
| VTQ2_1 4-outputs-per-thread kernel | ✅ committed | building/pending bench |

**Cumulative wins:** 
- 3B Q4_K_M PP@2k: ~160 → 2168 t/s (~14× over pre-Phase-5)
- 35B-A3B PP@4k: 839 → 1437 t/s (+71% vs single-GPU)

## Research-team findings (6 agents this session)

### NICHT pursuen (NEGATIV ROI bestätigt):
1. **Tensor Parallelism** — PR maintainer + Issue #22391 sagen MoE-broken; asymmetric PCIe x16+x4 macht's hopeless
2. **SageAttention SM_75 port** — RHT/smoothing-kollision, 4-6 wochen für vermutlich +15-25%, nicht 2.1×
3. **Speculative decoding auf 35B-A3B** — andere haben 19 configs getestet auf RTX 3090, **-3 bis -12% NEGATIV** wegen MoE expert-loading overhead
4. **Stream-overlap compute/memory split** — TG ist memory-bandwidth-bound auf RTX 2060

### Hauptbottleneck identifiziert:
**VTQ-V dequant** ist -26% slower als f16 (NICHT KTQ-K, NICHT scratch buffer).
- Root cause: 10-byte struct ist **nicht alignable**, jedes 3. block straddet 32-byte sector
- Solution-space:
  - Reines kernel-rewrite: **ceiling -10%** (sm_75 hat kein cp.async für byte-strided)
  - **SoA staging bei slot-load: -3 to -5% (q4_0-niveau)** — principled fix

## Priorisierte nächste schritte

### Phase A: Validate aktuelle commits (1-2 hours)
1. Build durchwarten (läuft)
2. Bench VTQ2_1 x4 kernel auf 35B-A3B PP@4k — recovery von -26% → ?
3. PPL test (wikitext-2 100 chunks) als safety net
4. Wenn x4 kernel >50% recovery liefert → fertig für heute
5. Wenn ≤30% recovery → SoA-staging plan starten

### Phase B: SoA staging (1-2 days, fallback)
Falls Phase A nicht reicht. Plan in `2026-05-13-soa-staging-design.md`:
1. Add `ggml_cuda_pool_alloc<ggml_half> d_buf` + `<uint8_t> qs_buf`
2. Reformat kernel: `block_vtq2_1[]` → `(d_buf, qs_buf)` SoA. ~1ms for 200k tokens
3. Modify `flash_attn_ext_f16_vtq_load_tile_V_vtq2_1` to consume SoA inputs
4. Cache SoA buffer between FA calls (only invalidate on cache-grow)

### Phase C: Decode TG opts (low-hanging, after Phase A)
Aus VTQ-decode analysis:
1. **Codebook → shared memory** beim kernel-entry (+3-6% TG, low risk)
2. **Warp-broadcast `x[ib].d`** statt 32× redundant LDG (+1-3%)
3. **3-bit decode unrollen** mit single uint32 load (+2-4% VTQ3_1 only)

### Phase D: Future research-only
- Upstream tensor-parallelism sync (b8738) — falls Lance NCCL bekommt
- Async weight-prefetch dual-stream auf x4 GPU1 (+3-5% TG, 1-2 days)
- POD-Attention prefill-decode overlap — research, kein code

## Decision points

**Wenn x4 kernel gewinnt (Phase A erfolgreich):**
- Commit, dokumentieren, fertig für heute
- Phase B (SoA) ist für später wenn user es will

**Wenn x4 kernel nicht hilft:**
- Investigate via `ncu` profiling
- Move to Phase B (SoA staging) als principled fix

## Konkrete deploy-empfehlungen (single-user OidaNice-GPT-34B)

Aktueller "best" stack basierend auf research-konvergenz:
```bash
# Single-GPU0 (kein PCIe-bottleneck), kombiniert mit Phase 5:
CUDA_VISIBLE_DEVICES=0 llama-server \
    -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
    --mmproj Qwen3.6-35B-A3B-mmproj-F16.gguf \
    -ngl 99 -ub 1024 -b 2048 -fa \
    --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
    --moe-pin-experts --backend-sampling \
    --tq-protect-layers 16 \
    -c 100000 --no-mmap --parallel 1
```

**Validated peak performance (this session):**
- Single-GPU0: PP@2k ~1300, TG 81 t/s
- Dual-GPU: PP@2k 1443, TG 76 t/s (-6% TG cost!)

→ **Für single-user: SINGLE-GPU IST BESSER**. Dual-GPU nur für massive ctx oder multi-user mit --parallel=N.

## Sources

- agent reports archived in 2026-05-13-research-team-findings.md + 2026-05-13-soa-staging-design.md
- thc1006 speculative bench: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090
- HackMD speculative report: https://hackmd.io/ODXuOQNzSiyUITz7g9mtBw
- Issue #22391 TP Qwen3.6 broken: https://github.com/ggml-org/llama.cpp/issues/22391
