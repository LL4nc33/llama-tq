# Phase 4 — realistic scope assessment (2026-05-13 22:48)

## Was wir gefunden haben (3 agents + code-deep-dive)

1. **Existing fusion in mmvq.cu macht silu+mul+up im EINEN kernel** — aber nur für n_tokens ≤ 8 (TG only)
2. **PP@4k geht in mmq.cu pfad** — kein fusion vorhanden
3. **`unary_gated_op_kernel` macht silu+mul schon kombiniert** (unary.cu:255) — also nur 3 kernel-launches im hot path: mul_mat_id_up + mul_mat_id_gate + swiglu_split
4. **HBM-traffic estimation:** 2× intermediate tensors (gate + up) × f32 × 4k × 1536 × 64 layers = 3.2 GB write+read pro forward bei PP@4k

## Realistic LOC + zeit-estimate für full mmq+GLU fusion port

| component | LOC | risk |
|---|---|---|
| `ggml_cuda_mm_fusion_args_*` extension (mmq) | ~30 | low |
| mmq.cuh template `has_fusion` param | ~200 | **high** (4188 LOC file) |
| Gate-tile shared-memory budget | ~100 | high (turing SMEM 64KB limit) |
| GLU epilog inline | ~50 | medium |
| Dispatch update in ggml-cuda.cu | ~100 | low |
| Bench-gate + fallback flag | ~50 | low |
| **Total** | **~530 LOC** | mixed |

**Build-zeit pro iteration:** ~2-3h (template-header touched).
**Test-zyklen:** mindestens 3-5 (ptxas register-check, smoke, bench, regression, tune)
**Real calendar-time:** 1-2 wochen

## Lighter-weight alternatives

### A) Intermediate tensor type → f16 statt f32
- Halbiert HBM-traffic für intermediates
- **Problem:** mul_mat_id ist baked f32 output
- Scope: dispatch change in ggml.c builder, BUT downstream ops müssen f16-input handhaben
- Risk: subtle precision-issues
- **Est win: +3-7% PP**, **scope: 1-2 tage**

### B) Sort intermediates ins fp16 tile statt full f32 buffer
- Compute in fp32, write fp16
- 1 line change in mul_mat_id kernel epilog
- **Risk:** floats sind teilweise overflow im fp16 range (vor silu)
- **Est win: +2-4% PP**, **scope: 0.5 tage**

### C) Activation-fusion in mul_mat_id epilog
- gate's mul_mat_id schreibt direkt silu(result) statt result
- Mul_mat_id template-param `has_unary_epilog`
- **Scope: ~100 LOC**, mmq template touched still
- Risk: similar register-budget issue
- **Est win: +3-5% PP**

### D) Bench GRAPH_OPT=1 first (zero code change)
- env-var test, single-GPU + force-graphs
- Schon implementiert für QKV-fanout (intra-attention-layer)
- **Scope: 5 min**
- **Est win: 0-3% TG**, vermutlich nichts

## Lehre

**Volle mmq+GLU fusion (ik_llama PR #229 port) ist 1-2 wochen.**
Das ist KEIN "quick-win" sondern echtes feature-development.

## Empfehlung für jetzt

Quick-win first sequenz:
1. **Bench GRAPH_OPT=1** nach build (5 min, sanity-check)
2. **Bench checkpoint-fix** smoke-test mit multi-turn (verify Issue #22384)
3. **Decision point:** lohnt 1-2 wochen für phase 4 wirklich, oder gibt's bessere ROI?

Alternativen die wir noch nicht ausprobiert haben:
- **MoE-pin-experts** flag-bench-sweep (Issue #20757, +3-20% TG hot-experts)
- **Adjust MMVQ_MAX_BATCH_SIZE** auf turing (current=8, höher = mehr fusion-eligibility für medium-batch)
- **Activation-fusion (Alternative C, ~100 LOC)** als kleinerer phase-4-skalpell-fix

Best ROI/effort ratio:
1. GRAPH_OPT bench: 0-3% / 5 min = ∞ ratio (free)
2. MMVQ_MAX_BATCH_SIZE tune: 0-10% / 30 min = sehr hoch
3. Expert-pinning sweep: 3-20% / 2-3h = hoch
4. Activation-fusion: 3-5% / 1 tag + 3h build = medium
5. Full mmq+GLU fusion: 8-15% / 1-2 wochen = niedrig (mehr risk + scope)

## Konkreter neuer plan

Statt direkt phase-4-coding:
- Phase 4.0 (5 min nach build): GRAPH_OPT=1 bench
- Phase 4.1 (30 min): MMVQ_MAX_BATCH_SIZE experiment (dispatch.cu change → 5min build)
- Phase 4.2 (2-3h): expert-pinning sweep
- Phase 4.3 (1 tag): activation-fusion in mul_mat_id epilog (lighter version)
- Phase 4.4 (deferred): full mmq+GLU fusion only if 4.1-4.3 nicht ausreicht
