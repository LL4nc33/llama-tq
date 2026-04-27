# Phase 3A1 Validation Results

**Datum:** 2026-04-22 03:21-03:25 UTC
**Build:** `build-e11/` mit `-DFATTN_VTQ2_CACHED=1`, branch phase2 @ 60258e002
**Kernel:** `flash_attn_ext_vec_vtq2_cached<128, 1, KTQ2_1, VTQ3_2, softcap∈{0,1}>`

## Partial Gate Status

| Metric | 3A1 Target | Measured | Status |
|---|---|---|---|
| Build green | yes | yes | ✅ |
| test-backend-ops FLASH_ATTN_EXT | PASS | PASS (alle OK) | ✅ |
| Regs/thread | ≤ 150 | **128** | ✅ |
| Blocks/SM (derived) | ≥ 2 | **4** (genau wie launch_bounds) | ✅ |
| TG128 tok/s (Qwen3.5-35B-A3B, D=256) | ≥ 25 | 4.35 | ❌ (not applicable — kernel guards D=128 only, fallback to legacy) |
| TG128 tok/s (Qwen3-0.6B, D=128, KTQ2_1×VTQ3_2) | ≥ 25 | **1.60** | ❌ **GATE RED** |
| PPL delta vs legacy | ≤ 0.5% | not-yet-measured | — |

## 🚨 GATE RED — E11 Kernel is 60× SLOWER than baseline on D=128

- Baseline 0.6B with VTQ3_2 usually ~100 tok/s (small model, no memory pressure)
- E11 cached decode delivers **1.60 tok/s** on same model
- Dispatch hook DOES fire (model is D=128, KTQ2_1, VTQ3_2 — exactly the single combo we gated on)
- Kernel IS being called, but has a severe performance bug (or correctness bug causing retries)

## Root-Cause Candidates (not yet isolated)

1. **`__syncwarp()` in warm loop** — unnecessary if only 32 lanes, but maybe triggers broader sync
2. **Shmem bank conflicts** — 1 KiB `smem_V_cache[nwarps][128]` with stride 128 on fp16 could conflict
3. **Launch config mismatch** — kernel declared `__launch_bounds__(128, 4)` but launcher passes different nthreads
4. **LUT access regression** — maybe the warp-wide decode hammers the 256KB `ggml_trellis_table` inefficiently
5. **Correctness bug cascade** — wrong V values cause softmax/sampler to retry or produce degenerate logits (unlikely — test-backend-ops passes for f16 combos, KTQ/VTQ combos not in default suite)

## Recommendation: REVERT dispatch hook

Per `docs/plans/2026-04-22-e11-phase3a1-validation-runbook.md` §7 Gate decision for TG<10:

> **RED (TG < 10 OR regs > 180 OR PPL > 1%)** → design wrong. Revert 3A1 dispatch hook, keep kernel code for reference. Re-evaluate: E14 (split decode → fp16 buffer → cuBLAS GEMM) may be the correct pattern for GQA=8 instead.

Kernel code (fattn-vec-vtq2.cuh) keep in-tree for diagnosis. Dispatch hook in fattn-vec-dispatch-vtq2.cu → guard with `#ifdef FATTN_VTQ2_CACHED_ENABLED` (OFF default) or comment out.

**Diagnosis-Priority 1:** nsight-compute kernel profile to find the actual slowdown cause before designing Phase 3B.

## Notes on Test Setup

- Prod-Server auf gpu00 (Qwen3.5-35B-A3B, PID 167491→1600480) wurde für Messung gestoppt und nach 2min wieder hochgefahren
- gpu00 checkout auf `fc1c512c1` + uncommitted E11 files (scp workaround wegen DNS-breakage von vorgestern)
- 3A1 commits 31c6790c0 + 60258e002 existieren in origin/phase2, sind aber lokal auf gpu00 nicht als commit sichtbar


## Regs-Diagnose (cuobjdump)

```
Function ...flash_attn_ext_vec_vtq2_cached<128, 1, KTQ2_1, VTQ3_2, softcap=1>...
  REG:128 STACK:0 SHARED:3328 LOCAL:0 CONSTANT[0]:560
Function ...flash_attn_ext_vec_vtq2_cached<128, 1, KTQ2_1, VTQ3_2, softcap=0>...
  REG:128 STACK:0 SHARED:3328 LOCAL:0 CONSTANT[0]:560
```

Vergleich zu baseline-legacy path: 249 regs → 128 regs (**-49%**).

Blocks/SM Rechnung (Turing sm_75):
- Register-File/SM = 65536
- 128 threads × 128 regs = 16384 regs/block
- 65536 / 16384 = **4 blocks/SM** — exakt was `__launch_bounds__(128, 4)` forciert

Shmem-Budget 3328 B/block × 4 blocks = 13 KiB, weit unter 64 KiB SM-Cap.

## Build-Anmerkungen

Erster Build-Versuch mit -j4 crashte um 29% auf ggml-cuda target, ohne sichtbare compile error im Log (stderr vom Log getrennt). Retry mit -j4 lief ohne Änderung grün durch → vermutlich transient (ccache race oder CPU-spike). Nicht reproduzierbar.

Kein Code-Fix nötig, kein Revert nötig.

## TG-Messung blockiert

Beide RTX 2060 GPUs voll belegt vom Prod-Server (PID 167491, Qwen3.5-35B-A3B IQ2_XS, 11.5+10.8 GiB). 
Kein Modell größer als 300 MiB lädt.

User-Ping abgesetzt: 5min Prod-Downtime für Messung freigegeben?

## Parallel abgeschlossen

- README root-cause-note mit cuobjdump-Diagnose (commit `60258e002`, gepusht)
- Phase 3A2 Spec (`docs/plans/2026-04-22-e11-phase3a2-spec.md`) ready-to-execute
- Validation Runbook (`docs/plans/2026-04-22-e11-phase3a1-validation-runbook.md`)
- Memory-Update `project_vtq2_e11.md` in session memory

## Nächster Schritt

Nach Zustimmung zu Prod-Stop:

```bash
ssh claude@gpu00.node
sudo systemctl stop cortex-llama-prod   # or pkill -f llama-server
./build-e11/bin/llama-bench \
  -m ~lance/models/Qwen3.5-35B-A3B-IQ2_XS.gguf \
  -fa 1 -ctk ktq2_1 -ctv vtq3_2 \
  -ngl 99 -mg 0 -sm none -p 0 -n 128 -r 2
# Prod wieder starten mit same args
```

Oder schneller: Messung vs. Baseline am selben Modell mit `build/` (master) statt `build-e11/` für A/B.
