# v6 Final Findings & Recommendation

**Datum:** 2026-04-23
**Status:** Phase A+B+C research complete

## Key Numbers (alle auf post-RHT Qwen3.5-27B V-samples, 131k vectors)

| V-Cache Type | bpw | rel MSE | Use Case |
|---|---|---|---|
| VTQ2_1 | 2.5 | **13.25%** | Extreme memory savings |
| VTQ_MIXED | 3.0 | 10.98% | Nischenoption, weder Fisch noch Fleisch |
| **VTQ3_1** | **3.5** | **3.07%** | **Recommended default** — 4.3× besser als VTQ2_1 bei +1.0 bpw |
| VTQ4_1 | 4.5 | 0.84% | Near-lossless, +1.0 bpw over VTQ3_1 |

## Experimente und deren Erkenntnisse

### A: VTQ_MIXED implementiert + getestet
- End-to-end funktioniert (block struct + quant/dequant + kv-cache integration)
- CPU roundtrip: 10.98% rel MSE (matcht Python prediction binnen 0.02%)
- **Verdict:** funktioniert, aber **dominated by VTQ3_1**. CUDA-kernel nicht wert.

### B: Codebook Optimierung (Laplace vs Gaussian)
- 3-bit Gaussian: 3.07% → 3-bit Laplace (fitted): 2.92% = **-4.7%**
- 4-bit Gaussian: 0.84% → 4-bit Laplace (fitted): 0.77% = **-8.6%**
- **Verdict:** margin zu klein, skip. Maintenance-Kosten > Gewinn.

### B2: Calibrated Outlier Selection (vs stride-4)
- Per-position variance ratio in QK=32: **8.57x** (erhebliche Outliers existieren!)
- Top-8 calibrated positions: 10.34% rel MSE vs stride-4 10.98% = **-5.8%**
- **Verdict:** minimaler Gewinn; Paper-level 22% improvement war synthetic-artifact.
  Real RHT verteilt outliers zu gleichmäßig für calibration-basierten Gain.

## Lessons Learned

1. **Paper's outlier-split ist für head_dim=128 ohne RHT.** Unser QK=32 mit RHT diffundiert outliers bereits effektiv. Paper-gain nicht replizierbar.

2. **RHT + Laplace-optimiertes 2-bit Codebook** war schon **der** kritische Trick. Für 3/4-bit würde Laplace nur marginale Verbesserung bringen.

3. **Regularization-Effekt auf 35B (VTQ2_1 = -1% PPL vs f16)** ist Glück, nicht Qualität. Auf 0.8B ist VTQ2_1 +21% PPL. Für generelle Deployment-Empfehlung **VTQ3_1 ist safer choice**.

4. **Die 4.3× MSE-Reduktion VTQ2_1 → VTQ3_1** bei nur +1.0 bpw ist der eigentliche Value-lever. Das ist nicht neu, aber wir haben es jetzt **numerisch bestätigt** mit real data.

## Recommendation: C — VTQ3_1 als recommended Default

**Nicht:** Marketing-Umstellung auf "less accurate but memory-efficient" VTQ2_1.
**Sondern:** Dokumentation macht **VTQ3_1 zur ersten Empfehlung**, VTQ2_1 zur "memory-extreme" option.

### Änderungen

1. **README.md Banner:**
   - "Qwen3.5-35B-A3B @ 70 tok/s mit VTQ2_1 (2.5 bpw, -1% PPL*)" 
   - *Regularization-artifact on large models, nicht accuracy claim
   - **Empfohlen:** VTQ3_1 (3.5 bpw) for reproducible accuracy across model sizes
   
2. **docs/turboquant.md:**
   - Quality/memory tradeoff section: Table mit unseren echten numbers
   - **Recommended: VTQ3_1 für neue deployments**
   - VTQ2_1 nur bei extreme Memory constraints

3. **common/arg.cpp:** keine Default-change — user setzt explizit `--cache-type-v`

### Optional: VTQ_MIXED als "mid-tier option" lassen

- Keine CUDA kernel bauen, aber CPU-impl bleibt für Experimente
- Docs: "Bei Bedarf für 3.0 bpw zwischen VTQ2_1 und VTQ3_1"
- Niedrigste Priorität

## Production Deploy Impact

**Aktueller gpu00:8791 prod:** Qwen3.5-35B-A3B VTQ2_1 @ 70 tok/s, -1% PPL
**Wenn Switch zu VTQ3_1:**
- Memory: 2.5 → 3.5 bpw = 40% mehr V-cache
- Für 200k ctx: V-cache von 8 GB → 11 GB (passt noch in 12 GB VRAM)
- Für 400k ctx: von 16 GB → 22 GB → **passt nicht mehr**
- TG: vermutlich leicht langsamer (mehr data dequant per sample), geschätzt ~65 tok/s
- PPL: deutlich besser (messbar) auf kleineren Modellen

**Recommendation:** VTQ3_1 für contexts ≤ 200k, VTQ2_1 für contexts > 200k.

## Nächste Schritte

1. **Jetzt:** Dokumentation schreiben (README, turboquant.md)
2. **Heute/Morgen:** Test deploy auf gpu00 mit `--cache-type-v vtq3_1` für 30min-Session
3. **Bei Erfolg:** Production default ändern (optional — kann user bleiben lassen)
4. **Task #145:** closed — VTQ_MIXED implementiert aber nicht deployed
5. **Task #127 (TQW2):** bleibt long-term research, separate von v6
