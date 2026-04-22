# Trick 2 PR1 — Per-Head Variance Profile Analysis

**Datum:** 2026-04-20
**Modell:** Qwen3-0.6B Q8_0 (nur Test, small model; Production mit 35B-A3B war GPU-besetzt)
**Config:** 50 decode samples, CPU path, f16 K/V cache, c=1024
**Branch:** `trick2-pr1-profile-heads` (Commit 71b56f7fb)
**Binary:** gpu00:~/llama-tq/build-cuda-trick2/bin/llama-cli
**JSON:** `2026-04-20-trick2-profile-qwen3-0.6b.json`

---

## Zusammenfassung

Profiling-Hook funktioniert. Per-Head Varianz und Kurtosis auf 28 Layern × 8 KV-Heads
erfolgreich gemessen über 50 Decode-Samples.

**Kernzahlen:**
- Median Varianz-Ratio (max/min pro Layer): **2.19×**
- Max Varianz-Ratio: **12.28×** (Layer 13)
- Layer mit Ratio > 4×: **6 von 28** (21%)
- Kurtosis-Bereich: **90-7421** (durchgängig hochgradig leptokurtic)

---

## Interpretation

### Varianz-Heterogenität rechtfertigt Per-Head Precision Mixing

Ein Go-No-Go Kriterium für Trick 2 war ein Varianz-Ratio ≥ 4× auf "vielen" Layern.
6/28 Layer erfüllen das auf kleinem Modell — bei einem größeren Modell (35B-A3B,
mehr Layer, mehr Heads) ist mit mehr Heterogenität zu rechnen.

**Konkrete Layer mit > 4× Ratio:**
- Layer 2: 5.13× (ein Head kurt=4488, extreme Ausreißer)
- Layer 11: 5.20×
- Layer 12: 9.02×
- Layer 13: 12.28× ← **Maximum**
- Layer 14: 6.45× (kurt bis 7421)
- Layer 25: 4.10×

Diese Layer würden in der Trick-2-Implementation statt VTQ3_2 (3.06 bpw) auf
VTQ4_2 (4.06 bpw) upgegradet werden. Overhead: 6/28 × (4.06-3.06) ≈ 0.21 bpw
zusätzlich im V-cache → **~7% mehr V-VRAM**, aber deutlich bessere Fidelity auf
den kritischen Heads.

### Kurtosis bestätigt Trellis-Wahl

Kurtosis(normal) = 3. Alle Heads zeigen Kurtosis 90-7421 — extreme schwere Tails.
Das validiert die Architektur-Entscheidung aus Phase 1 (Trellis + RHT), da
Codebook-basierte Quantizierer bei solchen Verteilungen systematisch underfitten.

---

## Nächste Schritte

1. **Größerer Run:** Profile auf Qwen3.5-35B-A3B sobald Production free (oder Deploy-Window)
   → mehr Layer, mehr Heads, repräsentativere Datenbasis. Expected 48 Layers × 4 KV-Heads.
2. **Längerer Run:** 500 decode samples statt 50 → stabilere Kurtosis-Schätzung
   (M4/Welford convergence braucht >> 50 Samples)
3. **Per-Token Residual-Varianz:** statt nur V-cache-State auch das Dequant-Residual
   messen — ist für Trick 4 (Correction Overlay) relevant
4. **Trick 2 PR2 Design:** Layer-Selection-Heuristik basierend auf Top-N Varianz-Heads
   → `--tq-upgrade-layers N` CLI flag für Mixed-Precision V-Cache

---

## Command used

```bash
# CPU-only run because gpu00:8791 Production had both GPUs saturated
CUDA_VISIBLE_DEVICES=0 timeout 180 build-cuda-trick2/bin/llama-cli \
  -m qwen3-0.6b-q8_0.gguf \
  -ngl 0 --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 \
  --tq-profile-heads 50 \
  -p 'Erkläre Quantenmechanik.' -n 50 \
  --no-mmap -c 1024
```

Output: `tq-profile-heads-<PID>.json` (14 KB for this run)
