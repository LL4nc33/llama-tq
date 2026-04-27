# KTQ_3 Outlier-Channel-Split — Research + Design

Stand: 2026-04-25. Phase 7 Kandidat der post-Gemma4 Roadmap.

VTQ_3 hat heute -28-32% rel-MSE bei +0.75 bpw geliefert. Frage: lohnt sich das Pattern auch für K-cache? Diese Doc analysiert KTQ-Architektur, bewertet Outlier-Split, und vergleicht mit drei Alternativen.

---

## 1. K-cache Architektur-Status

### 1.1 Block Layouts (`ggml-common.h`)

KTQ hat aktuell **eine** Generation (`_1`), Block-Size 32:

```c
#define QK_KTQ 32
typedef struct {
    ggml_half d;              // L2 norm (v5 norm correction)
    uint8_t   qs[QK_KTQ * b/8]; // b-bit codebook indices, b ∈ {1,2,3,4}
    uint8_t   sb[QK_KTQ / 8];   // RHT sign bits (precomputed, no Philox at dequant)
} block_ktq{1,2,3,4}_1;
```

bpw = b + 1.5 (KTQ1_1=2.5, KTQ2_1=3.5, KTQ3_1=4.5, KTQ4_1=5.5).

### 1.2 Score-Computation (Hadamard-Domain, `fattn-vec.cuh:104-180`)

Der entscheidende Unterschied zu V: **K wird NICHT dequantisiert für Q·K**.

```cuda
constexpr bool Q_tq = type_K == GGML_TYPE_KTQ1_1 || ... || GGML_TYPE_KTQ4_1;
// TQ v7: Q as f32 in registers for Hadamard-domain dot product.
// Thread t holds Q_f32[j][bi] = Q value at position bi*32 + t (scaled).
float Q_f32[ncols][D/WARP_SIZE];
```

Q wird einmal RHT-rotiert (Hadamard-Domain), dann `vec_dot_KQ_t` macht das Score in Hadamard-Domain mit:
1. `qs[]` → centroid index → centroid value (`PQ_CODEBOOK_*BIT[c]`)
2. Sign-flip via `sb[]`
3. Multiplikation mit `d`
4. Akkumulation gegen `Q_f32[t]` (selbe Hadamard-Basis)

**Konsequenz:** der Score-Path sieht nie das volle dequantisierte K. Outlier-Werte in der Original-K-Domäne werden nach RHT auf alle Channels verschmiert (Hadamard ist orthogonal/dense). Es gibt keine "Channel-Outliers" mehr im KTQ-Block — alle 32 samples sehen lookup+sign+scale.

### 1.3 MMA-KTQ Pfad (Tasks #154-157 done, #158 deploy pending)

`fattn-mma-ktq.cu` instantiiert tensor-core Q·K für KTQ via expandiertes Q (`Q_tq` path) — 2× speedup auf K-side. Auch hier: **kein K-Dequant**, nur Hadamard-Q × indexed centroids.

**Warum keine "v2 Trellis" Version für K wie bei V?**
- Trellis ist sequenziell (Viterbi state-decode, byte-by-byte).
- Score-Sweep ist warp-parallel (32 lanes × 32 samples = 1 block tick).
- Sequenzielles Trellis-Decode passt schwer in den parallelen Score-Tick — würde 4× Latenz draufschlagen ODER massive shared-memory Spills für decoded-K-buffer brauchen.
- VTQ_2 (Trellis) funktioniert für V weil das Mat-Vec im **Combine-Step** danach läuft, nicht im Score-Tick.

---

## 2. KTQ_3 Outlier-Split Hypothese

### 2.1 Block-Layout-Vorschlag

Analog VTQ_3 mit K_OUT=4 fp16 outliers:

```c
#define KTQ_OUTLIER_K 4
typedef struct {
    ggml_half d;
    uint8_t   qs[QK_KTQ * b / 8];
    uint8_t   sb[QK_KTQ / 8];
    uint8_t   outlier_pos[KTQ_OUTLIER_K];   // 4 B
    ggml_half outlier_val[KTQ_OUTLIER_K];   // 8 B
} block_ktq{1,2,3,4}_3;
```

### 2.2 bpw-Kosten

KTQ block-size = 32 (vs. VTQ Trellis = 128). Outlier-overhead skaliert linear mit Header, aber verteilt sich auf 4× weniger samples:

| Type    | Base bpw | +Outlier (4× pos+val = 12 B) | Total |
|---------|---------:|-----------------------------:|------:|
| KTQ1_3  | 2.5      | +3.0                         | 5.5   |
| KTQ2_3  | 3.5      | +3.0                         | 6.5   |
| KTQ3_3  | 4.5      | +3.0                         | 7.5   |
| KTQ4_3  | 5.5      | +3.0                         | 8.5   |

**+3 bpw overhead ist viel.** KTQ2_3 (6.5 bpw) ist nahe f16 (16 bpw) und teurer als ktq4_1 (5.5 bpw). Outlier-Split bei QK_KTQ=32 ist wirtschaftlich nicht überzeugend.

**Mitigation 1: K_OUT=2** → +6 B overhead = +1.5 bpw. KTQ2_3 = 5.0 bpw. Knapp unter ktq4_1. Aber nur 2 outliers pro 32-sample-block ist wenig protection.

**Mitigation 2: QK_KTQ=64** → +12 B / 64 = +1.5 bpw. Erfordert aber Verdopplung des Score-Block-Size, was warp-mapping (32 lanes ↔ 32 samples) bricht. Major refactor.

### 2.3 Score-Path Konsumierbarkeit (zentrale Frage)

Outliers sind in der **rohen K-Domäne**, nicht in Hadamard-K. Der Score läuft in Hadamard-Domain. Drei Optionen:

**Option A — Outlier RHT-rotiert speichern.** Encoder rotiert outliers durch das gleiche RHT. Dann sind sie in der Hadamard-Basis konsistent. Aber: 4 outlier-Channels in raw K werden nach RHT auf alle 32 Hadamard-Channels verschmiert → outliers sind nicht mehr Sparse. Konzept bricht.

**Option B — Outlier-Overlay nach Score (Post-Score-Correction).** Encoder speichert `(pos_i, raw_val_i)` und einen Offset-Vector `delta = raw_val_i · row(RHT, pos_i)`. Decoder addiert `Q_f32 · delta` zum Score nach dem main-Sweep. Korrekt rechenbar — aber **+4 dot-products pro Block** (4 dense vectors of length 32 against Q). Cost: ~4× compute des regulären Block-Sweep × 0.125 = +50% K-Score-Latenz. Ist auf D=128 schon dominanter Pfad. NO-GO.

**Option C — Outlier nur für non-MMA, MMA-KTQ ignoriert sie.** KTQ_3 wirkt nur auf älteren GPUs (<sm_80, kein MMA). On Pascal/Turing schon. Aber: MMA-KTQ ist der Production-Pfad auf den Zielen RTX 2060+. KTQ_3 hätte auf gpu00 (Turing) tatsächlich Effekt, auf modernen RTX nicht.

### 2.4 Erwartete Quality-Gewinne

K-cache-PPL-Headroom ist klein. README PPL-Tabelle:
- ktq2_1/f16 vs f16/f16: **+0.15% PPL**
- vtq2_2/f16 vs f16/f16: +0.02% PPL (V hat noch weniger Headroom — aber VTQ_3 zielt auf Distribution-Tail-MSE, nicht PPL direkt)

K hat zwar Outlier-Channels (Attention-Sink-artig), aber:
1. RHT verteilt sie schon Hadamard-weit → kein lokaler outlier mehr
2. PPL-Gewinn-Obergrenze realistisch -0.05 bis -0.1% PPL bei 2× bpw cost

**Hypothese:** KTQ_3 ist dominiert von ktq{b+1}_1 in jedem Pareto-Punkt. KTQ2_3 (6.5 bpw) > KTQ4_1 (5.5 bpw) bei besserer PPL und niedrigerer bpw — gewinnen sehr unwahrscheinlich.

**Verdict:** KTQ_3 Outlier-Split ist **kein Pareto-Move**. Skip.

---

## 3. Algorithmische Alternativen für K-cache

### A) Vector Quantization (VQ) statt 1D Lloyd-Max

Aktuell: 1D Codebook (Laplace/Gauss), pro sample independent quantisiert. VQ über 2-4 Channels gleichzeitig würde 2D-Korrelation nutzen.

- **Aufwand:** hoch (3-5 Tage). Codebook-Design (k-means in 4D), Encoder-Rewrite, Decoder-Lookup-Tabellen.
- **Gain:** post-RHT ist die K-Distribution approx-Gauss-isotropic → Channels approx-unabhängig → VQ-Gain klein (PolarQuant Paper bestätigt: post-RHT VQ-Gain <0.1% PPL).
- **Verdict:** wenig Hebel, hohe Komplexität.

### B) Trellis-K mit Q·K-Reverse-Path

Volles K-Dequant + standard Q·K. Verliert MMA-Speed, gewinnt Trellis-Quality.

- **Aufwand:** 5-7 Tage (FA-vec rewrite, MMA-KTQ Pfad invalidiert).
- **Gain:** hypothetisch +0.5% PPL bei 2.0 bpw K. ABER Speed-Hit ~30% TG (Tasks #154-158 wären zurückgedreht).
- **Verdict:** Quality-Win, aber Speed-Regression killt den Pareto-Move. Skip.

### C) Per-channel scale/offset im KTQ block header

Aktuell: nur block-scale `d`. Kein offset, keine per-channel-mean.

- **Aufwand:** +6 B per block (4-channel mean fp16) = +1.5 bpw bei QK_KTQ=32.
- **Gain:** RHT zentriert die Marginale schon — Block-Mean ist nicht 0, aber klein. Erwarteter PPL-Gewinn <0.05%.
- **Verdict:** marginaler Gewinn bei nicht-trivialer Cost. Skip.

### D) Importance Matrix (imatrix) — Calibration-aware Lloyd-Max

Mainline llama.cpp hat **imatrix tooling** (`tools/imatrix/imatrix.cpp`, ~600 LOC). Aktiviert via `llama-imatrix -m model.gguf -f calib.txt`. Output ist `imatrix.gguf` mit per-tensor Activation-Statistiken.

Aktuelle TQ-Quants ignorieren imatrix:

```c
// ggml-quants.c:2353
// quant_weights (imatrix) is currently unused — the RHT + Lloyd-Max codebook
```

Lloyd-Max centroids sind hardcoded:
- KTQ1/2_1: Laplace-fit (`PQ_CODEBOOK_*BIT`)
- KTQ3/4_1: Gauss-fit
- VTQ Codebooks: Laplace-Optimized (Section 5955+)

**imatrix-aware Calibration** würde:
1. Calibration-pass (imatrix collect) → per-tensor variance stats
2. Per-tensor Lloyd-Max re-fit auf model-specific K-distribution
3. Pack centroid-table in GGUF (per-tensor 8 floats = 32 B/tensor)
4. CUDA loader liest table aus tensor-meta, ersetzt PQ_CODEBOOK lookup mit indexed lookup

- **Aufwand:** 1-2 Tage Python (imatrix-collect + Lloyd-Max-fitter) + 1 Tag CUDA (table-loader, lookup-Pfad).
- **Gain:** estimated +0.05 bis +0.2% PPL — codebook fits actual model statistics statt theoretischer Distribution. Größter Gewinn auf nicht-RHT-canonical models (e.g., V vor RMS-norm wie Gemma4).
- **Synergy:** löst auch das Phase 4 Calibration-Problem (Gemma4 V vor RMS-norm). Eine Codepath für beide.
- **Risiko:** moderate. Bestehende Models brauchen re-Calibration; default-fallback auf Hardcoded-Codebook für non-calibrated tensors trivial.
- **Verdict:** beste Aufwand/Gain Ratio.

---

## 4. Empfehlung & Roadmap-Update

### Ranking

| Option | Aufwand (Tage) | Pareto-Move | Risiko | Score |
|--------|---------------:|------------:|-------:|------:|
| KTQ_3 Outlier-Split | 4-5 | **negative** (dominiert von ktq{b+1}_1) | mid | ❌ |
| VQ statt 1D | 3-5 | klein (post-RHT isotropic) | mid | ⚠️ |
| Trellis-K | 5-7 | quality+ aber speed-30% | high | ❌ |
| **imatrix-Calibration** | **2-3** | **+0.05 bis +0.2% PPL** + Phase 4 Synergie | **low-mid** | ✅ |

### Empfehlung: Phase 7 = imatrix-aware KTQ+VTQ Calibration

Begründung:
- **Niedrigster Aufwand** (2-3 Tage vs 4-7 für die anderen)
- **Echter Pareto-Move** (PPL-Gain ohne bpw-cost ohne speed-cost)
- **Synergie mit Phase 4** (Gemma4 V-pre-RMS-norm calibration ist Spezialfall)
- **Ports zu mainline-imatrix** (gut wartbares Tooling, kein neues Format)
- **Default-fallback trivial** — alte Models brechen nicht
- KTQ_3 Outlier-Split ist konzeptionell elegant, aber +3 bpw bei QK_KTQ=32 ist zu teuer → in nächst-höherer KTQ-Stufe immer dominiert.

### Folge-Phasen

- Phase 4 (Gemma4 V-RMS-norm calibration) wird Subset von Phase 7 (statt eigener Phase).
- Phase 8 könnte VQ-K (Option A) reaktivieren falls imatrix nicht reicht — aber wahrscheinlich skip.

---

## 5. Konkrete Sub-Tasks für Phase 7

1. **imatrix-extension für TQ:** `tools/imatrix/imatrix.cpp` extension um K/V activation statistics zu sammeln (variance + percentile-9 für outlier-detection).
2. **Lloyd-Max-fitter:** Python `tools/calibrate_tq_codebook.py` — input imatrix.gguf, output per-tensor codebook overrides als GGUF-Metadata.
3. **CUDA loader:** `convert.cu` + `fattn-vec.cuh` — wenn tensor hat custom codebook in metadata, lade in `__constant__` table und replace lookup.
4. **Default fallback:** ohne metadata fallback auf Hardcoded `PQ_CODEBOOK_*BIT` — keine Regression.
5. **Bench:** Qwen3.6 + Gemma4, PPL + TG vs uncalibrated. Gate: -0.1% PPL bei <1% TG cost.

Total: **2-3 Tage** PR.

---

## Anhang — Rejected Block Layout

KTQ_3 für Vollständigkeit dokumentiert (NICHT implementieren):

```c
typedef struct {
    ggml_half d;
    uint8_t   qs[QK_KTQ * b / 8];
    uint8_t   sb[QK_KTQ / 8];
    uint8_t   outlier_pos[4];  // raw-K channels
    ggml_half outlier_val[4];  // raw-K values
} block_ktq{b}_3;
// bpw cost: +3.0 — KTQ2_3 (6.5) > KTQ4_1 (5.5) at worse Pareto.
```
