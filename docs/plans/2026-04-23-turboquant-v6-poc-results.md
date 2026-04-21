# TurboQuant v6 — Python PoC Results

**Datum:** 2026-04-23
**Scope:** Validierung der zwei v6-Ideen aus `2026-04-23-turboquant-v6-ideas.md` per Python-PoC
**Tools:** numpy + scipy, Lloyd-Max k-means für Gaussian centroids, Random orthogonal Π via QR

## TL;DR

| Idee | Validierung | Verdict |
|---|---|---|
| **1. QJL Residual (TQ_prod)** | RED | Attention-Aggregation **3-5× schlechter** als Stage-1-only bei gleicher bpw |
| **2. Outlier Channel Split** | GREEN | **5-24% besseres Cosine-Match** an Attention-Output bei +0.06-0.25 bpw |

**Konsequenz:** Implementiere Idee 2 als v6-Hauptfeature. Verwerfe Idee 1.

## Idee 1: QJL Residual — RED

### Theorem-2 Test (Per-Vector Inner-Product Distortion)
PASS für alle bit-widths — D_prod_s2 matcht Paper-prediction binnen 13%:
- b=2: 4.39e-3 vs 4.38e-3 (1.00x)
- b=3: 1.43e-3 vs 1.41e-3 (1.02x)
- b=4: 4.15e-4 vs 3.67e-4 (1.13x)

Stage-2 ist also korrekt implementiert und liefert die theoretisch vorhergesagte Distortion-Variance.

### Attention-Aggregation Test (das was zählt)

Bei fixed bpw-Budget (Stage-1 nutzt b Bits, Stage-2 nutzt b-1 + 1 QJL = b Bits):

| b | N | TQ_mse (Stage-1, current) | TQ_prod (Stage-2, paper) | Ratio |
|---|---|---|---|---|
| 2 | 4096 | 5.99e-2 | 1.95e-1 | 3.26x worse |
| 3 | 4096 | 1.80e-2 | 7.96e-2 | 4.43x worse |
| 4 | 4096 | 4.97e-3 | 2.56e-2 | 5.16x worse |

**Begründung:** Bei fixed bpw nutzt Stage-1 das volle Bit-Budget für MSE-Reduktion. Stage-2 opfert 1 Bit für QJL. Bei head_dim=128 und random orthogonal Π ist Stage-1 schon nahezu unbiased für Inner-Products von unit-sphere-Queries. Der Bias-Vorteil von Stage-2 wird durch den MSE-Verlust überkompensiert.

**Wann TQ_prod gewinnen würde:** Near-Neighbor Search über Millionen Vektoren (wo Bias-Akkumulation linear, MSE-Akkumulation √N skaliert). NICHT für KV-Cache-Attention über tausende Tokens.

**Reproduzierbar:** `tests/trellis-phase1/tqprod_poc.py` und `tqprod_attention_sim.py`.

## Idee 2: Outlier-Channel-Split — GREEN

### Setup
- Realistic V-cache: 8 oder 32 von 128 channels haben 3-8x höhere Variance (simuliert echte LLM-Aktivierungen)
- Per-channel scaling vor Quantisierung (matches real VTQ block-scale)
- Outlier-Identifikation via offline calibration: top-N channels nach post-rotation variance

### Ergebnisse

**8 outlier channels @ b+1 bits, 120 regular @ b bits** (eff bpw +0.06):
| b | Uniform | Mixed | Improvement |
|---|---|---|---|
| 2 | 6.43e-2 | 5.88e-2 | **8.5%** |
| 3 | 1.89e-2 | 1.74e-2 | **7.9%** |
| 4 | 4.99e-3 | 4.73e-3 | **5.2%** |

**32 outlier channels @ b+1 bits, 96 regular @ b bits** (eff bpw +0.25, matches Paper Table 1):
| b | Uniform | Mixed | Improvement |
|---|---|---|---|
| 2 | 6.01e-2 | 4.61e-2 | **23.4%** |
| 3 | 1.66e-2 | 1.29e-2 | **22.5%** |
| 4 | 4.57e-3 | 3.56e-3 | **22.0%** |

**Reproduzierbar:** `tests/trellis-phase1/outlier_channels_poc.py`.

### Skalierung auf echte VTQ_1 production Werte

Aktuelle VTQ2_1 (uniform 2-bit, 2.5 bpw mit scales):
- PPL: -1% vs f16 (besser, vermutlich regularization)
- TG: 70 tok/s

v6-Hypothese (32 outlier @ 3-bit + 96 regular @ 2-bit):
- bpw: ~2.75 (+0.25)
- PPL: vermutlich -2 bis -3% vs f16 (Verbesserung dort wo unsere VTQ_1 schon bestehen — outlier dominate die residual error)
- TG: ähnlich (Decode-Cost minimal höher durch dual-codebook, FA-loop muss outlier-mask checken)

### Risiken

- **Calibration-Komplexität:** muss pro Modell einmalig outlier-channels identifizieren (wie Trick-3 RHT-seed)
- **GQA-Interaction:** Qwen3.5-35B hat 16 KV-heads. Outlier-Mask muss per-KV-head laufen, nicht global. Storage: 16 · 128 / 8 = 256 bytes Metadata pro layer.
- **CUDA-Dispatch:** FA inner-loop muss bei jedem channel das Codebook auswählen. Bei sm_75 könnte das einen 5-10% TG-Hit verursachen.

## Empfehlung Phase C: Implementation Plan

### v6 = VTQ2_1_OUT (Outlier-Variant)

**Block-Layout** (`block_vtq2_1_out`, 32 sample/block):
```c
typedef struct {
    ggml_half d;                  // global block scale
    uint8_t   outlier_mask[16];   // 128-bit mask: which of d=128 channels are outliers
    uint8_t   qs_outlier[12];     // 32 samples × 3 bit (für outlier-channels) — varies per block
    uint8_t   qs_regular[8];      // 32 samples × 2 bit (für regular-channels)
} block_vtq2_1_out;
```

Tatsächlich komplexer da outlier-count pro block variiert. Alternative: **outlier-mask fest per layer** (nicht per block). Dann:

```c
// Per-layer global metadata:
uint8_t outlier_channels[32];     // indices of outlier channels (max 32)
uint8_t n_outliers;               // 0..32

// Per-block:
typedef struct {
    ggml_half d;
    uint8_t qs_outlier[12];       // up to 32 × 3 bit
    uint8_t qs_regular[24];       // up to 96 × 2 bit
} block_vtq2_1_out;
```

Storage: `2 + 12 + 24 = 38 bytes` für 128 samples = **2.375 bpw** (gegenüber VTQ2_1's 32 sample×2bit + 2 byte = 2.5 bpw).

Wait — das ist sogar **niedriger** als aktuelles VTQ2_1, weil ich den scale-overhead pro outlier-block genauer rechne. Brauche realistisches Layout-Design.

### Aufgaben (Phase C, 1-2 Wochen)

1. **Block-Layout design:** entscheide ob outlier-mask per-block, per-layer, oder per-model
2. **Calibration script:** `scripts/tq_calibrate_outliers.py` — extract per-channel variance auf 1k-token wikitext sample, output outlier-channel-list
3. **CPU reference impl:** `ggml-vtq-out.c` — encode + decode outlier-aware blocks
4. **CUDA dequant kernel:** branched lookup outlier-vs-regular codebook
5. **FA integration:** dispatch path auf VTQ2_1_OUT
6. **Roundtrip test + PPL sweep auf gpu00**

**Estimated:** 1.5-2 Wochen, +0.25 bpw, vermutlich -2% PPL bei Performance-Neutralität (5% TG-Slowdown maximal).

## Was als nächstes (sofort)

Schreib einen Implementation-Plan (`2026-04-23-vtq-out-implementation.md`) mit konkreten file paths + commit-strategy. Dann start Block-Layout design + outlier-Calibration script.

Pause für User-Decision: **Phase C starten oder zuerst v6-PoC auf echtes Qwen3.5-35B V-cache sample validieren?** Ich empfehle zweitens — das ist in 2-3h erledigt und gibt uns echte outlier-statistics statt synthetischen.
