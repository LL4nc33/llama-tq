# TurboQuant v6 — Ideen-Extraktion aus Paper + Google Blog

**Datum:** 2026-04-23
**Quellen:** arxiv 2504.19874 (full PDF), research.google TurboQuant blog
**Ziel:** Konkrete Optimierungsvektoren die wir **noch nicht** probiert haben.

## Kritische Erkenntnis: TurboQuant ≠ PolarQuant

Unser Fork basiert auf **PolarQuant** (MSE-optimiertes Stage 1). Das Paper beschreibt **TurboQuant = PolarQuant + QJL residual (Stage 2)**. Wir haben das 2-Stage-Design **komplett übersehen**.

| Method | Stages | bpw | Property |
|---|---|---|---|
| PolarQuant (was wir haben) | 1 (RHT + Lloyd-Max) | b | MSE-optimal, **biased** inner product |
| TurboQuant (Paper) | 2 (stage 1 mit b-1 bits + 1-bit QJL residual) | b | **Unbiased** inner product, ähnliche MSE |

**Attention ist inner-product** (Q·Kᵀ und Softmax·V). Biased quantizer → systematischer Fehler den auch mehr Bits nicht eliminieren. QJL residual kostet nur **1 bit/coord extra** und macht den Estimator unbiased.

## Idee 1: QJL Residual Stage hinzufügen (TQ_prod statt TQ_mse)

**Was:** Algorithm 2 aus dem Paper implementieren:
```
idx = Quant_mse(x, bits=b-1)      # wir haben schon
r   = x - DeQuant_mse(idx)         # residual
qjl = sign(S · r)                  # 1-bit QJL, S ist dxd Gaussian matrix
γ   = ||r||₂                       # scalar
output = (idx, qjl, γ)

# Dequant:
x̃_mse = DeQuant_mse(idx)
x̃_qjl = √(π/2)/d · γ · Sᵀ · qjl
x̃     = x̃_mse + x̃_qjl
```

**Kosten:**
- +1 bit/coord Storage (aber Stage 1 nutzt nur b-1 bits, Netto-Total = b bits)
- Extra `sign(S·r)` beim Quantize — 1 Gaussian-Matrix-Multiply
- Extra `γ · Sᵀ · qjl` beim Dequant — dasselbe
- S kann mit Philox fly-generated werden (wie RHT seed bei uns)

**Ertrag (laut Paper Theorem 2):**
- **Unbiased inner-product** (unsere aktuelle Bias ~2/π bei b=1, schrumpft bei höheren b)
- D_prod ≈ 0.56/d bei b=2, 0.18/d bei b=3, 0.047/d bei b=4 — vs unser aktuelles biased D_mse
- **Needle-in-Haystack:** TurboQuant 0.997 score = Full-Precision 0.997, vs PolarQuant nur 0.995

**Wie es bei uns aussieht:**
- `VTQ2_1` aktuell = Stage-1-only, 2-bit codebook, 2.5 bpw mit scales
- `VTQ2_1_prod` v6 = Stage-1 bei 1-bit (2 centroids) + QJL residual 1-bit = 2 bpw effektiv mit **unbiased Attention**
- `VTQ3_1_prod` = Stage-1 bei 2-bit (4 centroids) + QJL residual = 3 bpw unbiased

**Implementierungs-Risiko:** MITTEL
- S-Matrix: bei d=128 head-dim ist S 128x128 = 16k Gaussian samples pro head-call. Mit Philox fly-computed wie RHT ist das vielleicht ok.
- Dequant-Kosten: jetzt zwei MatVecs (Πᵀ·ỹ + Sᵀ·qjl) statt einer. FA inner-loop muss beide machen.

**Zeitaufwand Schätzung:** 3-5 Tage Implementation + 1 Tag Testing.

## Idee 2: Outlier-Channel Split (der GEHEIME 0.5 bpw-Trick)

**Was das Paper macht (Abschnitt 4.3, line 1095-1103):**
> "2.5-bit and 3.5-bit quantization during text generation. These non-integer bit precisions result from our strategy of splitting channels into outlier and non-outlier sets, and applying two independent instances of TurboQuant to each, allocating higher bit precision to outliers."

**Beispiel:** head_dim=128, bei 2.5 bpw:
- 32 outlier channels → 3 bits each
- 96 regular channels → 2 bits each
- Average: (32·3 + 96·2)/128 = 2.5 bpw

**Warum das genial ist:**
Nach RHT-Rotation *sollen* alle Channels gleich-verteilt sein. Aber in der Praxis haben manche Tokens/Heads outlier-Bursts die durch die Rotation nicht komplett diffused werden (GQA-Effekte, Attention-Sinks, frühe-Layer-patterns).

**Wie es bei uns aussieht:**
- Unsere Trick-2 (per-head precision mixing, task #121) ist ein **Head-level** Split
- Paper-Trick ist **Channel-level** Split — orthogonal und feiner-granular
- Beide kombinieren: Outlier-Heads auf 3.5 bpw, Regular-Heads auf 2.5 bpw
- Innerhalb eines Heads: Outlier-Channels auf 3 bit, Regular-Channels auf 2 bit

**Welche Channels sind outlier?** Paper sagt: **statistische Bestimmung** einmal per Model. Wir können das mit einem Calibration-Run machen (1000 tokens wikitext sample, per-channel-variance messen, top-N nehmen). Das ist kein Online-Verfahren mehr, aber das ist **pro Model einmal** — wie unser Trick-3 RHT-seed-calibration.

**Storage-Cost:** Minimal. 7-bit-mask für welche Channels outlier sind (128-dim → 16 bytes per head). Global per model.

**Ertrag:** +0.2-0.5 PPL bei gleicher bpw, oder -0.5 bpw bei gleicher PPL. In der LongBench-Tabelle holt Paper damit **50.06 @ 3.5 bpw = exakt Full-Precision**.

**Implementierungs-Risiko:** NIEDRIG-MITTEL
- Zwei VTQ-Block-Layouts die koexistieren im selben Head
- Dispatch-Logik wird komplexer (outlier-vs-regular-Channels interleavt)
- Calibration-Script (Python offline oder C++ llama-quantize)

**Zeitaufwand:** 2-4 Tage.

## Idee 3: Entropy-Coded Codebook Indices

**Paper line 619-632:**
> "TurboQuant's efficiency can be further increased by applying entropy encoding to the indices. Probability of each codeword p_ℓ = ∫ f_X(x) dx. Optimally coding reduces average bit-width to nearly the entropy."

**Für unsere 4-bit codebooks:** Entropie der distribution ≈ 3.8 bit (statt 4). **~5% Reduktion** kostenlos.

**Paper-Autoren haben es abgelehnt:** "limited gain, we have chosen not to incorporate this technique to maintain simplicity and speed."

**Unsere Situation:** Wir sind schon bpw-limited. 5% bei 4-bit V = 3.8 bpw. Bei 3-bit K = 2.85 bpw.

**Trade-off:** Variable-length codes brechen die klare block-struct Layout. Dispatch wird langsamer (huffman-decode pro sample statt direct lookup). **Für CUDA wahrscheinlich net-negative** wegen branching.

**Verdict:** Interessant aber **nicht priorisiert**. Könnte für CPU-only path relevant sein (z.B. Memory-Mapped KV dumps).

## Idee 4: Precomputed Rotation Matrix Π — schon gemacht

Paper nutzt generisches Π via QR-decomp einer Gaussian. Wir nutzen **FWHT-basiertes RHT mit fly-computed sign bits** (v5 precomputed sb[4]) — das ist **schneller** als Paper-Π für ausreichend hohe d.

Allerdings: Paper's Π ist ein beliebiges orthogonal. Unser FWHT+random-sign-bits ist ein **structured random orthogonal** — funktioniert ebenso gut in hohen Dimensionen (Lemma 1 hält).

**Kein neues Territorium.**

## Idee 5: GPU-Batched Codebook Lookup via Tensor Cores (Ampere+)

Das ist **kein Paper-Idee**, aber das Paper läuft auf **A100**. Unser VTQ_2 scheitert auf **Turing (sm_75)** wegen fehlender async-copy + tensor-core support für INT-lookups.

**Hypothesen für Ampere+:**
- `cp.async` pipelined V-dequant überdeckt mit Q·K compute
- Tensor-core INT4/INT8 instructions → batched codebook lookup als matrix-mul
- Shared memory bank conflicts bei Turing haben wir nicht vermieden (LUT layout nicht bank-conflict-free)

**Test-Plan:** Nur möglich wenn wir Ampere-Hardware-Zugang bekommen. Aktuell nur sm_75 verfügbar.

## Idee 6: Streaming Generation-Time Quantization

**Paper line 1092-1094:**
> "Unlike existing approaches such as KIVI and PolarQuant, which **leave generated tokens unquantized**, our method applies quantization **even during the streaming generation process**."

**Das ist wichtig!** Viele KV-quant Arbeiten quantisieren nur den prefill, behalten neu-generierte Tokens in FP16 in einem "recent window". Wir quantisieren **alles sofort** — was bei langer Generation ein Vorteil ist (konstante Memory) aber bei kurzer Generation Overhead bedeutet.

**Frage für uns:** Lohnt sich ein Hybrid-Modus "last-N tokens fp16, rest VTQ"? 
- Pro: Niedriger Decode-Latency für die letzten N tokens (Generation hat bereits diesen pattern, siehe KIVI)
- Con: Memory spike bei burst-generation, complexity in KV-cache ring buffer

**Unser Trick-1 (attention-sink fp16 for first 4 tokens)** ist ein ähnlicher Spirit — erste Tokens bleiben fp16. Jetzt wäre der spiegelbildliche Trick: **last N fp16**.

**Zeitaufwand:** 1-2 Tage (ring-buffer integration), **unclear ROI** — müsste gemessen werden.

## Idee 7: Calibrate Rotation Seed per Layer (nicht per Model)

Trick-3 hat **ein RHT-seed pro Model**. Paper suggeriert **per-call Π** (online). Wir haben **per-model** (offline calibration).

**Middle ground:** Per-layer Seed — jede der 40+ layers bekommt ihre eigene RHT-Permutation. Kostet ~160 bytes extra Metadata (seed pro layer·K/V). Paper-Theorem braucht nur dass Π **random** ist, nicht dass es **über calls** fix ist.

**Hypothese:** Layer-specific seeds reduzieren inter-layer interference wenn mehrere Layers ähnliche outlier-patterns haben.

**Test:** PPL-diff messbar? Wahrscheinlich marginal (<0.1%) aber leicht testbar.

**Zeitaufwand:** 0.5 Tage.

## Priorisierung (subjektiv nach ROI)

| Idee | Effort | Accuracy ROI | Performance ROI | Status |
|---|---|---|---|---|
| **1. QJL Residual (TQ_prod)** | HIGH (3-5d) | **HIGH** (unbiased attention) | 0 (Decode leicht teurer) | **Empfehlung #1** |
| **2. Outlier Channel Split** | MID (2-4d) | **HIGH** (-0.2-0.5 PPL or -0.5 bpw) | 0 | **Empfehlung #2** |
| 6. Last-N fp16 streaming window | LOW-MID | LOW | MID (decode latency) | Nice-to-have |
| 7. Per-layer RHT seed | LOW | LOW (<0.1%) | 0 | Quick-win optional |
| 3. Entropy coding | MID | NEGATIVE (CPU only) | NEG (CUDA dispatch) | **Ablehnen** |
| 4. Precomputed Π | — | — | — | **Schon gemacht** |
| 5. Tensor Core codebook | HIGH | 0 | **HIGH (Ampere)** | **Hardware-blocked** |

## Empfehlung Phase B

**Start mit Idee 1 (QJL Residual).** Gründe:
1. Paper-Beweis sagt explizit dass nur dieser Schritt Full-Precision-Accuracy erreicht (Needle-in-Haystack 0.997 = f16)
2. Stage-2 addiert nur 1 bit — passt in unsere bestehende vtq-block-struct (sign bit kann in gap gepackt werden)
3. Orthogonal zu allem was wir schon haben — ersetzt nichts, erweitert

**Als zweites Idee 2 (Outlier Channels).** Kombiniert mit Idee 1: v6 = TQ_prod mit Outlier-Split = Paper-reference implementation.

**Baseline der Messung:** VTQ_1 production path, 70 tok/s, -1% PPL vs f16.

**Akzeptanz-Kriterien v6:**
- PPL ≤ f16 baseline (unbiased attention sollte das liefern)
- TG ≥ 60 tok/s (20% slower akzeptabel wenn accuracy-neutral)
- bpw ≤ 3.5 (match Paper Table 1 config)

## Nächste konkrete Schritte

1. Python-PoC: Idee 1 QJL-Residual auf einer V-cache sample validieren, MSE + biased-vs-unbiased inner product messen. **1 Tag**.
2. Falls Python-Validation greift: CPU reference implementation in `ggml-trellis.c` (NEIN, neue file: `ggml-tqprod.c`). **1 Tag**.
3. Block-layout design für `VTQ2_1_PROD` + `VTQ3_1_PROD` — plus Storage der QJL-sign-bits + γ-scalar. **Design-Doc**. **0.5 Tag**.
4. CUDA dequant kernel. **1-2 Tage**.
5. FA integration. **1-2 Tage**.
6. Benchmark vs production VTQ_1. **0.5 Tag**.

Gesamt: **~1 Woche** bis measurable result.

## Offene Fragen

- **QJL matrix S storage:** fly-Philox oder precomputed? Paper nicht spezifiziert — wir haben Erfahrung mit Philox, sollte funktionieren.
- **γ = ||r||₂ Precision:** fp16 reicht? Das ist ein per-vector scalar, kosten marginal bei head_dim=128.
- **Interaction mit GQA:** unser Qwen3.5-35B-A3B hat GQA=8. Outlier-Channel-Calibration muss pro-KV-head laufen, nicht pro-Q-head.
- **Inverse S^T berechnen on-the-fly:** S^T hat dieselben Gaussians, nur transponiert. Philox-Generation-Pattern muss deterministisch reproducable sein.
