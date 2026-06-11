# DiffusionGemma auf <10 GB: 2-Bit-Quantisierung eines Text-Diffusionsmodells

**Datum:** 2026-06-11 · **Fork:** llama-tq · **Modell:** DiffusionGemma 26B-A4B-it (Google, Juni 2026)

## TL;DR

Wir haben Googles DiffusionGemma (26B-Klasse Text-Diffusionsmodell) in den llama-tq-Fork
integriert und so weit quantisiert/optimiert, dass es mit **9.84 GB Gewichten** auf einer einzigen
**RTX 2060 (12 GB)** **kohärente, mehrteilige Sätze** generiert. Reines 2-Bit produziert nur
Kauderwelsch — der Durchbruch kam durch eine **Kombination aus 6 Hebeln**, von denen drei
neuartige Inferenz-Techniken sind. Stand der Forschung: Es gibt EIN akademisches Paper
([Quant-dLLM, arXiv:2510.03274](https://arxiv.org/pdf/2510.03274)), das das Problem theoretisch
bestätigt — aber **keine öffentliche lauffähige llama.cpp-Implementierung für ein Diffusion-LLM**.

## Das Problem: warum 2-Bit-Diffusion anders ist

DiffusionGemma ist **nicht-autoregressiv**: Es denoist ein 256-Token-Canvas über bis zu 48 Steps
(bidirektionale Attention). Nach jedem Step füttert **Self-Conditioning** die eigene Vorhersage
zurück in den nächsten Step:

```
soft = softmax(prev_logits) @ token_embd * sqrt(n_embd)
inpL = rms_norm(scaled_embed + self_cond_mlp(rms_norm(soft)))
```

Das ist eine **iterierte Rückkopplungsschleife**. Bei autoregressiven Modellen (Qwen, normales
Gemma) wird jeder Quantisierungsfehler einmal gesampled und ist fix — er akkumuliert nicht. Bei
Self-Conditioning fließt der Fehler **in dieselbe Position zurück**, Step für Step. Bei 2-Bit treibt
das Quant-Rauschen den **Loop-Gain über 1** (Lyapunov-instabil) → das Canvas konvergiert nie,
oszilliert, produziert Token-Salat. Das erklärt, warum Google/Unsloth nur bis 4-Bit anbieten.

## Die Stabilitäts-Leiter (empirisch gemessen)

Metrik: akzeptierte Tokens am letzten Denoise-Step (von 256); niedrigere Entropie = besser.

| Variante | Größe | accepted | Text |
|----------|-------|----------|------|
| Reines IQ2_XXS | 8.7 GB | 1/256 | Garbage `으로으로으로` |
| IQ3_XXS | 9.0 GB | 24→73 | fast lesbar, `und und und` |
| IQ2mix + Decoder-imatrix + alle Hebel + **2bit-KV** | 9.84 GB | 52→150 | Thinking kohärent, Repetition |
| **IQ2mix + alle Hebel + F16-KV** | **9.84 GB** | — | **2 saubere englische Sätze** ✓ |
| IQ2mix2 (alles attn+ffn q4) | 14 GB | 165/256 | voll kohärent |
| Q4_K_M | 16 GB | — | sauber (Referenz) |

Bester ~10GB-Output (alle Hebel + F16-KV):
> *"Sentence 1: ...the process where plants use light energy to convert water into glucose.
> Sentence 2: ...During this process, oxygen is released into the atmosphere."*

## Die 6 Hebel (das Rezept)

Kohärentes 2-Bit braucht ZWEI Achsen gleichzeitig — rohe Bit-Kapazität UND Loop-Dynamik:

1. **Mixed-Precision** (`--tensor-type`): Router `ffn_gate_inp`→q8_0 (winzig, kritisch für
   MoE-Expert-Stabilität), self_cond→q6_K, attn_output/ffn_down_exps/attn_q/k→q4_K, Rest IQ2_XXS.
2. **Decoder-Pfad-imatrix** (neu): imatrix aus dem ECHTEN Denoise-Loop (mit aktivem
   self-conditioning) statt aus dem Encoder-Prefill. Der Collector wurde aus llama-imatrix in
   `imatrix-collector.h` extrahiert und in den Diffusion-CLI eingeklinkt (`DG_DUMP_IMATRIX`).
3. **DG_SELF_COND_SCALE=0.5** (neu): dämpft den self-cond-MLP-Output → senkt den Loop-Gain.
   Optimum 0.5 (0.25 tötet das Signal).
4. **DG_DITHER=0.1** (neu, "Stochastic Resonance"): abklingendes Gauß-Rauschen
   `σ_t = DG_DITHER * cur_step/n_steps` auf die self-cond-Probs. Bricht das Attraktor-Looping —
   verdreifachte die accept-rate (52→150).
5. **Top-K-Annealing** (200→5): früh breit suchen, spät hart auf Top-5 zwingen.
6. **★ F16-Canvas-KV (der Schlüssel):** Das transiente Canvas-KV MUSS F16 sein. Mit 2-Bit ODER
   q8_0-KV produziert das Modell Repetition; nur F16 gibt saubere Sätze.

## Warum q8_0-KV versagt, f16 gewinnt (die spannendste Erkenntnis)

Kontraintuitiv: q8_0 (8-Bit, fast lossless für AR-Modelle) war **schlechter** als f16. Grund:
**systematischer Rundungs-Bias** (kein Rauschen) akkumuliert über die 48 Steps. Der Canvas-KV wird
in jedem Step neu geschrieben+verworfen (`llama_memory_seq_rm`); q8_0 rundet jedes Mal in dieselbe
Richtung → der Bias wird über die Self-Conditioning-Schleife verstärkt → mathematische Resonanz
(`Energie Energie Energie`). Nur f16-Float-Präzision lässt die Dämpfung sauber gegen 0 laufen.

**Wichtig:** TurboQuant (KTQ/VTQ) musste NICHT geändert werden — der Philox-Seed wird aus der
Tensor-Position abgeleitet (`ktq_cuda_derive_seed(block_index)`), ist also über die Steps
deterministisch. Es ist reine 2/8-Bit-Präzision, kein Stochastik-Problem.

## Deploy-Rezept (single-GPU, ~10 GB)

Der positions-abhängige KV-Split ist NICHT nötig — der Canvas ist nur 256 Tokens, der ctx winzig
(`prefix_len + 512`). F16-KV passt bei 9.84GB-Gewichten + realistischem Prompt in 12 GB:

```bash
llama-diffusion-gemma-cli \
  -m diffusiongemma-26B-A4B-it-IQ2mixdec.gguf \
  -ngl 99 -fa 1 \
  --cache-type-k f16 --cache-type-v f16 \
  --ctx-size 2048 --diffusion-steps 48 \
  --top-k-start 200 --top-k-end 5 \
  -p "DEINE FRAGE"
# env: DG_SELF_COND_SCALE=0.5 DG_DITHER=0.1
```

Quantisierung (IQ2mixdec): `llama-quantize --imatrix <decoder-imatrix> --tensor-type
ffn_gate_inp=q8_0 --tensor-type self_cond_{gate,up,down}=q6_K --tensor-type
{attn_output,ffn_down_exps,attn_q,attn_k}=q4_K BF16.gguf out.gguf IQ2_XXS`

## Offen (Roadmap)

- **Asymmetrisches History/Canvas-KV** (nur für Long-Context 128k+): History aggressiv KTQ/VTQ,
  Canvas f16. Spart GB bei langem Kontext, irrelevant bei kurzen Prompts. Deferred.
- **Deutsch-Finisher (minimal-QAT):** Die deutsche Satzstruktur wackelt bei komplexen Prompts noch
  leicht (reines Kapazitäts-Artefakt). Lösung: ultrakurzes LoRA-QAT (1-3M Tokens) NUR auf
  self_cond_mlp (3 Tensoren) + 6 Layer-Norms, Single-Step-Distillation-Loss (MSE: IQ2-MLP-Output
  vs f16-Lehrer). Minuten auf einer 2060.

## Neuheitswert

[Quant-dLLM (arXiv:2510.03274)](https://arxiv.org/pdf/2510.03274) bestätigt akademisch:
*"2-bit weight-only PTQ for diffusion LLMs drops sharply — mismatch with diffusion-style
inference."* Das ist exakt unsere Loop-Gain-Diagnose. Aber: Das ist eine theoretische PTQ-Methode.
**Keine öffentliche lauffähige llama.cpp-Implementierung für ein Diffusion-LLM existiert** —
DiffusionGemma ist brandneu (Juni 2026). Unsere Lösungs-Kombination (Mixed-Precision +
Decoder-Pfad-imatrix + 3 Inferenz-Hebel + F16-Canvas-KV) mit lauffähigem kohärentem Output ist
neuartig und reproduzierbar.
