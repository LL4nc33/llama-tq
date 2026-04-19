# TurboQuant Nano — Research Backlog (Trick 17 Edition)

Genau **siebzehn** experimentelle Tricks über der reinen Trellis-Quantisierung.
Hard-limit: **17 ist das Maximum**. Keine Trick 18. Nie.

> **Trick 17** — deutsches Idiom
> (<https://de.wikipedia.org/wiki/Trick_17>) für die eine geniale,
> scheinbar banale Lösung die das eigentliche Problem wegzaubert.
> Alle anderen Tricks sind Vorarbeit. Trick 17 am Ende ist reserviert
> für den großen Wurf — wenn wir dort ankommen, ist das Paper geschrieben.

Nummerierung ist nicht Reihenfolge der Wichtigkeit — ROI/Risk ist
hoch für Trick 1, "the big one" ist Trick 17. Dazwischen: enabling
infrastructure + incremental improvements. **Neue Ideen werden bestehende
Tricks erweitern oder ersetzen, nicht hinzugefügt** — siebzehn bleiben
siebzehn.

## Already in tree

**Layer-boundary protection** (`tq_protect_layers` in `llama_kv_cache`).
First N + last N layers keep `q8_0` instead of TQ. CLI exposes it,
default off. Spatial protection for critical layers.

## Trick 1 — Attention-sink token protection (highest ROI, trivial)

Keep the first 4 tokens of V-cache at `f16`, rest at `vtq_2`. Attention
sinks (StreamingLLM paper) carry outsized quality weight — 4 × 16 B
per head is peanuts but protects the hot-path tokens.

**Cost:** ~2 KB per layer.
**Expected gain:** recovers ~30-50 % of the vtq2_2 PPL delta on long
context (untested on our stack).

**Implementation sketch:** in `llama-kv-cache.cpp` write path, branch
on `token_idx < N_SINKS`. Needs a parallel f16 side-buffer and a
hybrid read path in the FA V-dequant. Couple dozen LOC.

## Trick 2 — Per-head precision mixing

Classify heads by V-variance on the first forward pass. High-variance
heads → `vtq4_2`, low-variance → `vtq2_2`. Keep average bpw ≈ 3.

**Cost:** one profiling pass + mixed-type layout support.
**Expected gain:** substantial, since variance is heavy-tailed.
**Blockers:** current `kv_cache` allocates one type per layer, not
per head. Needs new abstraction or per-head stream.

## Trick 3 — Per-model RHT seed calibration (free quality)

RHT uses a fixed random seed. Different seeds give different post-RHT
marginals; pick the one that minimizes kurtosis (closest to Gaussian,
where 1-D codebooks are optimal).

**Cost:** ~minutes per model, one-shot.
**Expected gain:** 0.1-0.5 % PPL improvement at zero runtime cost.
**Storage:** one `uint32_t` metadata tensor in GGUF.

## Trick 4 — Correction overlay buffer

Keep `correction_buf[N]` fp16 per layer storing the top-N worst quant
errors + their positions. Dequant checks the buffer first (cheap hash
lookup), else falls back to trellis decode.

**Cost:** ~512 B per layer × 48 layers ≈ 24 KB total.
**Expected gain:** if the error distribution is heavy-tailed (it is),
a handful of corrections can close the gap to near-lossless even at
`vtq2_2`. Lossless overlay on top of lossy base.

## Trick 5 — Learned λ sharpening (paper-worthy if it works)

Per-head `lambda_v[n_heads]` fp16 multiplier in V-dequant. Initial
λ = 1 is identity. Fine-tune λ on ~1000 samples, everything else
frozen — 1 hour on 0.8B.

**Cost:** one-shot fine-tune step.
**Expected gain:** quantization that becomes lossless (or better)
after minimal recalibration. "Quantization as regularizer" literature
supports this direction (Chen et al. 2023).

**Deliverable:** λ as GGUF metadata tensor, fully backwards-compatible
(missing λ → treat as 1).

## Trick 6 — CUDA encoder receiver-side Viterbi

Invert the DP: instead of scattering `prev → next` via atomicMin,
gather `next ← 2^K candidate prev` states via warp shuffle. No
atomics, coalesced writes, 5-10× encoder speedup on Turing.

**Cost:** ~150 LOC rewrite of inner DP loop.
**Expected gain:** 5-10× encoder throughput.

## Trick 7 — Shared trellis LUT across FA instances

The 256 KiB inv-Normal-CDF LUT is replicated per-TU once templates
are instantiated. Move to `extern __device__` in trellis.cuh,
define once in trellis.cu. Saves binary size and L2 pressure.

**Cost:** 10 LOC.
**Expected gain:** faster cold-cache FA, smaller binary.

## Trick 8 — Position-aware V protection

Beyond the first-4 sink tokens, the last ~32 tokens of the context
also carry outsized weight (recency bias). Protect both ends at
`q8_0` or `fp16`, middle at VTQ.

**Cost:** same side-buffer as Trick 1 extended at the tail.
**Expected gain:** incremental over Trick 1; mostly helps at short
contexts where tail is still in decode-hot zone.

## Trick 9 — RHT-free decode via sign-bit cache

Already done for K-side (v5 stores precomputed signs in sb[]).
Extend to V-side: store per-block signs in the VTQ_2 block header
(one extra byte per block) → dequant skips Philox evaluation.

**Cost:** block layout bump (ABI break) or opt-in v2.1 variant.
**Expected gain:** ~10-15 % faster FA V-dequant.

## Trick 10 — Asymmetric block sizes per layer

Early layers tolerate smaller blocks (higher overhead, better
fidelity); deep layers tolerate larger blocks. Experiment with
QK_VTQ2 ∈ {256, 512, 1024} per-layer auto-tuned.

**Cost:** block size per-layer metadata in GGUF + dispatch fan-out.
**Expected gain:** unknown, needs ablation.

## Trick 11 — KV-cache query-dependent re-quantization

When a query hits a KV slot with high attention weight, mark that
slot "hot" and keep a higher-precision shadow. Over time, the
working set becomes precision-balanced vs attention distribution.

**Cost:** streaming counter + shadow buffer + eviction policy.
**Expected gain:** speculative — could either shine or flop on
non-stationary workloads.

## Trick 12 — Entropy-coded V (if blocks compress further)

After RHT + trellis, the codebook index distribution may still have
residual entropy. Add a per-block rANS coder on top. Breaks random
access unless block-local.

**Cost:** significant — rANS encode in CUDA is non-trivial.
**Expected gain:** 0.2-0.4 bpw if entropy is real.

## Trick 13 — Prefix-shared V-cache across sequences

Many requests share system prompts. Deduplicate the VTQ blocks via
content hash → one physical buffer backs multiple logical KV slots.
Saves VRAM linearly with duplicate prefix length.

**Cost:** hash-table + refcount in KV manager.
**Expected gain:** 2-5× effective cache capacity in multi-user.

## Trick 14 — K-V cross-dependency aware quantization

Currently K and V are quantized independently. Joint training:
when K quantization introduces error ε_K, compensate V
quantization by shifting V codebook by −E[ε_K · ∂softmax/∂K].
Turns two independent lossy channels into a correlated denoiser.

**Cost:** one-shot calibration pass; extra metadata per layer.
**Expected gain:** unknown — could recover 20-40 % of combined
quantization error if covariance is strong.

## Trick 15 — Speculative decode with VTQ-quantized draft

Use the same VTQ2_2 model as its own speculative draft. Bulk
rejection sampling → ~1.5× decode if draft accept rate > 0.6.
Quantization noise is structured (not iid), so rejection is
coherent → expect high accept rate.

**Cost:** speculative-decode loop + tree attention.
**Expected gain:** 1.5-2× decode tok/s, orthogonal to bpw.

## Trick 16 — VRAM-budget-driven auto-bpw

Given a VRAM target, solve the layer-wise bpw allocation that
minimizes Σ PPL_delta(layer, bpw_layer) subject to
Σ bpw_layer · params_layer ≤ budget. Convex relaxation + greedy
refinement. Fully automatic.

**Cost:** profiler sidecar + solver (Python) + runtime config.
**Expected gain:** 15-25 % better PPL at same VRAM budget,
compared to uniform bpw.

## Trick 17 — Generative refinement pass (the big one)

The paper-worthy endgame: after VTQ quantization, run a tiny
corrective network (<1 % of model params) over the quantized
V-cache activations at inference time. The corrector is trained
to predict the quantization residual from the decoded V plus
local context. Net effect: V becomes **better** than f16 (the
corrector learns model-specific rotation + denoising jointly).

This is "quantization as regularizer + learned residual",
combining the best of Tricks 5, 14, and mixing it with
test-time compute. At 1 % params overhead, if it closes 90 % of
the VTQ2_2 → f16 gap and adds 0.5-1 % PPL *improvement* over
f16 (the regularizer effect), this single trick beats everything
else combined and reframes the whole project.

**Cost:** training infra (2-4 GPU-days on 0.8B), corrector
inference kernel (simple MLP), GGUF metadata tensor for weights.
**Expected gain:** lossless-or-better at 2.125 bpw. Paper title:
*"Compressed is Better: Quantization as Implicit Regularizer
Recovery with Lightweight Residual Correctors."*

**Status:** aspirational. Trick 17. Der gute alte Trick 17.

## Sequencing

Short-term, measurable:
1. Finish Qwen3.5 + Qwen3.6 35B MoE sweeps (in flight).
2. Trick 1 (sink protection) — trivial implementation, quick payoff.
3. Trick 3 (RHT seed cal) — orthogonal, pure bonus.
4. Trick 7 (shared LUT) — 10 LOC, safe cleanup.

Mid-term, riskier:
5. Trick 6 (CUDA receiver-side Viterbi) — major encoder speedup.
6. Trick 4 (correction overlay) — needs careful dequant-path timing.
7. Trick 9 (precomputed V signs) — ABI bump.
8. Trick 13 (prefix-shared V) — big impact for multi-user serving.

Long-term, research:
9. Trick 2 (per-head mixing) — biggest refactor.
10. Trick 5 (learned λ) — fine-tune recovery.
11. Trick 14 (K-V cross-dependency) — joint calibration.
12. Trick 16 (auto-bpw solver) — productization polish.

Aspirational:
13. Trick 17 (generative refinement) — the whole reason the
    numbering goes this far. If it works, everything above this
    becomes the support acts.
