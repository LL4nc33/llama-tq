# TurboQuant Nano — Research Backlog

Five Naughty-Dog-style tricks to squeeze more out of the current
quantization stack. Ordered by ROI / risk. All experimental, all
measure-before-ship.

> _"We didn't make the model smaller; we made the hardware work harder
> in smarter moments."_

## Already in tree

**Layer-boundary protection** (`tq_protect_layers` in `llama_kv_cache`).
First N + last N layers keep `q8_0` instead of TQ. CLI exposes it,
default off. Naughty-Dog's "keep critical geometry in foreground RAM."

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
`vtq2_2`. Naughty-Dog's sprite-overlay-on-low-res-background analogue.

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

## Sequencing

Short-term, measurable:
1. Finish Qwen3.5 + Qwen3.6 35B MoE sweeps (in flight).
2. Trick 1 (sink protection) — trivial implementation, quick payoff.
3. Trick 3 (RHT seed cal) — orthogonal to everything, pure bonus.

Mid-term, riskier:
4. Trick 4 (correction overlay) — needs careful dequant-path timing
   analysis; don't break FA inner loop.
5. Trick 2 (per-head mixing) — biggest refactor; blocked on an
   internal KV-cache type-per-head abstraction.

Long-term, research:
6. Trick 5 (learned λ) — becomes most interesting if the simpler
   tricks plateau. Needs training infra.
