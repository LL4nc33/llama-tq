# KTQ_2 Design — Trellis-coded K-cache

Parked design doc for the K-side Trellis variant, planned for autoresearch-driven
iteration. Written 2026-04-24 alongside the KTQ × VTQ_2 measurement session.

## Motivation

VTQ_2 (Trellis v2) hits <1% PPL at 3 bits on V by spending one big bulk
Viterbi encode at the prefill→decode boundary (deferred V). That trick works
because the full V sequence is available at boundary time. K is different:
each decode step produces one new K token that must be quantized **online**
before the next token's attention can use it. No bulk window, no deferred
pass.

So KTQ_1 uses an RHT + Lloyd-Max codebook and gets +1.4% PPL at 3.0 bpw on
35B-A3B. The question: can we reach VTQ_2-class quality (<1% at 3 bits) for
K without breaking the online-per-token constraint?

## Hard constraints

1. **Online quantization** — one new K vector per decode step, must fit in
   one kernel launch inside or adjacent to the attention update.
2. **No growth in FA inner loop** — dequant cost on the K side is already
   amortized because `Q·K` runs in Hadamard domain (K never dequantized to
   attention). Whatever we do on the encoder side must not leak cost into
   the FA read path.
3. **Memory layout stability** — block structure must match the existing
   KV-cache allocator and the FA dispatcher's `D` branch expectations.
4. **Turing (sm_75) viability** — must run on RTX 2060, so no WGMMA or FP8
   tensor cores. Shared memory budget ≤ 64 KB per SM.

## Design axes to explore

### A. Single-token Viterbi with warp-parallel trellis

The VTQ_2 encoder is group-Viterbi over a 512-sample window. For K we have
one vector of `head_dim` samples per step. Warp-parallel single-vector
Viterbi is well-known (TCQ / QTIP); open question is how many trellis states
fit a 32-thread warp at `head_dim ∈ {64, 128, 256}` without spilling.

Knobs:
- Trellis states: 4 / 8 / 16 / 32 (bits/sample = log2(states))
- Warp role: one warp per head? one warp per 32 samples? per-sample 1 thread
  with dp reduction?
- Branch metric: L2 in Hadamard domain (same as VTQ_2) or signed L1 (cheaper)

Risk: if the single-vector Viterbi is slower than the current KTQ_1 online
encode, we lose throughput before we gain quality.

### B. Incremental trellis with running state

The decode-step encoder holds a Viterbi state from the previous step and
emits only one symbol per step. Like a streaming Viterbi decoder run in
reverse. Requires per-head state kept in registers or shmem across decode
steps — lightweight but needs kernel-state threading through the FA launch.

Knobs:
- State width: 4 / 8 / 16 paths kept
- Backtrace depth: how many past symbols can still be edited?
  (fixed-lag smoothing vs truncated Viterbi)

Risk: FA kernel already owns the decode fastpath; adding cross-step state
is invasive and multi-sequence support would need per-slot state.

### C. Learned per-head codebooks (no Viterbi)

Sidestep Trellis entirely. Run one-shot codebook calibration per attention
head on a small activation trace. Store 4-8-centroid codebooks in constant
memory. Online encode is a nearest-centroid lookup — same cost profile as
KTQ_1 but with per-head optimal codebooks instead of one global codebook.

Knobs:
- Centroid count: 4 / 8 / 16
- Calibration data: wikitext-2 / model-specific
- Kmeans vs Lloyd-Max-Max (heavy-tail aware)

Risk: per-head codebook storage is 32 heads × 16 centroids × fp16 =
~1 KB per layer × 64 layers = 64 KB per model — tight but fits constant mem.

### D. Outlier-channel split (per-head)

Already in the v6 VTQ_OUT design doc for V. Same principle for K: detect
which channels (positions in head_dim) carry outlier values and store
those in f16 while the rest get a Trellis/codebook encode. At 20% outlier
channels + 2-bit bulk = 4.4 bpw effective.

Shares infrastructure with v6 VTQ_OUT. Co-implementing both is cheaper than
either alone.

## Measurement plan (autoresearch-ready)

Fixed metric for agent loop:

```
score = ppl_delta_pct  +  0.5 * tg_slowdown_pct
```

Where `ppl_delta_pct` is measured on `Qwen3.6-35B-A3B-IQ2_XXS` wikitext-2
ctx=2048/5ch, and `tg_slowdown_pct` is measured on `llama-bench -gen 128`
for the same model. Lower is better. Weight on TG reflects "we already have
KTQ_1 at -1.4% PPL, any new design must not cost more than 2× that in
throughput to justify any PPL gain."

### Loop shape

1. Agent picks one axis (A/B/C/D) and one knob setting
2. Builds on gpu00 (`cmake --build build -j2 --target llama-perplexity llama-bench`)
3. Measures PPL + TG (≤15 min total)
4. Logs `score` vs baseline KTQ_1
5. Keeps if `score < 0.9 * previous_best`, otherwise reverts
6. Repeat

Time budget: 15 min/experiment × 12 experiments/session = 3 h. Start with
axis C (lowest risk, smallest code change) for calibration baseline.

### Baselines to hit

| Config | PPL Δ | TG | Interpretation |
|--------|:---:|:---:|---|
| ktq3_1 (existing, 4.5 bpw) | +1.4% | 1.0× | Current floor |
| ktq4_1 (existing, 5.5 bpw) | +0.3% | 0.97× | Near-lossless bar |
| **target: ktq3_2**         | **<0.5%** | **>0.85×** | Want this |
| stretch: ktq2_2            | <2%     | >0.85× | Competitive with VTQ_2 |

## Scope for first pass

Start narrow — axis C (per-head codebooks) only. It's the smallest code
change, touches only the encoder, leaves FA kernel untouched, and gives
a honest PPL floor against which Viterbi variants (A/B) can be judged.

**Out of scope for first autoresearch pass:**
- Multi-sequence state threading (axis B only works on single-slot decode)
- Outlier-channel split (belongs to v6 VTQ_OUT work)
- Per-model calibration (would require retraining dataset sweeps)

## Files that would change

Rough map based on current KTQ implementation layout:

- `ggml/src/ggml-cuda/ktq_encoder.cu` (new) — per-head codebook encode
- `ggml/src/ggml-cuda/fattn-vec-dispatch-ktq2.cu` (new) — FA dispatch for K_2
- `ggml/src/ggml-cuda/ktq_calibration.py` (new, offline) — compute codebooks
- `src/llama-kv-cache.cpp` — register `KTQ{2,3,4}_2` enum, add `is_ktq_k` detection
- `common/arg.cpp` — CLI `--cache-type-k ktq{2,3,4}_2`
- `tests/test-ktq2-roundtrip.cpp` — unit test

Roughly 500-800 LOC for axis C. Axes A/B add 200-400 LOC each.

## Open questions

- Can we reuse the VTQ_2 group-Viterbi kernel with `group_size = head_dim`
  to get axis A almost for free? Need to check if the group-Viterbi state
  fits register budget at small group sizes.
- Is attention-sink protection (`--tq-protect-layers`) necessary for KTQ_2,
  or does per-head calibration subsume the first-layer protection behavior?
- Does KTQ_2 need the `D·H·D` randomized rotation that KTQ_1 uses? The
  rotation was added so that Gaussian-like K entries cluster nicely for
  Lloyd-Max. Per-head codebooks adapt to whatever distribution exists, so
  the rotation might be unnecessary (or actively harmful if it flattens
  per-head structure).

## Status

Design parked — will be picked up when autoresearch loop infrastructure
is in place. Not blocking any shipped work.
