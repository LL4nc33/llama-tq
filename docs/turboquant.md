# TurboQuant — KTQ/VTQ KV Cache Quantization for CUDA

**Status (2026-10-11):** the KTQ/VTQ encoders now search the block scale (32 trial scales, unbiased scale). KL divergence
of `ktq4_1`/`vtq4_1` drops by 44 % on Qwen3-4B and 35 % on gpt-oss; the 2-bit types gain a lot on Qwen3-4B
(1.38 → 0.40) but little on gpt-oss (4.05 → 3.65). `ktq4_1`/`vtq4_1` (5 bpw) is now more accurate than `q4_0` on Qwen3-4B
and equal to it on gpt-oss and Gemma-4-12B, the three models measured (see [Accuracy](#accuracy)). Quality is measured with the KV cache
really quantized: in prefill K is staged in f16 by default (deferred K quantization) and only quantized at the
prefill→decode boundary, so a perplexity run without `--no-tq-deferred-k` measures f16 K. The attention-rotation feature
(Hadamard rotation of Q/K in the graph) is not applied to KTQ caches, which made them worse; the remaining K/V rotations run
as the fast Walsh-Hadamard transform.

## Recommendation

Accuracy first, then speed, then memory: use `f16` when the context fits, otherwise `-ctk q8_0 -ctv
q8_0` (at the numerical noise floor of the model, same speed). `-ctk q8_0 -ctv vtq4_1` (6.5 bpw) stays close to that and
decodes as fast as `q8_0`; it is the choice for models that need 8-bit K such as gpt-oss. `-ctk ktq4_1 -ctv vtq4_1`
(5 bpw) is as accurate as or more accurate than `q4_0`; use it when you need the memory. The 2-bit types still lose a lot.

## Overview

TurboQuant is the KV-cache quantization stack of the `llama-tq` fork. It compresses the KV cache via a Randomized Hadamard Transform (RHT) plus codebook or trellis quantization, enabling much longer contexts on the same VRAM.

Two type families exist, split by cache role:

- **KTQ** (K-cache) — RHT (one shared random sign pattern + FWHT) + Lloyd-Max codebook. Attention kernels rotate Q once per query and dot it against codebook values, so K is never explicitly dequantized on the decode path. Prompt batches dequantize K/V to f16 and use the tensor-core kernel.
- **VTQ** (V-cache) — three sub-families:
  - **v1** (`vtq*_1`, codebook lookup): fixed D·H·D rotation applied once at the graph level via `self_v_rot`, then a flat codebook lookup per entry inside attention. The fast decode kernels cover `vtq2_1`, `vtq3_1` and `vtq4_1` (not `vtq1_1`).
  - **v2 Trellis** (`vtq*_2`): group-level Viterbi trellis with shared state and shared scale, 16-state shift-register, 16-bit open-start state, inverse-Gaussian CDF code table.
  - **v3 Trellis + outlier-split** (`vtq*_3`): same Viterbi backbone as v2, plus a 4-fp16-outliers-per-block sidecar that captures the largest absolute V values losslessly.

Reference: PolarQuant (Han et al., arXiv:2502.02617, ICLR 2026) and TurboQuant (Zandieh et al., arXiv:2504.19874).

## Quick Start

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server
```

Pick a tier by passing two cache-type flags. `-fa on` is required.

```bash
# Accurate: at the noise floor of the model (see Accuracy), same decode speed as f16 at long context
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk q8_0 -ctv q8_0

# Accurate and compact: 8-bit K, 4-bit V (6.5 bpw), as fast as q8_0
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk q8_0 -ctv vtq4_1

# More context: 5 bpw, at or above q4_0 accuracy, fast decode kernels
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk ktq4_1 -ctv vtq4_1

# Models that are sensitive to quantized K: protect the first and last layers
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk ktq4_1 -ctv vtq4_1 --tq-protect-layers 4
```

`--cache-type-k` accepts the stock quants (`f16`, `q8_0`, `q5_0`, `q4_0`, …) plus `ktq{1,2,3,4}_1`. `--cache-type-v` accepts the stock quants plus `vtq{1,2,3,4}_1` (v1), `vtq{2,3,4}_2` (v2 Trellis), and `vtq{2,3,4}_3` (v3 Trellis + outlier-split). The short aliases from 2026-05 map `ktq1`-`ktq4` to `ktq*_1`, `vtq1` and `vtq4` to `vtq1_1` / `vtq4_1`, but `vtq2` to the trellis type `vtq2_2` and `vtq3` to `vtq3_v8`; use the long names for the fast `vtq*_1` types.

Related options:

| Option | Effect |
|---|---|
| `--tq-protect-layers N` | keep the first and last N layers in q8_0 (TurboQuant types only) |
| `--tq-protect-sinks N` | keep the V cache of layer 0 in f16 when N > 0 (off by default) |
| `--no-tq-deferred-k`, `--no-tq-deferred-v` | skip the f16 staging buffer used during prefill; saves VRAM for full context |
| `GGML_CUDA_TQ_WMMA=0` | disable the tensor-core decode kernel (falls back to the GQA kernel) |
| `GGML_CUDA_TQ_VEC_NO_GQA=1` | disable the GQA decode kernel (falls back to the generic vector kernel) |


## Available Types

### K-cache — KTQ

RHT (shared sign pattern + FWHT) + Lloyd-Max codebook. Block stores normalization factor `d`, packed indices `qs`, and the 32 sign bits in `sb[4]` (identical in every block since 2026-10-04; the field keeps the block layout unchanged).

| Type   | enum | Index bits | bpw | Block | Notes |
|--------|:---:|:---:|:---:|:---:|---|
| `ktq1_1` | 45 | 1 | 2.5 | 10 B | extreme K compression |
| `ktq2_1` | 42 | 2 | 3.5 | 14 B | maximum context |
| `ktq3_1` | 43 | 3 | 4.5 | 18 B | balanced |
| `ktq4_1` | 44 | 4 | 5.5 | 22 B | better than `q4_0` K on Qwen3-4B (see Accuracy) |

### V-cache v1 — VTQ codebook (`vtq*_1`)

Pre-rotated via `self_v_rot` at graph level; FA dequant is `codebook[idx] * scale` only. No FWHT, no sign bits inside FA.

| Type   | enum | Index bits | bpw | Block | Notes |
|--------|:---:|:---:|:---:|:---:|---|
| `vtq1_1` | 46 | 1 | 1.5 | 6 B  | extreme VRAM, sharp quality drop |
| `vtq2_1` | 47 | 2 | 2.5 | 10 B | maximum context |
| `vtq3_1` | 48 | 3 | **4.0** | 16 B | 14 B index-payload + padding |
| `vtq4_1` | 49 | 4 | 4.5 | 18 B | near `q8_0` on Qwen3-4B and gpt-oss (V only) |

Note: `vtq3_1` is **4.0 bpw**, not 3.5 — block layout is `[d:2B] [qs:14B 3-bit packed across 16 bytes]`.

### V-cache v2 — Trellis (`vtq*_2`)

Group-level Viterbi trellis. 16-state shift register; 16-bit `start_state` per block plus 16-bit `d`. `QK_VTQ_TRELLIS=128` (halved from 256 in April 2026). Index-rate is the bare `K` bits/sample; structural bpw includes the 4-byte (`d`+`start_state`) overhead per 128-sample block.

| Type   | enum | Index bits | bpw (index-rate) | bpw (struct) | Block |
|--------|:---:|:---:|:---:|:---:|:---:|
| `vtq2_2` | 50 | 2 | 2.0 | 2.25 | 36 B |
| `vtq3_2` | 51 | 3 | 3.0 | 3.25 | 52 B |
| `vtq4_2` | 52 | 4 | 4.0 | 4.25 | 68 B |

Not covered by the fast decode kernels. The April 2026 finding that `vtq2_2` / `vtq3_2` / `vtq4_2` give bit-identical
perplexity was an artifact of the deferred V staging (V stayed f16 during prefill); these types have not been re-measured
with the cache quantized in prefill. Prefer `vtq*_1`.

### V-cache v3 — Trellis + Outlier-Channel-Split (`vtq*_3`)

Same Viterbi backbone as v2, plus 4 fp16 outliers per block (largest absolute V values, stored losslessly).

| Type   | enum | Index bits | bpw avg | Block |
|--------|:---:|:---:|:---:|:---:|
| `vtq2_3` | 54 | 2 | 3.0 | 48 B |
| `vtq3_3` | 55 | 3 | 4.0 | 64 B |
| `vtq4_3` | 56 | 4 | 5.0 | 80 B |

`vtq3_v8` (enum 58, 3.625 bpw) is a trellis 3-bit variant with two fp16 outliers per block. Like the v2 types these are
not covered by the fast decode kernels, and their April 2026 perplexity numbers were taken with V staged in f16 during
prefill; not re-measured.

### Reserved / dormant

| Type        | enum | Status |
|-------------|:---:|---|
| (`vtq_mixed`) | 53 | **Removed** — dominated by `vtq3_1`, never had a CUDA path. Code deleted, enum id kept reserved. |
| (`xktq2_1`) | 57 | **Removed** — XQuant cross-layer subordinate K. Never produced pairs on the hybrid models in use; code deleted, enum id kept reserved. |

## Accuracy

KL divergence of the token distributions against an f16 KV cache (`llama-perplexity --kl-divergence`, 40 chunks of 512
tokens, `-fa on`) with the cache really quantized during prefill (`--no-tq-deferred-k --no-tq-deferred-v`). Lower is
better; in brackets the share of positions where the most likely token is unchanged. Measured 2026-10-11 with the encoder
that searches the block scale.

Text: Qwen3-4B on wikitext-2 (test). gpt-oss and Gemma 4 are chat models that are far out of distribution on raw wikitext
(perplexity 207 and 505), so they are measured on their own prose: answers they wrote with an f16 cache on 40 topics
([scripts/kld-gen-text.py](../scripts/kld-gen-text.py); perplexity 5.7 and 40). The first row is the same f16 cache with
flash attention off: the numerical noise floor of the model, below which no KV type can be told apart.

| K / V | bpw (K+V)/2 | Qwen3-4B Q4_K_M | gpt-oss-20b MXFP4 | Gemma-4-12B Q4_K_M |
|---|---:|---:|---:|---:|
| `f16` / `f16`, flash attention off | 16 | 0.0030 (97.8 %) | 0.013 (96.0 %) | 0.32 (82.9 %) |
| `q8_0` / `q8_0` | 8.5 | 0.0032 (97.5 %) | 0.021 (95.0 %) | 0.35 (82.5 %) |
| `q8_0` / `vtq4_1` | 6.5 | 0.0055 (96.8 %) | 0.039 (92.4 %) | 0.53 (76.5 %) |
| `q4_0` / `q4_0` | 4.5 | 0.034 (92.5 %) | 0.36 (73.4 %) | 0.59 (75.3 %) |
| `ktq4_1` / `vtq4_1` | 5.0 | **0.025** (93.6 %) | 0.36 (74.7 %) | 0.58 (75.3 %) |
| `ktq3_1` / `vtq3_1` | 4.25 | 0.084 (88.1 %) | 1.30 (50.7 %) | 0.99 (66.3 %) |
| `ktq2_1` / `vtq2_1` | 3.0 | 0.40 (73.9 %) | 3.65 (18.8 %) | 2.13 (49.2 %) |
| `f16` / `vtq4_1` | | 0.0053 | 0.043 | 0.54 |
| `f16` / `vtq2_1` | | 0.044 | 0.23 | 1.66 |
| `ktq4_1` / `f16` | | 0.023 | 0.35 | 0.48 |
| `ktq2_1` / `f16` | | 0.31 | 3.40 | 1.33 |

- `ktq4_1`/`vtq4_1` is more accurate than `q4_0` on Qwen3-4B and equal on gpt-oss and Gemma 4, at 5 instead of 4.5 bpw.
  Before the encoder change (2026-10-10) it was 0.045 (Qwen3-4B) and 0.55 (gpt-oss).
- `q8_0` is at the noise floor of every model; `vtq4_1` V costs little on Qwen3-4B and gpt-oss.
- gpt-oss is sensitive in K: every 4-bit K type (`q4_0`, `ktq4_1`) costs about 0.35, V only 0.04; protecting outer layers
  helps only in proportion (`--tq-protect-layers 4`: 0.145, 8 of 24 layers each side: 0.097), so use `q8_0` K with
  `vtq4_1` V there (0.039). Gemma 4 is sensitive in V
  (`vtq2_1` 1.66); it reacts to any numerical change (flash attention on vs off already 0.32) and its same-top-token share
  stays the better guide there.
- 2-bit K and 3-bit K lose clearly on all three models.
- Reproduce: base `llama-perplexity -m M -f TEXT -c 512 --chunks 40 -fa on --kl-divergence-base base.kld`, then the same
  with `-ctk K -ctv V --no-tq-deferred-k --no-tq-deferred-v --kl-divergence-base base.kld --kl-divergence`.
- Hybrid models with few attention layers lose much less (Qwen3.8-27B, 16 of 64 layers with attention: `ktq4_1`/`vtq4_1`
  PPL 6.039 vs 6.036 with f16, cache quantized in prefill, earlier encoder), see [docs/models.md](models.md).
- `q8_0` K with `vtq4_1` or `vtq2_1` V uses the GQA and tensor-core decode kernels (since 2026-10-11): Qwen3-4B at 16k context
  50.7 t/s with `vtq4_1` (`q8_0`/`q8_0` 52), gpt-oss at 8k 70.4 t/s (RTX 2060).

Decode speed per GPU and KV type: [benchmarks](benchmarks/README.md).

Older perplexity tables (until 2026-10-09) were not all taken with the cache quantized in prefill; they are kept in the
history below only.

The v2/v3 trellis V types (`vtq*_2`, `vtq*_3`) still work but are not covered by the fast decode kernels below; prefer the v1 codebook types for speed.

## Memory Savings

Bits per value of the KV cache (block scales included) and its size relative to an f16 cache. The KV cache of a model
grows with context length × layers with attention × KV heads × head size; hybrid models (Gated DeltaNet, sliding
window) need it only for some layers.

| K / V | K bpw | V bpw | (K+V)/2 | size vs f16 |
|---|---:|---:|---:|---:|
| `f16` / `f16` | 16 | 16 | 16 | 100 % |
| `q8_0` / `q8_0` | 8.5 | 8.5 | 8.5 | 53 % |
| `q8_0` / `vtq4_1` | 8.5 | 4.5 | 6.5 | 41 % |
| `q4_0` / `q4_0` | 4.5 | 4.5 | 4.5 | 28 % |
| `ktq4_1` / `vtq4_1` | 5.5 | 4.5 | 5 | 31 % |
| `ktq3_1` / `vtq3_1` | 4.5 | 4 | 4.25 | 27 % |
| `ktq2_1` / `vtq2_1` | 3.5 | 2.5 | 3 | 19 % |

## Benchmarks

Decode speed per GPU and KV type, models on 2× RTX 2060: [benchmarks](benchmarks/README.md). The April 2026 tables from
this page are in [history](benchmarks/history/2026-04-turboquant.md).

## How It Works

### KTQ — K-cache quantization

```
float[32] -> normalize -> per-block sign flip -> FWHT -> scale search over the Lloyd-Max codebook -> unbiased scale -> pack
                              |                                                    |
                        [store in sb[4]]                              [store corrected d]
```

1. **Normalize:** `x_hat = x / ||x||`.
2. **Sign flip:** Deterministic ±1 signs from Philox 2×32, 6-round counter-based PRNG (Salmon et al. 2011). Since 2026-10-04 all blocks share the pattern of block 0: the rotation stays a random orthogonal map per block, and attention rotates Q once per query instead of once per K block.
3. **FWHT:** 32-point Fast Walsh-Hadamard Transform, scaled by `1/√32` (orthonormal, self-inverse).
4. **Scale search over the codebook:** the codebooks are Lloyd-Max optimal for `Beta((d-1)/2, (d-1)/2) = Beta(15.5, 15.5)`
   at d=32 (the marginal of a unit vector coordinate after random rotation). The block is assigned to the nearest
   centroids at 32 trial scales of the codebook (0.36x .. 1.60x); the assignment with the smallest angle to the block wins
   (since 2026-10-11; before, one assignment at the nominal scale).
5. **Unbiased scale:** store `||x|| / <x_hat, c>`, which keeps the component of the reconstruction along the input exact.
   A least-squares scale would shrink every block, and the attention average over many tokens keeps that shrink. Before
   2026-10-11 the scale kept the length of the block (`||x|| / ||c||`, "norm correction").
6. **Sign bits:** Store the 32 precomputed signs in `sb[4]` (32 bits). Dequant reads bits, never re-runs Philox.

### KTQ dequant — Hadamard-domain dot product

Instead of inverse-FWHT on K, the FA kernel applies FWHT to Q once per tile and dots directly against codebook values:

```
score = norm * <K_dequant, Q>
      = norm * <sign · FWHT⁻¹(cb), Q>
      = norm * <cb, FWHT(sign · Q)>     // FWHT orthogonal: <Hx, y> = <x, Hᵀy>; H = Hᵀ when normalized
```

Shifts the 32-element FWHT from per-K-block (gather + butterfly in the hot loop) to a single per-Q FWHT amortized across all K blocks a Q tile attends to. With the shared sign pattern the rotated Q is the same for every K block, so it is computed once per query.

### KTQ MMA path (CC ≥ 7.5)

Split-dequant for prompt batches: bulk K (and TurboQuant V) → f16 conversion, then the standard MMA-F16 tensor-core kernel runs unchanged. Covers head sizes up to 512 (Gemma 4 global layers since 2026-10-04). Decode uses the kernels in the next section. Source: `ggml/src/ggml-cuda/fattn-mma-ktq.{cu,cuh}`.

### Decode kernels (2026-10-04)

Decode (one query token per sequence) has two dedicated CUDA kernels for quantized KV with grouped-query attention. `ggml_cuda_flash_attn_ext` tries them first and falls back to the generic vector kernel for anything they do not cover.

| Kernel | Source | How it works | Covers |
|---|---|---|---|
| Tensor-core decode | `fattn-tq-wmma.cu` | Tiles of 64 (D=128), 32 (D=256) or 16 (D=512) K/V rows are dequantized to f16 in shared memory; S = K·Qᵀ and O += Vᵀ·P run as 16×16×16 WMMA with 16 query columns per block (the whole GQA group). `ktq{2,3,4}_1` × `vtq{2,3,4}_1`, `q5_0`, `q4_0` at D=128/256/512 and D=64 with sinks (gpt-oss); KTQ or VTQ mixed with f16 at D=256/512; `q8_0` and `q8_0` K with `vtq{2,4}_1` V at D=64/512 |
| GQA decode | `fattn-vec-gqa.cu` | One block per GQA group (4 or 8 query heads), one warp per K/V row, the column dot products reduced together; 3/4-bit codebooks in shared memory. Supports the sparse index lists of Qwen3.8-Flash-Next. | KTQ, VTQ v1, `q8_0` (also with `vtq{2,4}_1` V), `q5_0`, f16 mixes at D=128/256 |

Attention time at 16k context (one RTX 2060, from `test-backend-ops perf`):

| Shape | f16 | q8_0 | `ktq4_1`/`vtq4_1` | q5_0 |
|---|---:|---:|---:|---:|
| D=128, GQA 4 (K2-Horizon) | 229 µs | 216 µs | 212 µs | 226 µs |
| D=512, GQA 16 (Gemma-4-12B global layers, 32k) | 315 µs | 410 µs | 372 µs (before: 6636 µs) | – |

Measured speeds: [benchmarks](benchmarks/README.md); per model setup: [models.md](models.md).

The KV cache write (`SET_ROWS`) quantizes one 32-value block per warp (`GGML_CUDA_TQ_SET_ROWS_SERIAL=1` restores the one-thread-per-block kernel for comparison).

### VTQ v1 — codebook lookup

```
float[d_head] -> R · v (graph-level via self_v_rot) -> per-block: normalize -> scale search over the codebook -> unbiased scale -> pack
                                                                                           [no signs, no FWHT]
```

FA V-dequant reduces to:

```cuda
v_approx[j] = CB[q_j] * scale;   // ~8 thread-local registers, __forceinline__
```

After FA accumulates the weighted sum, a graph-level `Rᵀ · VKQ` matmul applies the inverse rotation. Eliminates the ~40-register staging buffer that caused FA accumulator corruption in the original combined-KTQ V path.

### VTQ v2 — Trellis

Group-level Viterbi trellis with shared state and shared scale, 16-state shift register, 16-bit open-start state, inverse-Gaussian CDF code table. Block layout: `[d:fp16] [start_state:u16] [packed indices: K · group_size / 8 bytes]`.

The Viterbi path optimizes globally over the block, adapting implicitly to the running model's V-distribution — every V-element leverages local statistics at the same average bpw.

**Encoder cost is non-trivial** (~22 ms/call), so VTQ_2 auto-enables **f16 staging during prefill** and runs the bulk Viterbi exactly once at the prefill→decode boundary. Startup logs: `deferred V quantization enabled (N layers with f16 staging)`. Perplexity runs therefore need `--no-tq-deferred-v` to measure the quantized V cache; without it they measure f16 V.

### VTQ v3 — Trellis + outlier-channel-split

Same v2 backbone plus 4 fp16 outliers per block (largest absolute V values). Round-trip MSE drops a further 4× vs v2 at +1 bpw average (perplexity not re-measured with the cache quantized in
prefill, see above).

### Deferred K/V quantization

During prefill the K (and VTQ v2/v3 V) cache is staged in f16 and converted in bulk at the prefill→decode boundary; this
is automatic for any KTQ K type and any VTQ v2/v3 V type. It was introduced in April 2026 against repetition loops with
K quantized per token in prefill (observed before the KTQ sign fix of 2026-10-04). `--no-tq-deferred-k` /
`--no-tq-deferred-v` turn it off, which saves the staging memory at full context and is how the accuracy numbers above
are measured. The conversion runs once at the first decode step (about 0.1 s after a 16k prompt on an RTX 2060).

### Boundary-layer protection

`--tq-protect-layers N` stores the first and last N layers in q8_0 and the rest in the chosen TurboQuant types. The outer layers carry the most outlier-heavy K; on K2-Horizon, N=4 cut the PPL cost of `ktq4_1`/`vtq4_1` from +3.3 % to +0.8 % (2026-10-04, earlier encoder); on gpt-oss
it helps only in proportion to the protected layers (see Accuracy).

### Attention-sink protection

`--tq-protect-sinks N` (off by default) keeps the V cache of layer 0 in f16 when N > 0, so the sink tokens that draw
outsized attention on long contexts keep their full precision (StreamingLLM idea).

### Sparse V dequant

In the FA V-accumulation loop, dequant is skipped for positions where attention weight < 1e-6. At 32k+ context, >90% of positions are skipped. +22% decode throughput. Works with both KTQ-V (legacy) and VTQ V-types.

### Shared codebooks (`PQ_CODEBOOK_*`)

Both KTQ and VTQ v1 use the same Lloyd-Max codebooks, optimal for `Beta(15.5, 15.5)`:

- 1-bit (2 centroids): `{−0.7979, +0.7979}` (= ±√(2/π))
- 2-bit (4 centroids): `{−1.4896, −0.4514, +0.4514, +1.4896}`
- 3-bit (8 centroids): `{−2.0719, −1.3150, −0.7453, −0.2424, +0.2424, +0.7453, +1.3150, +2.0719}`
- 4-bit (16 centroids): symmetric 16-point set, `±{0.1284, 0.3880, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326}`

All scaled by `1/√32`. CPU constants: `PQ_CODEBOOK_*BIT`. CUDA constants: `PQ_CUDA_CB_*BIT`.

## CUDA Implementation Details

| Feature                     | Description |
|-----------------------------|---|
| Hadamard-domain KQ          | FWHT on Q once per tile; dot against codebook values. 39% fewer warp shuffles per `vec_dot` call vs naive inverse-FWHT-on-K. |
| Warp-cooperative FWHT       | 32 lanes × 5 `__shfl_xor_sync` rounds (XOR masks 1/2/4/8/16); zero shared memory. |
| AVX2-FWHT-32                | Host-side path used by deferred-K bulk conversion. |
| MMA split-dequant           | sm_75+ tensor-core path for prompt batches with KTQ K and/or VTQ V, head sizes up to 512. |
| Tensor-core decode          | WMMA decode kernel with grouped-query attention for TurboQuant, `q8_0`, `q5_0`, `q4_0` and `q8_0` K with VTQ V; head sizes 64 (with sinks), 128, 256, 512 (`GGML_CUDA_TQ_WMMA=0` disables it). |
| GQA decode                  | One block per GQA group, one warp per K/V row; KTQ, VTQ v1, `q8_0` (also with VTQ V), `q5_0`; sparse index lists. |
| Warp SET_ROWS               | One warp quantizes one 32-value block when writing the KV cache; each lane evaluates one trial scale. |
| Branchless sign × norm      | `(1 − 2·bit) · norm` replaces ternary; eliminates warp divergence. |
| Shared sign pattern         | All KTQ blocks use the same signs, so Q is rotated once per query. |
| VTQ v1 V-dequant            | `codebook[idx] * scale`, `__forceinline__`, ~8 registers, no LMEM spills. |
| VTQ pre-rotation            | `self_v_rot` matmul before cache write + inverse `Rᵀ` matmul after FA. |
| Sparse V dequant            | Skip dequant for `attn_weight < 1e-6`; +22% decode at 32k+ ctx. |
| FA dispatch                 | KTQ1_1..KTQ4_1 (K), VTQ1_1..VTQ4_1 (v1), VTQ2_2..VTQ4_2 (v2), VTQ2_3..VTQ4_3 (v3). |
| SET_ROWS kernels            | `k_set_rows_ktq*` + `k_set_rows_vtq*`. |
| `head_dim` support          | KTQ: 64/128/256/512. VTQ v2: D=64/128/256/512 (verified live on Gemma4 D=256/512 SWA + full-attn, Qwen3.6-35B D=128, gpt-oss-20b D=64). |
| Compute capability          | Developed on CC 7.5 (Turing, RTX 2060); tested on Ampere (RTX 3060, 3090), Ada (RTX 4060 Ti, 4090) and Blackwell (RTX 5090, RTX PRO 6000). Builds for CC 6.0+; the tensor-core paths need CC 7.5+. |

## Source Files

| File                                       | Description |
|--------------------------------------------|---|
| `ggml/include/ggml.h`                      | Type enums: KTQ1_1=45, KTQ2_1=42, KTQ3_1=43, KTQ4_1=44. VTQ1_1=46, VTQ2_1=47, VTQ3_1=48, VTQ4_1=49. VTQ2_2=50, VTQ3_2=51, VTQ4_2=52. 53 reserved (was VTQ_MIXED). VTQ2_3=54, VTQ3_3=55, VTQ4_3=56. 57 reserved (was XKTQ2_1). VTQ3_V8=58. |
| `ggml/src/ggml-common.h`                   | Block structs: `block_ktq*` (with `sb[4]`), `block_vtq*_1`, `block_vtq*_2` (Trellis), `block_vtq*_3` (Trellis + outliers). |
| `ggml/src/ggml-cuda/turboquant.cuh`        | CUDA kernels: KTQ Philox, FWHT, quantize, dequant; VTQ v1 quantize/dequant. |
| `ggml/src/ggml-cuda/fattn-common.cuh`      | FA: `vec_dot_KQ_ktq*`, `dequantize_V_ktq*`, `dequantize_V_vtq*` (v1/v2/v3), Sparse-V guard. |
| `ggml/src/ggml-cuda/fattn-mma-ktq.{cu,cuh}` | MMA-KTQ tensor-core split-dequant path (CC ≥ 7.5). |
| `ggml/src/ggml-cuda/convert.cu`            | CUDA dequant dispatch (contiguous + NC) for KTQ + all VTQ families. |
| `ggml/src/ggml-quants.c`                   | CPU quantize/dequantize for KTQ + VTQ; shared `PQ_CODEBOOK_*` constants. |
| `common/arg.cpp`                           | CLI: `--cache-type-k`, `--cache-type-v` parser; accepts `ktq{1,2,3,4}_1`, `vtq{1,2,3,4}_1`, `vtq{2,3,4}_2`, `vtq{2,3,4}_3`. |
| `docs/benchmarks/history/LIVE_NUMBERS.md` | Historical TG/PPL/HellaSwag snapshot. |

## Tried and discarded

Measured and not pursued (the history is in the [changelog](../CHANGELOG.md)):

- **Trellis-K (TCQ-style):** a trellis K is not linear in its indices, so the K·Q dot product in the Hadamard domain no
  longer works; about 0.5–1 % PPL for losing the fast decode path.
- **VTQ_MIXED:** dominated by `vtq3_1`, no CUDA path; enum 53 stays reserved.
- **Calibrated outlier selection** before the v3 trellis: marginal gain after the rotation.
- **XQuant cross-layer K reuse:** the hybrid Qwen3.x models alternate Gated DeltaNet and attention layers, which leaves no
  layer pairs; removed, enum 57 stays reserved.
- **Adaptive top-k MoE routing:** Qwen3's softmax over the selected experts is near uniform, so a cumulative-mass cut does
  not apply ([moe-expert-locality.md](research/moe-expert-locality.md)).
- **Sparse K skip** (skip K blocks with a small norm): no speed gain at equal perplexity.
- **Least-squares block scale** in the encoder (2026-10-11): lower reconstruction error, but it shrinks V and the attention
  average keeps the shrink; the unbiased scale is used instead.

Not planned: FA3 (sm_80+ only), paged attention, multi-node inference, MLA-style latent KV.

## When *not* to use TurboQuant

- VRAM is not a constraint — upstream f16 KV is simpler and equally fast.
- Multi-node serving — not a goal of this fork (tensor split across local GPUs works, see [tensor-split.md](tensor-split.md)).

## Known Issues

- **2- and 3-bit K** still lose a lot of accuracy on every measured model (see [Accuracy](#accuracy)); gpt-oss needs 8-bit K.
- **VTQ v1 on Qwen3-Next-80B with `-b 1 -ub 1`**: reported in April 2026 to crash through the fused Gated DeltaNet path; not re-checked since. Batched mode and other models were unaffected.

## References

This implementation is inspired by but deviates from the cited papers. KTQ uses RHT (FWHT + per-block random sign) + Lloyd-Max instead of QR rotation; VTQ v1 uses a fixed D·H·D rotation (specific to this fork); v2/v3 use a Viterbi trellis over an inverse-Gaussian CDF code table. QJL was used in v1–v4 and removed in v5.

| Paper | Authors | arXiv | Relevance |
|---|---|---|---|
| **PolarQuant: Quantizing KV Cache via Polar Coordinate Transformation** | Han, Kacham, Karbasi, Mirrokni, Zandieh | [2502.02617](https://arxiv.org/abs/2502.02617) (Feb 2025, ICLR 2026) | Primary inspiration for VTQ v1 D·H·D rotation. |
| **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** | Zandieh, Daliri, Hadian, Mirrokni | [2504.19874](https://arxiv.org/abs/2504.19874) (April 2025) | Random rotation + Lloyd-Max codebooks framework. |
| **QJL: 1-Bit Quantized JL Transform for KV Cache Quantization** | Zandieh, Daliri, Han | [2406.03482](https://arxiv.org/abs/2406.03482) (June 2024) | Used in v1–v4, removed in v5. |
| **XQuant: Cross-layer KV reuse** | (cf. 2510.11236) | [2510.11236](https://arxiv.org/abs/2510.11236) | Tried as a subordinate K path, later removed. |
| **Parallel Random Numbers: As Easy as 1, 2, 3** | Salmon, Moraes, Dror, Shaw | SC 2011 | Philox 2×32 counter-based PRNG used for KTQ sign generation. |

## Version History

The history of the TurboQuant types (v1 in April 2026 to the encoder scale search) is in the
[changelog](../CHANGELOG.md).
