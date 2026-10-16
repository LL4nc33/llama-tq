# TurboQuant — KTQ/VTQ KV Cache Quantization for CUDA

**Status (2026-10-12):** the KTQ/VTQ encoders now search the block scale (32 trial scales, unbiased scale). KL divergence
of `ktq4_1`/`vtq4_1` drops by 44 % on Qwen3-4B and 35 % on gpt-oss; the 2-bit types gain a lot on Qwen3-4B
(1.38 → 0.40) but little on gpt-oss (4.05 → 3.65). `ktq4_1`/`vtq4_1` (5 bpw) is now more accurate than `q4_0` on Qwen3-4B
and equal to it on gpt-oss and Gemma-4-12B, the three models measured (see [Accuracy](#accuracy)). Quality is measured with the KV cache
really quantized: in prefill K is staged in f16 by default (deferred K quantization) and only quantized at the
prefill→decode boundary; `llama-perplexity` turns the staging off, so it scores the quantized cache
(`--tq-deferred-k`/`--tq-deferred-v` keep the f16 staging there). The attention-rotation feature
(Hadamard rotation of Q/K in the graph) is not applied to KTQ caches, which made them worse; the remaining K/V rotations run
as the fast Walsh-Hadamard transform on CUDA.

CPU: all TurboQuant types can be written (`SET_ROWS` through `from_float`, the trellis encoders since 2026-10-12) and read
in CPU flash attention; there are no tuned CPU kernels, and the V rotation runs as a dense matmul.

## Recommendation

Accuracy first, then speed, then memory: use `f16` when the context fits, otherwise `-ctk q8_0 -ctv
q8_0` (at the numerical noise floor of the model; speed: [benchmarks](benchmarks/README.md)). `-ctk q8_0 -ctv vtq4_1` (6.5 bpw) stays close to that and
decodes as fast as `q8_0`; it is the choice for models that need 8-bit K such as gpt-oss. `-ctk ktq4_1 -ctv vtq4_1`
(5 bpw) is as accurate as or more accurate than `q4_0`; use it when you need the memory. The 2-bit types still lose a lot:
on Qwen3-4B, 2-bit K no longer retrieves exact text from a long context, on Qwen3.6-35B-A3B it does ([Long-context
retrieval](#long-context-retrieval)).

## Overview

TurboQuant is the KV-cache quantization stack of the `llama-tq` fork. It compresses the KV cache via a Randomized Hadamard Transform (RHT) plus codebook or trellis quantization, enabling much longer contexts on the same VRAM.

Two type families exist, split by cache role:

- **KTQ** (K-cache) — RHT (one shared random sign pattern + FWHT) + Lloyd-Max codebook. Attention kernels rotate Q once per query and dot it against codebook values, so K is never explicitly dequantized on the decode path. Prompt batches dequantize K/V to f16 and use the tensor-core kernel.
- **VTQ** (V-cache) — three sub-families:
  - **v1** (`vtq*_1`, codebook lookup): fixed D·H·D rotation applied once at the graph level via `self_v_rot`, then a flat codebook lookup per entry inside attention. The GQA decode kernel takes `vtq2_1`, `vtq3_1` and `vtq4_1` (not `vtq1_1`) next to f16, `q8_0`, `q5_0`, `q4_0` and `ktq{2,3,4}_1` K at head sizes 128 and 256; the tensor-core decode kernel covers the pairs listed in [Decode kernels](#decode-kernels-2026-10-04).
  - **v2 Trellis** (`vtq*_2`): group-level Viterbi trellis with shared state and shared scale, 16-bit shift register (L = 16, 65536 states), 16-bit open-start state, inverse-Gaussian CDF code table. `vtq*_2s` is the same trellis with a 12-bit shift register (L = 12, 4096 states) and the same block layout; its encoder is fast enough for decode.
  - **v3 Trellis + outlier-split** (`vtq*_3`): same Viterbi backbone as v2, plus a 4-fp16-outliers-per-block sidecar that keeps the largest absolute V values in fp16.

The suffix tells the method:

| Suffix | Method | Types | Decode |
|---|---|---|---|
| `_1` | Hadamard rotation + codebook | `ktq{1,2,3,4}_1`, `vtq{1,2,3,4}_1` | full speed |
| `_2` | trellis | `vtq{2,3,4}_2` | ~3× slower than `_1` (Qwen3-4B, RTX 2060: 20-27 t/s; 1-3 t/s until 2026-10-14) |
| `_2s` | trellis with 12 state bits ("s" = fast) | `vtq{2,3,4}_2s` | fast (Qwen3-4B: ~75 t/s) |
| `_3` | trellis + 4 fp16 outliers per block | `vtq{2,3,4}_3` | same encoder as `_2`, not measured separately |
| `_v8` | trellis 3-bit + 2 fp16 outliers (added in TurboQuant v8) | `vtq3_v8` | same encoder as `_2`, not measured separately |

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

# More context: 5 bpw, at or above q4_0 accuracy, GQA and tensor-core decode kernels
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk ktq4_1 -ctv vtq4_1

# Models that are sensitive to quantized K: protect the first and last layers
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk ktq4_1 -ctv vtq4_1 --tq-protect-layers 4
```

`--cache-type-k` accepts the stock quants (`f16`, `q8_0`, `q5_0`, `q4_0`, …) plus `ktq{1,2,3,4}_1` (and the experimental trellis types `ktq{2,3,4}_3s`). `--cache-type-v` accepts the stock quants plus `vtq{1,2,3,4}_1` (v1), `vtq{2,3,4}_2` and `vtq{2,3,4}_2s` (v2 Trellis), `vtq{2,3,4}_3` (v3 Trellis + outlier-split) and `vtq3_v8`; a `ktq*_1` type given as V is stored as f16 with a warning. The short aliases `ktq1`-`ktq4` and `vtq1`-`vtq4` map to the `*_1` types (`vtq2` and `vtq3` pointed to the trellis types `vtq2_2` / `vtq3_v8` until 2026-10-12).

Related options:

| Option | Effect |
|---|---|
| `--tq-protect-layers N` | keep the first and last N layers in q8_0 (TurboQuant types only) |
| `--tq-protect-sinks N` | keep the V cache of the first attention layer in f16 when N > 0 (off by default; the value of N does not matter otherwise) |
| `--no-tq-deferred-k`, `--no-tq-deferred-v` | skip the f16 staging buffer used during prefill; saves VRAM for full context (env `LLAMA_ARG_NO_TQ_DEFERRED_K`, `LLAMA_ARG_NO_TQ_DEFERRED_V`) |
| `--tq-deferred-k`, `--tq-deferred-v` | keep the f16 staging in `llama-perplexity`, which turns it off by default |
| `GGML_CUDA_TQ_WMMA=0` | disable the tensor-core kernel for decode and small query batches (they fall back to the vector kernel or to f16 conversion) |
| `GGML_CUDA_TQ_VEC_NO_GQA` | set to any value: disable the GQA decode kernel (falls back to the generic vector kernel) |


## Available Types

### K-cache — KTQ

RHT (shared sign pattern + FWHT) + Lloyd-Max codebook. Block stores normalization factor `d`, packed indices `qs`, and the 32 sign bits in `sb[4]` (identical in every block since 2026-10-04; the field keeps the block layout unchanged).

| Type   | enum | Index bits | bpw | Block | Notes |
|--------|:---:|:---:|:---:|:---:|---|
| `ktq1_1` | 45 | 1 | 2.5 | 10 B | extreme K compression |
| `ktq2_1` | 42 | 2 | 3.5 | 14 B | maximum context |
| `ktq3_1` | 43 | 3 | 4.5 | 18 B | balanced |
| `ktq4_1` | 44 | 4 | 5.5 | 22 B | with `vtq4_1` V more accurate than `q4_0`/`q4_0` on Qwen3-4B (see Accuracy) |

### K-cache — trellis KTQ (`ktq*_3s`, experimental)

Research types to find out whether trellis coding beats the codebook for K. Of a block of 128 K values the 4 largest
are stored in fp16 (as in `vtq*_3`); the rest is rotated with the KTQ transform over the whole block (shared sign pattern
+ 128-point FWHT, so the one trellis scale sees the energy of all 128 values) and coded like `vtq*_2s` (12 state bits,
Viterbi encoder). The sign pattern is not stored: layout and bpw equal `vtq*_3`. Dequantization decodes, undoes the
rotation and restores the outliers; Q·K dots the query with that row.

| Type   | enum | Index bits | bpw | Block |
|--------|:---:|:---:|:---:|:---:|
| `ktq2_3s` | 62 | 2 | 3.0 | 48 B |
| `ktq3_3s` | 63 | 3 | 4.0 | 64 B |
| `ktq4_3s` | 64 | 4 | 5.0 | 80 B |

Why the outliers: in real K one or two channels carry most of the energy of every token (Qwen3-0.6B: the largest
channel holds 27-87 % of ||k||², in up to 99 % of the tokens the same channel). Rotated, such a block is a near-binary
±c pattern. The codebook types fit it well with their per-32 scale search; a trellis, whose reconstruction values follow
a fixed Gaussian table, only reaches its Gaussian error rate, relative to the whole ||k||². The first version without
outliers (`ktq*_2s`, 2.25/3.25/4.25 bpw) therefore lost to `ktq*_1` on real models (Qwen3-4B KLD: 4.54 / 0.256 / 0.036 vs
`ktq2_1` 0.326, `ktq3_1` 0.073, `ktq4_1` 0.024) although it won on synthetic Gaussian data. Without the outliers the
remainder is near-Gaussian after the rotation, where the trellis beats the codebook per bit.

KL divergence against f16 K (V f16, CPU, `scripts/kld-kv-types-cpu.sh`, wikitext-2, 2 × 512 tokens), Qwen3-0.6B (a model
whose K is very sensitive to quantization):

| K type | bpw | Mean KLD | Same top p |
|---|:---:|:---:|:---:|
| `q8_0`    | 8.5 | 0.0041 | 97.1 % |
| `ktq2_1`  | 3.5 | 4.38   | 15.1 % |
| `ktq3_1`  | 4.5 | 0.759  | 64.9 % |
| `ktq4_1`  | 5.5 | 0.317  | 75.5 % |
| `ktq2_3s` | 3.0 | 0.227  | 79.6 % |
| `ktq3_3s` | 4.0 | 0.050  | 89.0 % |
| `ktq4_3s` | 5.0 | 0.016  | 95.1 % |

Offline on the same model's K/Q rows (softmax KL per head), the same 2-outlier split around the codebook (`ktq3_1` +
2 outliers, 4.875 bpw: 0.043) stays behind the trellis with 2 outliers (3.625 bpw: 0.029): the gain is not only the
outliers. Heads that are not a multiple of 128 fall back to `ktq*_1` when the cache is created. The CPU Viterbi
encoder needs ~12 ms per 3-bit block.

CUDA (since 2026-10-16): the K cache write runs the shared-memory Viterbi encoder of `vtq*_2s` with the rotation
added after the outlier pick (KTQ signs, 128-point FWHT across 128 threads, same operands and stage order as the CPU,
so the encoder aims at bit-identical blocks); the deferred K conversion after the prompt takes the same path. Flash
attention rotates Q once per query like the K blocks and dots it with the decoded samples; the CPU dequant overwrites
the inverse-rotated value x_p at each of the 4 outlier positions, so the readers add −q_p·x_p + q_p·val_p per outlier,
with x_p = sign(p)/√128 · Σ_m H(p,m)·t_m computed from the samples a lane already holds. Coverage:

| Path | K/V pairs | Head sizes |
|---|---|---|
| Decode, GQA kernel (GQA ratio ≥ 3, more than 256 cells) | `ktq{2,3,4}_3s` K × V f16, `q8_0`, `q5_0`, `q4_0`, `vtq{2,3,4}_1`, `vtq{2,3,4}_2s` (12-bit table in shared memory, shared with a `vtq*_2s` V) | 128, 256, 512 (since 2026-10-16) |
| Decode, vector kernel (GQA ratio 1-2, single KQ tile, `vtq1_1` V) | the GQA kernel's pairs plus `vtq1_1` V | 128, 256, 512 |
| Decode, other pairs | K and V converted to f16 every step, MMA kernel (correct, slow at long context) | any multiple of 128 |
| 2+ queries (speculative verify, prompt) | K (and a quantized V) converted to f16, MMA kernel | any multiple of 128 |

`GET_ROWS` and the f16/f32 conversions dequantize with one 128-thread block per trellis block. Not in the tensor-core
decode kernel.

### V-cache v1 — VTQ codebook (`vtq*_1`)

Pre-rotated via `self_v_rot` at graph level; FA dequant is `codebook[idx] * scale` only. No FWHT, no sign bits inside FA.

| Type   | enum | Index bits | bpw | Block | Notes |
|--------|:---:|:---:|:---:|:---:|---|
| `vtq1_1` | 46 | 1 | 1.5 | 6 B  | extreme VRAM, sharp quality drop |
| `vtq2_1` | 47 | 2 | 2.5 | 10 B | maximum context |
| `vtq3_1` | 48 | 3 | **4.0** | 16 B | 12 B indices + 2 B pad |
| `vtq4_1` | 49 | 4 | 4.5 | 18 B | near `q8_0` on Qwen3-4B and gpt-oss (V only) |

Note: `vtq3_1` is **4.0 bpw**, not 3.5 — block layout is `[d:2B] [qs: 12 B of 3-bit indices + 2 B pad]`.

### V-cache v2 — Trellis (`vtq*_2`, `vtq*_2s`)

Group-level Viterbi trellis. `vtq*_2`: 16-bit shift register (L = 16, 65536 states); `vtq*_2s`: 12-bit shift register
(L = 12, 4096 states), same block layout. 16-bit `start_state` per block (L bits used) plus 16-bit `d`. `QK_VTQ_TRELLIS=128` (halved from 256 in April 2026). Index-rate is the bare `K` bits/sample; structural bpw includes the 4-byte (`d`+`start_state`) overhead per 128-sample block.

| Type   | enum | State bits | Index bits | bpw (index-rate) | bpw (struct) | Block |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| `vtq2_2` | 50 | 16 | 2 | 2.0 | 2.25 | 36 B |
| `vtq3_2` | 51 | 16 | 3 | 3.0 | 3.25 | 52 B |
| `vtq4_2` | 52 | 16 | 4 | 4.0 | 4.25 | 68 B |
| `vtq2_2s` | 59 | 12 | 2 | 2.0 | 2.25 | 36 B |
| `vtq3_2s` | 60 | 12 | 3 | 3.0 | 3.25 | 52 B |
| `vtq4_2s` | 61 | 12 | 4 | 4.0 | 4.25 | 68 B |

Accuracy with V quantized in prefill (2026-10-11, Qwen3-4B Q4_K_M, wikitext-2, 60 chunks of 512, K in f16, KL divergence
against f16 KV; a separate run from the table under Accuracy, which uses 40 chunks):

| V | bpw | KLD |
|---|---:|---:|
| `vtq2_1` | 2.5 | 0.049 |
| `vtq2_2` (trellis) | 2.25 | 0.033 |
| `vtq3_1` | 4.0 | 0.019 |

The 16-bit trellis is more accurate per bit than the codebook types, but its Viterbi encoder searches all 65536 states
for every 128 values and runs for every new token. With the global-memory encoder used until 2026-10-14 that took 13.7 ms
per layer on an RTX 2060, so decode reached 1-3 t/s (2026-10-12; Qwen3-4B with `vtq3_2`: 1.0 t/s on an RTX 2060, 2.5 t/s on
an RTX 5080; `vtq3_1`: 97 t/s on the RTX 2060). Since 2026-10-15 the 16-bit types use the shared-memory encoder of
`vtq*_2s` (16 times its work per block): Qwen3-4B Q4_K_M on an RTX 2060 with `q8_0` K decodes at 26.6 / 27.0 / 19.9
t/s with `vtq2_2` / `vtq3_2` / `vtq4_2` (`vtq3_2s` 78, `vtq3_1` 89), and a 40-chunk perplexity run with V quantized in
prefill takes 3-6 minutes instead of about two hours. Each layer still encodes only one block per KV head per token, so
a decode step keeps a few SMs busy. Deferred staging (the default for these types) only moves the prompt's encoding to
one pass after the prompt. Since 2026-10-15 decode reads the trellis types with the GQA decode kernel (coverage in
[Decode kernels](#decode-kernels-2026-10-04)) instead of the VEC kernel, which decoded every V block once per Q head;
Qwen3-4B on two RTX 2060, `q8_0` K, decode at 0 / 8k / 32k context: `vtq3_2` 26.3 / 20.7 / 12.7 t/s (VEC: 26.5 / 12.1 /
4.3), `vtq3_2s` 78.0 / 62.5 / 39.2 (VEC: 77.7 / 24.4 / 7.3), `vtq3_1` 88.5 / 72.5 / 45.4. The prompt side stays slow
for the 16-bit types: their encoder runs over every prompt token once the prompt is done (a 7k-token prompt on Qwen3-4B:
69 s against 5 s with `vtq3_2s` and 2 s with `vtq3_1`), and a follow-up prompt on a filled cache reaches ~90 t/s. The April 2026
finding that `vtq2_2` / `vtq3_2` / `vtq4_2` give bit-identical perplexity was an artifact of that staging (V stayed f16
during prefill). Prefer `vtq*_2s` or `vtq*_1` for speed.

`vtq*_2s` trade a little of that accuracy for speed (2026-10-13, measured with the 12-bit trellis selected at build time
before the types existed; Qwen3-4B Q4_K_M, wikitext-2, 40 chunks of 512, K in f16, V quantized in prefill, KL divergence
against f16 KV):

| V | bpw | 12 state bits (`vtq*_2s`) | 16 state bits (`vtq*_2`) | codebook (`vtq*_1`, bpw) |
|---|---:|---:|---:|---:|
| 4-bit | 4.25 | 0.00482 | 0.00469 | 0.00514 (4.5) |
| 3-bit | 3.25 | 0.01014 | 0.00943 | 0.01197 (4.0) |
| 2-bit | 2.25 | 0.0286 | 0.02602 | 0.0471 (2.5) |

The 16-bit column was measured on 2026-10-15 with the shared-memory encoder; the global-memory encoder gave 0.00889 and
0.02592 for 3 and 2 bits. Both runs have the same median KLD; the mean moves with single outlier chunks, which land on
different chunks for every encoder variant. The 12-bit register costs 3-10 % KLD against 16 bits and is still 6-39 % below the codebook type with the same index
bits, at fewer bits per value. Its encoder searches 4096 instead of 65536 states and keeps the DP rows in shared memory,
so it runs for every new token without stalling decode: Qwen3-4B decodes at ~75 t/s with `vtq*_2s` against 20-27 t/s with
`vtq*_2` (1-2 t/s with the global-memory encoder until 2026-10-14). Decode kernels as for `vtq*_2`.

### V-cache v3 — Trellis + Outlier-Channel-Split (`vtq*_3`)

Same Viterbi backbone as v2, plus 4 fp16 outliers per block (largest absolute V values, stored in fp16).

| Type   | enum | Index bits | bpw avg | Block |
|--------|:---:|:---:|:---:|:---:|
| `vtq2_3` | 54 | 2 | 3.0 | 48 B |
| `vtq3_3` | 55 | 3 | 4.0 | 64 B |
| `vtq4_3` | 56 | 4 | 5.0 | 80 B |

`vtq3_v8` (enum 58, 3.625 bpw) is a trellis 3-bit variant with two fp16 outliers per block. Decode kernels as for the v2
types; their April 2026 perplexity numbers were taken with V staged in f16 during prefill; not re-measured. Before 2026-10-12 the CUDA encoder did not store the outliers of `vtq*_3` / `vtq3_v8`, so
earlier CUDA results for these types are without them.

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
- Kolibri-1 Q3_K_S: perplexity +0.5 % with `ktq4_1`/`vtq4_1` against f16 (cache quantized in prefill, 2026-10-05, before
  the encoder change).
- Mixed pairs such as `q8_0` K with `vtq{2,3,4}_1` or `q4_0` V, or `ktq4_1` K with `q8_0` V, use the GQA decode kernel
  (every pair of the listed types since 2026-10-14) and decode about as fast as `q8_0` / `q8_0`
  ([benchmarks](benchmarks/README.md)).

### Long-context retrieval

A copy task closer to real use than KLD: a synthetic changelog of about 29k tokens, the request to copy one section
verbatim and summarize it, greedy decoding in `llama-server`; a prompt counts when the copy is exact. Qwen3-4B Q4_K_M
on one RTX 2060, 36 prompts (6 documents, target sections at 10-85 % of the context); Qwen3.6-35B-A3B Q4_K_M on two,
12 prompts (3 documents, 15-90 %); 2026-10-14:

| K / V | Qwen3-4B | Qwen3.6-35B-A3B |
|---|---:|---:|
| f16 / f16 | 32/36 | |
| `q8_0` / `q8_0` | 31/36 | 12/12 |
| `q8_0` / `vtq4_1` | 32/36 | 12/12 |
| `ktq4_1` / `vtq4_1` | 32/36 | 12/12 |
| `q8_0` / `vtq2_1` | 31/36 | |
| `ktq3_1` / `vtq3_1` | 9/12 | |
| `ktq2_1` / `vtq2_1` | **0/36** | 12/12 |
| `ktq2_1` / f16 | **0/12** | |

- Up to 4 bits per K value every pair copies as well as f16 on Qwen3-4B; the prompts that fail are mostly the same
  ones for every type (the model picks the neighbouring section's version number).
- 2-bit K loses the retrieval completely on Qwen3-4B: the copies are fluent but wrong lines, with f16 V too, with every
  decode kernel and in the release of 2026-10-12 as well. On Qwen3.6-35B-A3B the same pair copies every section.
  Check your own model before relying on `ktq2_1`.

Decode speed per GPU and KV type: [benchmarks](benchmarks/README.md).

Older perplexity tables (until 2026-10-09) were not all taken with the cache quantized in prefill; they are kept in the
history below only.

The v2/v3 trellis V types with 16 state bits (`vtq*_2`, `vtq*_3`, `vtq3_v8`) give correct output but their encoder runs for every new token over 16 times the states of `vtq*_2s` (see How It Works); they decode about 3 times slower than `vtq*_2s` (Qwen3-4B on an RTX 2060: 20-27 t/s; 1-3 t/s with the global-memory encoder until 2026-10-14). For speed use `vtq*_2s` (12 state bits) or the v1 codebook types `vtq*_1`. A trellis block spans 128 values of one head; on models with smaller heads (gpt-oss:
64) those layers use the codebook type with the same bits instead, and the log says so.

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

Decode speed per GPU and KV type: [benchmarks](benchmarks/README.md); models on 2× RTX 2060: [models.md](models.md). The April 2026 tables from
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
   (since 2026-10-11; before, one assignment at the nominal scale). Only the 2-4 bit types search; 1 bit stores the sign.
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
| Tensor-core decode | `fattn-tq-wmma.cu` | Tiles of 64 (D=128), 32 (D=256) or 16 (D=512) K/V rows are dequantized to f16 in shared memory; S = K·Qᵀ and O += Vᵀ·P run as 16×16×16 WMMA with 16 query columns per block (the whole GQA group); attention sinks (gpt-oss) are supported. Also takes speculative-decoding verify batches of 2-16 queries: the columns are (query, head) pairs, one block of 16, or of 32 from 8192 cells (half the tile rows; not f16 K at D=256), each column with its own mask row. | GQA ratio 2-16. Same-bit pairs `ktq{2,3,4}_1` × `vtq{2,3,4}_1`, `q4_0` and `q5_0` at D=64/128/256/512 (at 128/256 only what the GQA kernel does not take: GQA ratio 2, a single KQ tile; the GQA kernel goes first since 2026-10-14); `q8_0` and `q8_0` K with `vtq{2,4}_1` V at D=64/512; f16 K with `vtq{2,3,4}_1` V at D=64, with `vtq{2,4}_1` V at D=256/512; `ktq{2,4}_1` K with f16 V at D=256 |
| GQA decode | `fattn-vec-gqa.cu` | One block per GQA group (4-column blocks for GQA ratio 3-4, 8-column blocks above 4), one warp per K/V row, the column dot products reduced together. Each lane reads several elements per load (2-byte loads for the 4-bit codes and `q4_0`/`q5_0`), the block scale is applied once per element, 2-bit codebooks sit in registers and 3/4-bit codebooks in shared memory; the scores are stored row by row so that the V loop reads the weights of all columns with one load. Supports the sparse index lists of Qwen3.8-Flash-Next. | GQA ratio ≥ 3, no sinks; contexts of a single KQ tile go to the generic vector kernel. Every pair of K f16, `q8_0`, `q5_0`, `q4_0`, `ktq{2,3,4}_1` and V f16, `q8_0`, `q5_0`, `q4_0`, `vtq{2,3,4}_1` at D=128/256 (since 2026-10-14; tried before the tensor-core decode kernel), except f16 / f16 (tensor-core MMA kernel). Since 2026-10-15 also the trellis V types (`vtq*_2`, `vtq*_2s`, `vtq*_3`, `vtq3_v8`) next to K f16, `q8_0` and `ktq{2,3,4}_1`: each V block is decoded once per GQA group, the 12-state-bit table (16 KB) is copied to shared memory, the 16-bit one (256 KB) is read through L2. Since 2026-10-16 also the trellis K types `ktq{2,3,4}_3s` next to V f16, `q8_0`, `q5_0`, `q4_0`, `vtq{2,3,4}_1` and `vtq{2,3,4}_2s`, and head size 512 (Gemma 4 global layers) for every pair above with a trellis K or V: 4-column blocks there (a GQA group of 8 takes two; Q and the V accumulators of 8 columns would need 256 registers, at most 255 exist), each trellis row decoded twice per group of 8 instead of once per Q head; sm_75 (ptxas): 232-255 registers, no spills, 26.8 KB static shared memory with the 12-state-bit table, 10.4 KB without it. The other pairs at D=512 stay with the tensor-core decode kernel. Not `ktq1_1`/`vtq1_1`; the trellis types not in the tensor-core decode kernel (verify batches convert V to f16, GQA ratio 1-2 and single-tile contexts use the vector kernel) |

Kernel-only microbenchmark of the attention op at 16k context (one RTX 2060, `test-backend-ops perf`, 2026-10-04,
before the encoder change; end-to-end speeds are on the [benchmarks](benchmarks/README.md) page):

| Shape | f16 | q8_0 | `ktq4_1`/`vtq4_1` | q5_0 |
|---|---:|---:|---:|---:|
| D=128, GQA 4 (K2-Horizon) | 229 µs | 216 µs | 212 µs | 226 µs |
| D=512, GQA 16 (Gemma-4-12B global layers, 32k) | 315 µs | 410 µs | 372 µs (before: 6636 µs) | – |

Measured speeds: [benchmarks](benchmarks/README.md); per model setup: [models.md](models.md).

The KV cache write (`SET_ROWS`) quantizes one 32-value block per warp (`GGML_CUDA_TQ_SET_ROWS_SERIAL`, set to any value, restores the one-thread-per-block kernel for comparison).

### VTQ v1 — codebook lookup

```
float[d_head] -> R · v (graph-level via self_v_rot) -> per-block: normalize -> scale search over the codebook -> unbiased scale -> pack
                                                                                           [no signs, no FWHT]
```

FA V-dequant reduces to:

```cuda
v_approx[j] = CB[q_j] * scale;   // ~8 thread-local registers, __forceinline__
```

The rotation works on blocks of 64 values and is applied only when the V head size is a multiple of 64. After FA accumulates the weighted sum, a graph-level `Rᵀ · VKQ` matmul applies the inverse rotation. Eliminates the ~40-register staging buffer that caused FA accumulator corruption in the original combined-KTQ V path.

### VTQ v2 — Trellis

Group-level Viterbi trellis with shared state and shared scale, 16-bit shift register (L = 16, 65536 states; `vtq*_2s`: L = 12, 4096 states), open-start state stored in 16 bits, inverse-Gaussian CDF code table (state s maps to the same code for every L, so the 12-bit types use the first 4096 entries of the 16-bit table). Block layout: `[d:fp16] [start_state:u16] [packed indices: K · group_size / 8 bytes]`.

The CUDA encoder is a shared-memory Viterbi over butterflies (the 2^K predecessors of a group of 2^K states share their
best predecessor, so the DP keeps one value per butterfly, 2^(L-K) per step). For L <= 14 both DP rows sit in shared
memory and each thread keeps its codes in registers. For L = 16 a row is 16-64 KiB: each thread keeps the DP values of
its butterflies in registers and exchanges them through a 32 KiB shared buffer (two rounds at 2 bits), keeps a quarter
of its codes in registers and reads the rest from L2 at every step, and the backtrack walk loads only the bytes it can
reach. The backtrack pool holds one 2 MiB slot per SM at L = 16 (32 MiB at L = 12).

The Viterbi path optimizes globally over the block, adapting implicitly to the running model's V-distribution — every V-element leverages local statistics at the same average bpw.

All trellis types (`vtq*_2`, `vtq*_2s`, `vtq*_3`, `vtq3_v8`) stage V in f16 during prefill and encode the prompt in one pass at the prefill→decode boundary; every decoded token still runs the encoder (cost: see [V-cache v2](#v-cache-v2--trellis-vtq_2-vtq_2s)). Startup logs: `deferred V quantization enabled (N layers with f16 staging)`. `llama-perplexity` turns the staging off by default, so it scores the quantized V cache.

### VTQ v3 — Trellis + outlier-channel-split

Same v2 backbone plus 4 fp16 outliers per block (largest absolute V values). Not re-measured with the cache quantized in prefill (see above). Before 2026-10-12 the CUDA encoder did not store the outliers of `vtq*_3` / `vtq3_v8`.

### Deferred K/V quantization

During prefill the K (and trellis V) cache is staged in f16 and converted in bulk at the prefill→decode boundary; this
is automatic for any KTQ K type and any trellis V type (`vtq*_2`, `vtq*_2s`, `vtq*_3`, `vtq3_v8`), except in `llama-perplexity`,
which turns it off by default (`--tq-deferred-k` / `--tq-deferred-v` keep it there). It was introduced in April 2026 against repetition loops with
K quantized per token in prefill (observed before the KTQ sign fix of 2026-10-04). `--no-tq-deferred-k` /
`--no-tq-deferred-v` turn it off (in `llama-bench`: `LLAMA_ARG_NO_TQ_DEFERRED_K=1`, `LLAMA_ARG_NO_TQ_DEFERRED_V=1`), which
saves the staging memory at full context and is how the accuracy numbers above are measured. The conversion runs once at the first decode step ([its cost](benchmarks/README.md#against-upstream-two-gpus)).

### Boundary-layer protection

`--tq-protect-layers N` stores the first and last N layers in q8_0 and the rest in the chosen TurboQuant types. On gpt-oss it helps only in proportion to the protected layers (see Accuracy).

### Attention-sink protection

`--tq-protect-sinks N` (off by default) keeps the V cache of the first attention layer (on hybrid models not necessarily
layer 0) in f16 when N > 0; the value of N does not matter otherwise. The sink tokens that draw
outsized attention on long contexts thus keep their full precision (StreamingLLM idea).

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
| Hadamard-domain KQ          | FWHT on Q once per tile; dot against codebook values. Fewer warp shuffles per `vec_dot` call than an inverse FWHT on K. |
| Warp-cooperative FWHT       | 32 lanes × 5 `__shfl_xor_sync` rounds (XOR masks 1/2/4/8/16); zero shared memory. |
| MMA split-dequant           | sm_75+ tensor-core path for prompt batches with KTQ K and/or VTQ V, head sizes up to 512. |
| Tensor-core decode          | WMMA kernel with grouped-query attention for decode and small verify batches (2-16 queries); type and head-size coverage in [Decode kernels](#decode-kernels-2026-10-04). Decode at head sizes 128 and 256 only what the GQA kernel does not take (GQA ratio 2, a single KQ tile); `GGML_CUDA_TQ_WMMA=0` disables it. |
| GQA decode                  | One block per GQA group, one warp per K/V row; sparse index lists; coverage in [Decode kernels](#decode-kernels-2026-10-04). |
| Warp SET_ROWS               | One warp quantizes one 32-value block when writing the KV cache; each lane evaluates one trial scale. |
| Branchless sign × norm      | `(1 − 2·bit) · norm` replaces ternary; eliminates warp divergence. |
| Shared sign pattern         | All KTQ blocks use the same signs, so Q is rotated once per query. |
| VTQ v1 V-dequant            | `codebook[idx] * scale`, `__forceinline__`, ~8 registers, no LMEM spills. |
| VTQ pre-rotation            | `self_v_rot` matmul before cache write + inverse `Rᵀ` matmul after FA. |
| FA dispatch                 | KTQ1_1..KTQ4_1 (K), VTQ1_1..VTQ4_1 (v1), VTQ2_2..VTQ4_2 and VTQ2_2S..VTQ4_2S (v2), VTQ2_3..VTQ4_3 (v3), VTQ3_V8. |
| SET_ROWS kernels            | `k_set_rows_pq` / `k_set_rows_pq_warp` (KTQ, VTQ v1) in `set-rows.cu`; trellis encoder in `trellis-encode.cuh`. |
| `head_dim` support          | KTQ: 64/128/256/512. Trellis V needs head sizes that are multiples of 128 (128/256/512); layers with smaller heads (gpt-oss: 64) use the codebook type with the same bits. |
| CPU                         | Inverse rotation in KTQ dequantization (`kktq_rht_inverse_sb`, AVX2 path, `ggml/src/ggml-quants.c`). |
| Compute capability          | Developed on CC 7.5 (Turing, RTX 2060); tested on Ampere (RTX 3060, 3090), Ada (RTX 4060 Ti, 4090) and Blackwell (RTX 5090) for decode speed. Untested below CC 7.5; the tensor-core paths need CC 7.5+. |

## Source Files

| File                                       | Description |
|--------------------------------------------|---|
| `ggml/include/ggml.h`                      | Type enums: KTQ1_1=45, KTQ2_1=42, KTQ3_1=43, KTQ4_1=44. VTQ1_1=46, VTQ2_1=47, VTQ3_1=48, VTQ4_1=49. VTQ2_2=50, VTQ3_2=51, VTQ4_2=52. 53 reserved (was VTQ_MIXED). VTQ2_3=54, VTQ3_3=55, VTQ4_3=56. 57 reserved (was XKTQ2_1). VTQ3_V8=58. VTQ2_2S=59, VTQ3_2S=60, VTQ4_2S=61. KTQ2_3S=62, KTQ3_3S=63, KTQ4_3S=64. |
| `ggml/src/ggml-common.h`                   | Block structs: `block_ktq*` (with `sb[4]`), `block_vtq*_1`, `block_vtq*_2` (Trellis; `block_vtq*_2s` are typedefs of it), `block_vtq*_3` (Trellis + outliers), `block_vtq3_v8`. |
| `ggml/src/ggml-trellis.{c,h}`              | Trellis code table, CPU group encoder/decoder and outlier selection; state bits `GGML_TRELLIS_L` = 16, `GGML_TRELLIS_L_S` = 12 (`vtq*_2s`). |
| `ggml/src/ggml-cuda/turboquant.cuh`        | CUDA kernels: KTQ Philox, FWHT, quantize, dequant; VTQ v1 quantize/dequant. |
| `ggml/src/ggml-cuda/fattn-tq.cuh`          | FA: `vec_dot_fattn_vec_KQ_ktq`, `vec_dot_fattn_vec_KQ_ktq_trellis` (+ Q rotation and outlier terms of `ktq*_3s`), `dequantize_V_ktq`, `dequantize_V_vtq*` (v1/v2/v3, `vtq3_v8`). |
| `ggml/src/ggml-cuda/fattn-tq-wmma.cu`      | Tensor-core decode kernel. |
| `ggml/src/ggml-cuda/fattn-vec-gqa.{cu,cuh}` | GQA decode kernel. |
| `ggml/src/ggml-cuda/fattn-vec-dispatch-ktq-trellis.cu` | Vector-kernel dispatch of the `ktq*_3s` K types. |
| `ggml/src/ggml-cuda/fattn-mma-ktq.{cu,cuh}` | MMA-KTQ tensor-core split-dequant path (CC ≥ 7.5). |
| `ggml/src/ggml-cuda/set-rows.cu`           | KV cache write: `k_set_rows_pq`, `k_set_rows_pq_warp`, trellis encoder instances. |
| `ggml/src/ggml-cuda/trellis-encode.cuh`    | CUDA trellis encoder (full Viterbi over butterflies): DP rows in shared memory for L <= 14, in registers with a shared exchange buffer for L = 16; the 128-point rotation of `ktq*_3s`. |
| `ggml/src/ggml-cuda/trellis.cu`            | Trellis code table per device and encoder backtrack pool per device and L (L = 12: 32 MiB, L = 16: 2 MiB per SM). |
| `ggml/src/ggml-cuda/convert.cu`            | CUDA dequant dispatch (contiguous + NC) for KTQ (incl. `ktq*_3s`) + all VTQ families. |
| `ggml/src/ggml-quants.c`                   | CPU quantize/dequantize for KTQ + VTQ (incl. the experimental `ktq*_3s`); shared `PQ_CODEBOOK_*` constants. |
| `src/llama-kv-cache.cpp`                   | Per-layer cache types (layer and sink protection, trellis head-size fallback), deferred K/V staging, V rotation. |
| `tools/perplexity/perplexity.cpp`          | Turns deferred staging off unless `--tq-deferred-k`/`--tq-deferred-v` is given. |
| `common/arg.cpp`                           | CLI: `--cache-type-k`, `--cache-type-v` parser; accepts `ktq{1,2,3,4}_1`, `ktq{2,3,4}_3s`, `vtq{1,2,3,4}_1`, `vtq{2,3,4}_2`, `vtq{2,3,4}_2s`, `vtq{2,3,4}_3`, `vtq3_v8`. |
| `docs/benchmarks/history/LIVE_NUMBERS.md` | Historical TG/PPL/HellaSwag snapshot. |

## Tried and discarded

Measured and not pursued (the history is in the [changelog](../CHANGELOG.md)):

- **Trellis-K (TCQ-style):** a trellis K is not linear in its indices, so the K·Q dot product in the Hadamard domain no
  longer works, and the GQA and tensor-core decode path is lost (April 2026).
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
- **Mixed K/V pairs outside the decode kernels' coverage** (head sizes other than 128 and 256, GQA ratio 1-2,
  `ktq1_1`/`vtq1_1`) run on CUDA by converting K and V to f16 at every step; correct, but decode slows down with
  context. Since 2026-10-14 the GQA decode kernel takes every pair of f16, `q8_0`, `q5_0`, `q4_0` and the 2-4-bit
  TurboQuant types at head sizes 128 and 256 (before, Qwen3-4B decoded `ktq4_1` / `q8_0` at 10 t/s at 32k, now 41 t/s).
  Building with `-DGGML_CUDA_FA_ALL_QUANTS=ON` compiles vector kernels for more pairs.
- **VTQ v1 on Qwen3-Next-80B with `-b 1 -ub 1`**: reported in April 2026 to crash through the fused Gated DeltaNet path; not re-checked since. Batched mode and other models were unaffected.

## References

This implementation is inspired by but deviates from the cited papers. KTQ uses RHT (FWHT + one shared random sign pattern) + Lloyd-Max instead of QR rotation; VTQ v1 uses a fixed D·H·D rotation (specific to this fork); v2/v3 use a Viterbi trellis over an inverse-Gaussian CDF code table. QJL was used in v1–v4 and removed in v5.

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
