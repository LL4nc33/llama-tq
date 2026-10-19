# TurboQuant — KTQ/VTQ KV Cache Quantization for CUDA

**Status (2026-10-16):** the short names `ktq2`-`ktq4` and `vtq2`-`vtq4` now select the trellis types (`ktq*_3s`,
`vtq*_2s`), the most accurate type of each bit width; the number is the bits per value. They run on CUDA in the GQA and
vector decode kernels; verify and prompt batches convert them to f16 for the tensor-core kernel. Against the codebook types of the same bits (`ktq*_1` / `vtq*_1`, still available by their full
names) the KL divergence drops 3-4 times on Qwen3-4B, and 2-bit K no longer loses long-context retrieval (0/36 → 26-29/36
copies). The price is speed, because the trellis encoder runs after the prompt and for every new token: on Qwen3-4B
decode is 12-14 % slower on an empty context, 20 % at 8k and 30-34 % at 32k (one RTX 2060, since the flash-attention
work of 2026-10-18), and the first prompt takes 1.5-1.9 times as long (before the encoder work of 2026-10-17: decode
30-40 % slower, first prompt 3-4.5 times; [benchmarks](benchmarks/README.md#trellis-and-codebook-types-two-gpus)).
Quality is measured with the KV cache really quantized: in prefill K is staged in f16 by default (deferred K
quantization) and only quantized at the prefill→decode boundary; `llama-perplexity` turns the staging off, so it scores
the quantized cache (`--tq-deferred-k`/`--tq-deferred-v` keep the f16 staging there). The attention-rotation feature
(Hadamard rotation of Q/K in the graph) is not applied to KTQ caches, which made them worse; the remaining K/V rotations
run as the fast Walsh-Hadamard transform on CUDA.

CPU: all TurboQuant types can be written (`SET_ROWS` through `from_float`, the trellis encoders since 2026-10-12) and read
in CPU flash attention; there are no tuned CPU kernels, and the V rotation runs as a dense matmul.

## Recommendation

Accuracy first, then speed, then memory. KL divergence (KLD) measures how far the model's next-token probabilities move
away from an f16 cache; 0 is identical ([Accuracy](#accuracy)).

- Use `f16` when the context fits, otherwise `-ctk q8_0 -ctv q8_0`: at the numerical noise floor of the model, full speed.
- `-ctk q8_0 -ctv vtq4_1` (6.5 bpw) stays close to that at full speed; it is the choice for models that need 8-bit K
  such as gpt-oss.
- For more context: `-ctk ktq4 -ctv vtq4` (4.6 bpw, Qwen3-4B KLD 0.0090) is clearly more accurate than `q4_0` / `q4_0`
  (4.5 bpw, 0.034). `-ctk ktq3 -ctv vtq3` (3.6 bpw, 0.025) is as accurate as the codebook pair `ktq4_1` / `vtq4_1`
  (5.0 bpw): take the codebook pair when the memory is there, it decodes 15-50 % faster.
- `-ctk ktq2 -ctv vtq2` (2.6 bpw, 0.091) is the smallest cache that still retrieves exact text from a long context on
  Qwen3-4B (26-29/36 copies, f16 32/36); with `q8_0` V 29-30/36. The 2-bit codebook K (`ktq2_1`) fails there completely.
- The trellis types (`ktq2`-`ktq4`, `vtq2`-`vtq4`) decode 12-14 % slower than the codebook types on an empty context, 20 %
  at 8k and 30-34 % at 32k, and take 1.5-1.9 times as long for the first prompt; when speed matters more than the last bit of accuracy, use the codebook types by
  their full names (`ktq4_1`, `vtq4_1`, …).

## Overview

TurboQuant is the KV-cache quantization stack of the `llama-tq` fork. It compresses the KV cache via a Randomized Hadamard Transform (RHT) plus codebook or trellis quantization, enabling much longer contexts on the same VRAM.

The types are split by cache role:

- **KTQ trellis** (K-cache, `ktq{2,3,4}_3s`, short names `ktq2`-`ktq4`): the 4 largest values of a 128-value block
  in fp16, the rest rotated (shared signs + 128-point FWHT) and coded like `vtq*_2s` (12 state bits); see
  [trellis KTQ](#k-cache--trellis-ktq-ktq_3s).
- **KTQ codebook** (K-cache, `ktq*_1`) — RHT (one shared random sign pattern + FWHT) + Lloyd-Max codebook. Attention kernels rotate Q once per query and dot it against codebook values, so K is never explicitly dequantized on the decode path. Prompt batches dequantize K/V to f16 and use the tensor-core kernel.
- **VTQ** (V-cache) — three sub-families:
  - **v1** (`vtq*_1`, codebook lookup): fixed D·H·D rotation applied once at the graph level via `self_v_rot`, then a flat codebook lookup per entry inside attention. The GQA decode kernel takes `vtq2_1`, `vtq3_1` and `vtq4_1` (not `vtq1_1`) next to f16, `q8_0`, `q5_0`, `q4_0`, `ktq{2,3,4}_1` and `ktq{2,3,4}_3s` K at head sizes 128 and 256 (with a trellis
    K also at 512); the tensor-core decode kernel covers the pairs listed in [Decode kernels](#decode-kernels-2026-10-04).
  - **v2 Trellis** (`vtq*_2`): group-level Viterbi trellis with shared state and shared scale, 16-bit shift register (L = 16, 65536 states), 16-bit open-start state, inverse-Gaussian CDF code table. `vtq*_2s` is the same trellis with a 12-bit shift register (L = 12, 4096 states) and the same block layout; its encoder is fast enough for decode.
  - **v3 Trellis + outlier-split** (`vtq*_3`): same Viterbi backbone as v2, plus a 4-fp16-outliers-per-block sidecar that keeps the largest absolute V values in fp16.

The suffix tells the method:

| Suffix | Method | Types | Decode |
|---|---|---|---|
| `_1` | Hadamard rotation + codebook | `ktq{1,2,3,4}_1`, `vtq{1,2,3,4}_1` (short names `ktq1`, `vtq1`) | full speed |
| `_3s` | K: 4 fp16 outliers + Hadamard rotation + trellis with 12 state bits | `ktq{2,3,4}_3s` (short names `ktq2`-`ktq4`) | 12-34 % slower than `_1` with `vtq*_2s` V |
| `_2` | trellis | `vtq{2,3,4}_2` | ~3× slower than `_1` (Qwen3-4B, RTX 2060: 20-27 t/s; 1-3 t/s until 2026-10-14) |
| `_2s` | trellis with 12 state bits ("s" = fast) | `vtq{2,3,4}_2s` (short names `vtq2`-`vtq4`) | 10-15 % slower than `_1` |
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

# Accurate and compact: 8-bit K, 4-bit codebook V (6.5 bpw), as fast as q8_0
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk q8_0 -ctv vtq4_1

# More context: 4.6 bpw, more accurate than q4_0 (trellis: slower decode and first prompt)
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk ktq4 -ctv vtq4

# Most context: 2.6 bpw, still retrieves exact text from long contexts
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk ktq2 -ctv vtq2

# Fast instead: the codebook types of the same bits
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk ktq4_1 -ctv vtq4_1

# Models that are sensitive to quantized K: protect the first and last layers
./build/bin/llama-server -m model.gguf -fa on -ngl 99 -ctk ktq4 -ctv vtq4 --tq-protect-layers 4
```

`--cache-type-k` accepts the stock quants (`f16`, `q8_0`, `q5_0`, `q4_0`, …) plus `ktq{1,2,3,4}_1` and the trellis types `ktq{2,3,4}_3s`. `--cache-type-v` accepts the stock quants plus `vtq{1,2,3,4}_1` (v1), `vtq{2,3,4}_2` and `vtq{2,3,4}_2s` (v2 Trellis), `vtq{2,3,4}_3` (v3 Trellis + outlier-split) and `vtq3_v8`; both flags parse the same list, VTQ types are V-only, and only a `ktq*_1` type given as V is stored as f16 (with a
warning). The short names pick the most accurate type of each bit width: `ktq2`, `ktq3`, `ktq4` are `ktq{2,3,4}_3s` and `vtq2`, `vtq3`, `vtq4` are `vtq{2,3,4}_2s` (trellis); `ktq1` and `vtq1` are `ktq1_1` / `vtq1_1`, as there is no 1-bit trellis. The codebook types `*_1` decode faster and finish the first prompt sooner; pick them by their full name. Until 2026-10-16 the short names meant the `*_1` types (and `vtq2` / `vtq3` the trellis types `vtq2_2` / `vtq3_v8` until 2026-10-12).

Related options:

| Option | Effect |
|---|---|
| `--tq-protect-layers N` | keep the first and last N attention layers of the model in q8_0 (TurboQuant types only) |
| `--tq-protect-sinks N` | keep the V cache of the first attention layer in f16 when N > 0 (TurboQuant V types only; off by default; the value of N does not matter otherwise); with `--tq-protect-layers` as well, that layer's V stays f16 |
| `LLAMA_ARG_TQ_PROTECT_LAYERS`, `LLAMA_ARG_TQ_PROTECT_SINKS` | the same two options as environment variables. `llama-bench` takes no `--tq-*` flags; it reads these two, `LLAMA_ARG_NO_TQ_DEFERRED_K/V` and `LLAMA_ARG_TQ_V_OVERRIDE=SPEC` (per-layer V types, applied there without `--tq-mixed-v`) |
| `-ctks`, `-ctvs TYPE` (`--cache-type-k-swa`, `--cache-type-v-swa`) | cache types of the sliding-window layers (Gemma 4, gpt-oss; default: the same as `-ctk` / `-ctv`); `DG_SWA_KV_F16=1` forces f16 there |
| `--no-tq-deferred-k`, `--no-tq-deferred-v` | skip the f16 staging buffer used during prefill; saves VRAM for full context (env `LLAMA_ARG_NO_TQ_DEFERRED_K`, `LLAMA_ARG_NO_TQ_DEFERRED_V`) |
| `--tq-deferred-k`, `--tq-deferred-v` | keep the f16 staging in `llama-perplexity`, which turns it off by default |
| `GGML_CUDA_TQ_WMMA=0` | disable the tensor-core kernel for decode and small query batches (they fall back to the vector kernel or to f16 conversion) |
| `GGML_CUDA_TQ_VEC_NO_GQA` | set to any value: disable the GQA decode kernel (falls back to the generic vector kernel) |
| `GGML_CUDA_VTQ_POOL_SLOTS=N` | slot count of the trellis encoder's backtrack pool (default: at least one slot per SM and 32 MiB; 2 MiB per slot at 16 state bits, 128 KiB at 12) |
| `--tq-mixed-v` (with `--tq-v-profile`, `--tq-v-strategy`, `--tq-v-base`/`-high`/`-low`, `--tq-v-override`, `--tq-v-budget-bpw`) | experimental: a V type per layer (trellis V, f16 or `q8_0`) from a head profile written by `--tq-profile-heads N` (per-head V variance and kurtosis of the first N decode calls, written to `tq-profile-heads-<PID>.json`); without `--tq-mixed-v` the other options are ignored. The default base / high / low types are `vtq3_2` / `vtq4_2` / `vtq2_2`, the 16-state-bit trellis with the slow encoder. Not measured for accuracy. |


## Available Types

### K-cache — codebook KTQ (`ktq*_1`)

RHT (shared sign pattern + FWHT) + Lloyd-Max codebook. Block stores normalization factor `d`, packed indices `qs`, and the 32 sign bits in `sb[4]` (identical in every block since 2026-10-04; the field keeps the block layout unchanged).

| Type   | enum | Index bits | bpw | Block | Notes |
|--------|:---:|:---:|:---:|:---:|---|
| `ktq1_1` | 45 | 1 | 2.5 | 10 B | extreme K compression |
| `ktq2_1` | 42 | 2 | 3.5 | 14 B | maximum context |
| `ktq3_1` | 43 | 3 | 4.5 | 18 B | balanced |
| `ktq4_1` | 44 | 4 | 5.5 | 22 B | with `vtq4_1` V more accurate than `q4_0`/`q4_0` on Qwen3-4B (see Accuracy) |

### K-cache — trellis KTQ (`ktq*_3s`)

Trellis coding for K, the types behind the short names `ktq2`-`ktq4`. Of a block of 128 K values the 4 largest
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
encoder needs ~0.6 ms per block at 12 state bits and ~9 ms at 16 (Ryzen 7 4800H, one thread; before 2026-10-17 it
searched every state instead of every butterfly: 13 and 380 ms).

CUDA (since 2026-10-16): the K cache write runs the shared-memory Viterbi encoder of `vtq*_2s` with the rotation
added after the outlier pick (KTQ signs, 128-point FWHT across 128 threads, same operands and stage order as the CPU);
the deferred K conversion after the prompt takes the same path. CPU and CUDA write bit-identical blocks (see
[VTQ v2](#vtq-v2--trellis)). Flash
attention rotates Q once per query like the K blocks and dots it with the decoded samples; the CPU dequant overwrites
the inverse-rotated value x_p at each of the 4 outlier positions, so the readers add −q_p·x_p + q_p·val_p per outlier,
with x_p = sign(p)/√128 · Σ_m H(p,m)·t_m computed from the samples a lane already holds. Coverage:

| Path | K/V pairs | Head sizes |
|---|---|---|
| Decode, GQA kernel (GQA ratio ≥ 3; since 2026-10-18 also a single KQ tile of ≤ 256 cells) | `ktq{2,3,4}_3s` K × V f16, `q8_0`, `q5_0`, `q4_0`, `vtq{2,3,4}_1`, `vtq{2,3,4}_2s` (12-bit table in shared memory, shared with a `vtq*_2s` V) | 128, 256, 512 (since 2026-10-16) |
| Decode, vector kernel (GQA ratio 1-2, `vtq1_1` V; before 2026-10-18 also a single KQ tile) | the GQA kernel's pairs plus `vtq1_1` V | 128, 256, 512 |
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
that searches the block scale; the trellis rows (short names `ktq2`-`ktq4`, `vtq2`-`vtq4`) on Qwen3-4B 2026-10-17 with
the same base and flags, after the CUDA encoder switched to IEEE norms and scales (identical blocks on CPU and GPU;
before: 0.0087 / 0.025 / 0.095; the encoder now picks a different, equally good path in some blocks). On gpt-oss
(head size 64) the trellis types fall back to the codebook types of the same bits; on Gemma 4 their KLD was not
measured (see the retrieval test on Gemma 4 26B below). The Qwen3-4B column was repeated on the release tree of
2026-10-16 on two RTX 2060 and on an RTX A40, RTX 4090 and RTX 5090: every pair within ±6 % of the table on every GPU
(`q8_0` 0.0031-0.0034, `ktq4` / `vtq4` 0.0085-0.0089, `ktq2_1` / `vtq2_1` 0.40-0.42).

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
| `ktq4` / `vtq4` (`ktq4_3s` / `vtq4_2s`) | 4.6 | **0.0090** (95.9 %) | | |
| `ktq3` / `vtq3` (`ktq3_3s` / `vtq3_2s`) | 3.6 | **0.024** (93.4 %) | | |
| `ktq2` / `vtq2` (`ktq2_3s` / `vtq2_2s`) | 2.6 | **0.091** (87.6 %) | | |
| `f16` / `vtq4_1` | | 0.0053 | 0.043 | 0.54 |
| `f16` / `vtq2_1` | | 0.044 | 0.23 | 1.66 |
| `ktq4_1` / `f16` | | 0.023 | 0.35 | 0.48 |
| `ktq2_1` / `f16` | | 0.31 | 3.40 | 1.33 |
| `ktq4` / `f16` | | 0.0069 | | |
| `ktq2` / `f16` | | 0.063 | | |
| `q8_0` / `vtq4` ¹ | 6.4 | 0.0049 (97.0 %) | | |
| `q8_0` / `vtq3` ¹ | 5.9 | 0.010 (96.0 %) | | |
| `q8_0` / `vtq2` ¹ | 5.4 | 0.028 (92.9 %) | | |
| `q8_0` / `vtq3_1` ¹ | 6.25 | 0.013 (95.4 %) | | |
| `ktq4` / `q8_0` ¹ | 6.75 | 0.0072 (96.4 %) | | |
| `ktq3` / `q8_0` ¹ | 6.25 | 0.019 (94.3 %) | | |
| `ktq2` / `q8_0` ¹ | 5.75 | 0.065 (89.3 %) | | |
| `ktq4` / `vtq4_1` ¹ | 4.75 | 0.0090 (96.1 %) | | |
| `q4_0` / `vtq2` ¹ | 3.4 | 0.056 (90.3 %) | | |
| `ktq1` / `vtq1` (`ktq1_1` / `vtq1_1`) ¹ | 2.0 | 4.34 (18.9 %) | | |

¹ 2026-10-16, release tree, two RTX 2060, same base and flags.

- `ktq4_1`/`vtq4_1` is more accurate than `q4_0` on Qwen3-4B and equal on gpt-oss and Gemma 4, at 5 instead of 4.5 bpw.
  Before the encoder change (2026-10-10) it was 0.045 (Qwen3-4B) and 0.55 (gpt-oss).
- `q8_0` is at the noise floor of every model; `vtq4_1` V costs little on Qwen3-4B and gpt-oss.
- gpt-oss is sensitive in K: every 4-bit K type (`q4_0`, `ktq4_1`) costs about 0.35, V only 0.04; protecting outer layers
  helps only in proportion: `ktq4_1` / `vtq4_1` with `--tq-protect-layers` 2 / 4 / 8 (4 / 8 / 16 of 24 layers in `q8_0`)
  gives 0.235 / 0.180 / 0.087 (2026-10-17, two RTX 2060). The first layers matter more than the last: the first 8 layers
  alone (what `--tq-protect-layers 4` selected before the fix of 2026-10-17) gave 0.144. So use `q8_0` K with
  `vtq4_1` V there (0.039). Gemma 4 is sensitive in V
  (`vtq2_1` 1.66); it reacts to any numerical change (flash attention on vs off already 0.32) and its same-top-token share
  stays the better guide there.
- 2-bit and 3-bit codebook K lose clearly on all three models. The trellis K types with 4 fp16 outliers per block lose
  3-5 times less on Qwen3-4B: `ktq3` / `vtq3` at 3.6 bpw is as accurate as `ktq4_1` / `vtq4_1` at 5.0 bpw, `ktq4` /
  `vtq4` lies between `q8_0` / `vtq4_1` and `q4_0` / `q4_0`.
- Reproduce: base `llama-perplexity -m M -f TEXT -c 512 -b 512 --chunks 40 -fa on --kl-divergence-base base.kld`, then
  the same with `-ctk K -ctv V --no-tq-deferred-k --no-tq-deferred-v --kl-divergence-base base.kld --kl-divergence`.
  `-b 512` keeps one sequence per batch; with the default batch of 2048 four chunks share a batch and the values differ
  by a few percent (`ktq4` / `vtq4`: 0.0092 instead of 0.0087).
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
| `ktq3_1` / `vtq3_1` | 9/12, 28/36 ² | |
| `ktq2_1` / `vtq2_1` | **0/36** | 12/12 |
| `ktq2_1` / f16 | **0/12** | |
| `ktq2` / `vtq2` (`ktq2_3s` / `vtq2_2s`) ¹ | 26/36, 29/36 ³ | |
| `ktq2` / `q8_0` ¹ | 30/36, 29/36 ³ | |
| `ktq2_1` / `vtq2_1` ¹ | 0/36 | |
| `ktq2_1` / `q8_0` ¹ | 1/36 | |
| f16 / f16 ¹ | 32/36 | |
| `ktq4` / `vtq4` ² | 32/36, 32/36 ³ | |
| `ktq3` / `vtq3` ² | 30/36, 31/36 ³ | |
| `q8_0` / `vtq3` ² | 33/36, 34/36 ³ | |

¹ 2026-10-16 on two RTX 2060, same prompts. ² release tree of 2026-10-16, two RTX 2060, same prompts (f16, `q8_0`,
`ktq4_1` / `vtq4_1` and `ktq2` / `vtq2` gave the same counts as above again). ³ 2026-10-17 after the CPU/CUDA encoder
parity change (the CUDA encoder picks a different, equally good path in some blocks), two RTX 2060, same prompts.

- With 4-bit K or more every pair copies as well as f16 on Qwen3-4B; 3-bit K loses a few prompts (`ktq3` / `vtq3`
  30-31/36, `ktq3_1` / `vtq3_1` 28/36). The prompts that fail are mostly the same ones for every type (the model picks the
  neighbouring section's version number).
- 2-bit codebook K loses the retrieval completely on Qwen3-4B: the copies are fluent but wrong lines, with f16 V too, with every
  decode kernel and in the release of 2026-10-12 as well. On Qwen3.6-35B-A3B the same pair copies every section.
  The trellis K `ktq2` keeps the retrieval on Qwen3-4B (29-30/36 with `q8_0` V, 26-29/36 with `vtq2` V).
- Gemma 4 26B-A4B Q4_K_M (36 prompts, 2026-10-16): f16 30/36, `q8_0` 30/36, `ktq3_1` / `vtq3_1` 31/36, `ktq3` / `vtq3`
  29-31/36 (four runs on different builds and decode kernels, the release tree included), `ktq4` / `vtq4` 30/36,
  `ktq3` / `q8_0` 29-31/36; the failed prompts are the same few for every type.

Reproduce: [scripts/eval](../scripts/eval/README.md#long-context-retrieval).

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
| `q8_0` / `vtq4` | 8.5 | 4.25 | 6.4 | 40 % |
| `q8_0` / `vtq3` | 8.5 | 3.25 | 5.9 | 37 % |
| `ktq4_1` / `vtq4_1` | 5.5 | 4.5 | 5 | 31 % |
| `ktq4` / `vtq4` | 5 | 4.25 | 4.6 | 29 % |
| `q4_0` / `q4_0` | 4.5 | 4.5 | 4.5 | 28 % |
| `ktq3_1` / `vtq3_1` | 4.5 | 4 | 4.25 | 27 % |
| `ktq3` / `vtq3` | 4 | 3.25 | 3.6 | 23 % |
| `ktq2_1` / `vtq2_1` | 3.5 | 2.5 | 3 | 19 % |
| `ktq2` / `vtq2` | 3 | 2.25 | 2.6 | 16 % |

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
| Tensor-core decode | `fattn-tq-wmma.cu` | Tiles of 64 (D=128), 32 (D=256) or 16 (D=512) K/V rows are dequantized to f16 in shared memory; S = K·Qᵀ and O += Vᵀ·P run as 16×16×16 WMMA with 16 query columns per block (the whole GQA group); attention sinks (gpt-oss) are supported. Also takes speculative-decoding verify batches of 2-16 queries: the columns are (query, head) pairs, one block of 16, or of 32 from 8192 cells (half the tile rows; not f16 K at D=256), each column with its own mask row. | GQA ratio 2-16. Same-bit pairs `ktq{2,3,4}_1` × `vtq{2,3,4}_1` and `q4_0` at D=64/128/256/512, `q5_0` at D=128/256/512 (at D=64 only verify batches) (at 128/256 only what the GQA kernel does not take: GQA ratio 2, a single KQ tile; the GQA kernel goes first since 2026-10-14); `q8_0` and `q8_0` K with `vtq{2,4}_1` V at D=64/512; f16 K with `vtq{2,3,4}_1` V at D=64, with `vtq{2,4}_1` V at D=256/512; `ktq{2,4}_1` K with f16 V at D=256 |
| GQA decode | `fattn-vec-gqa.cu` | One block per GQA group (4-column blocks for GQA ratio 3-4, 8-column blocks above 4), one warp per K/V row, the column dot products reduced together. Each lane reads several elements per load (2-byte loads for the 4-bit codes and `q4_0`/`q5_0`), the block scale is applied once per element, 2-bit codebooks sit in registers and 3/4-bit codebooks in shared memory; the scores are stored row by row so that the V loop reads the weights of all columns with one load. Supports the sparse index lists of Qwen3.8-Flash-Next. Since 2026-10-18, with a trellis K or a 12-state-bit trellis V, 256-thread blocks share the 16 KB table (16 instead of 12 warps per SM), chosen per launch where their KV tiles still fill every SM; the outlier correction of a trellis K builds the 4 sign patterns of a lane's samples once per block and takes the KTQ signs from registers. | GQA ratio ≥ 3, no sinks; contexts of a single KQ tile go to the generic vector kernel, except pairs with a trellis K or V (since 2026-10-18: there each trellis row decoded once per Q head cost more than the split saves). Every pair of K f16, `q8_0`, `q5_0`, `q4_0`, `ktq{2,3,4}_1` and V f16, `q8_0`, `q5_0`, `q4_0`, `vtq{2,3,4}_1` at D=128/256 (since 2026-10-14; tried before the tensor-core decode kernel), except f16 / f16 (tensor-core MMA kernel). Since 2026-10-15 also the trellis V types (`vtq*_2`, `vtq*_2s`, `vtq*_3`, `vtq3_v8`) next to K f16, `q8_0` and `ktq{2,3,4}_1`: each V block is decoded once per GQA group, the 12-state-bit table (16 KB) is copied to shared memory, the 16-bit one (256 KB) is read through L2. Since 2026-10-16 also the trellis K types `ktq{2,3,4}_3s` next to V f16, `q8_0`, `q5_0`, `q4_0`, `vtq{2,3,4}_1` and `vtq{2,3,4}_2s`, and head size 512 (Gemma 4 global layers) for every pair above with a trellis K or V: 4-column blocks there (a GQA group of 8 takes two; Q and the V accumulators of 8 columns would need 256 registers, at most 255 exist), each trellis row decoded twice per group of 8 instead of once per Q head; sm_75 (ptxas): 232-255 registers, no spills, 26.8 KB static shared memory with the 12-state-bit table, 10.4 KB without it. The other pairs at D=512 stay with the tensor-core decode kernel. Not `ktq1_1`/`vtq1_1`; the trellis types not in the tensor-core decode kernel (verify batches convert V to f16, GQA ratio 1-2 and single-tile contexts use the vector kernel) |

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
reach. The backtrack pool holds at least one slot per SM and 32 MiB: 2 MiB slots at L = 16 (68 MiB on an RTX 2060 12 GB with
34 SMs), 128 KiB slots at L = 12 (32 MiB); `GGML_CUDA_VTQ_POOL_SLOTS` sets the slot count.

At L = 12 (since 2026-10-17) the readers of a DP segment sit in one warp, so the DP reads shared memory without bank
conflicts, and one warp picks the outliers of the K types. A decoded token writes a few trellis blocks per cache and each
one takes the time of one block on one SM, so the K and the V write of a layer run in one launch when their blocks fit
on the SMs together (one fused node in the CUDA graph executor; `GGML_CUDA_DISABLE_FUSION=1` turns it off with the other
fusions). The conversion after the prompt runs CUDA blocks of 256 threads, two per SM, and on a model split over
several GPUs each GPU converts its layers at the same time. CPU and CUDA encoders write bit-identical blocks: the CPU
runs the same butterfly DP with the same fused multiply-adds, and the CUDA encoder, built with `-use_fast_math`,
computes the block norm and scale with IEEE operations and in the same order as the CPU.

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
It runs in every cache type (plain, sliding-window such as gpt-oss and Gemma 4, hybrid such as Qwen3.5/3.6/3.8) as soon
as every token of a batch wants an output (decode, several slots, speculative verify), converts only the used rows in up
to 8 chunks, and when memory is short keeps decoding from the f16 staging buffer with a warning. The f16 copy of a
quantized K/V cache that flash attention takes for prompt batches is reserved at startup; on several GPUs, if it would
not fit next to the pipeline-parallel buffers, the context starts without pipeline parallelism and logs why. Restoring a
saved sequence whose staging state differs from the cache's current state fails unless the cache holds no other data.

### Boundary-layer protection

`--tq-protect-layers N` stores the first and last N attention layers of the model in q8_0 and the rest in the chosen TurboQuant types. Until 2026-10-17 hybrid models (Qwen3.5/3.6/3.8) and sliding-window models (Gemma 4, gpt-oss) got the first N layers of each cache and never the last ones; models with attention in every layer were not affected. On gpt-oss it helps only in proportion to the protected layers (see Accuracy). A V that is already f16 (sink protection, or a
`ktq*_1` type given as V) stays f16.

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
| FA dispatch                 | KTQ1_1..KTQ4_1 and KTQ2_3S..KTQ4_3S (K), VTQ1_1..VTQ4_1 (v1), VTQ2_2..VTQ4_2 and VTQ2_2S..VTQ4_2S (v2), VTQ2_3..VTQ4_3 (v3), VTQ3_V8. |
| SET_ROWS kernels            | `k_set_rows_pq` / `k_set_rows_pq_warp` (KTQ, VTQ v1) in `set-rows.cu`; trellis encoder in `trellis-encode.cuh`. |
| `head_dim` support          | KTQ codebook: 64/128/256/512. Trellis K and V (`ktq*_3s`, `vtq*_2`/`_2s`/`_3`, `vtq3_v8`) need head sizes that are multiples of 128 (128/256/512); layers with smaller heads (gpt-oss: 64) use the codebook type with the same bits. |
| CPU                         | Inverse rotation in KTQ dequantization (`kktq_rht_inverse_sb`, AVX2 path, `ggml/src/ggml-quants.c`). |
| Compute capability          | Developed on CC 7.5 (Turing, RTX 2060); tested on Ampere (RTX 3060, 3090, A40), Ada (RTX 4060 Ti, 4090) and Blackwell (RTX 5090) for decode speed, on TITAN RTX, A40, RTX 4090 and RTX 5090 with `test-backend-ops` (2026-10-16). Untested below CC 7.5; the tensor-core paths need CC 7.5+. |

## Source Files

| File                                       | Description |
|--------------------------------------------|---|
| `ggml/include/ggml.h`                      | Type enums: KTQ1_1=45, KTQ2_1=42, KTQ3_1=43, KTQ4_1=44. VTQ1_1=46, VTQ2_1=47, VTQ3_1=48, VTQ4_1=49. VTQ2_2=50, VTQ3_2=51, VTQ4_2=52. 53 reserved (was VTQ_MIXED). VTQ2_3=54, VTQ3_3=55, VTQ4_3=56. 57 reserved (was XKTQ2_1). VTQ3_V8=58. VTQ2_2S=59, VTQ3_2S=60, VTQ4_2S=61. KTQ2_3S=62, KTQ3_3S=63, KTQ4_3S=64. |
| `ggml/src/ggml-common.h`                   | Block structs: `block_ktq*_1` (with `sb[4]`), `block_vtq*_1`, `block_vtq*_2` (Trellis; `block_vtq*_2s` are typedefs of it), `block_vtq*_3` (Trellis + outliers; `block_ktq*_3s` are typedefs of it, no sign bits), `block_vtq3_v8`. |
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
| `ggml/src/ggml-quants.c`                   | CPU quantize/dequantize for KTQ + VTQ (incl. the trellis `ktq*_3s`); shared `PQ_CODEBOOK_*` constants. |
| `src/llama-kv-cache.cpp`                   | Per-layer cache types (layer and sink protection, trellis head-size fallback), deferred K/V staging, V rotation. |
| `tools/perplexity/perplexity.cpp`          | Turns deferred staging off unless `--tq-deferred-k`/`--tq-deferred-v` is given. |
| `common/arg.cpp`                           | CLI: `--cache-type-k`, `--cache-type-v` parser; accepts `ktq{1,2,3,4}_1`, `ktq{2,3,4}_3s`, `vtq{1,2,3,4}_1`, `vtq{2,3,4}_2`, `vtq{2,3,4}_2s`, `vtq{2,3,4}_3`, `vtq3_v8`; short names `ktq1`-`ktq4`, `vtq1`-`vtq4`; the `--tq-*` options. |
| `common/tq-profile.cpp`                    | Per-layer V profile and strategy for `--tq-mixed-v`. |
| `ggml/src/ggml-cpu/ggml-cpu.c`, `ops.cpp`  | CPU type traits and flash-attention reads of all TurboQuant types. |
| `docs/benchmarks/history/LIVE_NUMBERS.md` | Historical TG/PPL/HellaSwag snapshot. |

## Tried and discarded

Measured and not pursued (the history is in the [changelog](../CHANGELOG.md)):

- **Trellis-K without outliers** (TCQ-style, April 2026; again as `ktq*_2s` in 2026-10): lost to `ktq*_1` on real K
  (see [trellis KTQ](#k-cache--trellis-ktq-ktq_3s)). With 4 fp16 outliers per block it became `ktq*_3s`: flash attention
  rotates Q once per query and dots it with the decoded samples; the tensor-core decode kernel does not take it.
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

- **2- and 3-bit codebook K** (`ktq2_1`, `ktq3_1`) still lose a lot of accuracy on every measured model (see
  [Accuracy](#accuracy)); the trellis K `ktq2` / `ktq3` lose 3-5 times less on Qwen3-4B. gpt-oss needs 8-bit K.
- **Trellis types are slower** (`ktq2`-`ktq4`, `vtq2`-`vtq4`): the Viterbi encoder runs over the whole prompt and then
  for every new token, so the first prompt takes 1.5-1.9 times as long as with the codebook types, and decode is 12-14 %
  slower on an empty context, 20 % at 8k and 30-34 % at 32k, where reading the trellis cache limits it
  ([benchmarks](benchmarks/README.md#trellis-and-codebook-types-two-gpus)).
- **Mixed K/V pairs outside the decode kernels' coverage** (head sizes other than 128 and 256, except the trellis
  pairs, which the GQA kernel also takes at 512; pairs without a vector-kernel instance at GQA ratio 1-2;
  `ktq1_1`/`vtq1_1`) run on CUDA by converting K and V to f16 at every step; correct, but decode slows down with
  context. Since 2026-10-14 the GQA decode kernel takes every pair of f16, `q8_0`, `q5_0`, `q4_0` and the 2-4-bit
  TurboQuant types at head sizes 128 and 256 (before, Qwen3-4B decoded `ktq4_1` / `q8_0` at 10 t/s at 32k, now 41 t/s).
  Building with `-DGGML_CUDA_FA_ALL_QUANTS=ON` compiles vector kernels for more pairs.
- **VTQ v1 on Qwen3-Next-80B with `-b 1 -ub 1`**: reported in April 2026 to crash through the fused Gated DeltaNet path; not re-checked since. Batched mode and other models were unaffected.

## References

This implementation is inspired by but deviates from the cited papers. KTQ uses RHT (FWHT + one shared random sign pattern) + Lloyd-Max instead of QR rotation, the trellis K `ktq*_3s` the KTQ rotation over 128 values plus the 12-bit trellis of `vtq*_2s`; VTQ v1 uses a fixed D·H·D rotation (specific to this fork); v2/v3 use a Viterbi trellis over an inverse-Gaussian CDF code table. QJL was used in v1–v4 and removed in v5.

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
