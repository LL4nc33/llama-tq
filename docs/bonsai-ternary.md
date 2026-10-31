# Ternary weights with Hadamard-rotated activations (PQ2_0, PTQ1_0)

llama-tq can load models whose weights are ternary ({-1, 0, +1} times one fp16 scale per 128
weights) and stored in a rotated basis, as published for Qwen3.8-27B (64 layers, hybrid Gated
DeltaNet / attention). Two packings exist:

| type | id | packing | bits per weight | 27B file |
|---|---|---|---|---|
| `PQ2_0` | 142 | 2-bit codes 0..3 = -1, 0, +1, +2 times d | 2.13 | 7.2 GB |
| `PTQ1_0` | 143 | 5 trits per byte (as TQ1_0), group 128 | 1.75 | 5.95 GB |

The weights are folded with a blockwise Sylvester Hadamard transform (block 1024, ±1 signs per
input column). The GGUF declares it in `*.hadamard.*` metadata; the loader inserts the matching
transform of the activations in front of every folded matmul (memoised when several weights share
an input), applies the inverse after the token-embedding lookup, and handles a tied output head.
Files without that metadata are unaffected.

## Running

```sh
llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf --mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
    -ngl 99 -fa on -c 204800 -ub 256 -ctk f16 -ctv f16 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 \
    --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

The model reasons at length; give it a generous output limit. `reasoning_effort` accepts `low`,
`medium` and `xhigh` (not `high`).

## CUDA support

- Activation rotation: the CUDA fast Walsh-Hadamard transform (rows of 64 to 1024), with the
  sign flip applied inside the transform (`MUL -> RESHAPE -> MUL_MAT` fused).
- Mat-vec (1..8 columns): `PTQ1_0` has a dedicated kernel on a planar activation layout (each
  weight decoded once for all columns, trits as digits {0, 1, 2} with the -1 folded into exact
  activation sums); `PQ2_0` uses the generic mmvq path with dp4a.
- Batches above 8 columns: MMQ (int8 tensor cores), weights expanded to the Q8_0 tile layout.
- Dequantize and get_rows for both types.

## Measurements (2x RTX 2060 12 GB, Turing)

Reference numbers with the publisher's own llama.cpp build (`prism` branch), 30k-token needle
prompt, f16 KV:

| model | GPUs | context | decode short | decode at 27k | prompt |
|---|---|---|---|---|---|
| PQ2_0 | 1 | 32k | 24.4 t/s | 20.7 t/s | 350 t/s |
| PQ2_0 | 2 (layer split) | 200k | 26.3 t/s | 22.7 t/s | 356 t/s |
| PQ2_0, q8_0 KV | 2 | 200k | 25.9 t/s | 18.6 t/s | 303 t/s |
| PTQ1_0 | 1 | 32k | 32.6 t/s | 27.1 t/s | 325 t/s |
| PTQ1_0 + mmproj | 2 | 200k | 36.2 t/s | 29.6 t/s | 351 t/s |

f16 KV fits 200k on two 12 GB GPUs (10.8 / 10.9 GiB with PQ2_0) and stays faster than q8_0 at long
context.

llama-tq, same machine, same prompts:

| model | GPUs | context | decode short | decode at 27k | prompt 27k | short prompt |
|---|---|---|---|---|---|---|
| PTQ1_0 | 1 | 32k | 34.1 t/s | 29.4 t/s | 309 t/s | 132 t/s |
| PQ2_0 | 1 | 32k | 22.6 t/s | 20.6 t/s | 326 t/s | 123 t/s |
| PTQ1_0 + mmproj, `-ub 256` | 2 (layer split) | 200k | 36.3 t/s | 31.0 t/s | 307 t/s | 133 t/s |

Through the same serving script both builds generate the same text; decode at short context is
equal (36.1 vs 36.0 t/s), llama-tq is ahead at long context and on short prompts, the reference
build on long prefills (348 vs 307 t/s at 27k). A 171k-token prompt fills in 19 minutes (152 t/s
averaged over the growing context) and decodes at 16 t/s afterwards.

With 200k context and the vision projector, use `-ub 256`: at 512 the compute buffer (the KQ mask
grows with context x ubatch) leaves the vision encoder no room on the first GPU.

Decode is bound by memory bandwidth: the PTQ1_0 mat-vec reads the 5.95 GB of weights at about
290 GB/s of the RTX 2060's 336 GB/s. A presence penalty costs noticeable CPU time per token with a
248k vocabulary (about 2 t/s here).

## Tensor split (`-sm tensor`)

With tensor split each GPU holds half of every weight matrix; the rotation of the row-parallel weights
(attention output, `ssm_out`, `ffn_down`) stays local because the split keeps whole 1024-column transform
blocks on each GPU, and those weights get their own sign vector split like their input. The partial
results of the row-parallel matmuls are summed across the GPUs twice per layer.

Without peer access or NCCL that sum goes through mapped pinned host memory: each GPU writes its part,
raises a flag and adds the other part once its flag is up, without a host thread
(`GGML_CUDA_HOST_ALLREDUCE=0` disables it). `GGML_CUDA_HOST_ALLREDUCE_BF16=1` sends the parts as bf16, half
the traffic; every GPU then adds the rounded parts of all GPUs, so they stay identical.

```sh
GGML_CUDA_HOST_ALLREDUCE_BF16=1 llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    --mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf -sm tensor \
    -ngl 99 -fa on -c 204800 -ub 512 -ctk f16 -ctv f16
```

Two RTX 2060, one on a chipset x4 gen2 link (about 1.7 GB/s for both directions together), the other on
x16 gen3, same scripts and build:

| | layer split | tensor split (bf16) |
|---|---|---|
| decode, short | 34.8 t/s | 37.8 t/s |
| decode at 24k | 30.9 t/s | 31.4 t/s |
| decode at 171k | 16.0 t/s | 21.6 t/s |
| prefill 24k | 328 t/s | 269 t/s |
| fill 171k (average) | 152 t/s | 184 t/s |

The prefill of moderate prompts is limited by the slow link (a 512-token batch moves 5 MB per sum and
direction); at long context the split attention wins. VRAM at 200k with the vision projector and `-ub 512`:
10.9 / 10.0 GB. A regular Qwen3.8-27B (UD-Q4_K_M) gains more, its weights are heavier: 24.0 instead of 16.5 t/s decode,
22.0 instead of 16.2 at 17k, prefill at 17k 282 instead of 355 t/s (bf16 sums; 23.5 t/s with f32).
