# Benchmarks

Decode, prefill and speculative decoding speed per GPU and KV type (2026-10-09 to 2026-10-16, the date is given per
table). Accuracy,
fine-tuning and per-model setups each have their own page (links at the end), so every number lives in one place. Older logs are in [history/](history/); their numbers
are not comparable with these.

## Decode speed, one GPU

Qwen3-4B-Instruct-2507 Q4_K_M, `llama-bench -fa 1 -n 128`, empty context (RTX 2060 row 2026-10-12, RTX 5090, RTX 4090
and RTX A40 2026-10-16 on the release tree, the other rows 2026-10-10 unless marked):

| GPU | f16 KV | `q8_0` KV | `ktq2_1` / `vtq2_1` KV |
|---|---:|---:|---:|
| RTX 5090 | 332 t/s | 298 t/s | 320 t/s |
| RTX 4090 | 261 t/s | 244 t/s | 239 t/s |
| RTX A40 | 175 t/s | 164 t/s | 159 t/s |
| RTX 3090 | 203 t/s | | 176 t/s ¹ |
| RTX 3060 | 103 t/s | | 96 t/s ¹ |
| RTX 4060 Ti | 95 t/s | | 90 t/s ¹ |
| RTX 2060 | 92 t/s | 89 t/s | 88 t/s |

¹ measured 2026-10-09, before the K/V rotations ran as the fast Walsh-Hadamard transform (RTX 5090 then: 272 t/s, RTX
4090 217 t/s).
The encoder scale search of 2026-10-11 costs 1–2 % of the TurboQuant numbers in the rows measured before it.

Same model on one RTX 2060, decode at empty / 16k context (`llama-bench -n 64 -r 2`, 2026-10-12; the `ktq4_1` row
2026-10-13):

| KV cache | empty | 16k |
|---|---:|---:|
| `q8_0` / `q8_0` | 88 t/s | 57 t/s |
| `q8_0` / `vtq4_1` | 86 t/s | 57 t/s |
| `q8_0` / `vtq2_1` | 88 t/s | 58 t/s |
| `ktq4_1` / `vtq4_1` | 87 t/s | 54 t/s |
| f16 / f16 | 91 t/s | 53 t/s |
| `q4_0` / `q4_0` | 88 t/s | 47 t/s |

Same model on one RTX 2060 at 32k context, before → after the GQA decode kernel change of 2026-10-14
(`llama-bench -fa 1 -t 4 -n 64 -d 32768 -r 1`, builds alternated, mean of two runs per build):

| KV cache | decode 32k | `pp512` at 32k |
|---|---:|---:|
| f16 / f16 | 37.2 t/s (tensor-core kernel, unchanged) | 753 t/s |
| `q8_0` / `q8_0` | 41.7 → 43.8 t/s | |
| `q8_0` / `vtq4_1` | 41.5 → 45.3 t/s | 669 → 705 t/s |
| `ktq4_1` / `vtq4_1` | 38.9 → 46.0 t/s | 705 → 727 t/s |
| `ktq2_1` / `vtq2_1` | 40.7 → 49.0 t/s | 724 → 730 t/s |
| `q8_0` / `q4_0` ¹ | 18.4 → 43.9 t/s | |
| `ktq4_1` / `q8_0` ¹ | 10.3 → 41.4 t/s | |
| f16 / `vtq4_1` ¹ | 13.0 → 41.4 t/s | |

¹ pairs that converted the whole cache to f16 at every decode step before; "after" measured with an earlier version
of the kernel on 2026-10-13 (the final version is at least as fast on every pair measured with both).
Qwen3.6-35B-A3B Q4_K_M on two RTX 2060 at 32k: `q8_0` / `q8_0` 69.8 → 70.4 t/s, `ktq4_1` / `q8_0` 41.3 → 70.9 t/s.


Same model at 32k context on newer GPUs, start of the 2026-10-14 work (34c14bd639) → now (`llama-bench -fa 1 -t 4
-r 2 -d 32768`, builds alternated, 2026-10-14; "verify n" is a batch of n queries as speculative decoding sends it):

| KV cache | GPU | decode | verify 2 | verify 4 | verify 8 | verify 16 | `pp512` |
|---|---|---:|---:|---:|---:|---:|---:|
| `ktq4_1` / `vtq4_1` ² | RTX A6000 | 82.5 → 99.5 | 52 → 139 | 59 → 255 | 238 → 339 | 513 → 644 | 2430 → 2514 |
| `ktq4_1` / `vtq4_1` ² | RTX 4090 | 141.7 → 158.3 | 107 → 202 | 132 → 404 | 416 → 556 | 836 → 1003 | 5737 → 5917 |
| `q8_0` / `vtq4_1` | RTX A6000 | 86.3 → 93.3 | 50 → 136 | 56 → 254 | 179 → 322 | 381 → 513 | 2267 → 2399 |
| `q8_0` / `vtq4_1` | RTX 4090 | 142.0 → 149.7 | 102 → 216 | 126 → 437 | 322 → 576 | 647 → 781 | 5335 → 5581 |
| `q8_0` / `q8_0` | RTX A6000 | 90.2 → 93.4 | 72 → 134 | 136 → 246 | 234 → 311 | 507 (=) | |
| `q8_0` / `q8_0` | RTX 4090 | 137.6 → 140.0 | 171 → 207 | 212 → 412 | 379 → 559 | 760 (=) | |
| `q4_0` / `q4_0` | RTX A6000 | 90.5 → 101.3 | | | | | |
| `q4_0` / `q4_0` | RTX 4090 | 159.9 → 166.2 | | | | | |
| f16 / f16 | RTX A6000 | 81.2 (=) | | | | | 2606 → 2609 |
| f16 / f16 | RTX 4090 | 113.7 (=) | | | | | 6265 → 6160 ¹ |

All in t/s. Decode on an empty context is unchanged on both GPUs (within ±0.5 %). ¹ f16 code path unchanged; the two
base runs alone spread 1.8 % (6209, 6320), the RTX A6000 and RTX 2060 show no change. ² Verify and `pp512` cells:
`llama-bench -p` keeps a KTQ K cache in f16 staging until the first decode step, so they show f16 K with `vtq4_1` V;
the decode cells run on the quantized K.

### Release check on newer GPUs

The release tree of 2026-10-16 (one CUDA 12.8 build for sm_75/80/86/89/120a) on one cloud GPU each, same model,
`llama-bench -fa 1 -t 4 -p 512 -n 128 -d 0,32768 -r 2`, the pairs run forwards and then backwards and the two rounds
averaged (RTX 5090: one round). Decode at empty / 32k context in t/s:

| KV cache | bpw | RTX A40 | RTX 4090 | RTX 5090 |
|---|---:|---:|---:|---:|
| f16 / f16 | 16 | 175 / 77 | 261 / 112 | 332 / 161 |
| `q8_0` / `q8_0` | 8.5 | 164 / 92 | 244 / 138 | 298 / 189 |
| `q8_0` / `vtq4_1` | 6.5 | 155 / 93 | 232 / 148 | 283 / 188 |
| `ktq4_1` / `vtq4_1` | 5.0 | 158 / 99 | 236 / 157 | 300 / 186 |
| `q4_0` / `q4_0` | 4.5 | 162 / 101 | 240 / 165 | 306 / 210 |
| `ktq3_1` / `vtq3_1` | 4.25 | 157 / 96 | 235 / 158 | 290 / 189 |
| `ktq2_1` / `vtq2_1` | 3.0 | 159 / 98 | 239 / 163 | 320 / 212 |
| `ktq4` / `vtq4` | 4.6 | 79 / 51 | 123 / 90 | 142 / 107 |
| `ktq3` / `vtq3` | 3.6 | 88 / 54 | 139 / 97 | 164 / 122 |
| `ktq2` / `vtq2` | 2.6 | 86 / 55 | 138 / 98 | 166 / 120 |

At 32k context the codebook pairs and `q4_0` decode 16-46 % faster than f16 (`q8_0` 17-23 %); the trellis pairs
(encoder per token) stay 35-45 % below the codebook pairs and below f16, as on the RTX 2060. Verify batches of 16 queries at 32k
(speculative decoding): `ktq3` / `vtq3` 706 / 1151 / 1413 t/s, `ktq4_1` / `vtq4_1` 574 / 979 / 1290, `q8_0` / `q8_0` 463 /
741 / 992 (A40 / 4090 / 5090). In the same run `test-backend-ops` passed every TurboQuant `SET_ROWS`, `GET_ROWS` and
flash-attention case it reached on all three GPUs and on a TITAN RTX (Turing), and the KL divergence of every pair
matched the two RTX 2060 within its error ([turboquant.md](../turboquant.md#accuracy)).

Reproduce: [scripts/eval](../../scripts/eval/README.md#release-check-on-newer-gpus).

## Trellis and codebook types, two GPUs

Since 2026-10-16 the short names `ktq2`-`ktq4` and `vtq2`-`vtq4` select the trellis types (`ktq*_3s`, `vtq*_2s`); the
codebook types `ktq*_1` / `vtq*_1` keep their full names. Qwen3-4B-Instruct-2507 Q4_K_M on two RTX 2060, builds and
types alternated A,B,B,A. Decode (2026-10-16): `llama-bench -fa 1 -t 4 -n 64 -r 2`. First prompt and follow-up
(release tree, 2026-10-17): `llama-server -c 40960`, a fresh server for each prompt, a prompt of 6.8k / 23.7k tokens
(30000 / 96000 characters of wikitext-2), then the same prompt plus about 460 tokens on the cached prefix; the generated
text is identical across repetitions of a type:

| KV cache | bpw | decode empty / 8k / 32k | first prompt 6.8k / 23.7k | follow-up at 6.8k / 23.7k |
|---|---:|---:|---:|---:|
| `q8_0` / `q8_0` | 8.5 | 90 / 72 / 44 t/s | 2.6 s / 10.3 s | 1520 / 842 t/s |
| `ktq4_1` / `vtq4_1` | 5.0 | 89 / 72 / 46 t/s | 2.6 s / 10.4 s | 1508 / 834 t/s |
| `ktq4` / `vtq4` (`ktq4_3s` / `vtq4_2s`) | 4.6 | 59 / 47 / 27 t/s | 11.7 s / 41.2 s | 507 / 387 t/s |
| `ktq3_1` / `vtq3_1` | 4.25 | 89 / 72 / 44 t/s | 2.6 s / 10.4 s | 1512 / 838 t/s |
| `ktq3` / `vtq3` (`ktq3_3s` / `vtq3_2s`) | 3.6 | 63 / 49 / 27 t/s | 9.0 s / 32.1 s | 631 / 456 t/s |

The decode columns were re-measured on the release tree of 2026-10-16 (same GPUs and settings): every cell within 2 %.
The first version of the server columns (2026-10-16) sent the long prompt to the same server right after the short one,
so about 7k of its tokens came from the cache (`ktq4` / `vtq4`: 28.4 s for 21.8k new tokens on a 29k context); the
table now starts a fresh server for each prompt. The prompt itself is computed equally fast (about 3300 t/s at 7k); the trellis types then spend the extra time in the
Viterbi encoder, which converts the f16-staged K and V of the prompt and encodes every later token. Accuracy of these
pairs: [turboquant.md](../turboquant.md#accuracy).
Gemma 4 26B-A4B IQ2_XXS (head size 512 on the global layers), same GPUs, decode empty / 8k / 32k: `ktq3` / `vtq3`
50 / 39 / 33 t/s, `ktq3_1` / `vtq3_1` 79 / 67 / 54 t/s.

Reproduce: [scripts/eval](../../scripts/eval/README.md#trellis-and-codebook-types-two-gpus).

## Speculative decoding at long context, one GPU

`llama-server` with ngram speculation (`--spec-type ngram-cache --draft-max 8 --draft-min 4`), Qwen3-4B-Instruct-2507
Q4_K_M on one RTX 2060, a 28,883-token prompt (a synthetic changelog: copy one section verbatim, then summarize it),
greedy, 345-400 generated tokens, release of 2026-10-12 → 2026-10-14 (builds alternated, 4 requests per cell,
2026-10-14):

| KV cache | decode, speculation off | decode, speculation on | accepted drafts |
|---|---:|---:|---:|
| `q8_0` / `q8_0` | 43.8 → 46.0 t/s | 74.8 → 78.8 t/s | 77.6 % |
| `q8_0` / `vtq4_1` | 43.6 → 47.3 t/s | 42.9 → 79.3 t/s | 75.5 % |
| `ktq4_1` / `vtq4_1` | 35.8 → 46.8 t/s | 31.6 → 74.4 t/s | 72.2 % → 68.2 % |

With a VTQ V cache, speculation gained nothing before (every verify step converted the whole cache to f16); see the
CHANGELOG of 2026-10-14 for the verify batches. With `q8_0` and `q8_0` / `vtq4_1` the text with speculation is
byte-identical to the text without. On this prompt the model picks the right version digit of the copied heading
with only 59 % probability even with an f16 cache, so `ktq4_1` / `vtq4_1` copies the neighbouring section in both
builds; copy accuracy over many prompts: [TurboQuant, long-context retrieval](../turboquant.md#long-context-retrieval).

Release tree of 2026-10-16, two RTX 2060, two prompts of 27.6k / 27.9k tokens, 4 requests per cell, every copy
exact: decode with speculation off → on `q8_0` / `q8_0` 47.8 → 79.2 t/s (64 % accepted), `q8_0` / `vtq4_1` 48.9 → 80.8,
`ktq4_1` / `vtq4_1` 46.9 → 75.3, `ktq3` / `vtq3` 8.8 → 9.5. The trellis pair spends most of a request in the encoder
over the prompt (49 s per request against 19 s), which speculation does not shorten.

Reproduce: [scripts/eval](../../scripts/eval/README.md#speculative-decoding-at-long-context-one-gpu).

## Speculative decoding, two GPUs

Qwen3.6-35B-A3B Q4_K_M on two RTX 2060, `llama-server`, `q8_0` KV unless noted. MTP (`--spec-type draft-mtp`, setup in
[speculative.md](../speculative.md#draft_mtp-guidance)):

| Prompt | `--draft-max` | acceptance | decode, MTP off → on |
|---|---:|---:|---:|
| short (2026-10-15) | 1 | 86 % (upstream 86 %) | 79.5 → 93 t/s |
| short (release tree, 2026-10-16) | 3 | 80 % | 79.4 → 80.4-81.9 t/s |
| short, `ktq4_1` / `vtq4_1` (release tree, 2026-10-16) | 3 | 86 % | 80.5 t/s with MTP |
| 29k tokens at 40k context (2026-10-15) | 1 | 97 % | 70.5 → 85.4 t/s (upstream MTP 74.6) |
| 29k tokens at 40k context (2026-10-15) | 3 | 93 % | 70.5 → 92.6 t/s |

With `-b 8192` (the default with MTP) the prompt costs ~6 % against no speculation instead of 16 % with 2048-token
batches (1649 t/s without speculation, 1390 t/s with MTP at 2048, 1550 t/s at 8192). Two concurrent requests (`-np 2`,
draft max 3): 104 t/s combined against 109 t/s without speculation.

n-gram speculation on the same model (`--spec-type ngram-cache --draft-max 8`, 29k-token prompt, 2026-10-15): prompt
processing of the second and later requests 950 → 1560 t/s with `q8_0` and 1282 → 1539 t/s with `ktq2_1` / `vtq2_1`
after the hybrid-model fix, decode 80.4 → 82.9 t/s, same text; wall time of a request 35.6 → 23.4 s (23.0 s without
speculation).

Reproduce: [scripts/eval](../../scripts/eval/README.md#speculative-decoding-two-gpus).

## Against upstream, two GPUs

2× RTX 2060 (pipeline parallelism), `llama-bench -fa 1 -t 4 -p 2048 -n 128 -b 2048 -ub 512 -d 0,32768 -r 1`,
2026-10-16 (release tree; the run of 2026-10-13 gave the same cells within 2 %); upstream llama.cpp at 23b0202 against
this fork. Prefill is `pp2048`, decode `tg128`, "32k" means after 32768
tokens of context, all numbers in t/s, each cell upstream → llama-tq. Each model ran in the order upstream, llama-tq,
llama-tq, upstream, and each cell is the mean of its two runs: on these GPUs the first build of a round is 2-3 % faster.

| Model | prefill f16 | prefill f16 32k | decode `q8_0` | decode f16 32k | decode `q8_0` 32k |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Q4_K_M | 3510 → 3570 | 1159 → 1166 | 88.6 → 89.9 | 37.3 → 37.4 | 19.9 → 43.7 |
| gpt-oss-20b MXFP4 | 2616 → 2652 | 1757 → 1774 | 76.9 → 78.4 | 65.2 → 65.2 | 43.4 → 56.2 |
| Gemma 4 12B Q4_K_M | 1305 → 1307 | 838 → 821 | 33.6 → 34.0 | 32.0 → 31.9 | 28.0 → 29.4 |
| Qwen3.6-35B-A3B Q4_K_M | 1879 → 1881 | 1281 → 1278 | 77.6 → 80.8 | 67.6 → 69.5 | 46.0 → 69.4 |
| Qwen3.8-27B Q4_K_M | 635 → 663 | 424 → 427 | 16.5 → 16.7 | 14.9 → 15.0 | 12.4 → 15.1 |

Decode with a quantized KV cache at long context is where the fork is ahead (its flash attention decode kernels read
`q8_0` and the TurboQuant types directly). Prefill with an f16 cache is within 1 % of upstream or ahead, except on Gemma 4 at 32k context,
which is 2 % behind; the cause is not known yet.

The fork's TurboQuant rows at 32k (same settings and run, one run per cell):

| Model | `q8_0` / `vtq4_1` decode 32k | `ktq4_1` / `vtq4_1` decode 32k | `ktq4_1` / `vtq4_1` prefill 32k | `ktq3` / `vtq3` decode 32k |
|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Q4_K_M | 45.0 | 46.0 | 1094 | 27.4 |
| gpt-oss-20b MXFP4 | 57.8 | 59.6 | 1728 | 56.3 ¹ |
| Gemma 4 12B Q4_K_M | 27.2 | 29.5 | 802 | 17.0 |
| Qwen3.6-35B-A3B Q4_K_M | 69.3 | 69.7 | 1265 | 56.5 |
| Qwen3.8-27B Q4_K_M | 15.2 | 15.2 | 422 | 13.4 |

¹ head size 64: the trellis types fall back to `ktq3_1` / `vtq3_1` there.

gpt-oss (attention sinks) is not covered by the GQA kernel. On Gemma 4 (mostly sliding-window layers), Qwen3.6-35B-A3B
and Qwen3.8-27B (hybrid, few attention layers) attention over the cache is a small part of a decode step, so the
change stays within noise there.

KTQ K is staged in f16 during prefill and converted at the first decode step (about 0.1 s after a 16k prompt on one
RTX 2060). `llama-bench` runs that conversion before it times a generation test; before 2026-10-13 it was counted in
the decode numbers.

Reproduce: [scripts/eval](../../scripts/eval/README.md#against-upstream-two-gpus).

## Accuracy and fine-tuning

KV cache accuracy per type and model: [turboquant.md#accuracy](../turboquant.md#accuracy). Fine-tuning results and the
comparison with PyTorch QLoRA: [finetune.md#results](../finetune.md#results). Command and result per model:
[models.md](../models.md).
