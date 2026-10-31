# Benchmarks

Decode, prefill and speculative decoding speed per GPU and KV type (2026-10-09 to 2026-10-30, the date is given per
table; most RTX 2060 tables are from the tree of 2026-10-29, the cloud GPU rows from 2026-10-09 to 2026-10-16). Accuracy,
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

Same model on one RTX 2060, decode at empty / 16k / 32k context and prefill at 32k (`llama-bench -fa 1 -t 4 -n 64
-d 0,16384,32768 -r 2`, tree of 2026-10-29, pairs run forwards and then backwards, mean of both rounds):

| KV cache | bpw | decode empty / 16k / 32k | `pp512` at 32k |
|---|---:|---:|---:|
| f16 / f16 | 16 | 92.2 / 53.0 / 37.3 t/s | 762 t/s |
| `q8_0` / `q8_0` | 8.5 | 89.8 / 59.5 / 43.9 t/s | 718 t/s |
| `q8_0` / `vtq4_1` | 6.5 | 87.4 / 60.2 / 45.3 t/s | 707 t/s |
| `q8_0` / `vtq2_1` | 5.5 | 89.1 / 61.7 / 46.5 t/s | 711 t/s |
| `ktq4_1` / `vtq4_1` | 5.0 | 88.3 / 60.8 / 46.4 t/s | 734 t/s |
| `q4_0` / `q4_0` | 4.5 | 89.5 / 64.0 / 50.1 t/s | 724 t/s |
| `ktq2_1` / `vtq2_1` | 3.0 | 88.6 / 63.4 / 49.2 t/s | 733 t/s |

f16 keeps the tensor-core decode kernel and is fastest on an empty cache; from 16k on every quantized pair decodes
faster (at 32k by 18-34 %), because the GQA decode kernel of 2026-10-14 reads one K/V block for all query heads that
share it. Pairs that mix f16 with a quantized type or K and V of different families (`q8_0` / `q4_0`,
`ktq4_1` / `q8_0`, f16 / `vtq4_1`) converted the whole cache to f16 at every decode step before that change
(10-18 t/s at 32k); they now decode at 41-44 t/s at 32k (2026-10-13). Qwen3.6-35B-A3B Q4_K_M on two RTX 2060 at 32k,
same change: `q8_0` / `q8_0` 69.8 → 70.4 t/s, `ktq4_1` / `q8_0` 41.3 → 70.9 t/s.


Same model at 32k context on newer GPUs, start of the 2026-10-14 work (20b81a751) → now (`llama-bench -fa 1 -t 4
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
codebook types `ktq*_1` / `vtq*_1` keep their full names. Qwen3-4B-Instruct-2507 Q4_K_M on two RTX 2060, tree
of 2026-10-29, types run forwards and then backwards, mean of both rounds. Decode: `llama-bench -fa 1 -t 4
-n 64 -d 0,8192,32768 -r 2`. First prompt and follow-up: `llama-server -c 40960`, a fresh server for each prompt, a
prompt of 6.8k / 23.7k tokens (30000 / 96000 characters of wikitext-2), then the same prompt plus about 460 tokens on
the cached prefix; the generated text is identical across repetitions of a type:

| KV cache | bpw | decode empty / 8k / 32k | first prompt 6.8k / 23.7k | follow-up at 6.8k / 23.7k |
|---|---:|---:|---:|---:|
| `q8_0` / `q8_0` | 8.5 | 91.6 / 73.1 / 44.4 t/s | 2.6 s / 10.3 s | 1512 / 832 t/s |
| `ktq4_1` / `vtq4_1` | 5.0 | 89.6 / 71.7 / 46.4 t/s | 2.6 s / 10.3 s | 1485 / 820 t/s |
| `ktq4` / `vtq4` (`ktq4_3s` / `vtq4_2s`) | 4.6 | 78.5 / 56.9 / 30.5 t/s | 4.2 s / 15.5 s | 885 / 564 t/s |
| `ktq3_1` / `vtq3_1` | 4.25 | 89.5 / 72.0 / 44.2 t/s | 2.6 s / 10.3 s | 1506 / 827 t/s |
| `ktq3` / `vtq3` (`ktq3_3s` / `vtq3_2s`) | 3.6 | 78.1 / 56.9 / 30.6 t/s | 4.6 s / 17.0 s | 802 / 532 t/s |
| `ktq2` / `vtq2` (`ktq2_3s` / `vtq2_2s`) | 2.6 | 76.7 / 56.2 / 30.4 t/s | 4.9 s / 18.3 s ¹ | 743 / 501 t/s ¹ |

¹ 2026-10-17, not re-measured (the other trellis server cells of that day are within 2 % of the new ones).

The prompt itself is computed equally fast (about 3300 t/s at 7k); the trellis types then spend the extra time in the
Viterbi encoder, which converts the f16-staged K and V of the prompt at the first decoded token and encodes every later
token. The server counts that conversion in the timing of the first generated tokens, so its decode figure for the
first answer is misleading for these types; the wall time of the first prompt above includes everything. The first
prompt takes 1.5-1.9 times as long as with the codebook types (3-4.5 times before the encoder work of 2026-10-17), and
at 32k context decode is limited by reading the trellis cache, not by the encoder: the codebook pairs decode 44-52 %
faster there. Accuracy of these pairs: [turboquant.md](../turboquant.md#accuracy).

Gemma 4 26B-A4B IQ2_XXS (head size 512 on the global layers), same GPUs and run, decode empty / 8k / 32k: `ktq3` /
`vtq3` 57.9 / 41.9 / 34.2 t/s, `ktq3_1` / `vtq3_1` 79.7 / 66.6 / 55.0 t/s.

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

Two RTX 2060, tree of 2026-10-29, two prompts of 27.6k / 27.9k tokens, 4 requests per cell, every copy
exact: decode with speculation off → on `q8_0` / `q8_0` 48.5 → 83.3 t/s (64 % accepted), `q8_0` / `vtq4_1` 49.7 → 87.6
(67 %), `ktq4_1` / `vtq4_1` 49.0 → 85.2 (67 %); prefill 2240-2320 t/s for all three. On 2026-10-16 the same cells were
47.8 → 79.2, 48.9 → 80.8 and 46.9 → 75.3 t/s.

Reproduce: [scripts/eval](../../scripts/eval/README.md#speculative-decoding-at-long-context-one-gpu).

## MTP, one GPU

`llama-server` with MTP (`--spec-type draft-mtp`, setup in [speculative.md](../speculative.md#draft_mtp-guidance)) on one
RTX 2060, `q8_0` KV, one slot, 450 generated tokens on the three prompts of `mtp-bench.py` (German prose, new code, the
rewrite of a short function), each at temperature 0 and 1, 2 rounds per cell (the rounds agree within 0.5 t/s);
upstream at 3f03b17724 and the fork of 2026-10-29, both measured 2026-10-30. Upstream takes its own flag `--spec-draft-n-max`
and its default `--spec-draft-p-min` 0 unless noted. Decode in t/s, accepted share of the drafted tokens in parentheses.

Gemma 4 12B Q4_K_M with its MTP assistant `mtp-gemma-4-12B-it-Q8_0.gguf`, 32k context:

| Build, draft setting | prose | code | rewrite |
|---|---:|---:|---:|
| upstream, MTP off | 33.0 | 33.0 | 32.9 |
| upstream, draft max 3 | 39.2 (39 %) | 60.8 (80 %) | 54.8 (70 %) |
| upstream, draft max 7, `--spec-draft-p-min 0.75` | 42.1 (65 %) | 58.9 (75 %) | 49.2 (62 %) |
| fork, MTP off | 35.5 | 35.5 | 35.5 |
| fork, draft max 3 | 49.8 (59 %) | 70.9 (84 %) | 63.2 (72 %) |
| fork, draft max 5, `LLAMA_MTP_DECAY=1.0` | 51.2 (64 %) | 80.8 (80 %) | 65.5 (69 %) |
| fork, draft max 7, `LLAMA_MTP_DECAY=1.0` | 51.4 (62 %) | 87.1 (75 %) | 72.5 (68 %) |

Qwen3.6-35B-A3B IQ2_XXS with the MTP layer of the model, 8k context:

| Build, draft setting | prose | code | rewrite |
|---|---:|---:|---:|
| upstream, MTP off | 74.8 | 74.8 | 74.6 |
| upstream, draft max 1 | 82.0 (63 %) | 95.2 (90 %) | 90.5 (81 %) |
| upstream, draft max 3 | 66.7 (38 %) | 97.4 (71 %) | 85.3 (58 %) |
| fork, MTP off | 92.6 | 92.7 | 92.9 |
| fork, draft max 1 | 95.1 (93 %) | 111.2 (98 %) | 106.2 (97 %) |
| fork, draft max 3 | 92.9 (76 %) | 122.5 (90 %) | 113.3 (81 %) |

On the MoE model MTP gains little on prose (draft max 1: +3 %) although 93 % of the drafts are accepted. A verify
batch costs far less than decoding its tokens one by one, but the draft itself does not: the MTP layer and its output
over the full vocabulary take about 6 ms per step, half a decoded token (measured 2026-10-20). Gemma 4 12B decodes a
token in 28 ms and its assistant drafts in about 2 ms, so there longer drafts pay off on code and rewrites.

Reproduce: [scripts/eval](../../scripts/eval/README.md#mtp-one-gpu).

## Speculative decoding, two GPUs

Qwen3.6-35B-A3B Q4_K_M on two RTX 2060, `llama-server`, `q8_0` KV unless noted. MTP (`--spec-type draft-mtp`, setup in
[speculative.md](../speculative.md#draft_mtp-guidance)):

| Prompt | `--draft-max` | acceptance | decode, MTP off → on |
|---|---:|---:|---:|
| short (2026-10-30) | 1 | 95 % | 83.3 → 102.8 t/s |
| short (2026-10-30) | 3 | 83 % | 83.3 → 102.6 t/s |
| short, `ktq4_1` / `vtq4_1` (2026-10-30) | 3 | 81 % | 105.1 t/s with MTP |
| 29k tokens at 40k context (2026-10-15) | 1 | 97 % | 70.5 → 85.4 t/s (upstream MTP 74.6) |
| 29k tokens at 40k context (2026-10-15) | 3 | 93 % | 70.5 → 92.6 t/s |

With `-b 8192` (the default with MTP) the prompt costs ~6 % against no speculation instead of 16 % with 2048-token
batches (1649 t/s without speculation, 1390 t/s with MTP at 2048, 1550 t/s at 8192). Two concurrent requests (`-np 2`,
draft max 3, 2026-10-30): 136.6 t/s combined against 132.3 t/s without speculation (2026-10-15: 104 against 109; since
2026-10-25 the drafts of both requests end equally long, so their verify batch runs in one forward pass).

n-gram speculation on the same model (`--spec-type ngram-cache --draft-max 8`, 29k-token prompt, 2026-10-15): prompt
processing of the second and later requests 950 → 1560 t/s with `q8_0` and 1282 → 1539 t/s with `ktq2_1` / `vtq2_1`
after the hybrid-model fix, decode 80.4 → 82.9 t/s, same text; wall time of a request 35.6 → 23.4 s (23.0 s without
speculation).

Qwen3.8-27B UD-Q4_K_M (dense hybrid, MTP layer in the model) on two RTX 2060, `q8_0` KV, 8k context, one slot, 300
generated tokens on the prompts of `mtp-bench.py` (setup as in [MTP, one GPU](#mtp-one-gpu), 2 rounds per cell, the
rounds agree within 0.2 t/s; upstream at 3f03b17724 and the fork of 2026-10-29, both measured that day). Layer split (the
default) unless the row says `-sm tensor`; the tensor rows are from a build without NCCL with
`GGML_CUDA_HOST_ALLREDUCE_BF16=1` ([tensor split](../tensor-split.md)):

| Build, draft setting | prose | code | rewrite |
|---|---:|---:|---:|
| upstream, MTP off | 16.4 | 16.5 | 16.5 |
| upstream, draft max 1 | 23.6 (69 %) | 27.1 (93 %) | 25.9 (85 %) |
| upstream, draft max 3 | 20.9 (40 %) | 32.5 (81 %) | 28.0 (65 %) |
| fork, MTP off | 17.2 | 17.2 | 17.2 |
| fork, draft max 3, `LLAMA_MTP_DECAY=1.0` | 23.8 (81 %) | 34.2 (94 %) | 29.9 (88 %) |
| fork, draft max 5, `LLAMA_MTP_DECAY=1.0` | 24.3 (80 %) | 38.5 (88 %) | 31.9 (78 %) |
| fork `-sm tensor`, MTP off | 27.9 | 27.8 | 27.6 |
| fork `-sm tensor`, draft max 3, `LLAMA_MTP_DECAY=1.0` | 37.2 (80 %) | 52.6 (93 %) | 46.3 (89 %) |
| fork `-sm tensor`, draft max 5, `LLAMA_MTP_DECAY=1.0` | 35.5 (78 %) | 57.7 (88 %) | 48.2 (83 %) |

Two concurrent requests in the same `-sm tensor` setup (98k context, `-np 2 -kvu`, draft max 5, a prose and a code
prompt at temperature 0): 52-57 t/s in total, 43 t/s before the drafts of the two requests were made equally long
(2026-10-25; a hybrid model otherwise verifies a batch with unequal sequence lengths in one forward pass per length).

With `-sm tensor` the verify batches cost little (1 token 41 ms, 4 tokens 59 ms; layer split 61 and 83 ms). With MTP
the batch size changes at most steps; since 2026-10-22 each size up to 16 tokens keeps its graph (in the compute
buffers of the main graph, no extra VRAM) instead of being built again, which with `-sm tensor` cost about 8 ms per
step (prose with draft max 5 30.5 → 34.5 t/s; layer split: under 2 ms per rebuild, 22.7 → 22.9). Since 2026-10-22 the
logits also cross to the host as the best candidates of each device's part of the vocabulary instead of whole rows, which
GPU 0 (slow slot, 1.6 GB/s to the host) took about 0.3 ms for per draft token: 34.5 / 50.4 / 41.7 → 35.6 / 52.9 / 43.4.

Reproduce: [scripts/eval](../../scripts/eval/README.md#speculative-decoding-two-gpus).

## Against upstream, two GPUs

2× RTX 2060 (pipeline parallelism), `llama-bench -fa 1 -t 4 -p 2048 -n 128 -b 2048 -ub 512 -d 0,32768 -r 1`,
2026-10-29; upstream llama.cpp at 3f03b17724 (2026-10-24) against this fork (tree of 2026-10-29). Prefill is `pp2048`, decode
`tg128`, "32k" means after 32768 tokens of context, all numbers in t/s, each cell upstream → llama-tq. Each model ran in
the order upstream, llama-tq, llama-tq, upstream, and each cell is the mean of its two runs: on these GPUs the first
build of a round is 2-3 % faster.

| Model | prefill f16 | prefill f16 32k | decode `q8_0` | decode f16 32k | decode `q8_0` 32k |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Q4_K_M | 3572 → 3645 | 1184 → 1190 | 90.7 → 91.8 | 37.6 → 37.8 | 20.3 → 44.3 |
| gpt-oss-20b MXFP4 | 2636 → 2666 | 1763 → 1789 | 78.0 → 78.8 | 65.5 → 64.9 | 43.7 → 56.1 |
| Gemma 4 12B Q4_K_M | 1315 → 1320 | 844 → 838 | 34.0 → 34.3 | 32.3 → 32.1 | 28.2 → 29.7 |
| Qwen3.6-35B-A3B Q4_K_M | 1839 → 1849 | 1314 → 1315 | 79.6 → 85.4 | 69.1 → 73.1 | 47.4 → 73.5 |
| Qwen3.8-27B Q4_K_M | 556 → 642 | 432 → 447 | 16.5 → 17.0 | 15.0 → 15.4 | 12.5 → 15.5 |

Decode with a quantized KV cache at long context is where the fork is furthest ahead (its flash attention decode
kernels read `q8_0` and the TurboQuant types directly): 1.24-2.2 times upstream at 32k. The hybrid models (Qwen3.6,
Qwen3.8) also decode faster on an empty cache (+3-7 %, recurrent states updated in place, matrix-vector kernels chosen
by shape) and Qwen3.8-27B prefills 16 % faster. Every other cell is within 1 % of upstream (Gemma 4 prefill at 32k
-0.7 %, gpt-oss f16 decode at 32k -1 %; run-to-run spread on these GPUs is about 1 %).

The fork's TurboQuant rows at 32k (same settings and run, one run per cell):

| Model | `q8_0` / `vtq4_1` decode 32k | `ktq4_1` / `vtq4_1` decode 32k | `ktq4_1` / `vtq4_1` prefill 32k | `ktq3` / `vtq3` decode 32k |
|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Q4_K_M | 45.6 | 46.5 | 1118 | 30.8 |
| gpt-oss-20b MXFP4 | 57.6 | 59.7 | 1747 | 56.3 ¹ |
| Gemma 4 12B Q4_K_M | 27.5 | 29.8 | 818 | 18.3 |
| Qwen3.6-35B-A3B Q4_K_M | 73.5 | 74.1 | 1316 | 44.1 |
| Qwen3.8-27B Q4_K_M | 15.5 | 15.4 | 431 | 11.1 |

¹ head size 64: the trellis types fall back to `ktq3_1` / `vtq3_1` there.

gpt-oss (attention sinks) is not covered by the GQA kernel. On Gemma 4 (mostly sliding-window layers), Qwen3.6-35B-A3B
and Qwen3.8-27B (hybrid, few attention layers) attention over the cache is a small part of a decode step.

Accuracy against upstream, same day and GPUs: KL divergence of each build against an f32 reference of the same model
(this fork with every matrix multiplication in f32 through cuBLAS), wikitext-2, 8 chunks of 512 tokens, upstream →
llama-tq. Prompt batches of 512 tokens: Qwen3.8-27B Q4_K_M f16 KV 0.0022 → 0.00042, `q8_0` KV 0.0022 → 0.00051; Gemma 4
12B Q4_K_M f16 KV 0.250 → 0.179, `q8_0` KV 0.301 → 0.260. One token per batch (the decode path), `q8_0` KV: Qwen3.8-27B
0.00048 → 0.00048, Gemma 4 12B 0.158 → 0.174 (±0.016 each, within the error of 8 chunks). The fork is as accurate as
upstream or closer to the reference; its speed does not come from cheaper arithmetic.

KTQ K is staged in f16 during prefill and converted at the first decode step (about 0.1 s after a 16k prompt on one
RTX 2060). `llama-bench` runs that conversion before it times a generation test; before 2026-10-13 it was counted in
the decode numbers.

Reproduce: [scripts/eval](../../scripts/eval/README.md#against-upstream-two-gpus).

## Accuracy and fine-tuning

KV cache accuracy per type and model: [turboquant.md#accuracy](../turboquant.md#accuracy). Fine-tuning results and the
comparison with PyTorch QLoRA: [finetune.md#results](../finetune.md#results). Command and result per model:
[models.md](../models.md).
