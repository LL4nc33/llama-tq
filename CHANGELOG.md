# Changelog

## 2026-10-17

- **Release check of the tree of 2026-10-16.** One CUDA 12.8 build for sm_75/80/86/89/120a, tested on two RTX 2060
  and on one TITAN RTX, RTX A40, RTX 4090 and RTX 5090 (cloud): `test-backend-ops` passed every TurboQuant `SET_ROWS`,
  `GET_ROWS` and flash-attention case it ran (0 failures; the large flash-attention groups ran until their time
  limit because of the slow CPU reference for the trellis types). The KL divergence of every K/V pair on Qwen3-4B
  matches across the four GPU generations within ±6 %, the 36-prompt retrieval test gives the earlier counts
  (`ktq4` / `vtq4` 32/36, `ktq3` / `vtq3` 30/36, `q8_0` / `vtq3` 33/36, f16 32/36), and on the RTX 2060 decode,
  prefill and the comparison with upstream are within 2 % of the earlier runs (RTX 4090 at 32k context within 2 % of
  2026-10-14; RTX 5090 on an empty context 6-9 % faster with `q8_0` and `ktq2_1` / `vtq2_1` than on 2026-10-10). New tables:
  [release check on newer GPUs](docs/benchmarks/README.md#release-check-on-newer-gpus), more mixed pairs in
  [TurboQuant accuracy](docs/turboquant.md#accuracy).
- Fixed: `--tq-protect-layers N` never protected the last N layers of hybrid models (Qwen3.5/3.6/3.8: it compared the
  position among the attention layers with the count of all layers) and of sliding-window models (Gemma 4, gpt-oss:
  each of the two caches protected its own first N layers instead). Both options now count the attention layers of the
  whole model; models with attention in every layer (Qwen3-4B, K2-Horizon) keep the same layers. The gpt-oss numbers
  with layer protection in [TurboQuant accuracy](docs/turboquant.md#accuracy) predate the fix.
- Fixed: with `--tq-protect-sinks` and `--tq-protect-layers` together, the layer protection stored the V cache of the
  first attention layer in `q8_0` instead of the f16 of the sink protection; an f16 V now stays f16.
- Fixed: the bit budget of the experimental per-layer V types (`--tq-mixed-v --tq-v-budget-bpw`) counted the trellis
  V types as 8.5 / 12.5 / 16.5 bits per value instead of 2.25 / 3.25 / 4.25; the bits now come from the type traits.
- Help texts of `llama-finetune` (`--checkpoint-every` names `--resume`, the `--lora-train-target` example matches
  `attn_output`, `--train-skip-regex` is ignored with a LoRA target) and of the deferred K/V options corrected.
- Documentation checked against the code: every fork option and environment variable is now described on its page
  (per-layer V types, sliding-window cache types, `GGML_CUDA_P2P`, `GGML_CUDA_VTQ_POOL_SLOTS`, `--tensor-read-lazy`,
  `--mmap-huge`, the MTP debug variables), the trellis K types in every TurboQuant section, MTP speed on the
  [benchmarks page](docs/benchmarks/README.md#speculative-decoding-two-gpus), fine-tuning limits (data format errors,
  activations without backward, what `llama-export-lora` writes), Blackwell with CUDA 12.8, and the roadmap with
  current plans and known issues.

## 2026-10-16

- **The short KV cache names select the trellis types.** `ktq2`, `ktq3`, `ktq4` are now `ktq{2,3,4}_3s` and `vtq2`,
  `vtq3`, `vtq4` are `vtq{2,3,4}_2s`, the most accurate type of each bit width; `ktq1` / `vtq1` stay `ktq1_1` /
  `vtq1_1`, and the codebook types keep their full names (`ktq4_1`, …). Qwen3-4B, KLD against f16: `ktq4`/`vtq4` 0.0087
  (`ktq4_1`/`vtq4_1` 0.025), `ktq3`/`vtq3` 0.025 (0.084), `ktq2`/`vtq2` 0.095 (0.40); exact copies of 36 sections from
  a 28k-token prompt: `ktq2`/`vtq2` 26/36 (`ktq2_1`/`vtq2_1` 0/36), `ktq2`/`q8_0` 30/36, f16 32/36. On two RTX 2060 the
  trellis pairs decode 30-40 % slower (`ktq3`/`vtq3` 63 / 49 / 27 t/s at empty / 8k / 32k context against 89 / 72 / 44)
  and the first 7k-token prompt takes 9.1 s instead of 2.6 s.
- **The trellis K types `ktq{2,3,4}_3s` run on CUDA.** They were CPU-only. The K cache write uses the shared-memory
  Viterbi encoder of `vtq*_2s` with the rotation after the outlier pick (KTQ signs and a 128-point FWHT across 128
  threads, with the operands and stage order of the CPU encoder); the deferred K conversion after the prompt takes the
  same path. Flash attention rotates Q once per query like the K blocks and dots it with the decoded samples; since the
  dequantized row has the stored outlier values at the 4 outlier positions, the readers subtract q·x at those positions
  (x computed from the samples a lane already decoded) and add q·value. Decode uses the GQA kernel (GQA ratio 3 and
  more, longer than one KQ tile) at head sizes 128 and 256 next to V f16, `q8_0`, `q5_0`, `q4_0`, `vtq{2,3,4}_1` and
  `vtq{2,3,4}_2s`, otherwise the vector kernel, which also takes `vtq1_1` V and head size 512 (Gemma 4 global layers);
  other V types convert K and V to f16 every step, verify and prompt batches convert K to f16 for the tensor-core
  kernel. `GET_ROWS` and the f16/f32 conversions dequantize them as well.
  Qwen3-4B on two RTX 2060, KLD against f16 over 40 chunks: `ktq3_3s` K 0.0188 with `q8_0` V (`ktq3_1` 0.0715),
  0.0215 with `q4_0` V (0.0715), 0.253 with `vtq1_1` V (0.365); `ktq4_3s`/`q8_0` 0.0072. Decode with `q8_0` V at 8k
  context 29.2 → 54.5 t/s against the f16 conversion every step.
- **The GQA decode kernel takes head size 512** (Gemma 4 global layers) for the pairs with a trellis K or V, which
  the vector kernel decoded once per Q head: blocks of 4 columns there (a GQA group of 8 takes two), sm_75 ptxas 232-255
  registers and no spills. The other pairs at head 512 keep the tensor-core decode kernel.
  Gemma 4 26B-A4B IQ2_XXS on two RTX 2060, decode at 8k / 32k context: `ktq3_3s`/`vtq3_2s` 24.0 → 39.2 / 10.2 → 32.6
  t/s, `ktq3_1`/`vtq3_2s` 36.6 → 54.2 / 16.9 → 47.8, `ktq3_3s`/`q8_0` 31.0 → 46.2 / 14.3 → 37.6 (`ktq3_1`/`vtq3_1`
  66.7 / 54.2, unchanged). Gemma 4 26B-A4B Q4_K_M copies 36 sections from a 29k-token prompt as often verbatim as
  through the vector kernel: 28-31 of 36 on either kernel across four trellis K/V pairs, f16 30.

## 2026-10-15

- **Trellis V types decode at long context 3-5 times faster.** `vtq*_2`, `vtq*_2s`, `vtq*_3` and `vtq3_v8` only ran
  through the vector flash-attention kernel, which decoded every V block once per Q head of a GQA group with three byte
  loads per value. The GQA decode kernel now reads them next to K f16, `q8_0` and `ktq{2,3,4}_1` at head sizes 128 and
  256: each block once per group, one aligned load for the four state windows of a lane, the 12-state-bit table (16 KB)
  in shared memory and the 16-bit one through L2; decoded values are unchanged. Qwen3-4B on two RTX 2060, `q8_0` K,
  decode at 8k / 32k context: `vtq3_2s` 24.4 → 62.5 / 7.3 → 39.2 t/s, `vtq2_2s` 24.7 → 61.8 / 7.6 → 38.9, `vtq3_2` 12.1
  → 20.7 / 4.3 → 12.7 (`vtq3_1` 72.5 / 45.4, unchanged). Speculative verify batches, GQA ratio 1-2 and single-tile
  contexts keep the previous paths. The trellis V readers of flash attention share one type table
  (`vtq_trellis<type>`) and one scale expression.
- **Prompts on a trellis V cache read it faster.** Batches convert the quantized V cache to f16 first; that kernel
  decoded a whole trellis block serially in one thread. Every thread now decodes its own sample with the expression of
  the flash-attention readers (same values, generated text unchanged). Qwen3-4B on two RTX 2060, `q8_0` K, `vtq3_2s`:
  480-token follow-up prompt at 7k / 22k context 877 → 915 / 511 → 554 t/s, 22k first prompt 1261 → 1340 t/s. The
  16-state-bit types stay bound by their encoder (`vtq3_2` 84 → 85 t/s).
- **The 16-state-bit trellis V types decode 15-20 times faster.** `vtq{2,3,4}_2`, `vtq{2,3,4}_3` and `vtq3_v8` now
  use the shared-memory Viterbi encoder of `vtq*_2s`: the DP values stay in registers with one shared exchange row per
  step, the codes for the 65536 states are read from the table per step instead of being held in registers, and the
  backtrack loads only the bytes it can reach. The global-memory kernel, its host-side waves with a stream sync per
  wave and its 536 MiB pool are gone; the workspace is one 2 MiB backtrack slot per SM (at least 32 MiB, 60 MiB on an
  RTX 2060), `GGML_CUDA_VTQ_POOL_SLOTS` now counts slots. The encoder is bit-identical to the CPU encoder (the old GPU
  kernel fused one multiply-add and could pick a different path on rare near-ties); the stored format is unchanged.
  Qwen3-4B Q4_K_M on an RTX 2060, `q8_0` K, decode tg32: `vtq2_2` 26.6, `vtq3_2` 27.0, `vtq4_2` 19.9 t/s (before: 1-2;
  `vtq3_2s` 78, `vtq3_1` 89). KL divergence (40 chunks, K f16) `vtq2_2` 0.02602, `vtq3_2` 0.00943, `vtq4_2` 0.00469
  (old encoder 0.02592 / 0.00889 / not measured; same median, the mean moves with single outlier chunks), and the
  40-chunk run takes 3-6 minutes instead of about two hours. Each layer still encodes one block per KV head per token,
  so decode keeps only a few SMs busy.
- **MTP speculative decoding on Qwen3.5/3.6 drafts from the right hidden state again.** Since the DFlash2 lattice fix
  the qwen35 and qwen35moe graphs no longer exported the hidden state the MTP head reads (`t_h_nextn`), so the head
  drafted from whatever the buffer held, and the getters for the nextn and eagle3 hidden states returned the host
  buffer without waiting for the device copies, so a draft read rows the GPU had not finished writing. The graphs now
  export the post-norm hidden state for every batch token, the getters synchronize like the logits getters, a request
  no longer pairs its first token with the hidden state of the previous request, and the server no longer asks for
  logits of every prompt token. The MTP draft context now takes the target batch size instead of the context size and
  reserves one output row per sequence, and the target reserves verify outputs as with the n-gram modes, so MTP fits
  at 40k context with `q8_0` and at 32k with `ktq2_1` / `vtq2_1` on two 12 GB GPUs (draft compute buffer 649 → 290
  MiB). Qwen3.6-35B-A3B on two RTX 2060, `q8_0`, short prompt, draft max 1: acceptance 34 % → 86 % (upstream: 86 %),
  decode 71 → 93 t/s (79.5 without speculation); 29k-token prompt at 40k context: acceptance 97 %, decode 85.4 t/s
  with draft max 1 and 92.6 t/s with draft max 3 (70.5 without speculation, upstream MTP 74.6). Two identical requests
  now draft identically. The draft context reads the target hidden states after every prompt batch, which drains the
  two-GPU pipeline, so with MTP the server now defaults to `-b 8192` unless `-b` is given: the prompt cost drops from
  16 % to ~6 % (no speculation 1649 t/s, MTP with 2048-token batches 1390 t/s, with 8192 1550 t/s). MTP also runs with
  several server slots: all slots share one draft state over the draft context, and the hidden-state rows of a batch
  that holds several sequences keep batch order. With two concurrent requests acceptance stays at 80 % (draft max 3),
  but their combined decode (104 t/s) is ~5 % below two requests without speculation (109 t/s); MTP pays off while one
  request is active.
- **Speculative decoding on hybrid models keeps one graph shape.** With n-gram (or MTP/draft) speculative decoding on
  models with Gated DeltaNet layers (Qwen3.5/3.6), the recurrent layers built a different graph for single-token
  decode, for each verify length and for prompt ubatches. Every change made the scheduler re-reserve the compute graph
  and synchronize all backends; on two GPUs the prompt of every request after the first lost its pipeline overlap, and
  decode reallocated whenever the verify length changed. Now every ubatch takes the same rollback path: the
  `gated_delta_net` op takes the snapshot count as a parameter and writes the newest state snapshots without a padded
  input state, the snapshots reach the cache with one copy, and the batch splitter keeps the last `n_rs_seq + 1` tokens
  of a sequence in one ubatch (upstream #24086 and #25278). A context with rollback raises `n_ubatch` to at least
  `n_rs_seq + 2`. Without speculative decoding nothing changes. Qwen3.6-35B-A3B on two RTX 2060, 29k-token prompt,
  n-gram speculation (draft max 8): prompt processing of the second and later requests 950 → 1560 t/s (`q8_0`) and
  `ktq2_1` / `vtq2_1` 1282 (alternating 930/1630) → 1539 t/s, decode 80.4 → 82.9 t/s, same text; wall time of a
  request with speculation 35.6 → 23.4 s (without it: 23.0 s). Writing the snapshots for every prompt ubatch costs
  ~5 % prompt speed against a context without speculation. The n-gram modes also reserve output rows for a verify
  batch instead of the whole batch, which shrinks the compute buffer of the output device (672 → 307 MiB).
- **New V cache types `vtq2_2s`, `vtq3_2s`, `vtq4_2s`: the trellis with 12 state bits, fast enough for decode.** Same
  block layout and bits per value as `vtq{2,3,4}_2` (2.25 / 3.25 / 4.25 bpw, enums 59-61), but a 12-bit shift register
  (4096 states) instead of 16 bits (65536), so the Viterbi encoder, which runs for every new token, keeps its DP in
  shared memory. Qwen3-4B (RTX 2060, `q8_0` K, tg32) decodes at 77.6 / 78.0 / 75.2 t/s with `vtq2_2s` / `vtq3_2s` /
  `vtq4_2s` against 1-2 t/s with `vtq*_2` (`vtq3_1`: 89.1). KL divergence (Qwen3-4B, 40 chunks, K f16): `vtq4_2s`
  0.00482, `vtq3_2s` 0.01014, `vtq2_2s` 0.0286, against `vtq3_2` 0.00889 and `vtq2_2` 0.02592 (16 state bits) and
  `vtq4_1` 0.00514, `vtq3_1` 0.01197, `vtq2_1` 0.0471 (codebook). `vtq*_2`, `vtq*_3` and `vtq3_v8` keep 16 state bits
  and their format; the state bits are now a property of the type (the compile-time override of `GGML_TRELLIS_L` used
  for the measurements is gone). The CUDA encoder workspace is allocated per state bit count (12 bits: 1 MiB per slot),
  so the 12-bit types alone do not reserve the 536 MiB pool of the 16-bit types.

## 2026-10-14

- **Every common K/V pair decodes with the grouped-query kernel.** The GQA decode kernel (`fattn-vec-gqa.cu`) now takes
  K f16, `q8_0`, `q5_0`, `q4_0` and `ktq2/3/4_1` with V f16, `q8_0`, `q5_0`, `q4_0` and `vtq2/3/4_1` at head sizes 128
  and 256 (f16 / f16 stays with the tensor-core kernel). Pairs such as `q8_0` / `q4_0`, `ktq4_1` / `q8_0` or f16 /
  `vtq4_1` used to convert the whole cache to f16 at every decode step. Qwen3-4B on one RTX 2060, decode at 32k:
  `q8_0` / `q4_0` 18.4 → 43.9 t/s, `ktq4_1` / `q8_0` 10.3 → 41.4 t/s, f16 / `vtq4_1` 13.0 → 41.4 t/s; Qwen3.6-35B-A3B
  on two GPUs: `ktq4_1` / `q8_0` 41.3 → 70.9 t/s.
- **Faster decode at long context with quantized caches.** The same kernel reads each type in its own lane layout
  (2-byte loads for 4 codes of the 4-bit types and for `q4_0`/`q5_0`), keeps the 2-bit codebooks in registers, applies
  the block scale once per element, stores the scores row by row so that softmax and the V loop read all columns with
  one load, and unrolls the V loop 4 times. Qwen3-4B, one RTX 2060, decode at 32k: `ktq4_1` / `vtq4_1` 38.9 → 46.0 t/s,
  `ktq2_1` / `vtq2_1` 40.7 → 49.0 t/s, `q8_0` / `vtq4_1` 41.5 → 45.3 t/s, `q8_0` / `q8_0` 41.7 → 43.8 t/s (f16 / f16:
  37.2). Tried and dropped: staging the rows in shared memory (shared memory and L1 share the load pipe on Turing, so it
  only moved the narrow loads), element pairs or 4 consecutive elements per lane for `q8_0` K (3-7 % slower), reducing
  the scores of several rows together (5-25 % slower: more live registers), the KQ loop unrolled 8 times (`q8_0` 5 %
  slower) or 2 times, the V loop 8 times (no gain).
- **Faster prompt processing with a `vtq*_1` V cache.** Prompt batches convert V to f16 for the tensor-core kernel at
  every ubatch; the conversion now decodes 8 elements per thread with one 16-byte store and keeps the codebook in
  shared memory (lanes reading different entries of constant memory are serialized). Qwen3-4B, one RTX 2060, `pp512`
  at 32k context: `ktq4_1` / `vtq4_1` 705 → 727 t/s, `q8_0` / `vtq4_1` 669 → 705 t/s (f16 / f16: 753). The f16 copy of a
  `q8_0`, `q5_0` or `q4_0` K next to such a V now also takes 8 elements per thread (the upstream kernel kept 64 of 256
  threads busy at head size 128).
- **Speculative decoding verifies faster with TurboQuant caches.** Batches of 2-7 queries with a TurboQuant K or V
  ran the per-head vector kernel, which reads the cache once per Q head and query; they now take the split-dequant
  tensor-core path like longer batches. Qwen3-4B, one RTX 2060, at 32k context: 2 queries `ktq4_1` / `vtq4_1`
  25.4 → 45.8 t/s, 4 queries 27.0 → 80.4 t/s, `q8_0` / `vtq4_1` 23.8 → 61.4 t/s.
- **Small verify batches read the quantized cache directly.** With a quantized K or V cache, the split-dequant path
  converted the whole cache to f16 at every verify step. Batches of 2-16 queries whose (query, head) columns fit one
  block of the tensor-core kernel (`fattn-tq-wmma.cuh`, now a template with one instance file per head size and block
  width) now run it on the quantized cache: 16 columns always, 32 columns from 8192 cells (not f16 K at head size
  256); larger batches keep the conversion. Qwen3-4B (GQA 4), one RTX 2060, at 32k context, t/s of the verify batch:

  | K / V | 2 queries | 4 queries | 5 queries | 8 queries |
  |---|---|---|---|---|
  | `q8_0` / `q8_0` | 34.6 → 57.3 | 63.5 → 99.3 | 77.3 → 91.4 | 105.6 → 122.0 |
  | `ktq4_1` / `vtq4_1` | | | 76.9 → 98.9 | 104.9 → 128.0 |
  | f16 / `vtq4_1` ¹ | 45.9 → 69.6 | 80.4 → 118.3 | 97.5 → 105.0 | 129.1 → 135.7 |
  | `q8_0` / `vtq4_1` | 33.8 → 60.7 | 55.3 → 104.7 | 74.8 → 95.1 | 103.1 → 126.7 |

  ¹ measured as `ktq4_1` / `vtq4_1` with `llama-bench -p`, which keeps a KTQ K cache in f16 staging until the first
  decode step; the `ktq4_1` row has the K cache quantized (`LLAMA_ARG_NO_TQ_DEFERRED_K=1`, as in a server after the
  first token). At 2k context +4-11 % for 2 and 4 queries, 5-8 queries unchanged. Qwen3.6-35B-A3B (GQA 8, head 256)
  on two GPUs at 32k: `q8_0` 2 queries 84.8 → 99.6 t/s, 4 queries 133.4 → 139.8 t/s; `ktq4_1` / `vtq4_1` 2 queries
  87.0 → 101.5 t/s, 4 queries 135.8 → 141.7 t/s. Batches of 9 and 16 queries, decode and prefill unchanged. In `llama-server` with ngram speculation on a
  29k-token prompt (release of 2026-10-12 → now, [benchmarks](docs/benchmarks/README.md)): `q8_0` / `vtq4_1`
  42.9 → 79.3 t/s, `ktq4_1` / `vtq4_1` 31.6 → 74.4 t/s, `q8_0` / `q8_0` 74.8 → 78.8 t/s.
- Flash attention with a logit softcap at head sizes other than 128, 256 and 512 is reported as unsupported on CUDA:
  no kernel compiles the softcap there, and a softcapped prompt batch at head size 64 with a quantized cache launched
  an MMA instance without device code. No model in use combines the two (Gemma 2: 256, Grok: 128).
- `scripts/check-cuda-instances.py` checks the GQA decode instances per K type (the old pattern matched none).
- **The GQA decode kernel goes before the tensor-core decode kernel** at head sizes 128 and 256, where it is faster for
  every pair; the tensor-core kernel keeps GQA ratio 2, very short contexts, heads 64 and 512 and attention sinks.
  Qwen3-4B, one RTX 2060, decode at 32k: `q4_0` / `q4_0` 32.4 → 49.4 t/s, `q5_0` / `q5_0` 26.8 → 43.7 t/s; at 1k
  context +3.5 % for `ktq4_1` / `vtq4_1`. Qwen3.6-35B-A3B (head 256) on two GPUs: +0.5-6.5 % at 32k.
- The f16 copy of a KTQ K cache (batches after the first decode step) decodes 8 elements per thread like the VTQ one.
- The head offsets of both decode kernels are 64-bit (they overflowed int32 from about a million cells).
- New tests: decode with the GQA kernel (1024 cells, GQA 4 and 8, softcap on and off) for every covered pair;
  prompt batches with `vtq1_1` and `vtq3_1` V; prompt batches with `ktq1_1`, `ktq2_1` and `ktq3_1` K; 2-16 queries
  with a causal mask (`causal=1`: every query row ends at a different cell, whole KV tiles fully masked) at heads
  64-512, GQA 2-16 and eight K/V pairs, and 17-32 columns at 8192 cells.
- Checked on an RTX A6000 (sm_86) and an RTX 4090 (sm_89) besides the RTX 2060: every flash attention test passes;
  at 32k context the TurboQuant pairs decode 5-21 % faster and verify 4 queries 3-4.5× faster than at the start of
  the day (`q8_0` / `vtq4_1`; [benchmarks](docs/benchmarks/README.md)).

## 2026-10-13

- **f16 prefill on par with upstream.** The CUDA flash attention MMA kernel (`fattn-mma-f16.cuh`) and the tile helpers
  (`mma.cuh`) are now upstream's versions; the fork's copies were older upstream states with the swizzle and the sparse
  gather ported on top. On Turing they compiled to two to four times more register spills (Gemma 4 global layers, head
  size 512: 704-856 bytes per thread instead of ~180), mostly because upstream #29152 (thinner tiles for large head
  sizes) was missing. 2× RTX 2060, pp2048 at 32k with an f16 cache, upstream → llama-tq: Qwen3.8-27B 427 → 429 (before:
  4.8 % behind), Qwen3.6-35B-A3B 1286 → 1276, Gemma 4 12B 843 → 824 (before: 5.4 % behind). Gemma 4 is still 1-2 %
  behind; see [benchmarks](docs/benchmarks/README.md#against-upstream-two-gpus).
- The q6_K, q4_0, q5_0 and q3_K matrix kernels subtract packed bytes without saturation where the result cannot
  overflow (upstream #29478); on Turing the saturating form is emulated with more instructions. Prefill +0.4 to +0.7 %
  (Gemma 4, Qwen3-4B, Qwen3.6; not yet in the benchmark table), results unchanged.
- **Fixed: some K/V cache type pairs aborted.** `q4_0` K with a trellis V (`vtq*_2`, `vtq*_3`, `vtq3_v8`) and
  `ktq2_1` K with `vtq3_1`/`vtq4_1` V (also `ktq3_1` with `vtq4_1`) passed the CUDA support check but had no vector
  kernel, so decode at short context ended in `GGML_ABORT` (in `fattn.cu`). The vector kernel's own case list now decides
  where it runs, and every other pair converts K and V to f16 for the MMA kernel (tile kernel before Turing).
- Every K/V cache type pair now runs on CUDA: mixed stock types (e.g. `q8_0` / `q4_0`, f16 / `q8_0`), KTQ K with a stock
  V and `q5_0` K with a VTQ V had no kernel and ran on the CPU. A new test covers all 94 pairs of 7 K and 14 V types
  (decode and prompt, head 128 and 256). These pairs convert the cache at every decode step, see
  [known issues](docs/turboquant.md#known-issues).
- **Faster KTQ decode at long context with head size 128** (for example Qwen3, Llama): from 2k context the decode
  uses the grouped-query vector kernel instead of the tensor-core one (2× RTX 2060, Qwen3-4B: +3-5 % at 4k, +8-14 % at
  32k; the tensor-core kernel stays below 2k, where it is 0.5-4 % faster, and for heads 256 and 512).
- Flash attention for head sizes 192 (K) / 128 (V) and 320 / 256 on CUDA, as upstream has it (MiMo-V2.5, Mistral Small
  4): the MMA kernel on Turing and newer, the tile kernel before. Both need grouped-query attention with a GQA ratio that
  is a multiple of 8 (192) or 32 (320); other shapes ran on the CPU and still do.
- Fixed: the CUDA build for Pascal and older (`sm_61`) stopped in `fattn-tq-wmma.cu`, which named the tensor-core API
  that only exists from Volta on. The release archives start at Turing, so they were not affected.
- `llama-bench` counted the one-time conversion of the f16 K staging to KTQ in the decode time of generation tests
  (Qwen3-4B at 32k: tg64 28.4, tg256 32.8 t/s for the same speed); it now runs that conversion before timing. Together
  with the kernel choice: `ktq4_1` / `vtq4_1` at 32k 31.4 → 39.9 t/s, `q8_0` / `vtq4_1` 42.8. The TurboQuant rows in
  [benchmarks](docs/benchmarks/README.md#against-upstream-two-gpus) are measured again.
- The earlier comparison (2026-10-12) ran upstream first in every round; on these GPUs the first build of a round is
  2-3 % faster, so part of the gap given there was the order. The benchmark table now alternates the builds (upstream,
  llama-tq, llama-tq, upstream) and averages.
- Fixed: `--split-mode tensor` aborted on DeepSeek V4 (hyper-connection ops missing in the meta backend, upstream
  #25585) and on views of host tensors inside a tensor-split graph (upstream #30217). `test-llama-archs` caught both; it
  only runs these paths with a GPU, so the GitHub CI (CPU only) did not. It still aborts on a later architecture with
  tensor split; that is open.

## 2026-10-12

- **Fixed: deferred K/V quantization never converted on most models.** The f16 prefill buffer was converted to KTQ/VTQ
  only in the plain KV cache; sliding-window caches (gpt-oss, Gemma) and hybrid caches (Qwen3.5/3.6, Qwen3.8,
  Qwen3.8-Flash-Next) kept decoding from the f16 buffer, with both buffers in VRAM, and with two or more slots the
  conversion never ran on any model. It now runs in every memory type as soon as every token of a batch wants an output
  (decode, several slots, speculative verification).
- Fixed: at long context the conversion cast the whole cache to f32 and ran out of memory (1.84 GiB at 262k); the state
  stayed "ready", every token retried it, and decode read a cache without the prompt. It now converts only used rows, in
  up to 8 chunks, and keeps the f16 buffer if memory is short. A first decode on an empty cache crashed; a session restore
  did not restore the staging state.
- **`llama-perplexity` scores the quantized cache by default.** Without `--no-tq-deferred-k` it measured f16 K
  (Qwen3-4B `ktq2_1`: 9.12 instead of 11.48). `--tq-deferred-k` / `--tq-deferred-v` bring the staging back. The accuracy
  table in [turboquant.md](docs/turboquant.md#accuracy) was already measured with the cache quantized; KTQ perplexities
  in older notes that did not pass the flag are too good.
- `llama-bench` decode rows with KTQ K on sliding-window and hybrid models measured the f16 buffer until this fix; the
  two-GPU comparison in [benchmarks](docs/benchmarks/README.md) is measured with the fix, and the single-GPU tables use
  dense Qwen3-4B, which the bug did not affect.
- **Two GPUs, pipeline parallelism:** prefill was a third slower than upstream (Qwen3.6-35B-A3B pp2048 1230 vs 1908 t/s on
  2× RTX 2060). The scheduler shrank its reservation after the all-experts warmup graph and after empty ubatches, and from
  then on every growing KV view forced a re-reservation that synchronized both GPUs. Now 1903 t/s. These numbers and
  those in the relocatable device code entry below come from separate runs; the comparison table in
  [benchmarks](docs/benchmarks/README.md#against-upstream-two-gpus) is from the final build.
- **Maximum context closer to upstream:** the MTP layer is loaded only when it is used (upstream #26296; 856 MiB on
  Qwen3.6), the compute buffer reserves logits only for the outputs the context can produce (#23861), and the f16 copy
  of a quantized K/V cache for flash attention is reserved at startup (#23907). Pipeline parallelism is turned off at
  startup when that copy would not fit with it.
- CUDA flash attention fixes from upstream: divergent barrier in the f16 kernel (#27870), two Volta cases (#29803), the
  Gemma E4B MTP case (#25148), V-type validation (#24403, extended to the TurboQuant types), PDL wait before the mask scan
  (#25185).
- Fixed: Qwen3.8-Flash-Next sparse attention and DFlash rotated V with the plain Hadamard matrix instead of D·H·D, so VTQ
  V caches gave wrong output there; gpt-oss with KTQ K decoded through the slow VEC kernel while K was still staged.
- **The CUDA backend is built without relocatable device code.** It was needed only for one shared trellis table, and it
  cost registers and load scheduling in every kernel: the `q8_0` flash attention decode kernel ran at half speed on
  Turing. The table is now reached through a pointer per translation unit. 2× RTX 2060: Qwen3-4B `q8_0` decode 85.8 ->
  91.1 t/s (upstream 90.3), at 8k context 65.9 -> 70.9 t/s (upstream 57.4); gpt-oss f16 prefill at 32k 1729 -> 1783 t/s
  (upstream 1776). `libggml-cuda` shrinks from 230 MB to about 100 MB. Still open: f16 prefill at 32k is 2.6-5.4 %
  below upstream (Gemma 4 5.4 %, Qwen3.6 4.1 %, Qwen3.8 2.6 %) ([benchmarks](docs/benchmarks/README.md#against-upstream-two-gpus)).
- Decode with a single KV tile (up to 256 cells) uses the VEC kernel instead of the GQA kernel, which is faster there.
- **`vtq2` and `vtq3` now mean `vtq2_1` and `vtq3_1`.** They pointed to the trellis types `vtq2_2` and `vtq3_v8`, which
  decode at 1-3 t/s: their Viterbi encoder runs for every new token (13.7 ms per layer on an RTX 2060), and deferred
  staging only covers the prompt. The trellis types stay available under their full names
  ([turboquant.md](docs/turboquant.md)).
- Fixed: with `vtq*_3` / `vtq3_v8` V on CUDA the outliers were never stored (left for a CPU pass that did not exist):
  every block decoded its first sample as 0, and each write cleared the first blocks of the whole V cache. The GPU
  encoder now stores the outliers like the CPU encoder. Qwen3-4B, wikitext-2, 3 chunks of 512, V quantized in prefill:
  `vtq3_3` 10.09 -> 9.89, `vtq3_v8` 10.09 -> 9.83 (f16 9.91, `vtq3_2` 9.90). Heads that are not a multiple of the
  128-value trellis block (gpt-oss: 64) gave garbage with a trellis V; those layers now use the codebook type with the
  same bits.
- Tests: `test-llama-archs` and `test-quantize-perf` run through all types and architectures again; test-backend-ops
  covers the trellis V types (SET_ROWS, flash attention decode and prompt batches), which had no case before.

## 2026-10-11

- **TurboQuant encoders search the block scale:** each KTQ/VTQ block is assigned to the codebook at 32 trial scales and the
  assignment with the smallest angle wins; it is stored with an unbiased scale (a least-squares scale shrinks every V
  block, and the attention average does not remove that shrink). KL divergence vs f16 KV with the cache quantized in
  prefill: `ktq4_1`/`vtq4_1` 0.045 -> 0.025 on Qwen3-4B (`q4_0` 0.034) and 0.55 -> 0.36 on gpt-oss (`q4_0` 0.36);
  `ktq2_1`/`vtq2_1` 1.38 -> 0.40 on Qwen3-4B. Format unchanged; decode with the cache quantized per token -1.7 %
  (`ktq4_1`/`vtq4_1`, RTX 2060). CUDA and CPU encoders pick bit-identical codes.
- **Fast decode for `q8_0` K with `vtq4_1` / `vtq2_1` V**, the accurate compact setting (6.5 bpw): the GQA and tensor-core
  decode kernels take the pair (head 64 with sinks, 128, 256, 512). Qwen3-4B at 16k context 17.5 -> 50.7 t/s (`vtq4_1`), gpt-oss at 8k
  50.8 -> 70.4 t/s, both as fast as `q8_0`/`q8_0` (RTX 2060). gpt-oss needs 8-bit K: every 4-bit K type costs it KLD ~0.35,
  `q8_0`/`vtq4_1` 0.039.
- KV accuracy table re-measured on wikitext-2 (Qwen3-4B) and on the models' own prose (gpt-oss, Gemma 4; raw wikitext is
  far out of distribution for these chat models, [scripts/kld-gen-text.py](scripts/kld-gen-text.py)), with the noise
  floor of each model (flash attention on vs off). The earlier table used the appointment dataset as text, on which
  gpt-oss and Gemma 4 flip their answer under any numerical change; the statement that they react strongly to any KV
  quantization was an artifact of that text: `q8_0` is at their noise floor.
- Fixed: SET_ROWS into a KTQ/VTQ tensor on the CPU called a null pointer (no CPU encoder registered); test-backend-ops had
  no SET_ROWS cases for these types; test-quantize-fns overflowed its buffer for types with an F32 dot product type.
- Fixed: tool-call parameters that start with indentation lost one space (Qwen3.5 / Qwen3.8 XML tool calls, e.g. Python
  code); Nemotron-3-Nano put an empty reasoning block into the content with thinking off; Nemotron-Nano-v2 tool calls and
  reasoning were not parsed (its handler from upstream #24674 had been dropped, now restored with the matching parser
  pieces of upstream #21892 and #22654). The full chat test suite passes again.
- Release archives: the binaries find their libraries from any directory, the CUDA archives no longer need NCCL,
  `llama-finetune` is in every archive, and each archive is checked before upload (scripts/check-release-archive.sh).
  Releases and Docker images are built only on demand. The Blackwell archive builds with CUDA 12.9: the CUDA 12.8 linker
  fails the sm_120 device link ("Internal FNLZR error"); the 12.9 build was checked on an RTX 5060 Ti. `--version`, the server log and `/props` report the release
  version; the web interface is named llama-tq and has an About page with version, source links and license.
- The Claude Code launcher is now `scripts/llama-tq-launch-claude.sh` (environment variables `LLAMA_TQ_SERVER_URL`,
  `LLAMA_TQ_MODEL`; before: `onllama-launch-claude.sh`, `ONLLAMA_*`).
- CI for this fork: a Linux build with all tests on every push and pull request that changes code, and a docs check (links, internal
  details, the same measurement on two pages; scripts/check-docs.py).
- Trellis V re-measured with V quantized in prefill: `vtq2_2` (2.25 bpw) KLD 0.033 vs `vtq2_1` (2.5 bpw) 0.049 on
  Qwen3-4B; more accurate per bit, but without the fast decode kernels ([turboquant.md](docs/turboquant.md)).
- Docs: removed the sparse V dequant sections (the code is gone); prompt caching for `/v1/messages` does not apply to
  streaming requests such as Claude Code's; speculative-decoding speed-ups from 2026-06 were measured with the KTQ sign
  bug and are marked as historical.

## 2026-10-10

- **All 12 evaluated models reach 100 % exact match** on the held-out appointment task (some after tuning learning rate or epochs): Qwen3-4B,
  Qwen3.5-0.8B and Qwen3.8-27B (Gated DeltaNet), Qwen3.8-Flash-Next (hyper-connections, sparse attention, 72.5 GB),
  Ternary-Bonsai-2-27B (ternary), Qwen3.6-35B-A3B and K2-Horizon-36B (MoE), Kolibri-1 78B (MoE, 8 min on one 96 GB GPU),
  Gemma-4-12B and Gemma-4-26B-A4B, gpt-oss-20b (MXFP4), Ministral-3B. Settings per model in [docs/finetune.md](docs/finetune.md).
- Backward for the hyper-connection ops (`DSV4_HC_PRE`, `DSV4_HC_POST`), checked against float64 gradients; the CPU `ACC`
  takes a strided first dimension of src1.
- **Faster decode with quantized KV caches:** the Hadamard rotations of K and V (q8_0, q4_0, KTQ/VTQ) run as the fast
  Walsh-Hadamard transform instead of small matmuls, the VTQ V rotation D*H*D as one fused kernel. Qwen3-4B, RTX 2060:
  `ktq2_1`/`vtq2_1` 84.2 -> 86.8 t/s (f16 90.5); on an RTX 5090 the rotation cost 11-13 % of the decode speed before.
  Fixed: models with head size 64 rotated K with the signed V rotation as well.
- **KV cache accuracy re-measured** with the cache quantized during prefill (`--no-tq-deferred-k/-v`; without it K stays
  f16 in prefill and perplexity runs measure f16 K). The graph Hadamard rotation is no longer applied to KTQ K, which made
  it clearly worse (Qwen3-4B KLD vs f16: `ktq2_1` 1.97 -> 0.60, `ktq4_1` 0.057 -> 0.030). `ktq4_1`/`vtq4_1` is at the
  `q4_0` level; the recommendation is now `f16`, then `q8_0`, and TurboQuant when the memory is needed
  ([docs/turboquant.md](docs/turboquant.md#accuracy)).
- Fixed: the last answer is now trained after the prompt exactly as the server renders it (Gemma 4 with reasoning off adds
  an empty thought channel only in the generation prompt; the answers were trained without it and the model partly kept
  its own output format); MoE training on a single GPU aborted under CUDA graph capture (`MUL_MAT_ID_GRAD_B`).
- CUDA releases: the release workflow builds one archive per GPU generation with native code (Turing + PTX, Ampere sm_80/86, Ada, Blackwell)
  instead of a single Turing build that newer GPUs had to JIT-compile; first published with the next release.
- [examples/training/termine](examples/training/termine/README.md): the evaluation task as a reproducible package
  (generator with checksums, prompts and conventions, training input as rendered, evaluation script, PyTorch reference).

## 2026-10-09

- **Flash attention backward** (`GGML_OP_FLASH_ATTN_BACK`, CPU and CUDA): the softmax is recomputed from q and k instead of storing the attention probabilities (masks, GQA, softcap, ALiBi, sinks). The CUDA backward runs on cuBLAS GEMMs per block of query rows, so `-fa on` is as fast as the non-flash path with less memory (Qwen3-4B, `-c 4096 -ub 512`: 9.8 GB instead of 18.7 GB). `llama-finetune` uses flash attention by default on CPU and CUDA. New `test-flash-attn-back` against float64 central differences.
- **Training speed-ups:** CUDA `ACC` for strided and permuted second operands (these nodes ran on the CPU), `out_prod` through quantized weights on f16 tensor cores with power-of-two scaling, labels kept on the device of the output projection, `LLAMA_TRAIN_TIMING=1` prints the time per step by phase.
- **Gradient checkpointing** (`--grad-checkpoint`): only the layer outputs are kept, each layer is recomputed in the backward pass. Logits are computed only for positions with a trained label.
- **Packed chat windows:** whole chat examples per window without overlap (each token once per epoch instead of twice); `--train-stride N` sets the stride.
- Qwen3-4B, 2 epochs, 100 held-out examples, 100 % exact match: RTX 2060 321 s (PyTorch QLoRA 260 s; 1585 s before these changes), RTX 5090 54 s (PyTorch QLoRA 102 s). See [docs/finetune.md](docs/finetune.md).
- Fixed: training with `-ngl 0` in a CUDA build (no op offload, Gated DeltaNet q made contiguous); KV cache writes no longer need `GGML_BACKWARD_SKIP_INPLACE=1`; `--moe-pin-experts` had no effect since the per-model tensor loading and works again.
- CUDA builds: the VTQ flash attention vec dispatch compiles in eleven parallel TUs (one per V type) instead of three long ones.
- Cleanup: removed the unused VTQ correction overlay and trellis encode receiver, the standalone Vulkan TQ dequant prototype, XQuant cross-layer K reuse (`--xquant`), the MoE expert cache (`--moe-cache-mib`), the `vtq_mixed` type, the expert hotness prefetch and router profiler, and upstream-only GitHub automation.

## 2026-10-08 (later)

- K2-Horizon-MoVA-36B-A4B (MoE with routed value experts) learns the appointment task as well: 0 -> 100 % exact match after one attention-LoRA run (39 min).
- MoE routing: the normalization of the selected expert weights had no gradient (`ggml_clamp` works in place, so the backward pass skipped it). New `CLAMP` backward; `SUB`/`DIV` backward for a broadcast second operand.
- Fixed: builds with `GGML_RPC=ON` (also the release builds) stopped at the RPC op-count assert; the server crashed on exit when a `--lora` file failed to load.

## 2026-10-08

- **Finetuning learns real tasks on dense, MoE and hybrid models** (see [docs/finetune.md](docs/finetune.md)). Extracting an appointment from a German message as JSON with a fixed schema, 100 held-out examples, exact match before -> after one LoRA run: Qwen3-4B 0 -> 100 % (also after merging with `llama-export-lora`), Gemma-4-12B 0 -> 94 %, Qwen3.6-35B-A3B IQ2_XXS with LoRA on the routed experts only 0 -> 89 %, Ministral-3-3B and gpt-oss-20b 0 -> 100 %; Qwen3.5-0.8B (Gated DeltaNet hybrid) 99 % validation accuracy.
- Chat data: `-f data.jsonl` with `{"messages": [...]}`, rendered with the model's chat template and `--reasoning` like the server, loss on the assistant turns only (`LLAMA_FINETUNE_SHOW_MASK=1` shows the mask). Train with the reasoning setting you serve with.
- `--grad-clip`, `--lr-warmup`, `--early-stop N` (the best adapter by validation loss is kept as `<adapter>.best`), exact `--resume` after `--stop-after N` or a signal (adapter, AdamW moments and position; bit-identical to an uninterrupted run), `GGML_OPT_PRINT_GRAD_NORM=1|2`.
- Gradient fixes: K and V now get a gradient through the KV cache (before, `attn_k`/`attn_v` adapters never moved and every layer below an attention missed that path); CUDA `out_prod` read LoRA activations with a wrong stride (A gradients were garbage); CPU `rms_norm_back` destroyed its input when run inplace; CUDA `rms_norm_back` mis-read strided inputs; the optimizer state was looked up by node index instead of per parameter.
- New backward passes: gated delta net (new op `GATED_DELTA_NET_BACK`, CPU and CUDA), `SSM_CONV`, `CONCAT`, `TANH`, `GEGLU`, `REGLU`, `GEGLU_QUICK`, `SWIGLU_OAI`, `ADD_ID`; exact cross-entropy gradient for masked rows; accuracy over labeled positions only.
- Long context: `-c 16384 -ub 128` trains Qwen3-4B in 23 GB on two GPUs (the attention probabilities of every layer dominate the memory); an out-of-memory training graph now reports itself instead of overrunning buffers. `test-opt` runs again.

## 2026-10-07

- **LoRA finetuning directly on quantized GGUFs works end to end**, including the experts of MoE models and models larger than VRAM (see [docs/finetune.md](docs/finetune.md)). Fixed:
  - gradient accumulators were never cleared with `llama-finetune`'s per-batch graphs, so every step used the sum of all previous gradients and longer runs diverged;
  - LoRA A was initialised ~17x too large (now Kaiming-uniform as in PEFT);
  - the input gradient through quantized MoE experts (and the broadcast expert input) was dropped; new op `MUL_MAT_ID_GRAD_B` on CPU and CUDA; the routing-weight gradient (`get_rows_back`, batched) as well;
  - the input gradient of quantized dense matmuls (`out_prod`) ran on the CPU; now on CUDA (Qwen3-4B: ~90 s -> ~1 s per step);
  - AdamW with two GPUs (optimizer state now next to its parameter) and the scheduler size for training graphs.
- Verified: Qwen3-4B attention LoRA perplexity 15.3 -> 12.7 on held-out text; Qwen3-Coder-30B-A3B expert and attention LoRA converge; Kolibri-1 (31.5 GiB, experts in RAM) trains with `--no-op-offload` at lr 1e-5.

## 2026-10-06

- **`-fit` works:** automatic placement of layers, experts and context to free device memory (upstream `common/fit.cpp`); until now the option was a stub without effect. Parameters set by hand (`-ngl`, `-ts`, `-ot`, `-ncmoe`) are kept. Kolibri-1 Q3_K_S with `-fit` and the GPUs listed so that the RAM-expert layers land on the x16 GPU: prompts at 596 t/s instead of 454 with the hand placement; `GGML_OP_OFFLOAD_MIN_BATCH=256` cuts the time to the first token of short prompts from 3.7 s to 1.0 s.
- q4_0 K/V in the tensor-core decode kernel: Qwen3.8-27B with q4_0 KV decodes 18.3 t/s at 74k context instead of 10.8 (attention 865 -> 249 µs per step at 32k).
- Fixed: the CUDA `ktq1_1` quantizer still rounded stochastically; it now matches the CPU reference.
- TurboQuant code cleanup: one template per role for the KTQ helpers instead of one copy per bit width, type-family macros (`GGML_TYPE_IS_KTQ`, `GGML_TYPE_IS_VTQ`) instead of hand-written type chains (about 1200 lines less). Perplexity unchanged.

## 2026-10-05

- **MoE expert cache** (`--moe-cache-mib N`, upstream PR #29887) (removed 2026-10-09: no gain in our setups): an LRU cache in VRAM for experts kept in host memory (`-cmoe` / `-ncmoe` / `-ot`). Decode batches (up to 32 tokens) remap the selected experts to cached copies instead of reading them over PCIe. With several GPUs the cache sits on the device whose layers keep the most experts in host memory; pipeline parallelism is turned off while the cache is active.
- Fixed: the server crashed with a null context when the context could not be created (for example out of memory); it now exits with an error.
- Removed: the DiffusionGemma text-diffusion model, its examples, CUDA sampler, context API and WebUI preview.
- Removed: the inline KTQ MMA flash-attention path (dead since the split dequant + tensor-core path); `vtq_mixed` is no longer offered on the command line (the type enum stays for compatibility).

## 2026-10-04

- **TurboQuant KV fix:** the CUDA readers applied the KTQ sign bits inverted, so every dequantized K value was negated, and the CUDA quantizers used stochastic rounding. CUDA now writes bytes identical to the CPU reference (Qwen3.8-27B `ktq2_1` K: PPL 16.8 -> 6.09, f16 6.04).
- **TurboQuant KV speed (2x RTX 2060, Qwen3.8-27B Q4_K_M, tensor split, 118k context):** decode 6.5 -> 15.8 t/s with `ktq2_1`/`vtq2_1` (f16 KV: ~18.5 t/s at a 112k maximum context), prefill 18 -> 194 t/s.
  - All KTQ blocks share one RHT sign pattern, so Q is rotated once per query instead of per K block.
  - Batches with TurboQuant V dequantize K/V to f16 and use the tensor-core kernel.
  - New decode kernel for quantized KV with grouped-query attention: one block per GQA group, a whole warp per K/V row, the column dot products reduced together, codebooks in shared memory. It also takes q8_0 and q5_0 K/V (q8_0 decode is now faster than f16) and the sparse attention of Qwen3.8-Flash-Next.
  - Warp-cooperative TurboQuant quantization when writing the KV cache.
  - Tensor-core decode kernel for TurboQuant and q5_0 KV (`GGML_CUDA_TQ_WMMA=0` disables it): attention 12-29 % faster than the GQA kernel.
- KV type guidance from perplexity: `ktq4_1`/`vtq4_1` equals f16 on Qwen3.8-27B and Ternary-Bonsai-2-27B; `ktq2_1`/`vtq2_1` costs about +2.6 % there (these perplexity runs kept K in f16 during the prompt, see 2026-10-10; superseded by the KL divergence table of 2026-10-11).
- q5_0 K/V flash attention without `GGML_CUDA_FA_ALL_QUANTS` (tensor cores for batches, the GQA kernel for decode).
- **K2-Horizon-MoVA-36B-A4B** (`k2-horizon`, MoE with routed value experts in attention). Q3_K_M on 2x RTX 2060: 47 t/s decode, 840 t/s prefill. With `ktq4_1`/`vtq4_1` KV plus `--tq-protect-layers 4` the PPL is +0.8 % over f16 at about a third of the KV memory (whether K was quantized during the prompt is not recorded, see 2026-10-10).
- Tensor split with a quantized KV cache (attention rotation, staging cache, views of row-split tensors) and for routed value experts.
- Qwen3.8-Flash-Next: the TurboQuant deferred-staging options now reach its hybrid memory (`--no-tq-deferred-k/v` were ignored).
- Fixed: the fused Gated DeltaNet state gather could read freed row ids (illegal memory access on long prompts with hybrid Qwen3.5 / 3.8 models).
- Fixed: Gemma 4 prompt processing with TurboQuant KV fell back to the vector kernel at head size 512 and was several times slower.
- Fixed: gpt-oss (and other graphs without an output-ids null check) crashed while reserving the compute graph.
- Tensor-core decode kernel at head size 512 (Gemma 4 global layers with GQA 8-16): Gemma-4-12B with `ktq4_1`/`vtq4_1` decodes 28.8 t/s at 32k context instead of 16.4 (f16 KV: 31.0). q5_0 K/V at head size 512 no longer falls back to the CPU.
- **Aleph Alpha Kolibri-1** (`kolibri1`): 78B MoE, 3.5B active, German/English; router that selects on logits plus bias and weights by the unbiased sigmoid, NoPE full-attention layers. Port based on the patches by Seraphiel102. Q3_K_S on 2× RTX 2060 with part of the experts in RAM: ~38 t/s decode.
- Tensor-core decode kernel at head size 64 with attention sinks: gpt-oss-20b with `ktq4_1`/`vtq4_1` decodes 40 t/s at 64k context instead of 6.8.
- Fixed: the vector flash-attention kernel with KTQ K at head size 64 trapped on the GPU (gpt-oss crashed on the first decode with TurboQuant KV).
- Chat templates: a null left operand of `in` is a plain lookup (upstream fix, needed for the Kolibri-1 template).

## 2026-10-03

- Tensor split (`-sm tensor`) across GPUs without P2P or NCCL. Partial sums are reduced through mapped pinned host memory; `GGML_CUDA_HOST_ALLREDUCE_BF16=1` sends bf16 to halve link traffic, `GGML_CUDA_HOST_ALLREDUCE=0` disables the path.
- Ternary weight types `PQ2_0` and `PTQ1_0` with Hadamard-rotated activations and CUDA mat-vec / MMQ kernels (Ternary-Bonsai-2-27B).
- Qwen3.8 and Qwen3.8-Flash-Next (`qwen4exp`) support, including sparse flash attention for the compressed-attention indexer.
- Faster Gated DeltaNet layers for the hybrid Qwen3.5 / 3.8 models (state gather and gate activations inside the kernel).
- DFlash / DFlash2 block-diffusion speculative decoding.
- MoE LoRA fine-tuning on quantised models (`--train-skip-regex`, `MUL_MAT_ID` backward). See [docs/finetune.md](docs/finetune.md).

### Known issues

- TurboQuant KV (`ktq2_1` / `vtq2_1`) produced garbage output on Qwen3-4B-Instruct: this was the inverted KTQ sign convention, fixed on 2026-10-04. Small models with strong outlier channels (Qwen3-4B) stay sensitive to 2-3 bit KV; use `ktq4_1` or q8_0 there.

## 2026-09

- Qwen3.8-Flash-Next (`qwen4exp`) port: fused hyper-connection ops, per-layer n-gram embeddings, the compressed-attention
  indexer with sparse flash attention at head size 256, conversion of real checkpoints, model saver and
  `test-llama-archs` coverage (released 2026-10-03).
- Ternary weight types `PQ2_0` / `PTQ1_0` (groups of 128, Hadamard-rotated activations) for Ternary-Bonsai-2-27B.
- PCIe peer-to-peer between two RTX 2060 investigated on four paths: Turing consumer GPUs cannot do it in hardware, which
  led to the tensor split through pinned host memory (2026-10-03).

## 2026-08

- **DFlash / DFlash2 block-diffusion speculative decoding:** draft model and speculator, features from several target
  layers, token embeddings shared with the target, MRoPE, trimming of the draft KV tail. Code generation on 2× RTX 2060
  about 1.7-1.8x faster (26-28 t/s instead of 15.5 t/s, acceptance up to 100 %).
- Qwen3.8-27B (dense Gated DeltaNet hybrid) runs; Gated DeltaNet tests with several draft tokens per step (MTP).
- Upstream picks for vision, reasoning output, correctness and the server.

## 2026-07

- Unified Gemma 4 vision + audio projector (`gemma4uv` / `gemma4ua`).
- Upstream picks for KQ masks, sliding-window attention and n-gram drafts (out-of-bounds read when the prompt shrinks).
- VTQ: codebook scale folded into the V dequantization.

## 2026-06

- **MTP speculative decoding:** upstream MTP integration (model classes, libllama MTP API, draft flags, server
  speculation) plus an n-gram hybrid with a pretrained static cache. Qwen3.6-35B-A3B IQ2_XXS on 2× RTX 2060: 80 t/s on
  free text, 176 t/s on repeated content. The Gemma 4 assistant MTP head is a net loss on real content in our tests.
- Vision projector and speculation in the same server (speculation is skipped for requests with media).
- Fixed: the `ngram-*` and `draft-simple` speculation types could corrupt the output.
- Fixed: `llama-finetune` crashed in the warm-up pass after upstream merges (fused Gated DeltaNet detection) and two
  follow-up regressions; the MoE training pipeline runs end to end again.
- Dequantization kernels with several warps per block and pre-scaled VTQ codebooks; Vulkan KTQ2_1 dequant test harness.
- Eagle3 draft-head infrastructure: hidden states from three layers, GGUF keys, head graph fusion, converter for HF
  checkpoints (dormant until a trained head loads).

## 2026-05

- **TurboQuant v8:** short type names `ktq1`-`ktq4` / `vtq1`-`vtq4`, new `vtq3_v8` (3.625 bpw, trellis with two fp16
  outliers per block); `--no-tq-deferred-k/-v` for GPUs with little memory.
- `ktq2_1` + `vtq2_1` could end long code generations early; the recommendation moved to `vtq2_2` (2026-05-03).
- **LoRA fine-tuning on quantized MoE models:** `MUL_MAT_ID` adapter gradient, LoRA on the expert tensors (3D adapters),
  adapters saved as `.lora.gguf`, `--lora-train-target/-rank/-alpha`, adapter flush on SIGTERM / SIGINT, layer split over
  two GPUs; Qwen3.6-35B-A3B IQ2_XXS trains on one 12 GB GPU.
- `GGML_OP_QUANTIZE_DEQUANTIZE_FAKE` with a straight-through backward (groundwork for quantization-aware training).
- Vulkan backend tuned for Turing (KV rows per thread).
- Inline tensor-core kernel for KTQ+VTQ prompt processing: 2-3.5× faster prefill from 1k to 59k tokens (replaced on
  2026-10-05 by f16 split dequantization plus the standard tensor-core kernel).

## 2026-04

- **TurboQuant KV cache, v1-v7 (2026-04-06 to 04-14):** K quantized after a randomized Hadamard transform with Lloyd-Max
  codebooks (inspired by PolarQuant and TurboQuant), CPU reference, warp-parallel FWHT, precomputed sign bits and compact
  blocks (3.5 / 4.5 / 5.5 bpw), K·Q dot product in the Hadamard domain (decode +65 % on Turing over v6). QJL was tried and
  removed.
- 2026-04-15: garbage output with flash attention and TurboQuant K and V together traced to the interaction of the K and
  V paths; fixed in steps, with an asymmetric K/V workaround in between.
- 2026-04-16: KTQ/VTQ split; the VTQ V types are rotated in the graph and dequantize with a single codebook lookup.
- 2026-04-23: VTQ trellis types (`vtq*_2`, group Viterbi) and trellis with fp16 outliers (`vtq*_3`).
- 2026-04-24: tensor-core path for KTQ prompt processing on Turing.
- 2026-04-26: MoE speed-ups for models larger than VRAM (huge pages, `mul_mat_id` prefetch, adaptive layer split, AVX2
  FWHT): Qwen3-Next-80B decode +18.5 %.
- 2026-04-27: `--moe-pin-experts`, head size 64 fix for gpt-oss, Anthropic-compatible `/v1/messages` with prompt
  caching. XQuant cross-layer K reuse (removed 2026-10-09).
