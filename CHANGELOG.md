# Changelog

## 2026-10-13

- **f16 prefill on par with upstream.** The CUDA flash attention MMA kernel (`fattn-mma-f16.cuh`) and the tile helpers
  (`mma.cuh`) are now upstream's versions; the fork's copies were older upstream states with the swizzle and the sparse
  gather ported on top. On Turing they compiled to two to four times more register spills (Gemma 4 global layers, head
  size 512: 704-856 bytes per thread instead of ~180), mostly because upstream #29152 (thinner tiles for large head
  sizes) was missing. 2× RTX 2060, pp2048 at 32k with an f16 cache, upstream → llama-tq: Qwen3.8-27B 427 → 429 (before:
  4.8 % behind), Qwen3.6-35B-A3B 1286 → 1276, Gemma 4 12B 843 → 824 (before: 5.4 % behind). Gemma 4 is still 1-2 %
  behind; see [benchmarks](docs/benchmarks/README.md#against-upstream-two-gpus).
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
