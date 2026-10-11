# Changelog

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
- CUDA releases: one archive per GPU generation with native code (Turing + PTX, Ampere sm_80/86, Ada, Blackwell + PTX)
  instead of a single Turing build that newer GPUs had to JIT-compile.
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
- KV type guidance from perplexity: `ktq4_1`/`vtq4_1` equals f16 on Qwen3.8-27B and Ternary-Bonsai-2-27B; `ktq2_1`/`vtq2_1` costs about +2.6 % there.
- q5_0 K/V flash attention without `GGML_CUDA_FA_ALL_QUANTS` (tensor cores for batches, the GQA kernel for decode).
- **K2-Horizon-MoVA-36B-A4B** (`k2-horizon`, MoE with routed value experts in attention). Q3_K_M on 2x RTX 2060: 47 t/s decode, 840 t/s prefill. With `ktq4_1`/`vtq4_1` KV plus `--tq-protect-layers 4` the PPL is +0.8 % over f16 at about a third of the KV memory.
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
  roughly doubled (26-28 t/s instead of 15.5 t/s, acceptance up to 100 %).
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
