# Changelog

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

- **MoE expert cache** (`--moe-cache-mib N`, upstream PR #29887): an LRU cache in VRAM for experts kept in host memory (`-cmoe` / `-ncmoe` / `-ot`). Decode batches (up to 32 tokens) remap the selected experts to cached copies instead of reading them over PCIe. With several GPUs the cache sits on the device whose layers keep the most experts in host memory; pipeline parallelism is turned off while the cache is active.
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
