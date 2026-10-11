# Delta from upstream llama.cpp

Compact list of what this fork adds on top of upstream. The overview with numbers lives in
[README.md](../README.md); status and known issues in [ROADMAP.md](../ROADMAP.md).

| Area | What the fork adds | Main locations | Details |
|------|--------------------|----------------|---------|
| TurboQuant KV cache (v8) | KTQ (K) and VTQ (V) cache types: `ktq{1,2,3,4}` and `vtq{1,2,3,4}` short aliases plus the long forms (`ktq*_1`, `vtq*_1` codebook, `vtq*_2` trellis, `vtq*_3` trellis + outliers, `vtq3_v8`). Hadamard-domain K dot product in flash attention, deferred f16 staging during prefill. | `ggml/src/ggml-cuda/turboquant.cuh`, `fattn-*`, `ggml-quants.c`, `src/llama-kv-cache.cpp` | [turboquant.md](turboquant.md) |
| Speculative decoding | MTP + n-gram hybrid with static cache, mmproj + speculation coexistence, DFlash / DFlash2 block-diffusion drafting, Eagle3 draft-head plumbing. | `common/speculative.*`, `src/models/dflash.cpp`, `tools/server/` | [speculative.md](speculative.md) |
| Tensor split without P2P | `-sm tensor` on GPUs without peer access: allreduce through mapped pinned host memory, optional bf16 transfer. | `ggml/src/ggml-cuda/allreduce-host.cu` | [tp-tq-design.md](tp-tq-design.md) |
| Ternary weights | `PQ2_0` / `PTQ1_0` group-128 ternary types with Hadamard-rotated activations, CUDA mat-vec / MMQ kernels, fast Walsh-Hadamard transform. | `ggml/`, `src/llama-quant.cpp` | [README.md](../README.md) |
| Qwen3.8 family | Qwen3.8 and Qwen3.8-Flash-Next (`qwen4exp`): per-layer n-gram embeddings, hyper-connections, compressed-attention indexer with sparse flash attention. Faster Gated DeltaNet kernels for hybrid Qwen3.5 / 3.8. | `src/models/qwen4exp.cpp`, `src/models/qwen35.cpp`, `ggml/src/ggml-cuda/` | [README.md](../README.md) |
| MoE LoRA fine-tuning | LoRA on `ffn_*_exps` of quantised MoE models: `MUL_MAT_ID` backward, adapter save as `.lora.gguf`, `--train-skip-regex`. | `ggml/src/ggml.c`, `examples/training/` | [finetune.md](finetune.md) |
| Vulkan (experimental) | Turing tunings, dormant KTQ/VTQ port on the `vulkan` branch. | `ggml/src/ggml-vulkan/` | [vulkan.md](vulkan.md) |

Upstream fixes are cherry-picked case by case; see [upstream-integration.md](upstream-integration.md).
