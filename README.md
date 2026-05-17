# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)

A [llama.cpp](https://github.com/ggml-org/llama.cpp) fork with two independent additions:

1. **TurboQuant** — independent K and V cache type families with Hadamard-domain Q·K dot product and Trellis-quantized V cache. 38% smaller KV than upstream's most aggressive quant at lossless quality.
2. **Sparse fine-tuning for hybrid MoE+SSM** — an extended `llama-finetune` that can train hybrid Mamba/MoE architectures (Qwen3.5/3.6-A3B/A35B, Nemotron-Nano, Bamba, RWKV-hybrids) directly on GGUF, without writing new backward kernels.

---

## What this enables

- **Long context on small GPUs.** A 35B-class MoE with 100k context and vision on a single 12 GB GPU. 200k parallel slots × 2 on a dual-12 GB setup.
- **Multi-tenant on cheap hardware.** Four concurrent 65k slots on a 20B-class MoE on a single 12 GB GPU.
- **Fine-tuning hybrid architectures.** LoRA-style sparse fine-tuning (Embed + LM-head + Norms) on a 35B hybrid MoE+SSM model in ~6 h on 2× RTX 2060 12 GB — no Mamba backward kernels required.
- **Drop-in upgrade.** Two extra flags (`--cache-type-k ktq2 --cache-type-v vtq2`); the rest of the llama.cpp CLI is unchanged.

---

<details>
<summary><h2>TurboQuant — how &amp; why</h2></summary>

`--cache-type-k ktq2 --cache-type-v vtq2` gives you a 2.78 bpw KV cache that fits a 35B-class MoE in a single 12 GB GPU at 100k context with vision, at f16-equivalent quality (−0.33% PPL drift vs f16, within stderr on wikitext-2).

**How.** K and V are decoupled into independent type families with their own flash-attention dispatch matrix (KTQ × VTQ). KTQ uses a Randomized Hadamard Transform plus a Lloyd-Max codebook; the attention dot product is computed in the Hadamard domain, so K never needs to be dequantized during attention. VTQ comes in three sub-families: codebook (v1), group-Viterbi Trellis (v2, current default), and Trellis with outlier-channel-split (v3, quality tier).

**Why.** Upstream's `q4_0` / `q8_0` KV options are uniform-quant baselines that either give up quality (`q4_0`) or eat half the budget (`q8_0`). KTQ and VTQ target the actual distribution shape of Q/K/V activations — RHT decorrelates K so a tiny codebook captures it without quality loss, and Trellis-coded V exploits the heavy-tail structure that uniform quantization wastes bits on. The result is the same quality as f16 at a quarter of the KV memory, which is what unlocks long context on small GPUs.

Full design notes: [docs/turboquant.md](docs/turboquant.md).

</details>

---

<details>
<summary><h2>Training Support: Hybrid MoE+SSM Models</h2></summary>

llama-tq ships an extended `llama-finetune` that supports **sparse LoRA-style fine-tuning** of hybrid Mixture-of-Experts + State-Space-Model architectures (Qwen3.5/3.6-A3B/A35B, Nemotron-Nano, Bamba, RWKV-hybrids). Upstream `llama.cpp` fails on these architectures because:

- `ggml_ssm_scan` / `ggml_ssm_conv` / `flash_attn_ext` have no backward implementation
- Mamba/SSM ops set `view_src` (inplace state) which the autograd whitelist rejects
- `MUL_MAT_ID` (MoE expert routing) has no backward
- `UNARY_OP_SIGMOID` had no backward
- The model-saver explicitly refused `LLM_ARCH_QWEN35MOE` and others

### What this adds

1. **`--train-skip-regex REGEX`** — Freeze tensors by name pattern. Frozen tensors transitively prune the backward graph via `grads_needed=false` propagation, so unimplemented backward ops never get called.

2. **`GGML_BACKWARD_SKIP_INPLACE=1` env** — Opt-in bypass for the inplace-op autograd assert. Required for any model with Mamba/SSM state propagation.

3. **`UNARY_OP_SIGMOID` backward** — Implemented analytically: `d/dx σ(x) = σ(x)(1-σ(x))`.

4. **Graceful skip for unsupported backward ops** — When `GGML_BACKWARD_SKIP_INPLACE=1` is set, ops like `MUL_MAT_ID` are silently dropped from the backward graph instead of aborting.

5. **`GGML_OPT_LINE_PROGRESS=1` env** — Per-step newlines instead of carriage-returns, for tee/pipe-friendly logs.

6. **`LLAMA_SAVER_ALLOW_UNTESTED=1` env** — Save GGUFs for architectures the saver hasn't been validated against (Qwen3.5MoE etc.).

7. **Saver bugfixes:**
   - Fixed duplicate `LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH` write that was clobbering `n_ff_shexp` with `n_ff_chexp=0` (broke MoE GGUFs)
   - Preserve original `n_ctx_train` when saving (was being overwritten by training batch ctx)

8. **Dataset underflow fix** — `common_opt_dataset_init` had `(tokens.size() - n_ctx - 1) / stride` with unsigned underflow on small inputs.

### Example: fine-tune Qwen3.6-A35B-A3B IQ2_XXS

```bash
GGML_BACKWARD_SKIP_INPLACE=1 \
LLAMA_SAVER_ALLOW_UNTESTED=1 \
GGML_OPT_LINE_PROGRESS=1 \
llama-finetune \
  -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
  -f train.txt \
  -o out.gguf \
  -ngl 99 -ts 6,5 \
  -c 256 -b 8 -ub 4 \
  -epochs 1 -lr 1e-5 -opt sgd \
  --train-skip-regex 'blk\.'
```

This trains only `token_embd` + `output` + `output_norm` on a 35B MoE+Mamba model in ~6 h on 2× RTX 2060 12 GB. All 40 transformer blocks (Mamba + MoE) remain bit-identical to the input GGUF.

### Verified hardware envelope

- 2× RTX 2060 12 GB (24 GB total VRAM)
- 35 GB host RAM peak (training)
- Tensor split: `-ts 6,5` (proportional VRAM allocation)
- 6 h 21 min, 250 samples × 1 epoch, Loss 5.44 → 1.40

### Trade-offs

- Embed + LM-head + Norms only training is a **weak** training surface — surface-distribution drift, not new capability
- For full MoE-expert training, `MUL_MAT_ID` backward still needs to be implemented
- All flags are opt-in via env vars — default behavior unchanged

Full mechanics + iteration log: [docs/finetune.md](docs/finetune.md).

</details>

---

<details>
<summary><h2>Build</h2></summary>

Standard llama.cpp build — see the [upstream build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md). TurboQuant kernels are CUDA-only (sm_75+ tested on Turing; should work on Ampere/Ada/Hopper). The Vulkan backend is a work-in-progress on the `vulkan` branch.

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j2 --target llama-server llama-finetune
```

</details>

---

## Maintenance &amp; roadmap

This fork is actively maintained alongside its own roadmap. Upstream `llama.cpp` fixes (CUDA, server, build) are cherry-picked when they apply cleanly; larger features (MTP, fusion infrastructure) are integrated case-by-case. See [ROADMAP.md](ROADMAP.md) for what's working, what's in flight, and the path toward real capability gains in fine-tuning.

---

## License

MIT — inherited from upstream llama.cpp.
