# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)

A [llama.cpp](https://github.com/ggml-org/llama.cpp) fork with two independent additions:

1. **TurboQuant** — independent K and V cache type families with Hadamard-domain Q·K dot product and Trellis-quantized V cache. 38% smaller KV than upstream's most aggressive quant at lossless quality.
2. **MoE LoRA fine-tuning directly on quantised GGUFs** — an extended `llama-finetune` that trains LoRA adapters on the expert weights of hybrid Mamba/MoE architectures (Qwen3.5/3.6-A35-A3B, Nemotron-3-MoE, Bamba, RWKV-hybrids) without dequantising the base. Adapter saves as a portable `.lora.gguf` loadable via `--lora`.

---

## What this enables

- **Long context on small GPUs.** A 35B-class MoE with 100k context and vision on a single 12 GB GPU. 200k parallel slots × 2 on a dual-12 GB setup.
- **Multi-tenant on cheap hardware.** Four concurrent 65k slots on a 20B-class MoE on a single 12 GB GPU.
- **Fine-tuning hybrid MoE architectures directly in IQ2 quant.** Full LoRA training of the MoE expert weights (`ffn_*_exps`) on Qwen3.6-A35B-IQ2_XXS, on a single 12 GB GPU. Train → save (.lora.gguf) → load via `--lora` → inference all green. No dequantization roundtrip to BF16/FP16. (Updated 2026-05-18.)
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

llama-tq ships an extended `llama-finetune` that supports **sparse LoRA-style fine-tuning** of hybrid Mixture-of-Experts + State-Space-Model architectures (Qwen3.5/3.6-A35-A3B, Nemotron-3-MoE, Bamba, RWKV-hybrids). Upstream `llama.cpp` fails on these architectures because:

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

9. **`--lora-train-target REGEX`** — Bootstrap a fresh LoRA training adapter at training start: every tensor matching the regex gets a freshly initialised `(lora_a, lora_b)` pair (A ~ Normal(0, 1/√rank), B = 0), the base is frozen, and only A/B receive gradients. Backed by a new `llama_adapter_lora_init_for_training` API.

10. **`MUL_MAT_ID` backward (CPU + CUDA)** — `ggml_mul_mat_id_grad_as` for the `as`-gradient, plus a graceful skip-and-warn for the `b`-broadcast case used by Qwen3.6-A35B (`n_used_b=1, n_used=8`) where the full gradient would need a reduce-sum kernel. For LoRA training the dropped `grad_b` is mathematically safe — the LoRA path flows via `grad_as`.

11. **Quantised-source backward skip** — When `MUL_MAT_ID`'s `src0` is a quantised tensor (the base MoE weight), the backward path that would build `cont(transpose(W_q))` is dropped because it would force a multi-GiB strided block-copy of the quantised weight per step. The base is frozen during LoRA training anyway, so this just truncates a sub-graph that the optimiser would have ignored. This is the root fix that makes Qwen3.6-A35B-IQ2 trainable in place.

12. **`llama_adapter_lora_save_to_file`** — Serialises the trained adapter to `.lora.gguf` with the metadata `--lora` expects, so the trained weights can be reloaded into any `llama-cli` / `llama-server`. `llama-finetune` writes this automatically as `<out_file>.lora.gguf` when a training adapter is active.

13. **SIGTERM / SIGINT safety flush in `llama-finetune`** — A signal handler writes the adapter to disk before exit, so multi-hour training runs that hit a shell `timeout` or `Ctrl+C` still leave a usable checkpoint behind instead of losing all progress.

14. **`GGML_OP_QUANTIZE_DEQUANTIZE_FAKE` (Stage-4 QAT op)** — Forward round-trips an F32 tensor through `target_quant` (Q4_0, IQ2_XXS, …) to bake the quantisation error into the activation; backward is a Straight-Through Estimator (identity). The op is fully wired into the CPU compute path and the autograd; the `--qat-target-quant` CLI flag and LoRA-graph integration are queued as a follow-up.

### Example: full MoE-expert LoRA finetune of Qwen3.6-A35B IQ2_XXS (2026-05-18)

```bash
GGML_BACKWARD_SKIP_INPLACE=1 \
LLAMA_SAVER_ALLOW_UNTESTED=1 \
llama-finetune \
  -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
  -f train.txt -o out.gguf \
  --lora-train-target '^blk\.[0-9]+\.ffn_(gate|down|up)_exps\.weight$' \
  --train-skip-regex '(mamba|^token_embd|^output|_norm|attn_)' \
  --lora-train-rank 2 --lora-train-alpha 4 \
  --optimizer sgd -lr 5e-6 --epochs 3 \
  -ngl 99 --flash-attn 0 -c 128 -b 16 -ub 16
```

This bootstraps a fresh LoRA adapter for the MoE expert weights (282 (lora_a, lora_b) pairs across all 94 layers × {gate, down, up} × 128 experts), trains via SGD directly on the IQ2_XXS base, and saves the result as `out.lora.gguf` — loadable in any llama.cpp binary via `--lora out.lora.gguf`.

### What works end-to-end

- **Training** of `ffn_*_exps` on Qwen3.6-A35B-IQ2_XXS converges (loss 4.0 → 2.4) on a single RTX 2060 12 GB
- **Adapter save** via new `llama_adapter_lora_save_to_file` API → portable `.lora.gguf`
- **Adapter load** via `llama-cli --lora` / `llama-server --lora`
- **SIGTERM/SIGINT safety flush** so `timeout` hits or Ctrl+C never lose progress

### Verified hardware envelope

- 1× RTX 2060 12 GB sufficient (single-GPU; multi-GPU split is a follow-up VRAM headroom improvement, not a correctness gate)
- 11.8 GB peak VRAM with rank=2 + 128 ctx + 282 LoRA pairs
- ~6:30 min for 100 lines × 3 epochs, ~33 min for 500 lines × 3 epochs
- Convergence sweet spot: 100-500 sample subsets, lr ≤ 1e-5, 3 epochs. Bigger single-runs (28k lines × 1 epoch) diverge — split into sequential subsets.

### Trade-offs

- `MUL_MAT_ID` `grad_b` is skipped when `src0` is a quantised tensor — necessary because `cont(transpose(iq3_s))` would force a multi-GiB strided block-copy per step. The LoRA path still flows via `grad_as` (the trainable A/B sit on top of a frozen base), which is mathematically correct for the LoRA setup. Dense gradient flow through quantised activations would need dequant-on-the-fly.
- `rank ≤ 2` for 282 LoRA-pairs on a 12 GB GPU (rank=4 OOMs by ~1.3 GB). AdamW also doesn't fit — SGD is mandatory at this VRAM budget.
- All flags are opt-in via env vars — default behaviour unchanged.

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
