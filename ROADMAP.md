# Roadmap

This file tracks what works, what's in flight, and what's on the wishlist. Maintained alongside the fork — feedback and PRs welcome.

## ✅ What works today

- **Full MoE-expert LoRA fine-tuning** of `ffn_*_exps` on Qwen3.6-A35B-IQ2_XXS, end-to-end on a single RTX 2060 12 GB. Train → save (`.lora.gguf`) → load via `--lora` → inference, all green (2026-05-18).
- **Sparse fine-tuning** of Embed + LM-head + Norms on hybrid MoE+SSM models (Qwen3.5/3.6, Bamba, Nemotron-Nano) — the older, simpler path, still supported.
- **`llama_adapter_lora_save_to_file` API** — adapters serialise to `.lora.gguf` with the metadata `--lora` expects.
- **SIGTERM / SIGINT safety flush** in `llama-finetune` — multi-hour runs survive `timeout` and `Ctrl+C` without losing the adapter.
- **TurboQuant KV cache** at 2.78 bpw (KTQ + VTQ v2 Trellis) with f16-equivalent quality.
- **CUDA backend** on sm_75+ (Turing tested daily); compiled binaries for sm_75/80/86/89/90/120.
- **Dual-GPU tensor split** for fine-tuning (verified on 2× RTX 2060 12 GB).

## 🚧 In flight

- **MTP (Multi-Token Prediction).** Upstream PR #22673 landed; integration into TurboQuant FA dispatch pending non-trivial merge resolution (54-file conflict surface).
- **Vulkan backend.** KTQ/VTQ kernel port in progress on the `vulkan` branch. PP parity reached; TG -23% gap remaining (upstream IQ-decode shader).
- **Sparse K-Skip.** Fixed-threshold FA-vec optimisation on `feature/s199-sparse-k-skip`.

## 🎯 Roadmap toward real capability gains in fine-tuning

Where we stand on the capability surfaces (2026-05-18):

| Component        | Today      | Needed for capability training              |
|------------------|------------|---------------------------------------------|
| Token embeddings | trainable  | extends vocab, but no new skills            |
| LM head          | trainable  | only output distribution shift              |
| Attention        | frozen     | **required** for reasoning + context tracking |
| MoE experts (`MUL_MAT_ID`) | **trainable via LoRA** (2026-05-18) | unlocks domain knowledge |
| Mamba / SSM      | frozen     | sequential state — nice-to-have             |

### ✅ Phase A — MUL_MAT_ID backward (done 2026-05-18)

`ggml_mul_mat_id_grad_as` implemented on CPU + CUDA for the `as`-gradient. The `b`-broadcast case used by Qwen3.6-A35B (`n_used_b=1, n_used=8`) is dropped with a one-time warning — mathematically safe for the LoRA setup because the base weight is frozen and the LoRA path flows via `grad_as`. The strided `cont(transpose(W_q))` path that would otherwise force a multi-GiB block-copy of the quantised weight is also gated out. Full LoRA training of `ffn_*_exps` converges on a single 12 GB GPU.

### Phase B — Attention backward without FlashAttn (1-2 weeks)

Either port FA backward to CUDA, or fall back to standard attention backward (exists but memory-expensive). Unlocks full block-level training. Expected bench gain: +15-25 pp.

### Phase C — Research-grade (months)

- **SSM_SCAN / SSM_CONV backward** for Mamba state training (mathematically non-trivial — selective state spaces).
- **Quantization-Aware Training** — Stage-4 `GGML_OP_QUANTIZE_DEQUANTIZE_FAKE` op committed with Straight-Through Estimator backward. CLI flag `--qat-target-quant` + LoRA-graph integration are queued; once wired up, the LoRA adapter can be trained to compensate for the base-model's quantisation error.
- **Dense LoRA gradient flow through quantised activations** — currently the autograd skips `MUL_MAT_ID grad_b` for quantised `src0`. Adding a dequant-on-the-fly path would let deeper LoRA stacks see end-to-end gradients.

## ⚠️ Known quality gaps

- **Saver audit** — `LLAMA_SAVER_ALLOW_UNTESTED=1` works around missing MoE hparam writes (`swiglu_clamp_shexp`, `expert_groups`, `n_layer_dense_lead`, `mamba_d_*`). A full audit would catch silent bugs.
- **Re-quantization error** — `token_embd` + `output` train in FP32, then re-quantize back to IQ2_XXS on save. Quant error may eat part of the learning signal.
- **LoRA-only on quantised models** — full FP32 tensor updates aren't usable on quantised base weights anyway (transpose+cont is prohibitive); the LoRA path is the practical one. Dense FP32 training still works for non-quantised setups.
- **No gradient checkpointing** — limits practical context length (256–512 today, 2048+ would need it).
- **No periodic mid-training checkpoint** — adapters are flushed at epoch boundaries and on SIGTERM/SIGINT, but a crash mid-batch loses since-last-flush progress. Periodic save-every-N-steps is queued.
- **rank ≤ 2 for 282 LoRA-pairs on 12 GB** — rank=4 OOMs by ~1.3 GB. AdamW also doesn't fit. Dual-GPU tensor-split or `bitsandbytes`-style 8-bit optimiser state would lift this ceiling.

## Maintenance policy

- **Upstream sync.** llama.cpp fixes (CUDA, server, build) are cherry-picked when they apply cleanly. Larger upstream features (MTP, fusion infrastructure) are integrated case-by-case as they stabilise.
- **Stability target.** TurboQuant kernels: CUDA sm_75+, daily-driven on Turing RTX 2060. Vulkan and HIP are experimental. macOS / Metal are upstream-stock.
- **Regressions are blockers.** Each merge to `turboquant` (the default branch) must pass the local PPL + speed gates before landing.
