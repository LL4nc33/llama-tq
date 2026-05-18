# Roadmap

This file tracks what works, what's in flight, and what's on the wishlist. Maintained alongside the fork — feedback and PRs welcome.

## ✅ What works today

- **Full MoE-expert LoRA fine-tuning** of `ffn_*_exps` on Qwen3.6-A35B-IQ2_XXS, end-to-end on a single RTX 2060 12 GB (2026-05-18). Train → save (`.lora.gguf`) → load via `--lora` → inference, all green. Validated hyperparams: rank=2, alpha=4, lr=5e-6, SGD, ctx=128. Loss 4.0 → 2.4 on a 500-line × 3-epoch run.
- **Sparse fine-tuning** of Embed + LM-head + Norms on hybrid MoE+SSM models (Qwen3.5/3.6, Bamba, Nemotron-Nano) — the older, simpler path, still supported.
- **`llama_adapter_lora_save_to_file` API** — adapters serialise to `.lora.gguf` with the metadata the `--lora` loader expects.
- **SIGTERM / SIGINT safety flush** in `llama-finetune` — multi-hour runs survive `timeout` and `Ctrl+C` without losing the adapter.
- **`GGML_OP_QUANTIZE_DEQUANTIZE_FAKE` op** — forward (CPU compute) and STE backward landed. Public API: `ggml_quantize_dequantize_fake(ctx, F32_tensor, target_quant)`. CLI flag + LoRA-graph integration still queued (see Phase C below).
- **TurboQuant KV cache** at 2.78 bpw (KTQ + VTQ v2 Trellis) with f16-equivalent quality.
- **CUDA backend** on sm_75+ (Turing tested daily); compiled binaries for sm_75/80/86/89/90/120.
- **Dual-GPU tensor split** for the sparse fine-tuning path (verified on 2× RTX 2060 12 GB). The LoRA-on-quantised-base path has been validated on single GPU only — dual-GPU there is a follow-up VRAM-headroom improvement, not a correctness gate.

## 🚧 In flight

- **MTP (Multi-Token Prediction).** Upstream change has landed; integration into TurboQuant FA dispatch is pending non-trivial merge resolution (large conflict surface).
- **Vulkan backend.** KTQ/VTQ kernel port lives on branches `tq-vulkan-port-cpp` (origin) and `tq-vulkan-port-tests` (gitea). PP parity reached; TG -23 % gap remaining (upstream IQ-decode shader path).
- **MMQ + GLU fusion experiment.** Tracked on gitea branch `feature/mmq-glu-fusion`.
- **MMA-inline KTQ/VTQ.** Tensor-core path WIP on `feature/ktq-vtq-mma-inline`.
- **IQ2 MMQ toggle.** Per-build env switch on `iq2-mmq-toggle` (currently kept as a known-good fallback).
- **Stage-4 QAT integration.** The op exists; what remains is the `--qat-target-quant` CLI flag and wrapping `ab_cur` (LoRA-output) in the fake-quant op inside `build_lora_mm` / `build_lora_mm_id`. Design decision is settled (LoRA-output QAT); implementation is deferred.

## 🎯 Roadmap toward real capability gains in fine-tuning

Where we stand on the capability surfaces (2026-05-18):

| Component | Today | Needed for capability training |
|-----------|-------|--------------------------------|
| Token embeddings | trainable | extends vocab, but no new skills |
| LM head | trainable | only output distribution shift |
| Attention | frozen | **required** for reasoning + context tracking |
| MoE experts (`MUL_MAT_ID`) | **trainable via LoRA** (2026-05-18) | unlocks domain knowledge |
| Mamba / SSM | frozen | sequential state — nice-to-have |

### ✅ Phase A — MUL_MAT_ID backward (done 2026-05-18)

`ggml_mul_mat_id_grad_as` implemented on CPU + CUDA for the `as`-gradient. The `b`-broadcast case used by Qwen3.6-A35B (`n_used_b=1, n_used=8`) is dropped with a one-time warning — mathematically safe for the LoRA setup because the base weight is frozen and the LoRA path flows via `grad_as`. The strided `cont(transpose(W_q))` path that would otherwise force a multi-GiB block-copy of the quantised weight is gated out for quantised `src0`. Full LoRA training of `ffn_*_exps` converges on a single 12 GB GPU.

### Phase B — Attention backward without FlashAttn

Either port FA backward to CUDA or fall back to standard attention backward (exists but memory-expensive). Unlocks full block-level (attention + FFN) LoRA training in addition to the current expert-only path. Not started.

### Phase C — Research-grade

- **Stage-4 QAT — wire-up.** The ggml op is in. Remaining work: `--qat-target-quant` CLI flag, `common_params.qat_target_quant` field, and the `ab_cur` wrap in `build_lora_mm` / `build_lora_mm_id`. Once landed, the LoRA adapter can be trained to compensate for the base-model's quantisation error.
- **Dense LoRA gradient flow through quantised activations.** The autograd currently skips `MUL_MAT_ID grad_b` when `src0` is quantised. A dequant-on-the-fly path would let deeper LoRA stacks see end-to-end gradients through activations. Open research.
- **SSM_SCAN / SSM_CONV backward** for Mamba state training (mathematically non-trivial — selective state spaces).
- **Periodic mid-training checkpoint** — flush adapter every N steps so a crash mid-batch keeps progress. Currently flushes only at epoch boundary and on SIGTERM/SIGINT.

## ⚠️ Known quality gaps

- **Saver audit.** `LLAMA_SAVER_ALLOW_UNTESTED=1` works around missing MoE hparam writes (`swiglu_clamp_shexp`, `expert_groups`, `n_layer_dense_lead`, `mamba_d_*`). A full audit would catch silent bugs.
- **Re-quantisation error in the sparse path.** `token_embd` + `output` train in FP32, then re-quantise back to IQ2_XXS on save. Quant error may eat part of the learning signal there. The LoRA path doesn't have this problem — the adapter is stored as F32 and applied at inference.
- **LoRA-on-quantised: rank ≤ 2 for 282 expert pairs on 12 GB.** rank=4 OOMs by ~1.3 GB. AdamW doesn't fit either. Dual-GPU tensor-split or 8-bit optimiser state would lift this ceiling.
- **LoRA-on-quantised: ctx=128 default.** Higher contexts need gradient checkpointing, not yet implemented.
- **LoRA-on-quantised: convergence sweet spot is 100–500 sample subsets × 3 epochs, lr ≤ 1e-5.** Bigger single-runs (28k lines × 1 epoch) diverge with the current `grad_b`-skip setup. Split into sequential subsets.
- **Convergence path through activations is truncated** by the quant-`src0` `grad_b` skip. Mathematically correct for single-target-layer LoRA; gives looser gradient flow for multi-layer LoRA stacks. Dequant-on-the-fly (Phase C) would address this.

## Maintenance policy

- **Upstream sync.** Upstream `llama.cpp` fixes (CUDA, server, build) are cherry-picked when they apply cleanly. Larger upstream features (MTP, fusion infrastructure) are integrated case-by-case as they stabilise.
- **Stability target.** TurboQuant kernels: CUDA sm_75+, daily-driven on Turing RTX 2060. Vulkan and HIP are experimental. macOS / Metal are upstream-stock.
- **Regressions are blockers.** Each merge to `turboquant` (the default branch) must pass the local PPL + speed gates before landing.
