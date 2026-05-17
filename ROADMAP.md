# Roadmap

This file tracks what works, what's in flight, and what's on the wishlist. Maintained alongside the fork — feedback and PRs welcome.

## ✅ What works today

- **Sparse fine-tuning** of Embed + LM-head + Norms on hybrid MoE+SSM models (Qwen3.5/3.6, Bamba, Nemotron-Nano).
- **End-to-end pipeline:** load IQ2_XXS → train → save GGUF → inference.
- **TurboQuant KV cache** at 2.78 bpw (KTQ + VTQ v2 Trellis) with f16-equivalent quality.
- **CUDA backend** on sm_75+ (Turing tested daily); compiled binaries for sm_75/80/86/89/90/120.
- **Dual-GPU tensor split** for fine-tuning (verified on 2× RTX 2060 12 GB).

## 🚧 In flight

- **MTP (Multi-Token Prediction).** Upstream PR #22673 landed; integration into TurboQuant FA dispatch pending non-trivial merge resolution (54-file conflict surface).
- **Vulkan backend.** KTQ/VTQ kernel port in progress on the `vulkan` branch. PP parity reached; TG -23% gap remaining (upstream IQ-decode shader).
- **Sparse K-Skip.** Fixed-threshold FA-vec optimisation on `feature/s199-sparse-k-skip`.

## 🎯 Roadmap toward real capability gains in fine-tuning

The current fine-tune path trains only Embed + LM-head + Norms. That's useful for surface-distribution drift (output style, format adherence, tool-call template fidelity) but **not new capability** — the model learns *how it sounds*, not *how it thinks*. Real capability training requires reaching the parts of the network where reasoning lives:

| Component        | Today    | Needed for capability training              |
|------------------|----------|---------------------------------------------|
| Token embeddings | trainable | extends vocab, but no new skills            |
| LM head          | trainable | only output distribution shift              |
| Attention        | frozen   | **required** for reasoning + context tracking |
| MoE experts (`MUL_MAT_ID`) | frozen | **required** for domain knowledge           |
| Mamba / SSM      | frozen   | sequential state — nice-to-have             |

Three phases to close that gap:

### Phase A — MUL_MAT_ID backward (~3-5 days)

Without this, no MoE expert can be trained. Skeleton documented; mathematically straightforward (routing gradient = gating × expert-output). Expected bench gain on tool-calling: +5-10 pp.

### Phase B — Attention backward without FlashAttn (1-2 weeks)

Either port FA backward to CUDA, or fall back to standard attention backward (exists but memory-expensive). Unlocks full block-level training. Expected bench gain: +15-25 pp.

### Phase C — Research-grade (months)

- **SSM_SCAN / SSM_CONV backward** for Mamba state training (mathematically non-trivial — selective state spaces).
- **Quantization-Aware Training** with straight-through estimator, so `token_embd` / `output` re-quantization doesn't eat the learning gain.

## ⚠️ Known quality gaps

- **Saver audit** — `LLAMA_SAVER_ALLOW_UNTESTED=1` works around missing MoE hparam writes (`swiglu_clamp_shexp`, `expert_groups`, `n_layer_dense_lead`, `mamba_d_*`). A full audit would catch silent bugs.
- **Re-quantization error** — `token_embd` + `output` train in FP32, then re-quantize back to IQ2_XXS on save. Quant error may eat part of the learning signal.
- **No LoRA adapters** — full FP32 tensor updates use 100× more VRAM than a LoRA path would.
- **No gradient checkpointing** — limits practical context length (256–512 today, 2048+ would need it).
- **No resume-from-checkpoint** — a crashed 6 h training run starts over from zero.

## Maintenance policy

- **Upstream sync.** llama.cpp fixes (CUDA, server, build) are cherry-picked when they apply cleanly. Larger upstream features (MTP, fusion infrastructure) are integrated case-by-case as they stabilise.
- **Stability target.** TurboQuant kernels: CUDA sm_75+, daily-driven on Turing RTX 2060. Vulkan and HIP are experimental. macOS / Metal are upstream-stock.
- **Regressions are blockers.** Each merge to `turboquant` (the default branch) must pass the local PPL + speed gates before landing.
