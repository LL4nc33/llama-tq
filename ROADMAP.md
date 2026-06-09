# Roadmap

This file tracks what works, what's in flight, and what's on the wishlist. Maintained alongside the fork — feedback and PRs welcome.

## ✅ What works today

- **Full MoE-expert LoRA fine-tuning** of `ffn_*_exps` on Qwen3.6-A35B-IQ2_XXS, end-to-end on a single RTX 2060 12 GB (2026-05-18). Train → save (`.lora.gguf`) → load via `--lora` → inference, all green. Validated hyperparams: rank=2, alpha=4, lr=5e-6, SGD, ctx=128. Loss 4.0 → 2.4 on a 500-line × 3-epoch run.
- **Sparse fine-tuning** of Embed + LM-head + Norms on hybrid MoE+SSM models (Qwen3.5/3.6-A35-A3B, Bamba, Nemotron-3-MoE) — the older, simpler path, still supported.
- **`llama_adapter_lora_save_to_file` API** — adapters serialise to `.lora.gguf` with the metadata the `--lora` loader expects.
- **SIGTERM / SIGINT safety flush** in `llama-finetune` — multi-hour runs survive `timeout` and `Ctrl+C` without losing the adapter.
- **`GGML_OP_QUANTIZE_DEQUANTIZE_FAKE` op** — forward (CPU compute) and STE backward landed. Public API: `ggml_quantize_dequantize_fake(ctx, F32_tensor, target_quant)`. CLI flag + LoRA-graph integration still queued (see Phase C below).
- **TurboQuant KV cache** at 2.78 bpw (KTQ + VTQ v2 Trellis) with f16-equivalent quality.
- **CUDA backend** on sm_75+ (Turing tested daily); compiled binaries for sm_75/80/86/89/90/120.
- **Dual-GPU layer-split** for both the sparse fine-tuning path and the LoRA-on-quantised-base path (verified on 2× RTX 2060 12 GB, Phase D resolved 2026-05-19). Useful for VRAM headroom (200k+ ctx, higher rank, AdamW momenta); not a wall-clock speedup on consumer rigs without working P2P — see Phase D below for numbers.

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

### Phase D — Multi-GPU LoRA training on quantised base (2026-05-19: FUNCTIONAL)

Layer-split (`-sm layer -ts a,b`) for the LoRA-on-quantised path is now **functional** on 2× RTX 2060 12 GB. Smoke: Qwen3.6-A35B-IQ2_XXS, `-ts 1,1`, 222 steps SGD, loss 3.247 → 1.379, acc 31% → 64%, ~5 GB peak per GPU. Same numerical trajectory as the single-GPU path.

Two scheduler fixes were needed (both on `feature/phase-d-multigpu-lora`):
- `9d136dee5` — `sched->graph_inputs[]` is now a dynamic array. The previous fixed-size `GGML_SCHED_MAX_SPLIT_INPUTS=30` cap fits inference but is overrun by training graphs (forward + backward + per-param `OPT_STEP`) on a layer-split MoE.
- `93388f61d` — new `ggml_backend_sched_invalidate_prev_backend_ids()` plus a sentinel-aware comparator. The scheduler previously cached `prev_node_backend_ids` from the prior shape; on a `gf → gb_grad → gb_opt` switch it missed the change, skipped the reserve-and-retry path, and segfaulted at `init_tensor` on a stale `buffer_id`. The invalidate helper is called from `ggml_opt_alloc` on every graph-shape change and forces the realloc path.

**Caveat — dual-GPU is not a speedup for fitted workloads.** Layer-split is sequential, and on consumer dual-2060 rigs without working P2P/NVLink, the cross-device sync cost (asymmetric PCIe x16+x4) outweighs the compute parallelism. Measured: ~0.35 step/s dual vs ~0.50 step/s single (~44% slower per step). Use dual-GPU when the workload does not fit single-GPU (200k+ ctx, higher rank, AdamW momenta), not as a free speedup. See `docs/phase-d-smoke-results.md`.

**Next step.** rank=4 + AdamW reachable now that VRAM doubles, which addresses the SGD-only drift seen in the earlier 10335-sample run. Validation pending.

## ⚠️ Known quality gaps

- **Saver audit.** `LLAMA_SAVER_ALLOW_UNTESTED=1` works around missing MoE hparam writes (`swiglu_clamp_shexp`, `expert_groups`, `n_layer_dense_lead`, `mamba_d_*`). A full audit would catch silent bugs.
- **Re-quantisation error in the sparse path.** `token_embd` + `output` train in FP32, then re-quantise back to IQ2_XXS on save. Quant error may eat part of the learning signal there. The LoRA path doesn't have this problem — the adapter is stored as F32 and applied at inference.
- **LoRA-on-quantised: rank ≤ 2 for 282 expert pairs on 12 GB single-GPU.** rank=4 OOMs by ~1.3 GB on single GPU. AdamW doesn't fit either. Dual-GPU layer-split lifts this ceiling (Phase D done) — actual rank=4 / AdamW validation on dual-GPU is the next milestone.
- **LoRA-on-quantised: ctx=128 default.** Higher contexts need gradient checkpointing, not yet implemented.
- **LoRA-on-quantised: convergence sweet spot is 100–500 sample subsets × 3 epochs, lr ≤ 1e-5.** Bigger single-runs (28k lines × 1 epoch) diverge with the current `grad_b`-skip setup. Split into sequential subsets.
- **Convergence path through activations is truncated** by the quant-`src0` `grad_b` skip. Mathematically correct for single-target-layer LoRA; gives looser gradient flow for multi-layer LoRA stacks. Dequant-on-the-fly (Phase C) would address this.

## Maintenance policy

- **Upstream sync.** Upstream `llama.cpp` fixes (CUDA, server, build) are cherry-picked when they apply cleanly. Larger upstream features (MTP, fusion infrastructure) are integrated case-by-case as they stabilise.
- **Stability target.** TurboQuant kernels: CUDA sm_75+, daily-driven on Turing RTX 2060. Vulkan and HIP are experimental. macOS / Metal are upstream-stock.
- **Regressions are blockers.** Each merge to `turboquant` (the default branch) must pass the local PPL + speed gates before landing.

## Upstream integration log

- 2026-06-06: Upstream CUDA fusion infrastructure integrated (PR 22468 refactor includes muls and relu+sqr fusion paths, PR 22478 SSM_CONV ADD SILU, PR 22667 snake activation, PR 22912 snake hardening). TurboQuant KTQ/VTQ kernels untouched, bench parity verified on 0.8B-Q8 and 35B-A3B-IQ2_XXS (PPL identical, TG/PP within plus-minus 0.6 percent).
- 2026-06-06: Upstream targeted fixes (PR 23610 fattn-mma-f16 KQ-mask int64 overflow safety, PR 23893 cli model-params propagation, PR 23822 mtmd Gemma 4 projector pre_norm fix). KTQ2_1 plus VTQ2_1 PPL byte-identical to pre-pick baseline.
- Deferred: MTP support (PR 22673 plus 32 follow-ups) upstream PR introduced concurrent model-system architectural refactor (llm_build_X to llama_model_X with nested graph and graph_mtp structs plus new llama_model_loader virtual interface). Forks TurboQuant cparams plus memory flags hang off the old model system. Integration requires multi-day model-system migration, scheduled as separate roadmap phase. No local MTP draft model available for runtime testing anyway.
- Deferred: PR 24087 mul_mat_vec_q_moe pdl enrollment requires upstream PDL helper infrastructure (ggml_cuda_kernel_launch_params) not present in fork.
- 2026-06-06: Upstream MTP foundation integrated via 7-commit chain on feature/upstream-mtp-full-integration-2026-06-06 (PR 21971 NVFP4, 21245 QKV refactor, 21970 single llm_build per arch, 22079 bias renames, 22004 model-class refactor + TQ wiring re-injection in memory/kv_cache constructors, 22673 MTP core). Library-level MTP API available. Bench parity verified on 0.8B-Q8 and 35B-A3B-IQ2_XXS (PPL byte-identical, TG/PP within plus-minus 0.4 percent). Adapter commit fork(model) provides post-22004 backfill: HUNYUAN_VL, ROPE_SCALING_ALPHA, LLM_TYPE_31B, LLM_TYPE_26B_A4B enum stubs plus llama_params_fit() success-returning stub plus mmap_huge in llama_model_loader call. TurboQuant kernels untouched.
- 2026-06-06: MTP-aware CLI flags plus server speculation paths NOT integrated in phase 2G (common/common.h plus arg.cpp plus speculative.* plus tools/server/server-context.cpp reverted to pre-MTP to preserve TQ-cparams plus xquant config plus fork server features). MTP-runtime exposure requires manual port of common_params_speculative restructure, planned as own roadmap phase. No local MTP draft model yet for runtime test.
- 2026-06-06 evening: MTP runtime port done. common/speculative.cpp+h ported to new MTP API. tools/server/server-context.cpp builds clean with 3 speculation calls stubbed (init/begin/accept arg-shape mismatches). CLI flags for draft model (--hf-repo-draft, --cache-type-k-draft, etc.) work. Server speculation runtime requires seq_id wiring fix in a follow-up; library MTP + llama-cli draft loading code path functional. Bench parity preserved (0.8B-Q8, 35B-A3B-IQ2_XXS, KTQ2_1+VTQ2_1 PPL byte-identical).
- 2026-06-06 evening 2: MTP runtime test done with Qwen3.6-35B-A3B-MTP-UD-IQ2_XXS (11.8GB GGUF from unsloth, MTP head verified via nextn_predict_layers=1). Model loads, runs coherently 70-78 t/s, server speculation can_spec gate now reached via has_dft() recognizing fork-compat mparams_dft path. MTP self-speculation NOT auto-triggered because arg.cpp fork-legacy --draft flags do not register a common_speculative_type in the new types vector. Full MTP CLI activation needs arg.cpp restructure (separate roadmap phase). Phase 2 declared complete at code-integration level.
- 2026-06-06 evening 3: MTP self-speculation FUNCTIONAL with 100 percent draft acceptance verified via Qwen3.6-35B-A3B-MTP-UD-IQ2_XXS. Test results on 2x RTX 2060 12GB Turing dual-GPU: baseline (target dual-GPU, no spec) 71 t/s; spec with target+draft each on 1 GPU 18 t/s (target halved by single-GPU constraint); spec with target dual-GPU + draft on CPU 9 t/s (CPU draft dominates wall-clock). Speculation framework correct (acceptance 38/38, draft KV sync via state_seq copy), but this hardware tier cannot host target + draft in parallel without splitting target compute. Reference benches (Unsloth +20 percent) require single GPU with sufficient VRAM to hold target plus MTP head in parallel (e.g. A100). On a local GPU host prod-grade speed remains baseline 71 t/s; speculation is available code-side for future hardware upgrades.
- 2026-06-07: Phase 33 — FA-kernel `minBlocksPerSM` raised from 1 to 2 for Turing occupancy. Phase 35 — target-step kernel bandwidth utilisation taken from 53% to 80%+ via fused dispatch. Phase 37 — kernel-fusion availability gate added for MoE-IQ2 TG path.
- 2026-06-07 evening: Phase 38-40 — bartowski Qwen3.6-35B-A3B-IQ2_XXS quant benched 5% faster than unsloth UD on RTX 2060 (86.6 vs 82.6 t/s). KV-cache pareto sweep on bartowski-IQ2_XXS settled on KTQ2_1 + VTQ2_1 as the prod default. mmproj + spec coexistence design landed.
- 2026-06-07 late: Phase 41 — server-context.cpp hard-blocks on `slot.has_mtmd` lifted. Phase 41b — per-request `has_media()` gate introduced via new `server_tokens` accessor so spec / ctx-shift / cache-reuse are only disabled for the requests that actually carry image tokens. Phase 42 — code review pass on 41b, dead-code cleanup. Phase 43-44 — upstream PR patch prepared (`docs/plans/upstream-PR-server-spec-mmproj-coexistence.{md,patch}`) and verified against upstream HEAD build.
- 2026-06-07 late 2: Phase 45-47 — comprehensive grep for residual `has_mtmd` asserts, pre-github-push private-pattern scrub for the upstream-PR branch, scrub of dirty files for github-pushability. Patch author corrected to GitHub no-reply identity.
- 2026-06-08: n-gram speculation static cache pretraining on wikitext (Phase 30-31, LLAMA_NGRAM_STATIC tuned 2→4). n-gram + MTP hybrid acceptance rate reaches 85%+ on creative prompts (Phase 17, 24). Prod deploy 2026-06-08: bartowski Qwen3.6-35B-A3B-IQ2_XXS + ngram-spec + dual-GPU + 200k ctx live on a local GPU host:8791 — 80 t/s creative, 176 t/s repeat (2.28x universal boost).
- 2026-06-08 evening: Phase 48-49 — Eagle3 hidden-state-extraction infrastructure (Step B.3) — graph builder taps in qwen35 dense + MoE forward pass. Phase 50 (Step B.4) — async tensor copies for the three taps + public C API `llama_get_embeddings_eagle3_{low,mid,high}_ith`. Phase 51 (Step B.5) — GGUF KV reader/writer for `eagle3_layer_{low,mid,high}` plus loader hookup on `feature/mtp-shared-ctx`. All Phase B steps committed with 0 regression on the prod TG bench (~80 t/s baseline preserved at every step).
- 2026-06-09 (Phase C, Eagle3 head graph): Step C.1 — `LLM_TENSOR_NEXTN_EAGLE3_FC` enum + name mapping `blk.%d.nextn.eagle3_fc`. Step C.2 — `llm_graph_input_embd_h` extended with `h_low/h_mid/h_high` slots. Step C.3 — graph fusion path in `graph_mtp` (dense + MoE) + loader for the optional `eagle3_fc` weight. Step C.4 (part 1) — `set_input` / `can_reuse` wiring for the three streams. Step C.5 — converter (`_Qwen35MtpMixin` + gguf-py tensor map): HF `eagle3.fc(.weight|.bias)` → per-block `nextn.eagle3_fc`, KVs read from either top-level `eagle3_layer_*` or nested `eagle3.layer_*`. The runtime graph picks the fusion path up automatically when both KVs and `eagle3_fc` are present. Non-eagle3 checkpoints leave the slot `nullptr` and use the single-stream MTP path unchanged. Driver-side fill of the embd buffer from the new ctx getters (C.4 part 2) deferred until after a trained eagle3 head exists.
- 2026-06-09: Vulkan KTQ2_1 dequant validation harness merged (PR #11) — productionised the Layer-3 PoC into `tests/test-vulkan-tq-dequant.cpp` with random-block + bit-pattern-golden test cases, gated behind `LLAMA_TESTS_VULKAN_TQ`. Lays the foundation for the dormant Vulkan KTQ/VTQ port.
- 2026-06-09: VTQ/KTQ dequant perf-kernels merged (PR #12) — multi-warp-per-CTA dequant for VTQ NC + KTQ2_1 convert (4× occupancy), pre-scaled VTQ codebook in read-path decoders (one mul/element eliminated), VTQ2_1 4-outputs-per-thread NC dequant kernel. Plus build-snapshot/restore scripts and the OidaNice-GPT-34B + Ministral-3-3B optimal deploy scripts. Smoke parity: 0.8B-Q8 TG 225.40 t/s vs 224.37 baseline (within noise).
