# Changelog

## [Unreleased] feature/tq-finetune — Sparse fine-tuning for hybrid MoE+SSM

### Added

- `--train-skip-regex REGEX` flag for `llama-finetune` — freeze tensors by ECMAScript regex pattern. Frozen tensors transitively prune the backward graph via `grads_needed=false` propagation, so unimplemented backward ops (Mamba/SSM/MoE/FlashAttn) never get called.
- `GGML_BACKWARD_SKIP_INPLACE=1` env var to bypass the inplace-op assert in `ggml_build_backward_expand`. Required for any model with Mamba/SSM state propagation.
- `GGML_OPT_LINE_PROGRESS=1` env var for tee/pipe-friendly per-step progress logs.
- `LLAMA_SAVER_ALLOW_UNTESTED=1` env var to force-save unvalidated architectures (Qwen3.5MoE etc.).
- `UNARY_OP_SIGMOID` backward pass (analytical: σ(1-σ)).
- Graceful skip of unsupported backward ops (`MUL_MAT_ID`, `FLASH_ATTN_EXT`, `SSM_*`) when env opt-in is set.
- New hparam `orig_n_ctx_train` to preserve model context length across fine-tune save cycles.

### Fixed

- `common_opt_dataset_init` unsigned underflow for inputs shorter than `n_ctx`. `tokens.size() - n_ctx` is now cast to `int64_t` with an explicit assert.
- `llama_model_saver` duplicate `LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH` write that clobbered MoE expert dims (`n_ff_shexp` overwritten with `n_ff_chexp=0`).
- `llama_context::opt_init` was overwriting `hparams.n_ctx_train` with training batch ctx, breaking saved GGUFs for downstream inference.

### Verified

- Fine-tuned Qwen3.6-A35B-A3B IQ2_XXS (35B MoE + GatedDeltaNet) on 2× RTX 2060 12 GB.
- 6 h 21 min training, 250 samples × 1 epoch, Loss 5.44 → 1.40.
- Output GGUF loads cleanly in `llama-server` with `--chat-template-file`.
- Fair bench on 210-sample tool-calling suite: Base 63.78 % → V1 63.35 % (Embed + Head-only delta within noise).
