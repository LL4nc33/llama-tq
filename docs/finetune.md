# llama-tq fine-tune patches

llama-tq extends `llama-finetune` so that hybrid Mixture-of-Experts + State-Space-Model architectures (Qwen3.5/3.6-A3B/A35B, Nemotron-Nano, Bamba, RWKV-hybrids) can be trained directly on quantized GGUFs without writing new backward kernels.

## Why upstream fails

Modern hybrid models combine Transformer attention with state-space (Mamba/SSM) layers and Mixture-of-Experts routing. Upstream `llama.cpp` cannot fine-tune these because:

1. `ggml_build_backward_expand` asserts on inplace ops with `view_src` (Mamba recurrent state).
2. `ggml_ssm_scan`, `ggml_ssm_conv`, `flash_attn_ext`, `mul_mat_id` have no backward implementations.
3. `UNARY_OP_SIGMOID` had no backward.
4. `llama_model_saver` explicitly rejects `LLM_ARCH_QWEN35MOE` and other unvalidated architectures.
5. Data-loader and saver bugs surface as soon as MoE GGUFs are written back.

Implementing missing backward kernels for every hybrid op is weeks of work with uncertain numerical correctness. Instead llama-tq takes the **selective sparse training** path: freeze the sub-graphs that have no backward, and train only what does.

## How the freeze works

The mechanism is autodiff bitmap propagation:

1. `--train-skip-regex REGEX` is parsed at dataset-init time. Tensor names that match are marked `is_param = false`.
2. `ggml_build_backward_expand` walks the graph and propagates `grads_needed[i] = false` transitively — if no consumer of a tensor needs its gradient, no backward op is emitted for it.
3. The whole sub-graph rooted at frozen tensors (Mamba scan, MoE experts, etc.) disappears from the backward pass. Missing backward kernels never run.
4. The remaining trainable surface is whatever isn't matched by the regex — for `'blk\.'`, that's `token_embd`, `output`, `output_norm`, etc.

This is conceptually similar to PyTorch's `requires_grad = False`, applied via regex at the GGML graph level.

## Patches

The fork adds 9 commits to upstream `llama.cpp` master:

| Commit message | Files touched |
|----------------|---------------|
| `finetune: --train-skip-regex flag + dataset-init underflow fix` | `common/`, `examples/training/finetune.cpp` |
| `ggml: GGML_BACKWARD_SKIP_INPLACE env to bypass inplace-op assert` | `ggml/src/ggml.c` |
| `ggml: extend GGML_BACKWARD_SKIP_INPLACE to also skip unsupported backward ops` | `ggml/src/ggml.c` |
| `ggml: fast-path skip for unsupported backward ops when no src needs grad` | `ggml/src/ggml.c` |
| `ggml: backward pass for UNARY_OP_SIGMOID` | `ggml/src/ggml.c` |
| `ggml-opt: GGML_OPT_LINE_PROGRESS env for per-step newlines` | `ggml/src/ggml-opt.cpp` |
| `model-saver: LLAMA_SAVER_ALLOW_UNTESTED env to attempt save for unvalidated archs` | `src/llama-model-saver.cpp` |
| `model-saver: fix duplicate LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH write` | `src/llama-model-saver.cpp` |
| `preserve original n_ctx_train when saving fine-tuned models` | `src/llama-context.cpp`, `src/llama-hparams.h`, `src/llama-model.cpp` |

Total diff: ~150 insertions across `common/`, `examples/training/`, `ggml/src/`, `src/` — all opt-in via flags or env vars, no behavior change for non-finetune callers.

## CLI flags and env vars

| Knob | Where | Effect |
|------|-------|--------|
| `--train-skip-regex REGEX` | `llama-finetune` CLI | Freeze tensors matching the ECMAScript regex. |
| `GGML_BACKWARD_SKIP_INPLACE=1` | env | Skip inplace-op assert; required for Mamba/SSM models. |
| `GGML_OPT_LINE_PROGRESS=1` | env | Emit newlines between training steps for tee/pipe-friendly logs. |
| `LLAMA_SAVER_ALLOW_UNTESTED=1` | env | Force-save GGUFs for architectures not on the saver whitelist. |

All env-gated patches default to upstream-equivalent behavior; setting the env vars is required to opt in.

## Example: fine-tune Qwen3.6-A35B-A3B IQ2_XXS

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

The regex `'blk\.'` matches every per-block tensor (Mamba scan, MoE experts, attention, norms inside blocks). Training surface reduces to `token_embd`, `output`, `output_norm`. All 40 transformer blocks (Mamba + MoE) remain bit-identical to the input GGUF.

## Quick smoke test

Verifies the save path without spending hours training:

```bash
echo "test sample. test sample. test sample." > /tmp/smoke.txt
GGML_BACKWARD_SKIP_INPLACE=1 LLAMA_SAVER_ALLOW_UNTESTED=1 \
  llama-finetune -m model.gguf -f /tmp/smoke.txt -o /tmp/smoke.gguf \
  -c 256 -ngl 99 -epochs 0 -val-split 0 --train-skip-regex 'blk\.'

llama-cli -m /tmp/smoke.gguf -p "test" -n 5 -ngl 99 --simple-io
```

If the smoke GGUF loads in `llama-cli` and generates output, the saver + loader path is healthy.

## Trade-offs

- **Embed + LM-head + Norms only is a weak training surface.** Useful for surface-distribution drift (output style, format adherence, tool-call template fidelity) but not for teaching new capability. The fair-bench delta on a 210-sample tool-calling suite was within noise (63.78 % → 63.35 %).
- **For full MoE-expert training**, `MUL_MAT_ID` backward needs to be implemented. The skeleton is documented; estimated ~3 days of work.
- **For real SSM_SCAN / SSM_CONV backward**, this is an open research problem (weeks of work, uncertain correctness).
- **The saver `add_kv_from_model` path is not complete for every MoE arch.** Some hparams (`swiglu_clamp_shexp`, `expert_groups`, `n_layer_dense_lead`) may not be written correctly. `LLAMA_SAVER_ALLOW_UNTESTED=1` works around this; a full saver audit would be cleaner.

## Verified hardware envelope

- 2× RTX 2060 12 GB (24 GB total VRAM), `-ts 6,5` proportional split
- 35 GB host RAM peak during training
- 6 h 21 min for 250 samples × 1 epoch on Qwen3.6-A35B-A3B IQ2_XXS
- Loss 5.44 → 1.40, output GGUF loads cleanly in `llama-server`

## Upstream issue references

The relevant upstream `llama.cpp` issues this fork addresses:

- ggml-org/llama.cpp#18805 — Mamba fine-tuning crashes on inplace assert
- ggml-org/llama.cpp#15279 — MoE expert routing has no backward
- ggml-org/llama.cpp#15090 — Hybrid model fine-tuning request
- ggml-org/llama.cpp#14424 — Sparse training proposal
- ggml-org/llama.cpp#9674 — Saver rejects modern architectures
