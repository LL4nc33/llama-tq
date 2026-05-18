# Phase D smoke-test results (2026-05-18)

Smoke-tests run on `feature/phase-d-multigpu-lora` branch (build `30b53877b`+, code-equivalent), test-rig (2× RTX 2060 12 GB).

## What works

| Test | Config | Status | Notes |
|------|--------|--------|-------|
| 0.8b dual-GPU SGD | `qwen3.5-0.8b-q8_0`, rank=2, `-ts 1,1 -sm layer` | ✅ PASS | 72 LoRA pairs, loss 5.87→0.16, adapter saved 2.6 MB |
| 0.8b dual-GPU AdamW | same + `--optimizer adamw` | ✅ PASS | Same convergence, AdamW does not OOM on small model |
| 0.8b dual-GPU AdamW rank=4 | same + rank=4 alpha=8 | ✅ PASS | Confirms rank=4 fits on 2× 12 GB for small model |
| 35b dual-GPU SGD (start) | `Qwen3.6-A35B-IQ2_XXS`, rank=2 | ⚠️ PARTIAL | Training starts (loss 4.29 step 1), segfaults at later step |

## What does not work

| Test | Config | Failure mode |
|------|--------|--------------|
| 35b single-GPU AdamW | `CUDA_VISIBLE_DEVICES=0`, rank=2 | `GGML_ASSERT(tensor->data) failed` in `ggml_set_zero` during `ggml_graph_reset`. AdamW momenta (`m`, `v`) not allocated for quantised base tensors. |
| 35b dual-GPU AdamW | `-ts 1,1 -sm layer` | `pre-allocated tensor (adamw step for ...lora_a) in a buffer (CUDA1) that cannot run the operation (OPT_STEP_SGD)`. Scheduler-device mismatch — LoRA tensors on CUDA1 but op routed to wrong backend. |
| 35b dual-GPU SGD (full run) | `-ts 1,1 -sm layer` | First step completes (`loss=4.29 acc=31%`), then segfault. Reproducible. |

## Required skip-regex addition

`ssm_` must be added to `--train-skip-regex` for any hybrid MoE+SSM model. The default `(mamba|...)` does not match `ssm_beta`, `ssm_*` tensors.

Working regex for Qwen3.6-A35B:
```
--train-skip-regex '(mamba|ssm_|^token_embd|^output|_norm|attn_|ffn_gate_inp|shexp)'
```

## Root causes identified

1. **AdamW + quantised base = no momenta allocation.** `ggml_opt_build` allocates `m`/`v` momenta only for `GGML_TENSOR_FLAG_PARAM` tensors, but the quantised base weight has no buffer/data for the momenta tensors that reference it. The momenta tensors need their own buffer allocation independent of `src0`.

2. **Layer-split + LoRA = device-affinity bug.** `OPT_STEP_SGD`/`OPT_STEP_ADAMW` nodes for LoRA tensors on CUDA1 cannot resolve their backend in `ggml_backend_sched_backend_id_from_cur`. Likely the optimizer-step op gets created on the default backend instead of inheriting from the LoRA tensor's buffer.

3. **35b dual-GPU SGD step-2 segfault.** Less clear — possibly related to (2): the second step's graph reset hits the same scheduler issue but instead of asserting it segfaults. Needs `gdb` session.

## Verdict

**Phase D is not deliverable as a drop-in for the 35B production run today.** The infrastructure works on small models, but the LoRA-tensor placement across devices needs explicit device-affinity handling in:

- `llama_adapter_lora_init_for_training` — pin lora_a/lora_b to the same device as the base tensor
- `ggml_opt_build` — route optimizer-step ops to the device of the param tensor, not the default backend
- AdamW momenta allocation — separate buffer alloc for quantised-base LoRA pairs

Estimated dev effort: 1-2 days for a proper fix.

## Workaround for distillery

The validated 2026-05-18 morning path remains the only stable option:
- Single-GPU 35B + SGD + rank=2 + lr ≤ 1e-5
- Subset training (100-500 samples × 3 epochs) within convergence envelope
- Sequential adapter merging if more data needed

This is exactly what `project_convergence_boundary.md` documented.
