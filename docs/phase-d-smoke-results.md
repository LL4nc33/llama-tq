# Phase D smoke-test results

Updated **2026-05-19** after multi-GPU fix.

Smoke-tests run on `feature/phase-d-multigpu-lora` branch, 2× RTX 2060 12 GB test rig (asymmetric PCIe x16+x4, NVLink not available).

## What works (2026-05-19 update)

| Test | Config | Status | Notes |
|------|--------|--------|-------|
| 0.8b dual-GPU SGD | `qwen3.5-0.8b-q8_0`, rank=2, `-ts 1,1` | ✅ PASS | 72 LoRA pairs, loss 5.87→0.16, adapter saved 2.6 MB |
| 0.8b dual-GPU AdamW | same + `--optimizer adamw` | ✅ PASS | Same convergence, AdamW does not OOM on small model |
| 0.8b dual-GPU AdamW rank=4 | same + rank=4 alpha=8 | ✅ PASS | Confirms rank=4 fits on 2× 12 GB for small model |
| **35b single-GPU SGD** | `Qwen3.6-A35B-IQ2_XXS`, rank=2 | ✅ PASS | 262 steps, loss 3.247→1.420, acc 31%→63%, ~11.8 GB VRAM |
| **35b dual-GPU SGD** | same + `-ts 1,1` | ✅ PASS | 222 steps, loss 3.247→1.379, acc 31%→64%, ~5 GB per GPU |

## Throughput comparison (35B-A3B-IQ2_XXS, rank=2, ctx=256, b=ub=16, SGD)

| Config | 100 steps | 222 steps | ETA epoch (16320 batches) | Throughput |
|--------|-----------|-----------|---------------------------|------------|
| Single-GPU | 3:21 min | ~7:25 min (extrap) | ~9:05 h | 0.50 step/s |
| Dual-GPU `-ts 1,1` | 4:49 min | 10:42 min | ~12:57 h | 0.35 step/s |

**Dual-GPU is ~44% slower per step.** Loss progression is identical (same numerical values up to numerical noise).

### Why dual-GPU is slower here

- `-ts 1,1` is **layer-split**, not tensor-parallel. GPUs work sequentially (layers 0-19 on GPU0 → cross-PCIe transfer → layers 20-39 on GPU1), not in parallel.
- The test rig has asymmetric PCIe (x16 + x4). The x4 link bottlenecks cross-device transfers per step.
- P2P direct GPU-GPU copy is blocked (IOMMU + NVIDIA consumer driver).
- For a model that already fits on one GPU (~11.8 GB peak with rank=2 + ctx=256), the cross-device sync overhead exceeds the compute parallelism win.

### When dual-GPU is the right call

- **VRAM-bound regimes** where the model + LoRA + opt state does not fit on a single GPU: 200k+ ctx, higher rank, AdamW momenta on quantised base (AdamW currently disabled for 35B by design but useful elsewhere).
- **Larger models** that single-GPU cannot host at all.
- Hardware with symmetric PCIe + NVLink or working P2P would amortize transfer cost much better; the 44% slowdown here is specific to this consumer-grade dual-GPU setup.

For routine 35B + rank=2 + ctx=256 LoRA finetuning on consumer dual-GPU rigs without NVLink/P2P, **single-GPU is the recommended path** — same convergence, ~30% less wall-clock.

## What does not work (unchanged from 2026-05-18 entry)

| Test | Config | Failure mode |
|------|--------|--------------|
| 35b single-GPU AdamW | `CUDA_VISIBLE_DEVICES=0`, rank=2 | `GGML_ASSERT(tensor->data) failed` in `ggml_set_zero` during `ggml_graph_reset`. AdamW momenta (`m`, `v`) not allocated for quantised base tensors. |
| 35b dual-GPU AdamW | `-ts 1,1 -sm layer` | `pre-allocated tensor (adamw step for ...lora_a) in a buffer (CUDA1) that cannot run the operation (OPT_STEP_SGD)`. Same momenta-allocation root cause. |

AdamW for quantised-base LoRA remains a separate workstream.

## Required skip-regex

`ssm_` must be added to `--train-skip-regex` for any hybrid MoE+SSM model. The default `(mamba|...)` does not match `ssm_beta`, `ssm_*` tensors.

Working regex for Qwen3.6-A35B:
```
--train-skip-regex '(mamba|ssm_|^token_embd|^output|_norm|attn_|ffn_gate_inp|shexp)'
```

## Root causes fixed (2026-05-19)

The dual-GPU 35B SGD step-2 segfault traced to two compounding issues in the scheduler:

1. **`sched->graph_inputs[]` was fixed-size** at `GGML_SCHED_MAX_SPLIT_INPUTS=30`. Training graphs (forward + backward + per-param OPT_STEP) on a layer-split MoE produce hundreds of cross-device input tensors — way past 30. Fixed in `9d136dee5` by replacing the fixed array with a heap-allocated one that doubles on overflow (mirrors the existing pattern for `sched->splits`).

2. **Scheduler comparator missed graph-shape switches.** `ggml-opt` reuses one scheduler across `gf → gb_grad → gb_opt` shape changes. `sched_alloc_splits` compares `node_backend_ids` against `prev_node_backend_ids`; both arrays still matched on overlapping indices, so the comparator skipped the reserve-and-retry path, and the galloc carried stale `buffer_id`s — `galloc->buffers[buffer_id]` was NULL for the new shape's wider node range. The first `init_tensor` then segfaulted at line 986. Fixed in `93388f61d` by adding `ggml_backend_sched_invalidate_prev_backend_ids` (sets both `node_*` and `prev_*` arrays to `-1` so the sentinel survives `split_graph`'s internal `node↔prev` swap), called from `ggml_opt_alloc` on every graph-shape change.

These two changes together make multi-GPU LoRA training functional. Single-GPU is unchanged (verified via regression smoke after each commit).

## Failed approaches before the fix (anti-patterns for future reference)

The fix took several iterations. Approaches that **did not work** and were reverted:

- Bumping `GGML_SCHED_MAX_SPLIT_INPUTS` from 30 → 128 → 512. Same assert moved to a different check site (line 1345 → 1351 → 1353); a compile-time constant was the wrong abstraction.
- Manual `ggml_backend_sched_reserve()` call from inside `ggml_opt_alloc`. Triggers single-GPU regression (CUDA `mul_mat_id` illegal memory access) because the reserve does its own reset+split internally and corrupts the calling scheduler state.
- Forcing `dup_graph` routing in the dynamic-mode opt path. Single-GPU regression (`tensor->buffer` NULL on `set_inputs`) because the duplicate's tensors carry no buffer-binding.

Bisecting back to the pre-Phase-D commit `7b2f7d0e0` confirmed single-GPU was green there and reverting the symptom-patches restored single-GPU before the real fix landed. Lesson logged.
