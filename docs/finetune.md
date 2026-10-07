# llama-tq fine-tune patches

`llama-finetune` is extended so hybrid Mixture-of-Experts + State-Space-Model
architectures (Qwen3.5/3.6-A35-A3B, Nemotron-3-MoE, Bamba, RWKV-hybrids) can be
trained directly on quantized GGUFs. Two training surfaces are supported, with
opt-in flags:

1. **Sparse training** of non-quantised tensors (`token_embd`, `output`,
   `output_norm`) — older, simpler path. Useful for surface-distribution drift.
2. **LoRA training of MoE expert weights** (`ffn_*_exps`) directly in the
   quantised base, with a portable `.lora.gguf` adapter saved at the end. This
   is the capability added on 2026-05-18.

Both share the same backward-graph pruning and saver-guard infrastructure.

## Why upstream cannot do this

Modern hybrid MoE+SSM models fail in upstream `llama.cpp` because:

1. `ggml_build_backward_expand` asserts on inplace ops with `view_src` (Mamba
   recurrent state).
2. `ggml_ssm_scan`, `ggml_ssm_conv`, `flash_attn_ext` have no backward.
3. `MUL_MAT_ID` (MoE routing) had no backward at all.
4. `UNARY_OP_SIGMOID` had no backward.
5. `llama_model_saver` explicitly rejects modern MoE architectures.
6. Even where backward exists, the `cont(transpose(W_q))` path it builds is
   prohibitive for quantised base weights (multi-GiB strided block-copy per
   step).

## The freeze mechanism

The first line of defence is autograd bitmap propagation:

1. `--train-skip-regex REGEX` is parsed at dataset-init. Tensor names that
   match are marked `is_param = false`.
2. `ggml_build_backward_expand` walks the graph and propagates
   `grads_needed[i] = false` transitively — if no consumer of a tensor needs
   its gradient, no backward op is emitted for it.
3. Whole sub-graphs (Mamba scan, attention, anything matching the regex)
   disappear from the backward pass.

Conceptually similar to PyTorch's `requires_grad = False`, applied via regex at
the GGML graph level.

## The LoRA-on-quantised-base mechanism

When `--lora-train-target REGEX` is set, a fresh LoRA adapter is bootstrapped
at training start:

1. Every tensor matching the regex gets a new `(lora_a, lora_b)` pair
   (A ~ Uniform(±1/√n_in) as in PEFT, B = zero-initialised, so the adapter starts
   as a no-op). 3D tensors (per-expert MoE weights) get one pair per expert slice.
2. The base tensor is added to the skip regex internally — only A and B
   receive gradients.
3. The LoRA-merged matmul lives in `build_lora_mm` / `build_lora_mm_id` and
   participates in the forward graph from the first step.
4. `MUL_MAT_ID` backward computes `grad_as` (the adapter gradient) via the
   new `ggml_mul_mat_id_grad_as` op (CPU + CUDA). The gradient into the
   activations goes through `ggml_mul_mat_id_grad_b`, which applies each
   selected (quantised) expert untransposed to the incoming gradient; CUDA groups
   the routing by expert, dequantises each used expert once and multiplies with
   cuBLAS. Dense matmuls with a frozen quantised weight get their input gradient
   from `out_prod`, which on CUDA dequantises the weight in row chunks. Without
   these input gradients the layers below an expert FFN would only receive
   gradient through the residual stream.
5. The trained adapter is serialised by `llama_adapter_lora_save_to_file` at
   the end of training, in the same GGUF format the `--lora` loader expects.
6. A SIGTERM/SIGINT handler triggers the same save path before exit, so
   training runs that hit a `timeout` or Ctrl+C still leave a usable
   checkpoint behind.

## CLI flags and env vars

| Knob | Where | Effect |
|------|-------|--------|
| `--train-skip-regex REGEX` | `llama-finetune` CLI | Freeze tensors matching the ECMAScript regex. |
| `--lora-train-target REGEX` | `llama-finetune` CLI | Bootstrap a fresh LoRA adapter for matching tensors. Base is frozen, only A/B train. |
| `--lora-train-rank N` | `llama-finetune` CLI | LoRA rank (default 8; use 1–2 for 282-pair MoE-expert sets on 12 GB). |
| `--lora-train-alpha FLOAT` | `llama-finetune` CLI | LoRA alpha (default 16). Effective scale = alpha / rank. |
| `-opt sgd` / `--optimizer sgd` | `llama-finetune` CLI | SGD optimiser. AdamW also supported but uses 2× VRAM. |
| `GGML_BACKWARD_SKIP_INPLACE=1` | env | Skip inplace-op assert; required for Mamba/SSM models. |
| `GGML_OPT_LINE_PROGRESS=1` | env | Emit newlines between training steps for tee/pipe-friendly logs. |
| `LLAMA_SAVER_ALLOW_UNTESTED=1` | env | Force-save GGUFs for architectures not on the saver whitelist. |

All env-gated patches default to upstream-equivalent behaviour.

## Example: full MoE-expert LoRA finetune of Qwen3.6-A35B IQ2_XXS

```bash
GGML_BACKWARD_SKIP_INPLACE=1 \
LLAMA_SAVER_ALLOW_UNTESTED=1 \
llama-finetune \
  -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
  -f train.txt \
  -o out.gguf \
  --lora-train-target '^blk\.[0-9]+\.ffn_(gate|down|up)_exps\.weight$' \
  --train-skip-regex '(mamba|^token_embd|^output|_norm|attn_)' \
  --lora-train-rank 2 --lora-train-alpha 4 \
  --optimizer sgd -lr 5e-6 --epochs 3 \
  -ngl 99 --flash-attn 0 -c 128 -b 16 -ub 16
```

The `--lora-train-target` regex matches the MoE expert weights — 282
`(lora_a, lora_b)` pairs across 94 layers × {gate, down, up}. Training writes
`out.lora.gguf` (~600 MB at rank=2), which loads in any llama.cpp binary via
`--lora out.lora.gguf`.

## Example: sparse Embed + LM-head + Norms finetune

The older, simpler training surface — useful for surface-distribution drift
without touching MoE experts:

```bash
GGML_BACKWARD_SKIP_INPLACE=1 \
LLAMA_SAVER_ALLOW_UNTESTED=1 \
GGML_OPT_LINE_PROGRESS=1 \
llama-finetune \
  -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
  -f train.txt -o out.gguf \
  -ngl 99 -ts 6,5 \
  -c 256 -b 8 -ub 4 \
  --epochs 1 -lr 1e-5 -opt sgd \
  --train-skip-regex 'blk\.'
```

`'blk\.'` matches every per-block tensor — training surface reduces to
`token_embd`, `output`, `output_norm`. The base GGUF is rewritten with the
trained tensors merged back.

## Quick smoke test

```bash
echo "test sample. test sample. test sample." > /tmp/smoke.txt
GGML_BACKWARD_SKIP_INPLACE=1 LLAMA_SAVER_ALLOW_UNTESTED=1 \
  llama-finetune -m model.gguf -f /tmp/smoke.txt -o /tmp/smoke.gguf \
  -c 256 -ngl 99 --epochs 0 --val-split 0 --train-skip-regex 'blk\.'

llama-cli -m /tmp/smoke.gguf -p "test" -n 5 -ngl 99 --simple-io
```

If the smoke GGUF loads in `llama-cli` and generates output, the saver +
loader path is healthy.

## Correctness fixes (2026-10-06)

Earlier runs only converged for a few hundred steps. The causes, all fixed:

- **Gradient accumulators were never cleared.** With graphs rebuilt per batch and
  `opt_period == 1`, every optimizer step used the sum of all previous gradients,
  so the effective step size grew until training diverged. This, not "gradient
  noise", was behind the old "100–500 samples × 3 epochs" limit.
- **LoRA A was initialised ~17× too large** (N(0, 1/√rank) instead of
  U(±1/√n_in)).
- **Missing input gradients:** `MUL_MAT_ID` dropped the gradient into the expert
  input for quantised experts and for the broadcast input of every MoE FFN, and
  `GET_ROWS` dropped it for the batched routing weights. Both now flow
  (`ggml_mul_mat_id_grad_b`, batched `get_rows_back`).
- **Speed:** the input gradient of every quantised dense matmul ran on the CPU
  (`out_prod` with a quantised weight); now on CUDA (Qwen3-4B: ~90 s → ~1 s per
  256-token step).
- **Two GPUs with AdamW:** moments and gradient accumulators are allocated next to
  their parameter (the optimizer step is in place); the scheduler is sized for the
  training graphs.

Verified on 2× RTX 2060 (layer split), AdamW lr 1e-4, 360 lines of wikitext, perplexity
of the base model vs. base + adapter on the held-out last 40 lines:

| Model | LoRA target | Context | s/step | Train loss | Perplexity base → adapter |
|---|---|---:|---:|---|---|
| Qwen3-4B Q4_K_M | attention, rank 8, 2 epochs | 256 | 1.2 | 2.92 → 2.41 | 15.26 → 12.72 |
| Qwen3-Coder-30B-A3B IQ4_XS | experts, rank 2, 1 epoch | 128 | 3.3 | 3.09 → 2.59 | 14.84 → 14.62 |
| Qwen3-Coder-30B-A3B IQ4_XS | attention, rank 8, 1 epoch | 128 | 1.5 | 2.95 → 2.49 | 14.84 → 14.34 |

The adapters load with `--lora`. Larger, task-specific datasets are needed for meaningful quality
numbers; these runs show convergence and that the gradient paths are complete.

**Models larger than VRAM.** Kolibri-1 Q3_K_S (31.5 GiB) trains with the routed experts of the first
32 layers in RAM (`-ot "blk\.(0|1|…|31)\.ffn_(up|gate|down)_exps\.weight=CPU"`, `-ts 36,14`). Pass
`--no-op-offload`: otherwise the scheduler copies every RAM expert used by the training graph to the GPU
at once (a 21.8 GB allocation). The RAM experts then run forward and backward on the CPU, ~26 s per
128-token step with 4 threads. Attention LoRA rank 8 on 60 lines: with lr 1e-4 the loss rose
(validation 5.94); with lr 1e-5 it fell from 4.5 to 3.7 (validation 3.54). Start large models at a low
learning rate.

## Trade-offs

### LoRA path

- **VRAM:** AdamW keeps two moments and a gradient accumulator per trainable
  element. For MoE expert LoRA keep the rank low (2–4) and the context short, or
  use SGD.

### Sparse path

- **Embed + LM-head + Norms is a weak training surface.** Useful for
  surface-distribution drift (output style, format adherence, tool-call
  template fidelity) but not for new capability. Fair-bench delta on a
  210-sample tool-calling suite was within noise (63.78 % → 63.35 %).

### Both paths

- **The saver `add_kv_from_model` path is not complete for every MoE arch.**
  Some hparams (`swiglu_clamp_shexp`, `expert_groups`, `n_layer_dense_lead`)
  may not be written correctly. `LLAMA_SAVER_ALLOW_UNTESTED=1` works around
  this; a full saver audit would be cleaner.
- **SSM_SCAN / SSM_CONV backward is still an open research problem**
  (selective state spaces, weeks of work, uncertain correctness). Mamba
  layers remain frozen.

## Verified hardware envelopes

### LoRA path single-GPU (2026-05-18, re-verified 2026-05-19)

- 1× RTX 2060 12 GB sufficient
- 11.8 GB peak VRAM with rank=2 + 128 ctx + 282 LoRA pairs
- ~6:30 min for 100 lines × 3 epochs, ~33 min for 500 lines × 3 epochs on
  Qwen3.6-A35B-A3B IQ2_XXS
- Loss curve: 4.0 → 2.4 over 3 epochs at lr=5e-6 (validated, converges)
- Smoke 2026-05-19: 262 steps loss 3.247 → 1.420, acc 31% → 63%, no regression after multi-GPU fixes landed
- Output: `out.lora.gguf` ~600 MB, loads cleanly in `llama-cli --lora`

### LoRA path dual-GPU layer-split (2026-05-19)

- 2× RTX 2060 12 GB, `-ts 1,1`
- ~5 GB peak per GPU at rank=2 + 256 ctx
- Smoke: 222 steps loss 3.247 → 1.379, acc 31% → 64%, same numerical trajectory as single-GPU
- **Throughput: ~0.35 step/s vs single-GPU ~0.50 step/s** — dual-GPU layer-split is ~44% slower per step on a consumer dual-2060 rig (asymmetric PCIe x16+x4, no working P2P). Use only when the workload does not fit single-GPU.
- Fix commits: `9d136dee5` (dynamic `sched->graph_inputs[]`) and `93388f61d` (sched comparator sentinel for graph-shape switches in opt mode)

### Sparse path

- 2× RTX 2060 12 GB (24 GB total), `-ts 6,5` proportional split
- 35 GB host RAM peak during training
- 6 h 21 min for 250 samples × 1 epoch on Qwen3.6-A35B-A3B IQ2_XXS
- Loss curve: 5.44 → 1.40
- Output: rewritten GGUF, loads cleanly in `llama-server`

## Stage-4 (QAT) op

`GGML_OP_QUANTIZE_DEQUANTIZE_FAKE` is wired into ggml: forward round-trips an
F32 tensor through `target_quant` (Q4_0, IQ2_XXS, KTQ2_1, …) to bake the
quantisation error into the activation; backward is a Straight-Through
Estimator (identity). CPU compute path and autograd are in place. The
`--qat-target-quant TYPE` CLI flag is wired through `common_params`. The
remaining wire-up is the LoRA-graph integration (wrapping `ab_cur` / `W_eff`
in the fake-quant op inside `build_lora_mm` / `build_lora_mm_id`); once
landed, the LoRA adapter can be trained to compensate for the base-model's
quantisation error.

## Upstream issue references

- ggml-org/llama.cpp#18805 — Mamba fine-tuning crashes on inplace assert
- ggml-org/llama.cpp#15279 — MoE expert routing has no backward
- ggml-org/llama.cpp#15090 — Hybrid model fine-tuning request
- ggml-org/llama.cpp#14424 — Sparse training proposal
- ggml-org/llama.cpp#9674 — Saver rejects modern architectures

## Roadmap — toward full-capability fine-tuning

Where we stand on the capability surfaces (2026-10-06):

| Component | Today | Needed for capability training |
|-----------|-------|--------------------------------|
| Token embeddings | trainable | extends vocab, but no new skills |
| LM head | trainable | only output distribution shift |
| Attention | **trainable via LoRA** (with `-fa off`) | reasoning + context tracking |
| MoE experts (`MUL_MAT_ID`) | **trainable via LoRA** (2026-05-18) | unlocks domain knowledge |
| Mamba / SSM | frozen | sequential state — nice-to-have |

### ✅ Phase A — MUL_MAT_ID backward (done 2026-05-18)

`ggml_mul_mat_id_grad_as` implemented on CPU + CUDA for the `as`-gradient; since 2026-10-06 `ggml_mul_mat_id_grad_b` provides the input gradient for quantised experts and the broadcast input (see "Correctness fixes").

### Phase B — Attention backward

Attention LoRA works with the standard (non-flash) attention backward (`-fa off`), verified 2026-10-06. A flash-attention backward would cut the activation memory for long contexts; not started.

### Phase C — Research-grade

- **Stage-4 QAT — wire-up.** The ggml op is in. Remaining work: `--qat-target-quant` CLI flag, `common_params.qat_target_quant` field, and the `ab_cur` wrap in `build_lora_mm` / `build_lora_mm_id`. Once landed, the LoRA adapter can be trained to compensate for the base-model's quantisation error.
- **SSM_SCAN / SSM_CONV backward** for Mamba state training (mathematically non-trivial — selective state spaces).
- **Periodic mid-training checkpoint** — flush adapter every N steps so a crash mid-batch keeps progress. Currently flushes only at epoch boundary and on SIGTERM/SIGINT.

### Phase D — Multi-GPU LoRA training on quantised base (2026-05-19: FUNCTIONAL)

Layer-split (`-sm layer -ts a,b`) for the LoRA-on-quantised path is now **functional** on 2× RTX 2060 12 GB. Smoke: Qwen3.6-A35B-IQ2_XXS, `-ts 1,1`, 222 steps SGD, loss 3.247 → 1.379, acc 31% → 64%, ~5 GB peak per GPU. Same numerical trajectory as the single-GPU path.

Two scheduler fixes were needed (both on `feature/phase-d-multigpu-lora`):
- `9d136dee5` — `sched->graph_inputs[]` is now a dynamic array. The previous fixed-size `GGML_SCHED_MAX_SPLIT_INPUTS=30` cap fits inference but is overrun by training graphs (forward + backward + per-param `OPT_STEP`) on a layer-split MoE.
- `93388f61d` — new `ggml_backend_sched_invalidate_prev_backend_ids()` plus a sentinel-aware comparator. The scheduler previously cached `prev_node_backend_ids` from the prior shape; on a `gf → gb_grad → gb_opt` switch it missed the change, skipped the reserve-and-retry path, and segfaulted at `init_tensor` on a stale `buffer_id`. The invalidate helper is called from `ggml_opt_alloc` on every graph-shape change and forces the realloc path.

**Caveat — dual-GPU is not a speedup for fitted workloads.** Layer-split is sequential, and on consumer dual-2060 rigs without working P2P/NVLink, the cross-device sync cost (asymmetric PCIe x16+x4) outweighs the compute parallelism. Measured: ~0.35 step/s dual vs ~0.50 step/s single (~44% slower per step). Use dual-GPU when the workload does not fit single-GPU (200k+ ctx, higher rank, AdamW momenta), not as a free speedup.

**Next step.** rank=4 + AdamW reachable now that VRAM doubles, which addresses the SGD-only drift seen in the earlier 10335-sample run. Validation pending.
