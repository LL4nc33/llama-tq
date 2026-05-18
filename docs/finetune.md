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
   (A ~ Normal(0, 1/√rank), B = zero-initialised). 3D tensors (per-expert MoE
   weights) get rank-1 / rank-2 LoRA pairs per expert slice.
2. The base tensor is added to the skip regex internally — only A and B
   receive gradients.
3. The LoRA-merged matmul lives in `build_lora_mm` / `build_lora_mm_id` and
   participates in the forward graph from the first step.
4. `MUL_MAT_ID` backward computes `grad_as` (the adapter gradient) via the
   new `ggml_mul_mat_id_grad_as` op (CPU + CUDA). The matching `grad_b`
   (gradient flowing into activations) is **skipped** when `src0` is a
   quantised tensor — building `cont(transpose(W_q))` would copy a multi-GiB
   block-quant tensor per step, and the base is frozen anyway, so dropping
   that path is mathematically a no-op for the LoRA setup.
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

## Trade-offs

### LoRA path

- **`MUL_MAT_ID` `grad_b` skipped for quantised `src0`.** Mathematically safe
  for LoRA-only training (base is frozen, LoRA gradient flows via `grad_as`),
  but means deeper LoRA stacks won't see end-to-end gradients through
  activations. Dequant-on-the-fly would lift this; not implemented yet.
- **rank ≤ 2 for 282 LoRA pairs on 12 GB VRAM** (rank=4 OOMs by ~1.3 GB).
  AdamW also doesn't fit; SGD is mandatory at this VRAM budget. Dual-GPU
  tensor-split or 8-bit optimiser state would lift the rank ceiling.
- **Convergence sweet spot is 100–500 sample subsets × 3 epochs, lr ≤ 1e-5.**
  Bigger single-runs (28k lines × 1 epoch) diverge with the current grad-skip
  setup. Split larger datasets into sequential subsets.

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

### LoRA path (2026-05-18)

- 1× RTX 2060 12 GB sufficient
- 11.8 GB peak VRAM with rank=2 + 128 ctx + 282 LoRA pairs
- ~6:30 min for 100 lines × 3 epochs, ~33 min for 500 lines × 3 epochs on
  Qwen3.6-A35B-A3B IQ2_XXS
- Loss curve: 4.0 → 2.4 over 3 epochs at lr=5e-6 (validated, converges)
- Output: `out.lora.gguf` ~600 MB, loads cleanly in `llama-cli --lora`

### Sparse path

- 2× RTX 2060 12 GB (24 GB total), `-ts 6,5` proportional split
- 35 GB host RAM peak during training
- 6 h 21 min for 250 samples × 1 epoch on Qwen3.6-A35B-A3B IQ2_XXS
- Loss curve: 5.44 → 1.40
- Output: rewritten GGUF, loads cleanly in `llama-server`

## Stage-4 (QAT) op — work in progress

`GGML_OP_QUANTIZE_DEQUANTIZE_FAKE` is wired into ggml: forward round-trips an
F32 tensor through `target_quant` (Q4_0, IQ2_XXS, …) to bake the quantisation
error into the activation; backward is a Straight-Through Estimator
(identity). CPU compute path and autograd are in place. The `--qat-target-quant`
CLI flag and the LoRA-graph integration (wrapping `ab_cur` or `W_eff` in the
fake-quant op) are queued — once landed, the LoRA adapter can be trained to
compensate for the base-model's quantisation error.

## Upstream issue references

- ggml-org/llama.cpp#18805 — Mamba fine-tuning crashes on inplace assert
- ggml-org/llama.cpp#15279 — MoE expert routing has no backward
- ggml-org/llama.cpp#15090 — Hybrid model fine-tuning request
- ggml-org/llama.cpp#14424 — Sparse training proposal
- ggml-org/llama.cpp#9674 — Saver rejects modern architectures
