# Fine-tuning quantized GGUF models

`llama-finetune` trains LoRA adapters directly on quantized GGUF models (k-quants, IQ quants, MXFP4,
ternary types) without converting them back to full precision: attention, dense and MoE expert
feed-forward layers and Gated DeltaNet layers, on one or two GPUs or the CPU, including models whose
experts sit in host memory. The adapter is written as a `.lora.gguf` that loads with `--lora` and merges
with `llama-export-lora`.

## Quick start

```bash
GGML_BACKWARD_SKIP_INPLACE=1 llama-finetune \
  -m Qwen3-4B-Instruct-2507-Q4_K_M.gguf -f train.jsonl -o task.gguf \
  --lora-train-target '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$' --lora-train-rank 16 --lora-train-alpha 32 \
  -opt adamw -lr 2e-4 --lr-warmup 20 --epochs 2 -val-split 0.1 \
  -ngl 99 -c 512 -b 512 -ub 512 --reasoning off
# writes task.lora.gguf (and task.lora.gguf.best, .opt, .state)

llama-server -m Qwen3-4B-Instruct-2507-Q4_K_M.gguf --lora task.lora.gguf --reasoning off
llama-export-lora -m Qwen3-4B-Instruct-2507-Q4_K_M.gguf --lora task.lora.gguf -o merged.gguf
```

Training data is JSONL, one example per line:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"text": "plain text, every token is trained"}
```

Chats are rendered with the model's chat template (`--chat-template` / `--chat-template-file` override
it) and only the assistant turns are trained, including the template's end-of-turn token. `content`
may also be `null` or a list of `{"type": "text", "text": ...}` parts; a line the loader or the template
cannot handle is skipped with its line number. `LLAMA_FINETUNE_SHOW_MASK=1` prints the start of the data
with the trained spans in `[[ ]]`. A plain `.txt` file trains on every token.

**Train with the `--reasoning` setting you serve with.** The template renders the prompt differently
with thinking on or off (Gemma 4 adds `<|think|>` to the system turn, Qwen changes the generation
prompt), and an adapter trained on one form only partly transfers to the other.

## Options

| Option | Effect |
|---|---|
| `--lora-train-target REGEX` | Create a LoRA pair for every model tensor matching the regex (3D expert tensors get one pair per expert). All model tensors stay frozen; only the adapter trains. |
| `--lora-train-rank N`, `--lora-train-alpha A` | Rank (default 16) and alpha (default 32); the adapter scale is alpha / rank. |
| `-opt adamw\|sgd`, `-lr`, `-wd`, `--epochs`, `-val-split` | Optimizer (AdamW keeps two moments per trained value), learning rate, weight decay, epochs, share of the data used for validation. |
| `--lr-warmup N` | Raise the learning rate linearly over the first N optimizer steps. |
| `--grad-clip N` | Clip the global gradient norm to N (default 1.0, 0 = off). |
| `--early-stop N` | Stop after N epochs without a lower validation loss. The adapter with the best validation loss is always kept as `<adapter>.best`. |
| `--reasoning on\|off\|auto` | Render chat data as the server does with this setting. |
| `--resume` | Continue a stopped run from the adapter, `<adapter>.opt` (step count, AdamW moments) and `<adapter>.state` (position, learning-rate step, early-stop state). The result is bit-identical to an uninterrupted run. Refuses an adapter trained with another alpha. |
| `--stop-after N` | Stop after N context windows and save everything for `--resume`. SIGINT/SIGTERM do the same after the current window (a second signal exits at once). |
| `--checkpoint-every N` | Also save adapter and state every N training ubatches (at the end of a window). |
| `--train-skip-regex REGEX` | Without a LoRA target: train the model tensors not matching the regex directly (see below). |
| `GGML_BACKWARD_SKIP_INPLACE=1` | Required for hybrid/recurrent models: inplace ops (cache writes) end the gradient instead of asserting. Each kind of skipped op is reported once. |
| `GGML_OPT_PRINT_GRAD_NORM=1\|2` | Log the global gradient norm of every optimizer step (2: also per parameter). |
| `GGML_OPT_LINE_PROGRESS=1` | One progress line per step, for logs. |

Checkpoint files are written to a temporary file and renamed, the position last; an interrupted save
keeps the previous consistent checkpoint.

Training forces flash attention off (it has no backward pass), memory mapping off and an F32 KV cache,
and disables CPU weight repacking.

## What gets a gradient

The backward graph is built by ggml from the forward graph. For training on quantized weights the fork
adds or fixes:

- **Frozen quantized matmuls** pass the gradient to their input: `out_prod` with a quantized weight
  (dequantized in row chunks on CUDA), and for MoE experts `MUL_MAT_ID_GRAD_B` (CUDA groups the tokens by
  expert and dequantizes each used expert once). The adapter of an expert gets `MUL_MAT_ID_GRAD_AS`. The
  routing weights get their gradient through a batched `get_rows_back`, including the normalization of the
  selected weights (`CLAMP`, and `DIV`/`SUB` with a broadcast operand).
- **K and V through the KV cache.** Training graphs read the cache with the current ubatch's rows taken
  from `k_cur`/`v_cur` (`ggml_set`), so K, V and the layers below receive the gradient. The rows of
  earlier ubatches of the same window are constants: within a window the gradient is truncated at the
  ubatch boundary (with `-ub` = `-c` it is complete).
- **Gated DeltaNet** (Qwen3.5/3.6/3.8 hybrids and similar): new op `GATED_DELTA_NET_BACK` on CPU and CUDA;
  the recurrent states are recomputed per segment of ~sqrt(n_tokens) tokens from checkpoints. Also `SSM_CONV` and the
  `CONCAT` of the conv state. Raw gate folding is turned off for training.
- **Activations and other ops:** `TANH` (logit soft-capping), `GEGLU`, `REGLU`, `GEGLU_QUICK`, `SWIGLU_OAI`
  (gpt-oss), `ADD_ID` (expert biases), `RMS_NORM` on strided views, and the cross-entropy loss with masked
  rows (positions with an all-zero label get no gradient; the loss is the mean over trained tokens).
- **Optimizer:** gradient accumulators are cleared after every step, moments and accumulators live next
  to their parameter (two GPUs), the state is kept per parameter (graphs may differ between ubatches),
  global-norm clipping, save/restore of the AdamW state.

Not covered: `SSM_SCAN` (Mamba-1/2) and flash attention have no backward; rows added by `ADD_ID` (the
expert biases themselves) and the kernel of `GELU_ERF` are not trainable.

## Results

Extracting an appointment from a German message as JSON with a fixed schema: 700 chat examples for
training, 100 held out, greedy decoding through `llama-server` (2× RTX 2060 12 GB).

| Model | Setup | Time | Exact match before → after | Validation loss / accuracy |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 Q4_K_M | attention q/k/v/o, rank 16, AdamW 2e-4, 2 epochs | 28 min | 0 % → **100 %** | 1.27 / 88.9 % → 0.131 / 98.0 % |
| same, merged with `llama-export-lora` | | | **100 %** | perplexity equal to `--lora` |
| Gemma-4-12B Q4_K_M | attention q/k/v/o, rank 16, AdamW 5e-5, 1 epoch, `--reasoning off` | 28 min | 0 % → **94 %** | 0.080 / 97.8 % |
| same, trained with thinking on, served with it off | | 29 min | 0 % → 73 % | 0.109 / 97.0 % |
| Qwen3.6-35B-A3B IQ2_XXS (MoE, Gated DeltaNet) | routed experts only, rank 2, AdamW 2e-4, 1 epoch, `--reasoning off` | 82 min | 0 % → **89 %** | 0.217 / 95.8 % |
| Qwen3.5-0.8B Q8_0 (Gated DeltaNet) | attention + GDN projections, rank 16, 1 epoch | 10 min | | 0.042 / 99.1 % |
| Ministral-3-3B Q4_K_M | attention q/k/v/o, rank 16, AdamW 1e-4, 1 epoch | 10 min | 0 % → **100 %** | 0.078 |
| gpt-oss-20b MXFP4 | attention q/k/v/o, rank 16, AdamW 1e-4, 1 epoch | 26 min | 0 % → **100 %** | 0.062 |
| K2-Horizon-MoVA-36B-A4B Q3_K_M (MoE, routed value experts) | attention q/k/o, rank 16, AdamW 1e-4, 1 epoch, `--reasoning off` | 39 min | 0 % → **100 %** | 0.130 / 96.0 % |

Short runs (60 windows, attention LoRA) also converge on Gemma-4-26B-A4B, Qwen3.8-27B and
Ternary-Bonsai-2-27B (PTQ1_0). Gemma 4 needs a lower learning rate (1e-4 diverged, 5e-5 trains).

**Long context** (Qwen3-4B, two GPUs, peak memory): `-c 4096 -ub 512` 18.6 GB, `-c 8192 -ub 256`
19.2 GB, `-c 16384 -ub 128` 23.0 GB. Without a flash-attention backward every layer keeps its attention
probabilities for the backward pass; lower `-ub` for longer contexts. An out-of-memory training graph
stops with a message.

**Models larger than VRAM.** Experts can stay in host memory (`-ot ...=CPU`); pass `--no-op-offload`,
otherwise the scheduler copies every host expert the training graph uses to the GPU at once. The host
experts then run forward and backward on the CPU (Kolibri-1 Q3_K_S, 31.5 GiB: ~26 s per 128-token step;
measured before the fixes of 2026-10-07).

## How it was checked

- `test-backend-ops grad` (numerical gradients) for every new or changed backward, `test-backend-ops`
  CPU vs. CUDA for the new kernels, `test-opt` (AdamW and SGD).
- Gated DeltaNet backward against a float64 reference: relative error 1–2e-7, across segment
  boundaries, shared q/k heads, per-row gates and the final-state gradient.
- Whole models: the gradient of a LoRA step on the CPU matches CUDA exactly (after the `rms_norm_back`
  fix); an SGD step along the gradient lowers the loss by 0.85–0.92 of the first-order prediction on
  F32/Q8 models. On Q4/IQ2 models the quantized activations make the loss of a window jitter by
  ±0.02–0.1 under tiny weight changes, so such steps only confirm the direction there.
- Resume: stopping (`--stop-after`, SIGTERM, within and across epochs) and resuming gives a bit-identical
  adapter with SGD and with AdamW + warmup.

## Training without LoRA

Without `--lora-train-target`, the model tensors themselves are trained, except those matching
`--train-skip-regex` (for example `'blk\.'` leaves `token_embd`, `output` and `output_norm`). The
trained model is written back as a GGUF (`LLAMA_SAVER_ALLOW_UNTESTED=1` for architectures the saver has
not been checked with). Quantized tensors cannot be trained this way.

## Experimental

`--qat-target-quant TYPE` wraps the LoRA delta in a fake-quantization op (straight-through estimator) so
that the adapter sees the quantization error of the target type. The op has a CPU kernel only and has
not been validated in training.

## History

- 2026-10-08 (later): `CLAMP` backward (the MoE weight normalization had no gradient), `SUB`/`DIV` backward with a
  broadcast operand, no double free when a `--lora` file fails to load, RPC op count.
- 2026-10-08: chat rendering follows `--reasoning`; crash-safe checkpoints; tolerant chat data; alpha
  check on resume; backward for `GEGLU`, `REGLU`, `GEGLU_QUICK`, `SWIGLU_OAI`, `ADD_ID`; CUDA
  `rms_norm_back` on strided inputs; out-of-memory message; early stopping.
- 2026-10-07: chat data with an assistant-only loss, gradient clipping, warmup, exact resume; gradient
  through the KV cache; CUDA `out_prod` stride for LoRA shapes; CPU `rms_norm_back` inplace; optimizer
  state per parameter; Gated DeltaNet, `SSM_CONV`, `CONCAT`, `TANH` backward; accuracy over trained
  tokens; `test-opt` re-enabled.
- 2026-10-06: gradient accumulators cleared after each step, LoRA A initialization, input gradients of
  quantized experts and dense matmuls on CUDA, AdamW on two GPUs.
- 2026-05: `MUL_MAT_ID` adapter gradient, LoRA training on quantized MoE experts, layer split over two
  GPUs.
