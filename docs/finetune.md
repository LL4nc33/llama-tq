# Fine-tuning quantized GGUF models

`llama-finetune` trains LoRA adapters directly on quantized GGUF models (k-quants, IQ quants, MXFP4,
ternary types) without converting them back to full precision: attention, dense and MoE expert
feed-forward layers and Gated DeltaNet layers, on one or two GPUs or the CPU, including models whose
experts sit in host memory. The adapter is written as a `.lora.gguf` that loads with `--lora` and merges
with `llama-export-lora`.

## Quick start

```bash
llama-finetune \
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
| `--grad-checkpoint` | Gradient checkpointing: only the layer outputs are kept, each layer is recomputed in the backward pass (about one more forward pass). Same gradients; Qwen3-4B, c=512: 8.2 → 4.4 GB. |
| `--train-stride N` | Tokens between window starts. Default: the context size for chat JSONL (whole examples are packed into windows, an example that does not fit starts the next window), half of it for plain text. Resume with the same value. |
| `--checkpoint-every N` | Also save adapter and state every N training ubatches (at the end of a window). |
| `--train-skip-regex REGEX` | Without a LoRA target: train the model tensors not matching the regex directly (see below). |
| `GGML_BACKWARD_SKIP_INPLACE=1` | Only for recurrent models whose state ops have no backward (Mamba, RWKV): other inplace ops end the gradient instead of asserting. Each kind of skipped op is reported once. KV cache writes never need it. |
| `GGML_OPT_PRINT_GRAD_NORM=1\|2` | Log the global gradient norm of every optimizer step (2: also per parameter). |
| `GGML_OPT_LINE_PROGRESS=1` | One progress line per step, for logs. |

Checkpoint files are written to a temporary file and renamed, the position last; an interrupted save
keeps the previous consistent checkpoint.

Training turns memory mapping off, uses an F32 KV cache and disables CPU weight repacking. Flash attention
is on by default on CPU and CUDA (it saves the attention probabilities of every layer at the same speed, see
below) and off when another GPU backend is present; `-fa on`/`-fa off` overrides.

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

- **Flash attention** (`-fa on`): `GGML_OP_FLASH_ATTN_BACK` on CPU and CUDA recomputes the softmax from q
  and k instead of storing the attention probabilities (masks, GQA, softcap, ALiBi, sinks).

Not covered: `SSM_SCAN` (Mamba-1/2) has no backward; rows added by `ADD_ID` (the expert biases themselves)
and the kernel of `GELU_ERF` are not trainable.

## Results

Extracting an appointment from a German message as JSON with a fixed schema: 700 chat examples for
training, 100 held out, greedy decoding through `llama-server` (2× RTX 2060 12 GB). Data generator, prompts, conventions,
the exact training input, evaluation script and the PyTorch reference are in
[examples/training/termine](../examples/training/termine/README.md).

| Model | Setup | Time | Exact match before → after | Validation loss / accuracy |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 Q4_K_M | attention q/k/v/o, rank 16, AdamW 2e-4, 2 epochs | 28 min | 0 % → **100 %** | 1.27 / 88.9 % → 0.131 / 98.0 % |
| same, merged with `llama-export-lora` | | | **100 %** | perplexity equal to `--lora` |
| Gemma-4-12B Q4_K_M | attention q/k/v/o, rank 16, AdamW 5e-5, 1 epoch, `--reasoning off` | 28 min | 0 % → **94 %** | 0.080 / 97.8 % |
| same, trained with thinking on, served with it off | | 29 min | 0 % → 73 % | 0.109 / 97.0 % |
| same, packed windows, prompt as rendered by the server, 1 epoch / 3 epochs | | 15 min / 47 min | 0 % → 98 % / 99 % ("3.5.2027" read month first) | 0.0004 / 0.00001 |
| same, AdamW 1e-4, 2 epochs / attention + MLP 5e-5, 2 epochs (1× RTX 5090) | | 20 min / 19 min | 99 % / 99 % ("12.10 Uhr" → 10:12 / "3.5.2027") | |
| same, AdamW 1e-4, 3 epochs / attention + MLP, AdamW 1e-4, 2 epochs (1× RTX 5090) | | 30 min / 19 min | 0 % → **100 %** / **100 %** | 0.0007 / 0.00002 |
| Qwen3.6-35B-A3B IQ2_XXS (MoE, Gated DeltaNet) | routed experts only, rank 2, AdamW 2e-4, 1 epoch, `--reasoning off` | 82 min | 0 % → **89 %** | 0.217 / 95.8 % |
| same, with the gradient of the expert weight normalization (CLAMP/DIV fix) | | 83 min | 0 % → 86 % | 0.196 / 96.0 % |
| same, whole examples per window (packed, `-c 512`), 1 epoch | | 43 min | 0 % → 95 % | 0.004 / 99.9 % |
| same, 2 epochs | | 86 min | 0 % → 99 % (the miss: "7 Uhr" read as 19:00) | 0.0006 / 99.98 % |
| same, 3 epochs (1× RTX 5090) | | 30 min | 0 % → **100 %** | 0.0003 |
| experts + attention q/k/v/o, rank 2, 2 epochs (1× RTX 5090) | | 20 min | 0 % → **100 %** | 0.00006 |
| Qwen3.5-0.8B Q8_0 (Gated DeltaNet) | attention + GDN projections, rank 16, 1 epoch | 10 min | | 0.042 / 99.1 % |
| Qwen3.5-0.8B Q8_0 (Gated DeltaNet), attention + GDN projections, rank 16, AdamW 1e-4, 2 epochs, packed (1× RTX 5090) | | 51 s | 0 % → **100 %** | 0.00006 |
| Ternary-Bonsai-2-27B PTQ1_0 (ternary, Gated DeltaNet), same setup, `--grad-checkpoint` (1× RTX 5090) | | 10 min | 0 % → **100 %** | 0.00002 |
| Qwen3.8-27B UD-Q4_K_M (dense, Gated DeltaNet), same setup, `--grad-checkpoint` (1× RTX 5090) | | 6 min | 0 % → **100 %** | 0.00001 |
| Qwen3.8-Flash-Next UD-IQ1_S (qwen4exp: MoE, Gated DeltaNet, hyper-connections, sparse attention indexer, 72.5 GB), same setup, `--grad-checkpoint` (1× RTX PRO 6000 96 GB) | | 15 min | 0 % → **100 %** | 0.00000 |
| Gemma-4-26B-A4B UD-IQ2_XXS (MoE), attention q/k/v/o, rank 16, AdamW 5e-5, 2 epochs, packed (1× RTX 5090) | | 6 min | 0 % → 99 % ("12.10 Uhr" → 12:00) | 0.0001 |
| same, AdamW 1e-4 (1× RTX 5090) | | 10 min | 0 % → **100 %** | 0.00006 |
| Ministral-3-3B Q4_K_M | attention q/k/v/o, rank 16, AdamW 1e-4, 1 epoch | 10 min | 0 % → **100 %** | 0.078 |
| same, packed windows, 1 epoch / 2 epochs (1× RTX 5090) | | 18 s / 31 s | 0 % → 99 % / **100 %** | 0.0002 / 0.00005 |
| gpt-oss-20b MXFP4 | attention q/k/v/o, rank 16, AdamW 1e-4, 1 epoch | 26 min | 0 % → **100 %** | 0.062 |
| same, packed windows (1× RTX 5090) | | 138 s | 0 % → **100 %** | 0.00003 |
| K2-Horizon-MoVA-36B-A4B Q3_K_M (MoE, routed value experts) | attention q/k/o, rank 16, AdamW 1e-4, 1 epoch, `--reasoning off` | 39 min | 0 % → **100 %** | 0.130 / 96.0 % |
| same, packed windows, `--grad-checkpoint` | | 19 min | 0 % → **100 %** | 0.0002 / 100 % |
| Kolibri-1 Q3_K_S (78B MoE, 31.5 GiB, experts of 32 layers in RAM) | attention, rank 16, AdamW 1e-4, 1 epoch on 300 examples, `--reasoning off` | 4 h 13 min | 0 % → **99 %** | 0.165 / 95.4 % |
| same, packed windows, all 700 examples, model fully in VRAM (1× RTX PRO 6000 96 GB), 1 epoch / 2 epochs | | 4 min / 8 min | 0 % → 99 % / **100 %** | 0.0001 / 0.00001 |

Short runs (60 windows, attention LoRA) also converge on Gemma-4-26B-A4B, Qwen3.8-27B and
Ternary-Bonsai-2-27B (PTQ1_0). Gemma 4 needs a lower learning rate (1e-4 diverged, 5e-5 trains).

**Long context** (Qwen3-4B, peak memory): without flash attention `-c 4096 -ub 512` takes 18.7 GB,
`-c 8192 -ub 256` 19.2 GB and `-c 16384 -ub 128` 23.0 GB, because every layer keeps its attention
probabilities. With `-fa on`, `-c 4096 -ub 512` takes 9.8 GB on an RTX 2060 and 10.1 GB on an RTX 5090
(−47 %). A whole window in one ubatch (`-ub` = `-c`, so that the gradient is not cut at ubatch boundaries) is limited by
the stored activations of all layers; `--grad-checkpoint` keeps only the layer outputs. On a 12 GB RTX 2060:
`-c 2048 -ub 2048` runs out of memory without it and takes 9.1 GB with it; `-c 3072 -ub 3072` fits with
`--grad-checkpoint -fa on` (12.0 GB). On a 32 GB RTX 5090, `-c 4096 -ub 4096 --grad-checkpoint` takes 15.6 GB and
`-c 8192 -ub 8192 --grad-checkpoint` 28.0 GB (13 s per window). Logits are computed only for positions with a trained label, which keeps
the output projection (n_vocab per position) small for chat data. An out-of-memory training graph stops with a
message.

The CUDA flash attention backward runs on cuBLAS GEMMs (per block of query rows: S = K^T Q, dP = V^T dO,
dV += dO P^T, dQ = K dS, dK += Q dS^T), so `-fa on` is no slower than the non-flash path. First training step on an
RTX 2060 (Qwen3-4B, including setup): `-c 2048 -ub 2048 --grad-checkpoint` 16 s (non-flash 17 s), `-c 3072 -ub 3072
--grad-checkpoint` 20 s, `-c 4096 -ub 512` 23 s (the first, row-wise kernel: 33 s, 58 s, 82 s;
`GGML_CUDA_FA_BACK_NAIVE=1` still selects it). The flash attention forward rounds Q/K/V to F16: the first-step
loss differs from the non-flash path by 0.7 % on the CPU backend (Qwen3-4B) and by 1–2 % on CUDA.

**Compared with PyTorch QLoRA** (same model, data, LoRA setup — attention q/k/v/o, rank 16, AdamW 2e-4,
warmup 20, 2 epochs — and the same exact-match metric on the 100 held-out examples):

| System | | Exact match | Training time | Peak memory |
|---|---|---|---|---|
| 1× RTX 2060 12 GB | llama-finetune, Qwen3-4B Q4_K_M, packed windows (2026-10-09) | 100 % | 321 s | 7.2 GB |
| | llama-finetune, overlapping windows, flash attention | 100 % | 509 s | 7.2 GB |
| | llama-finetune, overlapping windows, without flash attention | 100 % | 513 s | 8.3 GB |
| | llama-finetune, before the fixes of 2026-10-09 | 100 % | 1585 s | 8.2 GB |
| | PyTorch + PEFT + bitsandbytes, nf4 | 100 % | 260 s | 10.5 GB |
| 1× RTX 5090 32 GB | llama-finetune, Qwen3-4B Q4_K_M, packed windows (2026-10-09) | 100 % | 54 s | 7.8 GB |
| | llama-finetune, before the fixes of 2026-10-09 | 100 % | 396 s | 20.3 GB |
| | PyTorch + PEFT + bitsandbytes, nf4 | 100 % | 102 s | 11.0 GB |

The adapters are equally good. Profiling the 2060 run showed where llama-finetune lost time: 108 `ACC` nodes of
the backward pass ran on the CPU (strided operand not supported by the CUDA kernel), the input gradient through
the quantized weights ran as f32 SGEMM without tensor cores, and the dense labels (311 MB per window) were copied
from host memory on every step. With these fixed (CUDA `ACC` for strided and permuted operands, `out_prod` on
f16 tensor cores with power-of-two scaling of the gradient, labels kept on the device) the same run takes 513 s
instead of 1585 s (3.1×). The remaining factor 2 was the data layout: windows overlapped by half (every token
trained twice per epoch), while PyTorch trains each example once. Chat examples are now packed whole into
non-overlapping windows (16 % padding), which takes 321 s, 1.2× the PyTorch time, at the same accuracy. In a step
the GPU is busy almost all the time; 48 % of the kernel time is the quantized forward matmul (MMQ) and the f16
GEMM of the input gradient. On an RTX 5090 the same run takes 54 s (82 ms per step), half the PyTorch time (102 s); there the
per-step overhead that the fixes removed had dominated (396 s before).
`LLAMA_TRAIN_TIMING=1` prints the time per step by phase.

**Models larger than VRAM.** Experts can stay in host memory (`-ot ...=CPU`); pass `--no-op-offload`,
otherwise the scheduler copies every host expert the training graph uses to the GPU at once. The host
experts then run forward and backward on the CPU (Kolibri-1 Q3_K_S, 31.5 GiB, experts of 32 of 50 layers in
host memory: ~52 s per 256-token window). Use `--checkpoint-every` for such runs: a host reset after 2.6 hours
of the first Kolibri run lost everything, the second run kept a resumable checkpoint every 20 windows. Serve
the adapter with the same placement (`-ot` for the host experts, small `-ub`): `--lora` adds compute buffers
that an automatic placement made without the adapter does not reserve.

## How it was checked

- `test-backend-ops grad` (numerical gradients) for every new or changed backward, `test-backend-ops`
  CPU vs. CUDA for the new kernels, `test-opt` (AdamW and SGD).
- Gated DeltaNet backward against a float64 reference: relative error 1–2e-7, across segment
  boundaries, shared q/k heads, per-row gates and the final-state gradient.
- Whole models: the gradient of a LoRA step on the CPU matches CUDA exactly (after the `rms_norm_back`
  fix); an SGD step along the gradient lowers the loss by 0.85–0.92 of the first-order prediction on
  F32/Q8 models. On Q4/IQ2 models the quantized activations make the loss of a window jitter by
  ±0.02–0.1 under tiny weight changes, so such steps only confirm the direction there.
- Flash attention backward: `test-flash-attn-back` against float64 central differences of a float64
  forward (norm-relative error ~1e-7, GQA, sequences, softcap, ALiBi, sinks, no mask, DV != DK), CPU vs. CUDA
  in `test-backend-ops` on Turing (RTX 2060) and Blackwell (RTX 5090); a training step with `-fa on` matches
  the non-flash gradient norm within 0.5–1.4 % (the flash attention forward rounds K/V to F16).
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

- 2026-10-09 (evening): chat examples packed whole into windows without overlap (each token once per epoch,
  previously twice), flash attention on by default, KV cache writes need no `GGML_BACKWARD_SKIP_INPLACE`,
  `-ngl 0` in a CUDA build; 2 epochs Qwen3-4B: RTX 2060 321 s (PyTorch QLoRA 260 s), RTX 5090 54 s (PyTorch 102 s).
- 2026-10-09 (later): flash attention backward on cuBLAS GEMMs (2–3.5× faster for long windows); repository
  cleanup (unused experiments removed, `--moe-pin-experts` works again).
- 2026-10-09: gradient checkpointing (`--grad-checkpoint`), logits only for trained positions, training 3.1×
  faster (CUDA `ACC` for strided/permuted operands, `out_prod` on f16 tensor cores, labels on the device).
- 2026-10-09: flash attention backward (CPU, CUDA; `-fa on`), PyTorch QLoRA comparison, checked on an
  RTX 5090 (Blackwell) as a second system; the VTQ flash attention dispatch compiles in eleven parallel TUs.
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
