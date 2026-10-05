# Models and tested setups

Architectures and weight types that this fork adds or runs differently from upstream
llama.cpp, with the settings we use on 2× RTX 2060 12 GB (Turing, no P2P). Measurements for
every run, including the models upstream already supports, are on the
[interactive benchmark page](https://ll4nc33.github.io/llama-tq/docs/benchmarks/).

All commands assume a CUDA build and `-fa on`. TurboQuant KV types are described in
[turboquant.md](turboquant.md), tensor split in [tp-tq-design.md](tp-tq-design.md).

## Ternary-Bonsai-2-27B (`PQ2_0`, `PTQ1_0`)

Ternary weights in groups of 128 with Hadamard-rotated activations. `PTQ1_0` packs them at
about 1.6 bits and is both smaller and faster than the 2-bit `PQ2_0` container at the same
quality. Source: [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

Two slots with 200k context each, vision, KV equal to f16 in perplexity:

```bash
GGML_CUDA_HOST_ALLREDUCE_BF16=1 llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  -ngl 99 -fa on -sm tensor -c 409600 --parallel 2 -ub 512 \
  -ctk ktq4_1 -ctv vtq4_1 --no-tq-deferred-k --no-tq-deferred-v \
  --jinja --reasoning off
```

Decode: 34 t/s short, 28 t/s at 23k, 22 t/s at 74k.

## Qwen3.8-27B (hybrid, Gated DeltaNet)

Dense hybrid model: one in four layers has full attention, the rest are Gated DeltaNet, so the
KV cache is small. Source: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

Full 256k context with `ktq4_1`/`vtq4_1` (PPL 6.039 vs 6.036 with f16):

```bash
GGML_CUDA_HOST_ALLREDUCE_BF16=1 llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf \
  -ngl 99 -fa on -sm tensor -c 262144 \
  -ctk ktq4_1 -ctv vtq4_1 --no-tq-deferred-k --no-tq-deferred-v --jinja
```

Decode: 22.5 t/s short, 17.2 t/s at 118k. Prefill 190–250 t/s.

## Qwen3.8-Flash-Next (`qwen4exp`)

Hybrid MoE with a compressed-attention indexer that selects which cached tokens each query
reads. The fork runs this as sparse flash attention: the GQA decode kernel takes the index
lists directly. At IQ1_S the model is larger than 24 GB, so the down-projection experts stay
on the CPU:

```bash
llama-server -m Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  -ngl 99 -fa on -ts 24,24 -c 131072 -ub 256 -t 6 \
  -ot "blk\..*\.ffn_down_exps\.weight=CPU" --no-op-offload \
  -ctk ktq4_1 -ctv vtq4_1 --no-tq-deferred-k --no-tq-deferred-v --jinja
```

Decode: 15.5 t/s at 118k with a peak of 11.6 GB per GPU.

## K2-Horizon-MoVA-36B-A4B (`k2-horizon`)

MoE whose attention layers route the value projection through experts (mixture of value
experts). All 48 layers use full attention, so the KV cache grows quickly; the outer layers are
the most sensitive to a quantized K. Source:
[NANI-Nithin/K2-Horizon-MoVA-36B-A4B-GGUF](https://huggingface.co/NANI-Nithin/K2-Horizon-MoVA-36B-A4B-GGUF).

```bash
llama-server -m K2-Horizon-MoVA-36B-A4B-Q3_K_M.gguf \
  -ngl 99 -fa on -ts 25,23 -c 65536 -ub 512 -t 4 \
  -ctk ktq4_1 -ctv vtq4_1 --no-tq-deferred-k --no-tq-deferred-v \
  --tq-protect-layers 4 --jinja
```

| KV cache | PPL vs f16 |
|---|---:|
| `ktq4_1` / `vtq4_1` | +3.3 % |
| `ktq4_1` / `vtq4_1` + `--tq-protect-layers 4` | +0.8 % |
| `q5_0` / `q5_0` | +0.4 % |
| `q8_0` / `q8_0` | equal |

Decode: 47 t/s short, 22 t/s at 40k. Prefill 650 t/s short, 416 t/s at 38k.

## Qwen3-Coder-Next (`UD-TQ1_0`)

The Qwen3-Next coding model with ternary 1.6-bit weights fits completely into 24 GB. Tool
calls stay valid JSON in our tests. Source:
[unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF).

```bash
llama-server -m Qwen3-Coder-Next-UD-TQ1_0.gguf \
  -ngl 99 -fa on -ts 12,12 -c 131072 \
  -ctk ktq4_1 -ctv vtq4_1 --no-tq-deferred-k --no-tq-deferred-v --jinja
```

Decode: 61 t/s short, 37 t/s at 72k. KV PPL +0.1 % over f16.

## Aleph Alpha Kolibri-1 (`kolibri1`)

78B MoE with 3.46B active parameters, trained on English and German with a tokenizer tuned for
German (about 4.7 bytes per token). 384 routed experts (top 6) plus one shared expert per layer;
four of five layers use a 513-token sliding window with RoPE, the other ten layers attend to the
full context without positional encoding, so the KV cache stays small and 262k context needs no
RoPE scaling. The router selects experts on logits plus bias and weights them with the unbiased
sigmoid. Sources: [Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1),
GGUF [Eliasfpv28/Kolibri-1-Q3_K_S-GGUF](https://huggingface.co/Eliasfpv28/Kolibri-1-Q3_K_S-GGUF).
The port is based on the patches by Seraphiel102.

At Q3_K_S (31.5 GiB) part of the routed experts has to stay in RAM. The simplest setup lets `-fit`
(on by default) decide: it measures weights, KV cache and compute buffers and picks the layer split
and which layers keep their experts in RAM. Raise the CPU/GPU crossover for prompts with
`GGML_OP_OFFLOAD_MIN_BATCH`: below it, the RAM experts run on the CPU instead of being streamed to the
GPU for every batch, which cuts the time to the first token of short prompts by about 4x.

```bash
GGML_OP_OFFLOAD_MIN_BATCH=1024 llama-server -m Kolibri-1-Q3_K_S.gguf -fa on -c 131072 -b 4096 -ub 4096 -t 4 \
  -ctk ktq4_1 -ctv vtq4_1 --no-tq-deferred-k --no-tq-deferred-v \
  --temp 1.0 --top-p 0.97 --top-k 128 --jinja --reasoning off
```

By hand, put the RAM experts on the early layers: those sit on the first GPU under layer split, and
large prompt batches stream their weights over its link, which is faster when the first GPU has the
wider PCIe link (here x16 vs x4):

```bash
llama-server ... -ngl 99 -ts 37,13 \
  -ot "blk\.([0-9]|1[0-9]|2[0-7])\.ffn_(up|gate|down)_exps\.weight=CPU"
```

2x RTX 2060 12 GB, 131k context, `-ub 4096`, `GGML_OP_OFFLOAD_MIN_BATCH=1024`, server timings:

| Placement | Decode | Prompt 81 tok | Prompt 1.3k | Prompt 3.4k |
|---|---:|---:|---:|---:|
| `-fit` (automatic) | 37.4 t/s | 0.9 s | 7.0 s (191 t/s) | 10.3 s (331 t/s) |
| experts of layers 0-27 in RAM | 34.7 t/s | 1.0 s | 5.5 s (246 t/s) | 7.5 s (454 t/s) |

The MoE expert cache (`--moe-cache-mib`) does not pay off here: with 5 GB of cache and a 91 % hit
rate, decode drops to 25 t/s. The cache serves only the layers on its own GPU.

Decode barely drops with context (38 t/s at 10k). KV accuracy: perplexity +0.5 % with
`ktq4_1`/`vtq4_1` against f16. Reasoning effort is set per request through `chat_template_kwargs`
(`reasoning_effort`: none, low, medium, high); tool calls use the Hermes format.

## gpt-oss-20b

Head size 64 with attention sinks. Quantized KV goes through the tensor-core decode kernel:

```bash
llama-server -m gpt-oss-20b-MXFP4.gguf -ngl 99 -fa on -c 131072 \
  -ctk ktq4_1 -ctv vtq4_1 --no-tq-deferred-k --no-tq-deferred-v --jinja
```

Decode 77 t/s short, 57 t/s at 22k, 40 t/s at 64k. Prompt processing 1100-2300 t/s.

## Gemma 4

Global layers use head size 512 with GQA 8-16 and a quantized cache runs through the tensor-core
decode kernel. Decode at 32k context (llama-bench, `ktq4_1`/`vtq4_1`, f16 KV in brackets):
12B Q4_K_M 28.8 (31.0), 26B-A4B UD-IQ2_XXS 53.7 (62.9), 26B-A4B UD-Q4_K_M 45.0 (51.3),
31B UD-IQ2_XXS 11.1 (12.1), 31B Q4_K_M 11.3 (12.2).

## Known limitations

- **Gemma 4:** wikitext perplexity of the instruction-tuned models is in the hundreds to tens of
  thousands with any engine (upstream llama.cpp gives the same values, and they swing by a third
  between its own FA and non-FA paths), so it is no measure of KV quality for these models.
  Decode with a quantized cache at head size 512 is fixed since 2026-10-04: Gemma-4-12B with
  `ktq4_1`/`vtq4_1` decodes 28.8 t/s at 32k context (f16 KV 31.0, before the fix 16.4).
