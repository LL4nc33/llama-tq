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

## Known limitations

- **Gemma 4:** wikitext perplexity of the instruction-tuned models is in the hundreds to tens of
  thousands with any engine (upstream llama.cpp gives the same values, and they swing by a third
  between its own FA and non-FA paths), so it is no measure of KV quality for these models.
  Decode with a quantized cache at head size 512 is fixed since 2026-10-04: Gemma-4-12B with
  `ktq4_1`/`vtq4_1` decodes 28.8 t/s at 32k context (f16 KV 31.0, before the fix 16.4).
- **gpt-oss:** loading crashed while reserving the compute graph; fixed on 2026-10-04,
  benchmark pending.
