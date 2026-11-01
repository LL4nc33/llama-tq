# Tensor split without P2P

`-sm tensor` splits every weight matrix across the GPUs, so all GPUs work on each token at the same time (upstream's
default `-sm layer` gives each GPU a range of layers, which then run one after the other). It also runs without peer
access or NCCL, e.g. on two RTX 2060 on a B450 board with x16/x4 slots. Upstream adds up the partial results with its
own fallback there; this fork reduces them through mapped pinned host memory and can send them as bf16.

The host reduction is used only in builds without NCCL: the release archives, or a source build where NCCL is not
installed or `-DGGML_CUDA_NCCL=OFF` is set. When NCCL is compiled in, NCCL does the reduction and the variables below
have no effect.

```bash
GGML_CUDA_HOST_ALLREDUCE_BF16=1 llama-server -m model.gguf -ngl 99 -fa on -sm tensor -c 65536 -ctk q8_0 -ctv q8_0
```

| Variable | Effect |
| --- | --- |
| `GGML_CUDA_HOST_ALLREDUCE_BF16=1` | sends the partial sums as bf16, halving the traffic over the host link |
| `GGML_CUDA_HOST_ALLREDUCE=0` | disables the host reduction (upstream behaviour) |
| `GGML_CUDA_P2P=1` | enables peer access between the GPUs; off by default, because on boards with x16/x4 slots it routes traffic over the slow link (MoE prompt processing on Qwen3.6-35B-A3B was about 14 % slower) |

Qwen3.8-27B Q4_K_M on two RTX 2060, in t/s; with MTP the better draft length per text
([benchmarks](benchmarks/README.md#speculative-decoding-two-gpus)):

| | prose | code | rewrite |
| --- | ---: | ---: | ---: |
| upstream, `-sm layer` | 16.4 | 16.5 | 16.5 |
| llama-tq, `-sm layer` | 17.2 | 17.2 | 17.2 |
| upstream, `-sm tensor` | 26.5 | 26.6 | 26.5 |
| llama-tq, `-sm tensor` | 26.8 | 26.9 | 26.8 |
| llama-tq, `-sm tensor`, bf16 | **27.9** | **27.8** | **27.6** |
| upstream, `-sm layer`, MTP | 23.6 | 32.5 | 28.0 |
| llama-tq, `-sm layer`, MTP | 24.3 | 38.5 | 31.9 |
| upstream, `-sm tensor`, MTP | 35.8 | 49.5 | 43.6 |
| llama-tq, `-sm tensor`, MTP | 34.6 | 51.5 | 45.0 |
| llama-tq, `-sm tensor`, bf16, MTP | **37.2** | **57.7** | **48.2** |

Upstream has no bf16 option. The rows of 2026-11-01 (`-sm tensor` without bf16) and of 2026-10-29 (the others) are
separate runs.

bf16 costs no measurable accuracy here: KL divergence against an f32 reference (Qwen3.8-27B, `q8_0`, 8 chunks of 512
tokens, 2026-11-01), prompt batches of 512 tokens / one token per batch: `-sm tensor` 0.00044 / 0.00049, with bf16
0.00056 / 0.00049 (±0.00009 / ±0.00003), `-sm layer` 0.00051 / 0.00048. Upstream's `-sm tensor`: 0.0023 / 0.00050.

Where we measured it: decode of dense 27B models that fit the combined VRAM of 2× RTX 2060 (Qwen3.8-27B,
Ternary-Bonsai-2-27B, see [models.md](models.md)). For MoE models with experts in system RAM, or when one GPU holds the
whole model, we have no comparison; try `-sm layer` and a single GPU as well.
The TurboQuant KV types work with it; the 27B setups in [models.md](models.md) use both.

`test-llama-archs` passes with `-sm tensor` on every architecture that supports it (2026-10-21, two RTX 2060).

Speculative decoding (MTP) works with `-sm tensor` since 2026-10-21; it samples on the CPU there, since the logits are
split along the vocabulary. Since 2026-10-22 each device selects the best candidates of its part of the vocabulary, and only
those cross to the host (the samplers that read only the k largest logits, as MTP drafts and greedy or top-k sampling do).
