# Tensor split without P2P

`-sm tensor` splits every weight matrix across the GPUs, so all GPUs work on each token at the same time (upstream's
default `-sm layer` gives each GPU a range of layers, which then run one after the other). Upstream's tensor split needs
peer-to-peer access or NCCL to add up the partial results. This fork reduces them through mapped pinned host memory
instead, so it also runs on consumer boards where the GPUs have no peer access (e.g. two RTX 2060 on a B450 board with
x16/x4 slots).

```bash
GGML_CUDA_HOST_ALLREDUCE_BF16=1 llama-server -m model.gguf -ngl 99 -fa on -sm tensor -c 65536 -ctk q8_0 -ctv q8_0
```

| Variable | Effect |
| --- | --- |
| `GGML_CUDA_HOST_ALLREDUCE_BF16=1` | sends the partial sums as bf16, halving the traffic over the host link |
| `GGML_CUDA_HOST_ALLREDUCE=0` | disables the host reduction (upstream behaviour) |

When it helps: decode of dense models that fit the combined VRAM, where every token touches all weights. For MoE models
with experts in system RAM, or when one GPU holds the whole model, `-sm layer` or a single GPU is usually as fast.
The TurboQuant KV types work with it; the 27B setups in [models.md](models.md) use both.

Measured setups (Qwen3.8-27B, Ternary-Bonsai-2-27B on 2× RTX 2060): [benchmarks](benchmarks/README.md),
commands per model: [models.md](models.md).
