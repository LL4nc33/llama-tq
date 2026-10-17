# Tensor split without P2P

`-sm tensor` splits every weight matrix across the GPUs, so all GPUs work on each token at the same time (upstream's
default `-sm layer` gives each GPU a range of layers, which then run one after the other). Without peer access or NCCL,
upstream's tensor split falls back to slow copies to add up the partial results (it logs that NCCL is unavailable and
multi-GPU performance will be suboptimal). This fork reduces them through mapped pinned host memory instead, so it also
runs fast on consumer boards where the GPUs have no peer access (e.g. two RTX 2060 on a B450 board with x16/x4 slots).

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

Where we measured it: decode of dense 27B models that fit the combined VRAM of 2× RTX 2060 (Qwen3.8-27B,
Ternary-Bonsai-2-27B, see [models.md](models.md)). For MoE models with experts in system RAM, or when one GPU holds the
whole model, we have no comparison; try `-sm layer` and a single GPU as well.
The TurboQuant KV types work with it; the 27B setups in [models.md](models.md) use both.

Known issue: with `-sm tensor`, `test-llama-archs` still aborts on Kolibri-1 in the meta backend ("split axis
unknown"); DeepSeek V4 runs since 2026-10-13. Use `-sm layer` (the default) for Kolibri-1.
