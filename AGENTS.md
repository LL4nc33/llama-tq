# Instructions for llama-tq

llama-tq is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with a quantized KV cache (TurboQuant
KTQ/VTQ), LoRA fine-tuning directly on quantized GGUF models and additional model ports. These notes apply to humans
and to AI coding assistants working on the fork. Changes to code shared with upstream follow the
[upstream guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md) and are best proposed there.

AI assistance is welcome. The person submitting a change is responsible for it: they must understand it, have run the
checks below and be able to explain every line.

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # CPU only: -DGGML_CUDA=OFF
cmake --build build -j --target llama-server llama-finetune test-backend-ops
```

New source files are picked up by globs: re-run the configure step after adding one.

## Checks before a change is done

- Kernels and ops: `test-backend-ops -b CUDA0 -o <OP>` (CUDA against the CPU reference), `test-backend-ops grad -o <OP>`
  for ops with a backward pass, `test-flash-attn-back` for the flash attention gradients (float64 reference).
- Training code: compare the step-1 gradient norm of a short `llama-finetune` run between the CPU and the CUDA backend
  (`GGML_OPT_PRINT_GRAD_NORM=1`). Op tests alone have missed bugs (permuted views) that this comparison caught.
- KV cache types and attention: perplexity before and after (`llama-perplexity`) on the same text and settings.
- Performance claims: numbers measured before and after on the same machine, one run at a time.

## Conventions

- Commit messages and code comments in English; one topic per commit.
- Numbers in docs come from measurements; name the model, quantization, settings and GPU.
- No machine-specific paths, host names, addresses or credentials in code, docs or scripts.
- Prefer values derived from tensor shapes or model metadata over hard-coded constants.
- Keep the diff to upstream small in shared files: fork features go into separate files where possible.

## Where things are

- TurboQuant KV cache: `docs/turboquant.md`, CUDA kernels in `ggml/src/ggml-cuda/` (`fattn-*`, `turboquant*`, `trellis*`).
- Fine-tuning: `docs/finetune.md`, `examples/training/finetune.cpp`, autodiff in `ggml/src/ggml.c`, training loop in
  `src/llama-context.cpp`, optimizer in `ggml/src/ggml-opt.cpp`.
- Model ports and deployment notes: `docs/models.md`.
