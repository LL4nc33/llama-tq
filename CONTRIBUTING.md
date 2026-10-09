# Contributing to llama-tq

llama-tq is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). Changes to code that
llama-tq shares with upstream are best proposed upstream first; they reach llama-tq with the next sync.

For llama-tq specific work (TurboQuant KV cache, fine-tuning on quantized models, model ports):

- open an issue first for larger changes;
- follow the upstream [coding guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md#coding-guidelines);
- kernels and graph changes need a test (`test-backend-ops`, a dedicated test, or a perplexity / gradient
  comparison against the CPU backend) and before/after numbers for performance claims;
- one topic per pull request, commit messages in English.
