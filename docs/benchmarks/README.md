# Benchmarks

Current measurements (2026-10-10). Older logs and snapshots are in [history/](history/); their numbers are not
comparable with these (KV accuracy was measured differently, see below).

## Decode speed, one GPU

Qwen3-4B-Instruct-2507 Q4_K_M, `llama-bench -fa 1 -n 128`, empty context:

| GPU | f16 KV | `q8_0` KV | `ktq2_1` / `vtq2_1` KV |
|---|---:|---:|---:|
| RTX 5090 | 338 t/s | 281 t/s | 293 t/s |
| RTX 4090 | 252 t/s | | 217 t/s ¹ |
| RTX 3090 | 203 t/s | | 176 t/s ¹ |
| RTX 3060 | 103 t/s | | 96 t/s ¹ |
| RTX 4060 Ti | 95 t/s | | 90 t/s ¹ |
| RTX 2060 | 90 t/s | 84 t/s | 87 t/s |

¹ measured 2026-10-09, before the K/V rotations ran as the fast Walsh-Hadamard transform (RTX 5090 then: 272 t/s).

RTX 2060, empty / 16k context: f16 90 / 52, `q8_0` 84 / 52, `q4_0` 87 / 47, `ktq4_1`/`vtq4_1` 87 / 48 t/s.

## KV cache accuracy

KL divergence against an f16 KV cache with the cache quantized during prefill: table in
[turboquant.md#accuracy](../turboquant.md#accuracy). Summary: `q8_0` is near f16 on dense models, `ktq4_1`/`vtq4_1` is at
the `q4_0` level, the 2-bit types lose a lot; gpt-oss and Gemma 4 react strongly to any KV quantization.

## Models on 2× RTX 2060 12 GB

| Model | KV cache | Max context | Decode, short | Decode, long |
|---|---|---|---|---|
| Qwen3.8-27B Q4_K_M (tensor split) | f16 | 72k | 24 t/s | |
| Qwen3.8-27B Q4_K_M (tensor split) | `ktq2_1` / `vtq2_1` | 256k | 24 t/s | 15.8 t/s at 118k |
| Ternary-Bonsai-2-27B PTQ1_0 (tensor split) | f16 | 200k | 40 t/s | 21.6 t/s at 171k |
| Kolibri-1 78B MoE Q3_K_S | `ktq4_1` / `vtq4_1` | 128k | 35 t/s | |
| gpt-oss-20b MXFP4 | `ktq4_1` / `vtq4_1` | 128k | 77 t/s | 40 t/s at 64k |

More setups per model: [models.md](../models.md).

## Fine-tuning

LoRA directly on the quantized GGUF against PyTorch + PEFT + bitsandbytes (nf4), Qwen3-4B, attention q/k/v/o, rank 16,
AdamW 2e-4, 2 epochs, same data and metric (100 held-out examples):

| GPU | llama-tq | PyTorch QLoRA | Exact match |
|---|---:|---:|---:|
| RTX 5090 | 54 s, 7.8 GB | 102 s, 11.0 GB | 100 % / 100 % |
| RTX 2060 | 321 s, 7.2 GB | 260 s, 10.5 GB | 100 % / 100 % |

All 13 evaluated models (dense, MoE, Gated DeltaNet, ternary, up to 78B and 72.5 GB files) reach 100 % exact match on
the task after one LoRA run: [finetune.md](../finetune.md). The task and its scripts:
[examples/training/termine](../../examples/training/termine/README.md).
