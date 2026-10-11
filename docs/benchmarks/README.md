# Benchmarks

Decode and prefill speed per GPU and KV type (2026-10-10/11). Accuracy, fine-tuning and per-model setups each have their
own page (links at the end), so every number lives in one place. Older logs are in [history/](history/); their numbers
are not comparable with these.

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
The encoder scale search of 2026-10-11 costs 1–2 % of the TurboQuant numbers in this table.

Same model on the RTX 2060, decode at empty / 16k context (`llama-bench -n 64`, 2026-10-11):

| KV cache | empty | 16k |
|---|---:|---:|
| `q8_0` / `q8_0` | 83 t/s | 52 t/s |
| `q8_0` / `vtq4_1` | 82 t/s | 51 t/s |
| `q8_0` / `vtq2_1` | 82 t/s | 53 t/s |
| `ktq4_1` / `vtq4_1` | 86 t/s | 43 t/s ² |
| f16 / f16 (2026-10-10) | 90 t/s | 52 t/s |
| `q4_0` / `q4_0` (2026-10-10) | 87 t/s | 47 t/s |

² includes the one-time conversion of the f16 prefill buffer to KTQ at the first decode step (about 0.1 s after a 16k
prompt); over 256 generated tokens 47 t/s.

gpt-oss-20b MXFP4 on one RTX 2060, decode at empty / 8k context: `q8_0`/`q8_0` 78 / 70, `q8_0`/`vtq4_1` 77 / 70,
`ktq4_1`/`vtq4_1` 77 / 53 t/s (2026-10-11).

## Accuracy and fine-tuning

KV cache accuracy per type and model: [turboquant.md#accuracy](../turboquant.md#accuracy). Fine-tuning results and the
comparison with PyTorch QLoRA: [finetune.md#results](../finetune.md#results). Command and result per model:
[models.md](../models.md).
