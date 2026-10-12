# Benchmarks

Decode and prefill speed per GPU and KV type (2026-10-09 to 2026-10-12, the date is given per table). Accuracy,
fine-tuning and per-model setups each have their own page (links at the end), so every number lives in one place. Older logs are in [history/](history/); their numbers
are not comparable with these.

## Decode speed, one GPU

Qwen3-4B-Instruct-2507 Q4_K_M, `llama-bench -fa 1 -n 128`, empty context (RTX 2060 row 2026-10-12, the other rows
2026-10-10 unless marked):

| GPU | f16 KV | `q8_0` KV | `ktq2_1` / `vtq2_1` KV |
|---|---:|---:|---:|
| RTX 5090 | 338 t/s | 281 t/s | 293 t/s |
| RTX 4090 | 252 t/s | | 217 t/s ¹ |
| RTX 3090 | 203 t/s | | 176 t/s ¹ |
| RTX 3060 | 103 t/s | | 96 t/s ¹ |
| RTX 4060 Ti | 95 t/s | | 90 t/s ¹ |
| RTX 2060 | 92 t/s | 89 t/s | 88 t/s |

¹ measured 2026-10-09, before the K/V rotations ran as the fast Walsh-Hadamard transform (RTX 5090 then: 272 t/s).
The encoder scale search of 2026-10-10 costs 1–2 % of the TurboQuant numbers in the rows measured before it.

Same model on one RTX 2060, decode at empty / 16k context (`llama-bench -n 64 -r 2`, 2026-10-12):

| KV cache | empty | 16k |
|---|---:|---:|
| `q8_0` / `q8_0` | 88 t/s | 57 t/s |
| `q8_0` / `vtq4_1` | 86 t/s | 57 t/s |
| `q8_0` / `vtq2_1` | 88 t/s | 58 t/s |
| `ktq4_1` / `vtq4_1` | 87 t/s | 44 t/s ² |
| f16 / f16 | 91 t/s | 53 t/s |
| `q4_0` / `q4_0` | 88 t/s | 47 t/s |

² includes the one-time conversion of the f16 prefill buffer to KTQ at the first decode step (about 0.1 s after a 16k
prompt).

## Against upstream, two GPUs

2× RTX 2060 (pipeline parallelism), `llama-bench -fa 1 -t 4 -p 2048 -n 128 -b 2048 -ub 512 -d 0,32768 -r 1`,
2026-10-12; upstream llama.cpp at 23b0202 against this fork with the fixes of 2026-10-12. Prefill is `pp2048`, decode
`tg128`, "32k" means after 32768 tokens of context, all numbers in t/s, each cell upstream → llama-tq.

| Model | prefill f16 | prefill f16 32k | decode `q8_0` | decode f16 32k | decode `q8_0` 32k |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Q4_K_M | 3598 → 3624 | 1196 → 1207 | 90.1 → 91.1 | 37.4 → 37.6 | 20.1 → 42.8 |
| gpt-oss-20b MXFP4 | 2647 → 2678 | 1777 → 1791 | 79.0 → 80.4 | 65.8 → 66.8 | 44.6 → 58.3 |
| Gemma 4 12B Q4_K_M | 1334 → 1305 | 855 → 809 | 33.8 → 33.9 | 32.0 → 32.1 | 28.0 → 29.6 |
| Qwen3.6-35B-A3B Q4_K_M | 1802 → 1887 | 1294 → 1241 | 78.5 → 81.5 | 68.1 → 70.0 | 46.6 → 68.9 |
| Qwen3.8-27B Q4_K_M | 644 → 678 | 429 → 418 | 16.5 → 16.7 | 14.9 → 15.0 | 12.4 → 15.0 |

Decode with a quantized KV cache at long context is where the fork is ahead (its flash attention decode kernels read
`q8_0` and the TurboQuant types directly). Prefill at 32k with an f16 cache is still behind upstream on Gemma 4 (head size
256, 5.4 %), Qwen3.6 (4.1 %) and Qwen3.8 (2.6 %), and Gemma 4 prefill without context by 2.2 %; the cause is open.

The fork's TurboQuant rows at 32k (same run):

| Model | `q8_0` / `vtq4_1` decode 32k | `ktq4_1` / `vtq4_1` decode 32k | `ktq4_1` / `vtq4_1` prefill 32k |
|---|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Q4_K_M | 42.6 | 31.4 | 1109 |
| gpt-oss-20b MXFP4 | 60.0 | 58.9 | 1737 |
| Gemma 4 12B Q4_K_M | 27.9 | 29.0 | 773 |
| Qwen3.6-35B-A3B Q4_K_M | 68.1 | 64.0 | 1215 |
| Qwen3.8-27B Q4_K_M | 15.0 | 14.3 | 401 |

## Accuracy and fine-tuning

KV cache accuracy per type and model: [turboquant.md#accuracy](../turboquant.md#accuracy). Fine-tuning results and the
comparison with PyTorch QLoRA: [finetune.md#results](../finetune.md#results). Command and result per model:
[models.md](../models.md).
