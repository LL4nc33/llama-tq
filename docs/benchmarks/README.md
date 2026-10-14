# Benchmarks

Decode and prefill speed per GPU and KV type (2026-10-09 to 2026-10-14, the date is given per table). Accuracy,
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

Same model on one RTX 2060, decode at empty / 16k context (`llama-bench -n 64 -r 2`, 2026-10-12; the `ktq4_1` row
2026-10-13):

| KV cache | empty | 16k |
|---|---:|---:|
| `q8_0` / `q8_0` | 88 t/s | 57 t/s |
| `q8_0` / `vtq4_1` | 86 t/s | 57 t/s |
| `q8_0` / `vtq2_1` | 88 t/s | 58 t/s |
| `ktq4_1` / `vtq4_1` | 87 t/s | 54 t/s |
| f16 / f16 | 91 t/s | 53 t/s |
| `q4_0` / `q4_0` | 88 t/s | 47 t/s |

Same model on one RTX 2060 at 32k context, before → after the GQA decode kernel change of 2026-10-14
(`llama-bench -fa 1 -t 4 -n 64 -d 32768 -r 1`, builds alternated, mean of two runs per build):

| KV cache | decode 32k | `pp512` at 32k |
|---|---:|---:|
| f16 / f16 | 37.2 t/s (tensor-core kernel, unchanged) | 753 t/s |
| `q8_0` / `q8_0` | 41.7 → 43.8 t/s | |
| `q8_0` / `vtq4_1` | 41.5 → 45.3 t/s | 669 → 705 t/s |
| `ktq4_1` / `vtq4_1` | 38.9 → 46.0 t/s | 705 → 727 t/s |
| `ktq2_1` / `vtq2_1` | 40.7 → 49.0 t/s | 724 → 730 t/s |
| `q8_0` / `q4_0` ¹ | 18.4 → 43.9 t/s | |
| `ktq4_1` / `q8_0` ¹ | 10.3 → 41.4 t/s | |
| f16 / `vtq4_1` ¹ | 13.0 → 41.4 t/s | |

¹ pairs that converted the whole cache to f16 at every decode step before; "after" measured with an earlier version
of the kernel on 2026-10-13 (the final version is at least as fast on every pair measured with both).
Qwen3.6-35B-A3B Q4_K_M on two RTX 2060 at 32k: `q8_0` / `q8_0` 69.8 → 70.4 t/s, `ktq4_1` / `q8_0` 41.3 → 70.9 t/s.


## Against upstream, two GPUs

2× RTX 2060 (pipeline parallelism), `llama-bench -fa 1 -t 4 -p 2048 -n 128 -b 2048 -ub 512 -d 0,32768 -r 1`,
2026-10-13; upstream llama.cpp at 23b0202 against this fork. Prefill is `pp2048`, decode `tg128`, "32k" means after 32768
tokens of context, all numbers in t/s, each cell upstream → llama-tq. Each model ran in the order upstream, llama-tq,
llama-tq, upstream, and each cell is the mean of its two runs: on these GPUs the first build of a round is 2-3 % faster.

| Model | prefill f16 | prefill f16 32k | decode `q8_0` | decode f16 32k | decode `q8_0` 32k |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Q4_K_M | 3509 → 3544 | 1166 → 1167 | 89.2 → 90.9 | 37.3 → 37.5 | 20.0 → 41.9 |
| gpt-oss-20b MXFP4 | 2623 → 2660 | 1767 → 1778 | 78.6 → 80.1 | 65.3 → 65.6 | 43.4 → 56.4 |
| Gemma 4 12B Q4_K_M | 1318 → 1301 | 843 → 824 | 33.7 → 33.8 | 32.0 → 32.1 | 27.8 → 29.3 |
| Qwen3.6-35B-A3B Q4_K_M | 1836 → 1874 | 1286 → 1276 | 78.4 → 81.0 | 67.6 → 69.2 | 46.3 → 67.6 |
| Qwen3.8-27B Q4_K_M | 638 → 665 | 427 → 429 | 16.5 → 16.7 | 14.9 → 15.0 | 12.4 → 15.0 |

Decode with a quantized KV cache at long context is where the fork is ahead (its flash attention decode kernels read
`q8_0` and the TurboQuant types directly). Prefill with an f16 cache is within 1 % of upstream except on Gemma 4, which
is 1.3 % behind without context and 2.3 % behind at 32k; the cause is not known yet.

The fork's TurboQuant rows at 32k (same settings, 2026-10-14, after the GQA decode kernel change; mean of two runs,
builds alternated with the rows of 2026-10-13, which are given in brackets where they differ by more than 1 %):

| Model | `q8_0` / `vtq4_1` decode 32k | `ktq4_1` / `vtq4_1` decode 32k | `ktq4_1` / `vtq4_1` prefill 32k |
|---|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Q4_K_M | 46.1 (42.3) | 46.4 (39.5) | 1115 (1087) |
| gpt-oss-20b MXFP4 | 58.2 | 60.1 | 1748 (1726) |
| Gemma 4 12B Q4_K_M | 27.3 | 29.7 | 806 |
| Qwen3.6-35B-A3B Q4_K_M | 69.6 | 66.9 | 1270 |
| Qwen3.8-27B Q4_K_M | 15.2 | 14.7 | 426 |

gpt-oss (attention sinks) is not covered by the GQA kernel. On Gemma 4 (mostly sliding-window layers), Qwen3.6-35B-A3B
and Qwen3.8-27B (hybrid, few attention layers) attention over the cache is a small part of a decode step, so the
change stays within noise there.

KTQ K is staged in f16 during prefill and converted at the first decode step (about 0.1 s after a 16k prompt on one
RTX 2060). `llama-bench` runs that conversion before it times a generation test; before 2026-10-13 it was counted in
the decode numbers.

## Accuracy and fine-tuning

KV cache accuracy per type and model: [turboquant.md#accuracy](../turboquant.md#accuracy). Fine-tuning results and the
comparison with PyTorch QLoRA: [finetune.md#results](../finetune.md#results). Command and result per model:
[models.md](../models.md).
