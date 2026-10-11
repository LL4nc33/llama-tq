# TurboQuant benchmarks, April 2026

Moved from docs/turboquant.md on 2026-10-11. 
Historical measurements from 2026-04, before the KTQ fix of 2026-10-04. All measured on the test box: Ryzen 7 3700X host (Zen 2, 8C/16T), KVM guest 12 vCPUs, 40 GB DDR4-3200, 2× RTX 2060 12 GB on asymmetric PCIe (GPU0 x16 / GPU1 x4). llama-tq commit `1a1d49ef5`; upstream baseline `0c6ee1cad` (ggerganov/llama.cpp master, 2026-04-27).

### 35B-A3B full-GPU (ctx=2048, `-fa 1 -ts 12,12`)

| Variant                          | pp512 | tg128 | tg1024 | PPL (8ch) | KV-bpw | KV @ 32k |
|----------------------------------|---:|---:|---:|---:|---:|---:|
| upstream f16/f16                 | 934  | 57.76 | 57.30 | 7.0810 | 32.0 | 640 MiB |
| llama-tq f16/f16                 | 1009 | 76.58 | 75.09 | 7.0863 | 32.0 | 640 MiB |
| llama-tq `ktq2_1 + vtq2_2` ⭐    | 1010 | 75.40 | 75.18 | 7.1814 | 2.78 | 115 MiB |
| Δ tq vs upstream (quant)         | +8% | +31% | +31% | +1.34% | −91% | −82% |

### 80B-A3B (Qwen3-Next, hybrid + CPU expert offload, ctx=2048)

| Variant            | pp512 | tg128 |
|--------------------|---:|---:|
| upstream f16/f16   | 378 | 31.21 |
| llama-tq f16/f16   | 404 | 31.50 |
| Δ                  | +7% | +1% |

### 122B-A10B (Qwen3.5-A10B, GQA(2) + CPU expert offload, ctx=2048)

| Variant            | pp512 | tg128 |
|--------------------|---:|---:|
| upstream f16/f16   | 178 | 15.65 |
| llama-tq f16/f16   | 187 | 17.00 |
| Δ                  | +5% | +9% |

Historical snapshot of TG, PPL and HellaSwag for several models: [LIVE_NUMBERS.md](LIVE_NUMBERS.md).

