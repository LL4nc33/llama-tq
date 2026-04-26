# Safe bench (no VTQ_2, 1 rep, smaller ctx) Thu Apr 23 18:52:09 CEST 2026
Build: bc7c2e3d3

## Qwen3.6-35B single-GPU + expert-offload
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --------------------- | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        546.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.06 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        404.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         44.70 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        385.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         43.48 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        526.88 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.25 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        516.64 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         51.27 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        518.80 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         56.14 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        495.64 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         51.89 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        393.52 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.96 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        539.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.17 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        395.86 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         43.64 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        513.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.99 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        508.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.38 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        496.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         43.63 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        501.37 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         55.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        416.46 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.89 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        399.78 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.05 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        536.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         55.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        512.48 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        515.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         45.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        500.02 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        491.02 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         51.07 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        532.94 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.50 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        389.60 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        392.70 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.30 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        520.48 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.23 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        513.00 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.27 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        504.71 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.80 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        510.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        540.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         51.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        397.32 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.33 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        393.93 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.71 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        497.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         51.39 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        502.87 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         54.47 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        504.83 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.99 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        503.25 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         47.84 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        518.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.38 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        381.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.02 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        394.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.03 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        528.63 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         56.86 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        512.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         56.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        516.65 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.55 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        501.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         46.67 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        528.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.70 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        385.86 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        387.99 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         47.33 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        517.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         54.93 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        514.89 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         52.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        505.23 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         46.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        476.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        222.87 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.25 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        188.69 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         34.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        192.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        220.43 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.39 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        214.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.07 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        221.02 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.25 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        202.50 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.95 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        199.00 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         33.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        219.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        189.37 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         34.78 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        205.57 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.13 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        216.87 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         37.57 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        222.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.80 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        214.73 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         38.12 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        187.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         37.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        191.53 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         36.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        222.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        210.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         37.34 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        217.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.25 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        219.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         37.03 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        203.03 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         40.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        235.96 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.71 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        197.21 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         37.36 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        186.78 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         34.86 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        206.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.21 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        208.52 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         37.33 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        205.37 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.34 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        225.41 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         38.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        222.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.98 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        194.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         34.53 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        197.65 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.26 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        223.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         36.14 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        205.63 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.27 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        217.06 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         36.93 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        215.06 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.50 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        222.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         37.26 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        196.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         35.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        186.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         35.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        217.42 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.36 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        228.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.12 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        218.05 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        201.68 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         38.56 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        222.18 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         45.18 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        183.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         34.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        189.43 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         36.88 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        211.00 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         38.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        213.84 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         45.17 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        208.93 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         37.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        211.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.54 ± 0.00 |
               total        used        free      shared  buff/cache   available
Mem:            39Gi       2.3Gi        23Gi        25Mi        13Gi        36Gi

## Qwen3.6-35B dual-GPU no offload
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | ------------ | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |           pp512 |        985.71 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |           tg128 |         72.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |           pp512 |        656.00 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |           tg128 |         67.59 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |           pp512 |        652.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |           tg128 |         60.92 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |           pp512 |        908.99 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |           tg128 |         71.04 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |           pp512 |        908.98 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |           tg128 |         71.37 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |           pp512 |        892.11 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |           tg128 |         70.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |           pp512 |        840.08 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |           tg128 |         70.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |           pp512 |        721.59 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |           tg128 |         59.14 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |           pp512 |        973.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |           tg128 |         70.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |           pp512 |        666.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |           tg128 |         61.53 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |           pp512 |        881.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |           tg128 |         69.36 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |           pp512 |        874.27 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |           tg128 |         69.55 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |           pp512 |        842.73 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |           tg128 |         68.52 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |           pp512 |        827.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |           tg128 |         68.83 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |           pp512 |        687.94 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |           tg128 |         58.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |           pp512 |        671.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |           tg128 |         57.99 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |           pp512 |        960.96 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |           tg128 |         69.15 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |           pp512 |        868.96 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |           tg128 |         69.00 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |           pp512 |        878.63 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |           tg128 |         69.18 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |           pp512 |        841.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |           tg128 |         68.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |           pp512 |        820.38 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |           tg128 |         68.56 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |           pp512 |        949.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |           tg128 |         71.13 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        |           pp512 |        642.67 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        |           tg128 |         56.14 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        |           pp512 |        625.78 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        |           tg128 |         65.67 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        |           pp512 |        890.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        |           tg128 |         69.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        |           pp512 |        888.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        |           tg128 |         70.16 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        |           pp512 |        857.03 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        |           tg128 |         69.12 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        |           pp512 |        840.06 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        |           tg128 |         69.47 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        |           pp512 |        938.32 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        |           tg128 |         70.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        |           pp512 |        602.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        |           tg128 |         63.08 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        |           pp512 |        646.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        |           tg128 |         55.13 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        |           pp512 |        864.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        |           tg128 |         69.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        |           pp512 |        876.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        |           tg128 |         69.39 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        |           pp512 |        845.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        |           tg128 |         68.41 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        |           pp512 |        822.38 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        |           tg128 |         68.57 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        |           pp512 |        896.67 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        |           tg128 |         69.17 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        |           pp512 |        643.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        |           tg128 |         63.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        |           pp512 |        590.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        |           tg128 |         65.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        |           pp512 |        872.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        |           tg128 |         68.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        |           pp512 |        865.99 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        |           tg128 |         68.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        |           pp512 |        829.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        |           tg128 |         67.42 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        |           pp512 |        809.04 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        |           tg128 |         67.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        |           pp512 |        904.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        |           tg128 |         69.14 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        |           pp512 |        633.03 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        |           tg128 |         56.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        |           pp512 |        639.20 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        |           tg128 |         64.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        |           pp512 |        866.72 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        |           tg128 |         67.87 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        |           pp512 |        869.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        |           tg128 |         68.21 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        |           pp512 |        841.25 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        |           tg128 |         67.53 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        |           pp512 |        784.06 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        |           tg128 |         67.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |           pp512 |        922.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |           tg128 |         69.59 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |           pp512 |        630.26 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |           tg128 |         57.32 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |           pp512 |        642.68 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |           tg128 |         57.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |           pp512 |        865.41 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |           tg128 |         68.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |           pp512 |        838.07 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |           tg128 |         68.89 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |           pp512 |        836.36 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |           tg128 |         68.08 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |           pp512 |        797.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |           tg128 |         68.04 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |           pp512 |        717.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |           tg128 |         52.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |           pp512 |        921.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |           tg128 |         67.21 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |           pp512 |        628.65 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |           tg128 |         65.06 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |           pp512 |        823.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |           tg128 |         66.93 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |           pp512 |        838.72 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |           tg128 |         67.15 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |           pp512 |        813.06 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |           tg128 |         66.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |           pp512 |        789.67 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |           tg128 |         66.80 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |           pp512 |        664.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |           tg128 |         61.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |           pp512 |        661.72 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |           tg128 |         57.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |           pp512 |        915.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |           tg128 |         67.03 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |           pp512 |        820.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |           tg128 |         66.80 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |           pp512 |        823.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |           tg128 |         67.42 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |           pp512 |        791.52 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |           tg128 |         66.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |           pp512 |        792.25 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |           tg128 |         66.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |           pp512 |        940.15 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |           tg128 |         68.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        |           pp512 |        625.56 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        |           tg128 |         57.39 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        |           pp512 |        619.59 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        |           tg128 |         55.16 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        |           pp512 |        845.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        |           tg128 |         68.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        |           pp512 |        847.95 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        |           tg128 |         68.18 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        |           pp512 |        823.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        |           tg128 |         67.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        |           pp512 |        820.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        |           tg128 |         67.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        |           pp512 |        928.84 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        |           tg128 |         69.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        |           pp512 |        639.08 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        |           tg128 |         63.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        |           pp512 |        589.60 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        |           tg128 |         62.57 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        |           pp512 |        861.58 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        |           tg128 |         68.21 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        |           pp512 |        848.95 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        |           tg128 |         68.26 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        |           pp512 |        825.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        |           tg128 |         67.17 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        |           pp512 |        797.30 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        |           tg128 |         67.72 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        |           pp512 |        922.04 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        |           tg128 |         69.02 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        |           pp512 |        625.98 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        |           tg128 |         53.72 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        |           pp512 |        615.10 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        |           tg128 |         63.10 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        |           pp512 |        857.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        |           tg128 |         67.72 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        |           pp512 |        873.26 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        |           tg128 |         68.56 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        |           pp512 |        824.06 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        |           tg128 |         67.06 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        |           pp512 |        798.07 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        |           tg128 |         67.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        |           pp512 |        921.50 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        |           tg128 |         68.92 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        |           pp512 |        594.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        |           tg128 |         58.70 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        |           pp512 |        614.36 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        |           tg128 |         56.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        |           pp512 |        850.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        |           tg128 |         68.18 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        |           pp512 |        864.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        |           tg128 |         68.30 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        |           pp512 |        821.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        |           tg128 |         67.43 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        |           pp512 |        814.50 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        |           tg128 |         67.82 ± 0.00 |
               total        used        free      shared  buff/cache   available
Mem:            39Gi       2.4Gi        23Gi        25Mi        13Gi        36Gi

## Qwen3.6-35B dual-GPU + expert-offload
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa | ts           | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | ------------ | --------------------- | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        852.68 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.70 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        580.13 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.03 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        540.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.14 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        796.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         54.23 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        793.56 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        764.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         47.12 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        697.88 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         46.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        656.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.78 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        841.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         56.17 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        583.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.18 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        773.05 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         55.16 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        769.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.41 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        713.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        699.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        630.10 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         59.04 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        565.83 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.71 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        840.59 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.44 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        731.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         46.91 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        774.11 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         54.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        745.92 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        727.68 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        839.29 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         51.20 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        582.42 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.88 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        590.39 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         52.14 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        759.15 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        792.72 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         55.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        767.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.87 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        760.04 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.26 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        835.93 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.64 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        567.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        591.63 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         52.18 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        774.16 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         52.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        782.89 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         56.65 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        755.80 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        753.02 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.38 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        816.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.64 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        527.10 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.50 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        577.17 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.89 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        802.88 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.78 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        800.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        773.43 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         54.10 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        748.17 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.96 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        828.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         54.57 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        568.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        584.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         47.44 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        787.99 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.62 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        794.56 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         54.42 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        765.58 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         55.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        725.68 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         47.57 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        647.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.94 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        482.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         40.94 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        475.54 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         37.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        621.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.58 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        538.80 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        539.84 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.53 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        573.09 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         47.37 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        514.02 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         46.63 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        636.36 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.86 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        455.10 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        588.50 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.11 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        600.07 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         37.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        604.63 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        580.20 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        488.83 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         35.58 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        495.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         40.93 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        643.34 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.27 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        514.54 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         35.17 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        536.78 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         48.12 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        509.37 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         45.94 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        563.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         38.21 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        568.65 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         45.03 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        473.41 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.64 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        422.00 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.27 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        592.59 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.57 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        600.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        586.41 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        602.20 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        649.70 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        490.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         36.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        479.32 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.17 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        620.56 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         47.21 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        600.96 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.54 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        599.65 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         47.20 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        564.84 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         48.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        649.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        498.91 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        478.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        613.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.95 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        638.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         47.18 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        605.83 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         48.56 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        566.10 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         48.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        651.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.70 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        486.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        488.57 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.52 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        602.84 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.05 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        618.47 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        594.54 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         48.47 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        588.15 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         45.86 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        843.88 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         60.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        603.39 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.91 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        603.86 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.95 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        812.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.69 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        807.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         63.59 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        780.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         54.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        756.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         60.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        610.29 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         61.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        860.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         54.42 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        612.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         61.42 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        784.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.86 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        771.94 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.24 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        757.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         59.53 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        723.72 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         59.94 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        665.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         51.51 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        618.78 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         55.87 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        857.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         56.89 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        791.12 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         49.44 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        763.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.50 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        758.92 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        731.65 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         51.43 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        822.11 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.88 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        605.96 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.92 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        564.03 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         57.64 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        793.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         51.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        789.68 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         59.64 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        764.76 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.41 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        720.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         48.60 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        836.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         60.09 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        602.16 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         60.19 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        594.39 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         52.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        787.69 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.29 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        798.21 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.86 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        755.29 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         56.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        740.73 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         54.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        841.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         56.70 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        571.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         59.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        589.78 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         52.64 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        798.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         55.78 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        790.96 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        769.80 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         52.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        763.48 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        852.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        610.38 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         50.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        583.95 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         53.12 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        776.08 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         59.25 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        802.87 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         58.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        763.27 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         47.95 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           pp512 |        743.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |           tg128 |         60.95 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        629.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         50.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        482.11 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.79 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        488.12 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         48.26 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        596.63 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         47.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        595.85 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.07 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        587.55 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.60 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        580.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        554.07 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         46.44 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        648.03 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         47.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        475.49 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         45.50 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        598.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         46.90 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        595.05 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         40.43 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        597.87 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.88 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        560.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.74 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        517.87 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        498.39 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.98 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        650.37 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         47.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        607.44 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.94 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        586.06 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         48.93 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        599.02 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        582.31 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         40.08 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        654.87 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.73 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        491.09 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        468.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.88 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        613.89 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         47.56 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        611.15 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         46.59 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        603.34 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        585.86 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.08 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        618.71 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.28 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        478.95 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        467.73 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         40.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        603.54 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         40.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        601.80 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         45.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        578.89 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         44.89 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        586.35 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         45.45 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        649.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.32 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        484.83 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         36.97 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        470.91 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.02 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        616.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.09 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        609.29 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         38.43 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        609.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.53 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        582.22 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         41.00 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        655.66 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         38.41 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        481.05 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.01 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        478.40 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         43.18 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        606.77 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.55 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        616.61 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         45.73 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        607.75 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         42.82 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           pp512 |        584.81 ± 0.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        | blk.[2-4][0-9].ffn_.*_exps.=CPU |           tg128 |         39.45 ± 0.00 |
               total        used        free      shared  buff/cache   available
Mem:            39Gi       2.4Gi        23Gi        25Mi        13Gi        36Gi

## Ministral-14B single-GPU
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --------------: | -------------------: |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |    f16 |  1 |           pp512 |        748.78 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |    f16 |  1 |           tg128 |         25.80 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |   q8_0 |  1 |           pp512 |        216.93 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |   q8_0 |  1 |           tg128 |         13.58 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |   q4_0 |  1 |           pp512 |        211.35 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |   q4_0 |  1 |           tg128 |         13.93 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq1_1 |  1 |           pp512 |        619.22 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq1_1 |  1 |           tg128 |         25.32 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq2_1 |  1 |           pp512 |        606.83 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq2_1 |  1 |           tg128 |         25.24 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq3_1 |  1 |           pp512 |        560.81 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq3_1 |  1 |           tg128 |         25.19 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq4_1 |  1 |           pp512 |        514.57 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq4_1 |  1 |           tg128 |         24.96 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |    f16 |  1 |           pp512 |        242.00 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |    f16 |  1 |           tg128 |         13.49 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 |           pp512 |        707.82 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 |           tg128 |         24.65 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 |           pp512 |        218.07 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 |           tg128 |         14.98 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 |           pp512 |        510.65 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 |           tg128 |         24.46 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 |           pp512 |        500.96 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 |           tg128 |         24.49 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 |           pp512 |        470.79 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 |           tg128 |         24.31 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 |           pp512 |        435.21 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 |           tg128 |         24.11 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |    f16 |  1 |           pp512 |        247.43 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |    f16 |  1 |           tg128 |         13.77 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 |           pp512 |        209.59 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 |           tg128 |         15.74 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 |           pp512 |        727.14 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 |           tg128 |         24.38 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 |           pp512 |        507.28 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 |           tg128 |         24.36 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 |           pp512 |        505.31 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 |           tg128 |         24.38 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 |           pp512 |        467.72 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 |           tg128 |         24.18 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 |           pp512 |        434.31 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 |           tg128 |         24.14 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |    f16 |  1 |           pp512 |        727.00 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |    f16 |  1 |           tg128 |         24.32 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 |           pp512 |        205.28 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 |           tg128 |         13.48 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 |           pp512 |        211.03 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 |           tg128 |         14.77 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 |           pp512 |        605.29 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 |           tg128 |         23.88 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 |           pp512 |        596.38 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 |           tg128 |         23.96 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 |           pp512 |        563.47 ± 0.00 |
               total        used        free      shared  buff/cache   available
Mem:            39Gi       2.4Gi        19Gi        25Mi        18Gi        36Gi

## Ministral-14B dual-GPU
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | ------------ | --------------: | -------------------: |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |           pp512 |        739.71 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |           tg128 |         25.58 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |           pp512 |        251.60 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |           tg128 |         19.25 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |           pp512 |        243.27 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |           tg128 |         19.07 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |           pp512 |        604.48 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |           tg128 |         24.99 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |           pp512 |        594.36 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |           tg128 |         25.01 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |           pp512 |        556.19 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |           tg128 |         24.70 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |           pp512 |        501.97 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |           tg128 |         24.61 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |           pp512 |        322.14 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |           tg128 |         19.40 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |           pp512 |        705.68 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |           tg128 |         24.19 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |           pp512 |        265.75 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |           tg128 |         18.34 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |           pp512 |        492.26 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |           tg128 |         23.70 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |           pp512 |        485.99 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |           tg128 |         23.37 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |           pp512 |        446.43 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |           tg128 |         23.02 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |           pp512 |        409.67 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |           tg128 |         22.49 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |           pp512 |        309.97 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |           tg128 |         19.22 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |           pp512 |        267.16 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |           tg128 |         18.40 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |           pp512 |        670.22 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |           tg128 |         22.92 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |           pp512 |        473.10 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |           tg128 |         22.69 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |           pp512 |        466.42 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |           tg128 |         22.64 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |           pp512 |        438.28 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |           tg128 |         22.39 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |           pp512 |        407.04 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |           tg128 |         22.41 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |           pp512 |        674.20 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |           tg128 |         22.44 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        |           pp512 |        244.40 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        |           tg128 |         15.68 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        |           pp512 |        255.74 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        |           tg128 |         15.40 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        |           pp512 |        575.59 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        |           tg128 |         22.81 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        |           pp512 |        548.75 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        |           tg128 |         22.66 ± 0.00 |
| mistral3 14B IQ2_M - 2.7 bpw   |   4.57 GiB |    13.51 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        |           pp512 |        518.23 ± 0.00 |
               total        used        free      shared  buff/cache   available
Mem:            39Gi       2.5Gi        19Gi        25Mi        18Gi        36Gi

## Qwen3.5-27B-Dense single-GPU
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |    f16 |  1 |           pp512 |        409.39 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |    f16 |  1 |           tg128 |         15.03 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q8_0 |  1 |           pp512 |        236.75 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q8_0 |  1 |           tg128 |         11.55 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q4_0 |  1 |           pp512 |        231.76 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q4_0 |  1 |           tg128 |         11.71 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq1_1 |  1 |           pp512 |        378.39 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq1_1 |  1 |           tg128 |         14.80 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq2_1 |  1 |           pp512 |        375.06 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq2_1 |  1 |           tg128 |         14.74 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq3_1 |  1 |           pp512 |        360.85 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq3_1 |  1 |           tg128 |         14.68 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq4_1 |  1 |           pp512 |        347.25 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq4_1 |  1 |           tg128 |         14.71 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |    f16 |  1 |           pp512 |        258.49 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |    f16 |  1 |           tg128 |         11.36 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 |           pp512 |        392.91 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 |           tg128 |         14.55 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 |           pp512 |        241.66 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 |           tg128 |         11.65 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 |           pp512 |        364.52 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 |           tg128 |         14.47 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 |           pp512 |        360.40 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 |           tg128 |         14.53 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 |           pp512 |        352.08 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 |           tg128 |         14.46 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 |           pp512 |        340.78 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 |           tg128 |         14.45 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |    f16 |  1 |           pp512 |        258.40 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |    f16 |  1 |           tg128 |         11.51 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 |           pp512 |        239.45 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 |           tg128 |         11.68 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 |           pp512 |        391.20 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 |           tg128 |         14.51 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 |           pp512 |        358.75 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 |           tg128 |         14.50 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 |           pp512 |        363.74 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 |           tg128 |         14.52 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 |           pp512 |        352.02 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 |           tg128 |         14.43 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 |           pp512 |        341.40 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 |           tg128 |         14.45 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |    f16 |  1 |           pp512 |        397.51 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |    f16 |  1 |           tg128 |         14.61 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 |           pp512 |        227.91 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 |           tg128 |         11.19 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 |           pp512 |        233.16 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 |           tg128 |         11.36 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 |           pp512 |        372.29 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 |           tg128 |         14.48 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 |           pp512 |        374.02 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 |           tg128 |         14.56 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 |           pp512 |        361.00 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 |           tg128 |         14.49 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 |           pp512 |        347.57 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 |           tg128 |         14.52 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |    f16 |  1 |           pp512 |        391.78 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |    f16 |  1 |           tg128 |         14.65 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 |           pp512 |        232.74 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 |           tg128 |         11.31 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 |           pp512 |        228.33 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 |           tg128 |         11.21 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 |           pp512 |        372.47 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 |           tg128 |         14.52 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 |           pp512 |        374.12 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 |           tg128 |         14.58 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 |           pp512 |        356.98 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 |           tg128 |         14.54 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 |           pp512 |        343.16 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 |           tg128 |         14.55 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |    f16 |  1 |           pp512 |        397.31 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |    f16 |  1 |           tg128 |         14.64 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 |           pp512 |        238.52 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 |           tg128 |         11.18 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 |           pp512 |        238.02 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 |           tg128 |         11.31 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 |           pp512 |        367.76 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 |           tg128 |         14.56 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 |           pp512 |        374.15 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 |           tg128 |         14.58 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 |           pp512 |        361.21 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 |           tg128 |         14.53 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 |           pp512 |        347.45 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 |           tg128 |         14.54 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |    f16 |  1 |           pp512 |        397.14 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |    f16 |  1 |           tg128 |         14.62 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 |           pp512 |        234.22 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 |           tg128 |         11.25 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 |           pp512 |        233.32 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 |           tg128 |         11.36 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 |           pp512 |        372.13 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 |           tg128 |         14.52 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 |           pp512 |        373.97 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 |           tg128 |         14.55 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 |           pp512 |        360.86 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 |           tg128 |         14.49 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 |           pp512 |        347.87 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 |           tg128 |         14.51 ± 0.00 |
               total        used        free      shared  buff/cache   available
Mem:            39Gi       2.4Gi        11Gi        25Mi        26Gi        36Gi

## Qwen3.5-27B-Dense dual-GPU
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | ------------ | --------------: | -------------------: |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |           pp512 |        401.55 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |           tg128 |         14.84 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |           pp512 |        270.17 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |           tg128 |         12.75 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |           pp512 |        264.06 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |           tg128 |         12.50 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |           pp512 |        369.01 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |           tg128 |         14.50 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |           pp512 |        370.12 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |           tg128 |         14.44 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |           pp512 |        351.47 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |           tg128 |         14.02 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |           pp512 |        329.92 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |           tg128 |         13.70 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |           pp512 |        290.80 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |           tg128 |         11.91 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |           pp512 |        374.88 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |           tg128 |         13.59 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |           pp512 |        258.25 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |           tg128 |         12.09 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |           pp512 |        341.69 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |           tg128 |         13.54 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |           pp512 |        342.37 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |           tg128 |         13.56 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |           pp512 |        330.87 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |           tg128 |         13.52 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |           pp512 |        317.36 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |           tg128 |         13.54 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |           pp512 |        284.20 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |           tg128 |         11.87 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |           pp512 |        261.65 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |           tg128 |         11.95 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |           pp512 |        374.16 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |           tg128 |         13.56 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |           pp512 |        341.38 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |           tg128 |         13.55 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |           pp512 |        342.42 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |           tg128 |         13.56 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |           pp512 |        331.35 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |           tg128 |         13.50 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |           pp512 |        319.04 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |           tg128 |         13.54 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |           pp512 |        375.08 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |           tg128 |         13.69 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        |           pp512 |        255.31 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        |           tg128 |         11.92 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        |           pp512 |        255.57 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        |           tg128 |         11.92 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        |           pp512 |        350.41 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        |           tg128 |         13.61 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        |           pp512 |        352.22 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        |           tg128 |         13.64 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        |           pp512 |        339.30 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        |           tg128 |         13.58 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        |           pp512 |        324.87 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        |           tg128 |         13.61 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        |           pp512 |        375.19 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        |           tg128 |         13.70 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        |           pp512 |        261.79 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        |           tg128 |         11.81 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        |           pp512 |        260.68 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        |           tg128 |         11.97 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        |           pp512 |        350.32 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        |           tg128 |         13.61 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        |           pp512 |        352.08 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        |           tg128 |         13.63 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        |           pp512 |        339.24 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        |           tg128 |         13.59 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        |           pp512 |        323.53 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        |           tg128 |         13.60 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        |           pp512 |        374.64 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        |           tg128 |         13.70 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        |           pp512 |        262.61 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        |           tg128 |         11.67 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        |           pp512 |        256.38 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        |           tg128 |         12.08 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        |           pp512 |        350.41 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        |           tg128 |         13.61 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        |           pp512 |        352.55 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        |           tg128 |         13.64 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        |           pp512 |        338.68 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        |           tg128 |         13.58 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        |           pp512 |        323.68 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        |           tg128 |         13.60 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        |           pp512 |        375.22 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        |           tg128 |         13.69 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        |           pp512 |        259.54 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        |           tg128 |         12.07 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        |           pp512 |        255.86 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        |           tg128 |         12.01 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        |           pp512 |        350.07 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        |           tg128 |         13.60 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        |           pp512 |        351.79 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        |           tg128 |         13.66 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        |           pp512 |        338.34 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        |           tg128 |         13.58 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        |           pp512 |        323.78 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        |           tg128 |         13.61 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |           pp512 |        375.87 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |           tg128 |         13.76 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |           pp512 |        259.72 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |           tg128 |         12.29 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |           pp512 |        255.82 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |           tg128 |         12.19 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |           pp512 |        351.65 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |           tg128 |         13.66 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |           pp512 |        353.15 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |           tg128 |         13.69 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |           pp512 |        339.73 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |           tg128 |         13.63 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |           pp512 |        324.56 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |           tg128 |         13.65 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |           pp512 |        288.53 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |           tg128 |         12.19 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |           pp512 |        373.77 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |           tg128 |         13.59 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |           pp512 |        261.29 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |           tg128 |         12.46 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |           pp512 |        340.39 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |           tg128 |         13.56 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |           pp512 |        341.75 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |           tg128 |         13.56 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |           pp512 |        329.13 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |           tg128 |         13.52 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |           pp512 |        316.16 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |           tg128 |         13.54 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |           pp512 |        287.53 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |           tg128 |         11.94 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |           pp512 |        271.74 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |           tg128 |         12.21 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |           pp512 |        374.19 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |           tg128 |         13.55 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |           pp512 |        341.37 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |           tg128 |         13.56 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |           pp512 |        340.87 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |           tg128 |         13.57 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |           pp512 |        329.48 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |           tg128 |         13.50 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |           pp512 |        317.74 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |           tg128 |         13.53 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |           pp512 |        373.97 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |           tg128 |         13.69 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        |           pp512 |        258.98 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | 12.00        |           tg128 |         11.84 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        |           pp512 |        259.34 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | 12.00        |           tg128 |         11.92 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        |           pp512 |        350.01 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | 12.00        |           tg128 |         13.62 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        |           pp512 |        351.91 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | 12.00        |           tg128 |         13.64 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        |           pp512 |        338.27 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | 12.00        |           tg128 |         13.60 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        |           pp512 |        324.43 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | 12.00        |           tg128 |         13.62 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        |           pp512 |        374.16 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | 12.00        |           tg128 |         13.69 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        |           pp512 |        257.27 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | 12.00        |           tg128 |         11.80 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        |           pp512 |        260.46 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | 12.00        |           tg128 |         12.09 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        |           pp512 |        350.48 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | 12.00        |           tg128 |         13.60 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        |           pp512 |        351.87 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | 12.00        |           tg128 |         13.63 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        |           pp512 |        338.25 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | 12.00        |           tg128 |         13.57 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        |           pp512 |        324.16 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | 12.00        |           tg128 |         13.61 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        |           pp512 |        374.67 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | 12.00        |           tg128 |         13.69 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        |           pp512 |        259.88 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | 12.00        |           tg128 |         12.13 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        |           pp512 |        256.94 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | 12.00        |           tg128 |         11.91 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        |           pp512 |        350.50 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | 12.00        |           tg128 |         13.61 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        |           pp512 |        352.25 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | 12.00        |           tg128 |         13.67 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        |           pp512 |        338.54 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | 12.00        |           tg128 |         13.58 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        |           pp512 |        324.08 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | 12.00        |           tg128 |         13.60 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        |           pp512 |        374.27 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | 12.00        |           tg128 |         13.70 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        |           pp512 |        263.42 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | 12.00        |           tg128 |         11.98 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        |           pp512 |        258.24 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | 12.00        |           tg128 |         12.13 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        |           pp512 |        350.10 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | 12.00        |           tg128 |         13.62 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        |           pp512 |        352.83 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | 12.00        |           tg128 |         13.65 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        |           pp512 |        339.38 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | 12.00        |           tg128 |         13.58 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        |           pp512 |        324.82 ± 0.00 |
| qwen35 27B IQ2_XXS - 2.0625 bpw |   7.97 GiB |    26.90 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | 12.00        |           tg128 |         13.60 ± 0.00 |
               total        used        free      shared  buff/cache   available
Mem:            39Gi       2.4Gi        11Gi        25Mi        26Gi        36Gi
