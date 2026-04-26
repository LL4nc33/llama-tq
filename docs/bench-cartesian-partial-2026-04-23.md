Read from remote host gpu00.node: Connection reset by peer
client_loop: send disconnect: Broken pipe
s: 2, pp2048 + tg512


## Single-GPU + expert-offload 30 layers

| model                          |       size |     params | backend    | ngl | type_k | type_v | fa | ot                    | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --------------------- | ---: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        520.40 ± 2.33 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         52.63 ± 0.11 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        255.57 ± 0.25 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.45 ± 0.09 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        249.37 ± 0.19 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.52 ± 0.36 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        474.03 ± 0.89 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         52.18 ± 0.68 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        472.91 ± 0.46 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         51.28 ± 0.89 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        451.29 ± 1.23 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         50.73 ± 0.21 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        435.82 ± 1.80 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         50.63 ± 0.91 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        508.09 ± 1.94 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         52.62 ± 0.44 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        512.00 ± 9.93 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         51.90 ± 0.41 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        512.40 ± 2.93 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         51.89 ± 0.07 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        302.91 ± 0.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         43.37 ± 1.63 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        514.71 ± 6.09 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         48.79 ± 7.08 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        254.23 ± 0.55 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.68 ± 0.36 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        458.60 ± 2.15 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         52.76 ± 0.36 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        447.20 ± 1.50 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         50.36 ± 0.33 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |       453.92 ± 21.20 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         51.12 ± 1.24 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        423.59 ± 2.53 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         52.22 ± 1.86 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        302.28 ± 3.09 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.08 ± 0.52 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        300.50 ± 1.08 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.18 ± 0.57 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        304.85 ± 1.59 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.29 ± 0.82 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        297.60 ± 2.91 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         45.95 ± 0.81 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        268.03 ± 3.67 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         48.49 ± 0.14 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        511.69 ± 6.80 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.97 ± 1.36 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        459.15 ± 6.55 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.21 ± 0.24 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        451.06 ± 3.14 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.41 ± 1.15 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        434.23 ± 2.22 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.06 ± 1.11 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        424.71 ± 0.73 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.45 ± 0.43 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        298.38 ± 0.29 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         45.35 ± 0.46 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        297.26 ± 2.41 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         45.19 ± 0.11 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        296.86 ± 1.36 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         45.49 ± 0.45 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        512.81 ± 4.45 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.39 ± 0.46 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        256.17 ± 0.52 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.00 ± 0.18 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        249.82 ± 3.13 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         45.10 ± 0.48 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        466.14 ± 0.46 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.73 ± 1.22 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        469.76 ± 1.77 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         56.54 ± 1.19 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        448.50 ± 0.63 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.65 ± 1.56 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        432.83 ± 3.41 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.72 ± 0.91 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        515.33 ± 3.20 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.41 ± 1.79 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        510.90 ± 2.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.53 ± 2.47 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        510.42 ± 1.57 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.72 ± 1.52 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        509.46 ± 3.91 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         56.19 ± 1.88 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        256.32 ± 3.71 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.56 ± 0.25 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        249.13 ± 3.19 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         45.30 ± 0.37 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        471.47 ± 0.44 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.11 ± 1.65 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        470.68 ± 3.28 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.17 ± 1.66 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |       452.29 ± 10.34 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.70 ± 1.34 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        431.79 ± 5.68 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.25 ± 0.41 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        516.16 ± 5.92 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.79 ± 0.11 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        508.55 ± 9.07 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.40 ± 0.96 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        515.36 ± 2.07 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.34 ± 0.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        515.67 ± 6.41 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.65 ± 1.06 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        256.53 ± 2.58 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.04 ± 0.35 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        249.54 ± 0.85 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.87 ± 0.09 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        473.84 ± 2.07 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.40 ± 0.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        470.31 ± 3.76 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.94 ± 1.04 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        450.75 ± 7.61 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.59 ± 0.43 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        435.89 ± 2.26 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.84 ± 0.40 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        510.27 ± 7.61 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.38 ± 0.93 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        509.15 ± 1.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         56.04 ± 0.94 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        511.68 ± 4.24 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         56.40 ± 0.79 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        508.38 ± 0.47 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.70 ± 0.50 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        256.25 ± 0.16 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         43.84 ± 0.15 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        252.26 ± 1.24 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         45.03 ± 0.75 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        464.98 ± 3.94 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.92 ± 0.38 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        466.78 ± 3.40 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.88 ± 1.20 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        450.34 ± 0.26 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         55.51 ± 0.74 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        432.36 ± 3.60 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.62 ± 0.07 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        514.89 ± 4.22 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         56.31 ± 0.24 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        515.05 ± 7.36 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |        40.27 ± 10.17 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        516.11 ± 1.06 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_2 |  1 | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         56.07 ± 1.04 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        219.99 ± 5.46 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.67 ± 0.12 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        153.33 ± 0.96 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         35.76 ± 0.02 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        151.96 ± 1.99 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         36.65 ± 0.37 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        208.35 ± 9.74 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.18 ± 0.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        204.09 ± 4.36 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.09 ± 0.26 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        209.97 ± 3.13 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         43.49 ± 0.74 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        196.63 ± 1.78 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         43.36 ± 0.51 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        215.18 ± 2.29 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         44.73 ± 0.39 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        212.00 ± 1.98 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         43.80 ± 0.60 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        219.03 ± 4.37 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         43.90 ± 0.69 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        169.25 ± 5.72 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         35.34 ± 0.34 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        213.73 ± 1.00 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.88 ± 0.57 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        153.30 ± 5.59 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         37.36 ± 0.02 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        205.49 ± 4.81 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.57 ± 1.27 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        203.22 ± 1.76 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.50 ± 0.03 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        200.72 ± 2.11 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.64 ± 0.09 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        195.94 ± 5.96 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.69 ± 0.86 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        168.82 ± 4.65 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         35.54 ± 0.24 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        166.37 ± 5.45 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         35.59 ± 0.58 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        166.78 ± 4.19 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         35.45 ± 0.44 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        165.46 ± 2.32 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         36.72 ± 0.75 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        154.43 ± 0.19 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         37.98 ± 0.03 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        216.10 ± 6.02 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.83 ± 0.33 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        207.51 ± 0.04 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.57 ± 0.19 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        201.73 ± 1.50 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.79 ± 0.37 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        199.87 ± 1.01 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.55 ± 0.23 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        197.68 ± 0.41 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.00 ± 0.72 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        166.55 ± 4.30 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         36.54 ± 0.27 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        166.87 ± 1.03 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         36.07 ± 0.21 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        164.86 ± 3.15 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         36.02 ± 0.58 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        219.27 ± 1.64 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         43.70 ± 0.28 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        151.43 ± 2.89 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         34.94 ± 0.25 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        148.83 ± 4.50 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         35.87 ± 0.31 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        207.90 ± 5.90 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.08 ± 0.58 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        205.68 ± 0.92 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.35 ± 1.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        201.71 ± 0.30 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.36 ± 0.60 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        202.99 ± 1.62 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.15 ± 0.18 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        219.27 ± 1.38 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.75 ± 0.55 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        217.73 ± 2.83 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.98 ± 0.75 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        210.38 ± 4.70 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.74 ± 0.52 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        220.47 ± 8.41 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         43.27 ± 0.31 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        153.04 ± 1.10 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         35.43 ± 0.32 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        153.03 ± 0.13 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         36.19 ± 0.13 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        208.36 ± 0.76 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.24 ± 0.13 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        204.43 ± 1.42 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.21 ± 0.10 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        208.57 ± 8.14 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.96 ± 0.11 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        203.65 ± 4.23 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.43 ± 1.17 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        211.97 ± 3.21 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.46 ± 1.22 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        215.61 ± 5.63 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.98 ± 1.32 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        213.51 ± 0.37 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq2_1 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         40.86 ± 0.54 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        210.86 ± 3.40 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         39.14 ± 0.27 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        147.08 ± 3.89 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         33.31 ± 0.19 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        146.42 ± 4.60 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         33.39 ± 2.49 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        207.97 ± 3.40 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         37.51 ± 0.61 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        205.58 ± 0.57 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         38.40 ± 0.65 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        206.42 ± 0.31 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         35.94 ± 0.71 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        200.38 ± 1.29 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         37.23 ± 0.71 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        214.77 ± 8.44 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         36.67 ± 1.24 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        218.94 ± 4.46 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         39.27 ± 1.77 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        214.99 ± 4.86 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq3_1 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         38.32 ± 2.14 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        214.64 ± 0.93 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |    f16 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.66 ± 0.28 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        148.24 ± 0.90 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q8_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         33.96 ± 1.13 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        149.40 ± 2.10 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 |   q4_0 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         35.73 ± 0.16 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        207.27 ± 1.88 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq1_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.56 ± 0.49 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        206.16 ± 1.85 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         42.05 ± 0.28 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        203.47 ± 2.69 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         41.95 ± 0.10 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        200.82 ± 3.55 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_1 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         37.76 ± 0.32 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        211.52 ± 3.67 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq2_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         40.69 ± 0.41 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        212.90 ± 4.02 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq3_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         36.95 ± 2.02 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        214.65 ± 3.99 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq4_1 | vtq4_2 |  1 | blk.[2-4][0-9].ffn_.*_exps.=CPU |    0 |           tg512 |         38.69 ± 0.46 |
build: 0a15c5a55 (17928)

## Dual-GPU no offload

| model                          |       size |     params | backend    | ngl | type_k | type_v | fa | ts           | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | ------------ | ---: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |    0 |          pp2048 |        891.26 ± 3.10 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        |    0 |           tg512 |         68.51 ± 0.25 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |    0 |          pp2048 |        344.41 ± 0.21 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        |    0 |           tg512 |         58.37 ± 2.15 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |    0 |          pp2048 |        331.20 ± 2.71 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        |    0 |           tg512 |         55.73 ± 1.57 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |    0 |          pp2048 |        747.01 ± 4.07 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        |    0 |           tg512 |         66.00 ± 0.12 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |    0 |          pp2048 |        749.44 ± 1.40 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        |    0 |           tg512 |         66.28 ± 0.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |    0 |          pp2048 |        689.12 ± 0.87 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        |    0 |           tg512 |         65.26 ± 0.11 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |    0 |          pp2048 |        647.91 ± 1.49 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        |    0 |           tg512 |         65.32 ± 0.15 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_2 |  1 | 12.00        |    0 |          pp2048 |        855.75 ± 3.04 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_2 |  1 | 12.00        |    0 |           tg512 |         66.53 ± 0.01 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_2 |  1 | 12.00        |    0 |          pp2048 |        858.31 ± 2.74 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_2 |  1 | 12.00        |    0 |           tg512 |         66.48 ± 0.04 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_2 |  1 | 12.00        |    0 |          pp2048 |        854.69 ± 1.28 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_2 |  1 | 12.00        |    0 |           tg512 |         66.53 ± 0.01 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |    0 |          pp2048 |        433.03 ± 4.06 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |    f16 |  1 | 12.00        |    0 |           tg512 |         57.13 ± 1.70 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |    0 |          pp2048 |        853.82 ± 5.88 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 | 12.00        |    0 |           tg512 |         65.03 ± 0.02 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |    0 |          pp2048 |        339.48 ± 1.06 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 | 12.00        |    0 |           tg512 |         58.84 ± 0.79 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |    0 |          pp2048 |        701.32 ± 2.21 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq1_1 |  1 | 12.00        |    0 |           tg512 |         64.52 ± 0.03 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |    0 |          pp2048 |        695.83 ± 3.07 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_1 |  1 | 12.00        |    0 |           tg512 |         64.75 ± 0.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |    0 |          pp2048 |       659.11 ± 11.57 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_1 |  1 | 12.00        |    0 |           tg512 |         64.10 ± 0.07 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |    0 |          pp2048 |        616.32 ± 2.47 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_1 |  1 | 12.00        |    0 |           tg512 |         64.00 ± 0.14 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_2 |  1 | 12.00        |    0 |          pp2048 |        437.19 ± 0.78 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq2_2 |  1 | 12.00        |    0 |           tg512 |         57.26 ± 1.20 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_2 |  1 | 12.00        |    0 |          pp2048 |        437.56 ± 2.47 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq3_2 |  1 | 12.00        |    0 |           tg512 |         58.06 ± 0.48 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_2 |  1 | 12.00        |    0 |          pp2048 |        438.90 ± 3.48 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q8_0 | vtq4_2 |  1 | 12.00        |    0 |           tg512 |         57.23 ± 2.13 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |    0 |          pp2048 |        424.42 ± 0.37 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |    f16 |  1 | 12.00        |    0 |           tg512 |         59.29 ± 0.34 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |    0 |          pp2048 |        351.51 ± 2.66 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q8_0 |  1 | 12.00        |    0 |           tg512 |         59.66 ± 0.94 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |    0 |          pp2048 |        855.76 ± 4.55 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 | 12.00        |    0 |           tg512 |         64.93 ± 0.08 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |    0 |          pp2048 |        701.79 ± 1.68 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq1_1 |  1 | 12.00        |    0 |           tg512 |         64.67 ± 0.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |    0 |          pp2048 |        698.88 ± 0.46 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_1 |  1 | 12.00        |    0 |           tg512 |         64.81 ± 0.05 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |    0 |          pp2048 |        650.50 ± 4.44 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_1 |  1 | 12.00        |    0 |           tg512 |         64.11 ± 0.09 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |    0 |          pp2048 |        620.86 ± 2.37 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_1 |  1 | 12.00        |    0 |           tg512 |         64.06 ± 0.19 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_2 |  1 | 12.00        |    0 |          pp2048 |        415.89 ± 0.27 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq2_2 |  1 | 12.00        |    0 |           tg512 |         57.92 ± 0.33 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_2 |  1 | 12.00        |    0 |          pp2048 |        421.57 ± 2.77 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq3_2 |  1 | 12.00        |    0 |           tg512 |         59.47 ± 1.93 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_2 |  1 | 12.00        |    0 |          pp2048 |        422.62 ± 0.35 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |   q4_0 | vtq4_2 |  1 | 12.00        |    0 |           tg512 |         56.52 ± 2.45 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |    0 |          pp2048 |        855.02 ± 8.49 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 | ktq1_1 |    f16 |  1 | 12.00        |    0 |           tg512 |         66.71 ± 0.00 |

## Dual-GPU + expert-offload 20 layers

| model                          |       size |     params | backend    | ngl | type_k | type_v | fa | ts           | ot                    | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | ------------ | --------------------- | ---: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        807.40 ± 1.75 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |    f16 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         48.48 ± 0.57 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        329.25 ± 0.79 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q8_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         52.60 ± 1.83 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        323.30 ± 1.57 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 |   q4_0 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         54.49 ± 0.96 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        701.64 ± 2.90 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq1_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         47.12 ± 0.32 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        703.94 ± 1.38 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         47.93 ± 0.12 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        652.33 ± 0.89 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         46.64 ± 0.25 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        616.26 ± 1.42 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq4_1 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         46.65 ± 0.55 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_2 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        798.41 ± 3.06 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq2_2 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         46.56 ± 0.03 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_2 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |          pp2048 |        801.69 ± 4.52 |
| qwen35moe 35B.A3B IQ2_XXS - 2.0625 bpw |  10.01 GiB |    34.66 B | CUDA       |  99 |    f16 | vtq3_2 |  1 | 12.00        | blk.1[5-9].ffn_.*_exps.=CPU |    0 |           tg512 |         47.42 ± 0.40 |
| qwen35moe 35B.A3B IQ2_XX