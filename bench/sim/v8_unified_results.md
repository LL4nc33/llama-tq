# TurboQuant v8 Unified Sim — Results

Generated: 2026-05-02 05:06:07
Seed=42, head_dim=128, n_heads=32, n_tokens=2048
K-blocks tested: 2048 × 32 = 65536 samples
V-blocks tested (32): 2048 × 32 = 65536 samples
V-blocks tested (128): 256 × 128 = 32768 samples

## Round-trip MSE

| Type | Role | bpw | MSE | rel_MSE | PPL-Δ-Estimate (%) |
|------|:----:|:---:|----:|--------:|-------------------:|
| `ktq1_1` | K | 2.50 | 3.935884e-01 | 0.3914 | +23.48 |
| `ktq2_1` | K | 3.50 | 1.124465e-01 | 0.1118 | +6.71 |
| `ktq3_1` | K | 4.50 | 3.086675e-02 | 0.0307 | +1.84 |
| `ktq4_1` | K | 5.50 | 8.455500e-03 | 0.0084 | +0.50 |
| `ktq1_v8` | K | 5.50 | 1.799510e-01 | 0.1789 | +10.74 |
| `ktq2_v8` | K | 6.50 | 5.028850e-02 | 0.0500 | +3.00 |
| `ktq3_v8` | K | 7.50 | 1.392488e-02 | 0.0138 | +0.83 |
| `ktq4_v8` | K | 8.50 | 3.745357e-03 | 0.0037 | +0.22 |
| `vtq1_1` | V | 1.50 | 4.646009e-01 | 0.6393 | +38.36 |
| `vtq2_2` | V | 2.25 | 8.323756e-02 | 0.1181 | +7.08 |
| `vtq2_1` | V | 2.50 | 1.600047e-01 | 0.2202 | +13.21 |
| `vtq2_v8` | V | 2.62 | 2.864615e-02 | 0.0406 | +2.44 |
| `vtq2_3` | V | 3.00 | 2.422664e-02 | 0.0344 | +2.06 |
| `vtq3_2` | V | 3.25 | 4.669029e-02 | 0.0662 | +3.97 |
| `vtq3_v8` | V | 3.62 | 7.771631e-03 | 0.0110 | +0.66 |
| `vtq3_1` | V | 4.00 | 7.299177e-02 | 0.1004 | +6.03 |
| `vtq3_3` | V | 4.00 | 6.210119e-03 | 0.0088 | +0.53 |
| `vtq4_2` | V | 4.25 | 3.401887e-02 | 0.0482 | +2.89 |
| `vtq4_1` | V | 4.50 | 2.590437e-02 | 0.0356 | +2.14 |
| `vtq4_v8` | V | 4.62 | 5.217358e-03 | 0.0074 | +0.44 |
| `vtq4_3` | V | 5.00 | 1.489143e-03 | 0.0021 | +0.13 |

## Per-bpw-class winners

| bpw | Best existing | Best MSE | v8 candidate | v8 MSE | v8 wins (within 5%)? |
|:---:|---------------|---------:|--------------|-------:|:--------------------:|
| 2.50 | `vtq2_1` | 1.600047e-01 | `vtq2_v8` | 2.864615e-02 | YES |
| 3.50 | `ktq2_1` | 1.124465e-01 | `vtq3_v8` | 7.771631e-03 | YES |
| 4.50 | `vtq4_1` | 2.590437e-02 | `vtq4_v8` | 5.217358e-03 | YES |
| 5.50 | `ktq4_1` | 8.455500e-03 | `ktq1_v8` | 1.799510e-01 | NO |

## Gate decision

**HYBRID** — 3/4 v8 candidates win; 1 lose.

Losing candidates:
- `ktq1_v8` (5.50 bpw) loses to `ktq4_1` by +2028.21% MSE — keep existing for that tier.

## Caveats

- Trellis path uses beam-pruned Viterbi (beam=1024 instead of full 2^16). MSE is slightly pessimistic vs the full Viterbi CUDA path. SAFE bias for GO/NO-GO gate.
- Synthetic V is Gauss + Laplace + 1% 5σ-outlier mix. Real V from a trained transformer may have heavier tails — re-run with real tensors before CUDA commit.
- Calibration `PPL_drift_pct ≈ rel_mse * 0.6` is from `ktq2_1+vtq2_2` row in `bench/plots/benchmarks.csv` (PPL +0.16%).
