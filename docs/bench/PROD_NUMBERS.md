# Live Production Numbers

**Single source of truth for current production TG / quality numbers.**
Update only this file when a phase ships new results. README links here.

**Last updated:** 2026-04-26 (Phase 4 deployed — adaptive split + OMP_active + prefetch)

---

## Hardware

gpu00.node — KVM guest VM
- 12 vCPUs from Ryzen 7 3700X host (Zen 2, 8C/16T host, 2 CCDs × 2 CCXs)
- 40 GB DDR4-3200 (~40 GB/s real)
- 2× RTX 2060 12 GB on asymmetric PCIe (GPU0 x16 / GPU1 x4)
- Linux 6.8 / Ubuntu 24.04, transparent_hugepage=madvise

## Production deploy stack (Phase 4 final)

All deploys use `--cache-type-k ktq2_1 --cache-type-v vtq2_2` (2.78 bpw KV) +
`OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores`.

Adaptive expert-routing splits between short-ctx (≤8192, aggressive) and
long-ctx (>8192, safe-up-to-200k). See
[oidanice-distillery/scripts/deploy/](https://github.com/LL4nc33/oidanice-distillery/tree/main/scripts/deploy).

## TG / quality matrix (current production)

Measured 2026-04-25/26, KV `ktq2_1 + vtq2_2`, OMP_active, prefetch enabled.

| Model | Size | tg128 (short-ctx) | tg128 (long-ctx) | pp512 | PPL (4-chunk) | HellaSwag (200) | Deploy |
|---|---:|---:|---:|---:|---:|---:|---|
| **Qwen3.6-27B dense** | 9.4 GB | **15.26** | n/a | 400 | 7.67 (3-chunk) | 78.0% | `deploy-27b.sh` (single-GPU) |
| **Gemma4-26B-A4B** | 9.0 GB | **80.91** | — | 1331 | reasoning-broken on raw text | TBD | full-GPU |
| **Qwen3.6-35B-A3B** | 10 GB | **75.39** | 75.39 | 1005 | 6.018 | **83.5%** | `deploy-35b.sh` (full-GPU, dual) |
| **Qwen3-Next-80B-A3B** | 25 GB | **~36.5 (+18.5%)** | 32.62 | 415 → 463 | 5.0817 | TBD | `deploy-80b.sh` (adaptive) |
| **Qwen3.5-122B-A10B** | 34 GB | **18.24 (+9.3%)** | 17.43 | 196 | 4.0379 | TBD | `deploy-122b.sh` (adaptive) |

"+" % is vs Phase 3 baseline (vtq2_2 prod-default, no Phase 4 stack).

### Phase 4 win-stack on 80B (ctx ≤ 8192)

| Layer | tg128 | Δ |
|---|---:|---:|
| Phase 3 baseline (vtq2_2 alone) | 30.80 | — |
| + `OMP_WAIT_POLICY=active` | 32.62 | +5.9% |
| + `__builtin_prefetch` in `mul_mat_id` | 32.88 | +0.8% |
| + 18/18/12 layer split | ~36.5 | +11.0% |
| **Total stack** | **~36.5** | **+18.5%** |

### Phase 4 win-stack on 122B (ctx ≤ 8192)

| Layer | tg128 | Δ |
|---|---:|---:|
| Phase 3 baseline | 16.69 | — |
| + OMP + prefetch + 11/12/25 split | 18.24 | +9.3% |

## Quality references (HellaSwag-200, ktq2_1+vtq2_2)

Measured 2026-04-26, llama-perplexity `--hellaswag --hellaswag-tasks 200`.

| Model | HellaSwag |
|---|---:|
| Qwen3.6-27B-IQ2 (dense) | 78.0% [71.8%, 83.2%] |
| Qwen3.6-35B-A3B-IQ2 (MoE) | **83.5%** [77.7%, 88.0%] |
| 80B / 122B / 26B | TBD |

Reference points (HellaSwag-validation full):
- LLaMA-7B Q4 ≈ 76-78%
- LLaMA-2-13B Q4 ≈ 80-82%
- Mistral-7B Q4 ≈ 82%
- Llama-3-8B Q4 ≈ 80%

## What changed

- **Phase 4 (2026-04-26):** OMP_active env-default + `__builtin_prefetch` in `mul_mat_id` + adaptive layer-split (18/18/12 on 80B, 11/12/25 on 122B for ctx ≤ 8192). Net +18.5% TG on 80B, +9.3% on 122B.
- **Qwen3.6-27B added:** new dense option, single-GPU pfad (`deploy-27b.sh`), 5.1× faster than old Qwen3.5-27B-Q4-partial-offload (3.00 → 15.26 t/s).

## How to update

When a new phase ships new numbers:
1. Update this file's matrix
2. Append a "What changed" entry
3. README still links here — no README rewrite needed
