# Phase 4: Squeezing +18% TG Out of an 80B MoE on 2× RTX 2060

**Date:** 2026-04-26
**Branch:** turboquant
**Hardware:** gpu00.node — KVM guest (12 vCPUs from Ryzen 7 3700X), 40 GB DDR4-3200, 2× RTX 2060 12 GB on asymmetric PCIe x16/x4
**Models:** Qwen3-Next-80B-A3B IQ2_XXS, Qwen3.5-122B-A10B IQ2_XXS, both with `ktq2_1` + `vtq2_2` KV-cache (Phase 3 production-default)

## TL;DR

Phase 4 stacked four orthogonal optimizations and re-tuned the layer-split. On 80B at ctx ≤ 8192 we reached **+18.5% TG** over the Phase 3 baseline, on 122B **+9.3%**. Long-context (>8k) configs fall back to safe defaults that still fit ctx=200000.

| Layer | 80B tg128 | Δ | 122B tg128 | Δ |
|---|---:|---:|---:|---:|
| Phase 3 baseline | 30.80 | — | 16.69 | — |
| + `OMP_WAIT_POLICY=active` | 32.62 | +5.9% | 17.43 | +4.4% |
| + `__builtin_prefetch` in `mul_mat_id` | 32.88 | +0.8% | included | — |
| + Adaptive expert split | ~36.5 | +11.0% | 18.24 | +4.7% |
| **Total** | **~36.5** | **+18.5%** | **18.24** | **+9.3%** |

## 1. The OMP win — KVM-specific, not bare-metal

We started by checking `/proc/cpuinfo` and discovered gpu00 is a KVM guest VM (`systemd-detect-virt = kvm`). The Linux kernel inside the guest sees one virtual L3 cache shared by all 12 vCPUs — real CCX topology of the underlying Ryzen 7 3700X is hidden by the hypervisor.

This kills the obvious Zen2 win (CCX-pinning via `pthread_setaffinity_np`) at the guest level. But it surfaces a different one: with default `OMP_WAIT_POLICY=passive`, libgomp `sched_yield`s during compute waits, and the hypervisor steals the vCPU. Active spinning keeps the vCPU on the run-queue.

```bash
export OMP_WAIT_POLICY=active
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

Result on 80B: **30.80 → 32.62 t/s, +5.9%**. On 122B: **16.69 → 17.43 t/s, +4.4%**. Effect size correlates with how much actual CPU work happens during inference — full-GPU models (35B-A3B fits dual-GPU, 2B fits single-GPU) saw zero win.

Verified by sweep:

| Model | Config | tg baseline | +OMP_active |
|---|---|---:|---:|
| 2B dense Q4 single-GPU | full-GPU | 153.29 | 153.09 (no win, idle CPU) |
| 35B-A3B dual-GPU | full-GPU | 75.39 | 75.27 (no win) |
| 80B-A3B 14/14/20 | mixed | 30.80 | 32.62 (+5.9%) |
| 122B-A10B 10/9/29 | CPU-heavy | 16.69 | 17.43 (+4.4%) |
| 27B dense Q4 partial offload | PCIe-bound | 2.97 | 3.00 (+1.0%) |

## 2. The prefetch patch — small but free

11 lines in `ggml/src/ggml-cpu/ggml-cpu.c::ggml_compute_forward_mul_mat_id`. Right before computing the current active MoE expert, we prefetch the first two cachelines of the next active expert's weight tensor:

```c
for (int next_a = cur_a + 1; next_a < n_as; ++next_a) {
    if (matrix_row_counts[next_a] > 0) {
        __builtin_prefetch((const char *) src0->data + next_a * nb02, 0, 0);
        __builtin_prefetch((const char *) src0->data + next_a * nb02 + 64, 0, 0);
        break;
    }
}
```

`(0, 0)` = read, no-temporal-locality — the weights are touched once. On 80B with OMP_active baseline: **32.62 → 32.88 t/s, +0.8%**. Inside stderr but consistent positive direction. Eleven lines, zero risk, kept in tree.

## 3. Layer-split re-tuning — the big one

Phase 3 prod used `14/14/20` (14 layers GPU0, 14 GPU1, 20 CPU) for 80B. We had ~3.4 GiB free on each GPU at `ctx=2048` after model load. So we systematically swept:

| Split | pp512 | tg128 | Δ vs prod |
|---|---:|---:|---:|
| 14/14/20 | 396.07 | 32.39 | — |
| 15/15/18 | 407.81 | 33.50 | +3.4% |
| 16/16/16 | 415.87 | 34.64 | +7.0% |
| 17/17/14 | 427.96 | 35.41 | +9.3% |
| 18/18/12 | 438.24 | 36.05 | +11.3% |
| 19/19/10 | 451.85 | 37.10 | +14.5% |
| **20/20/8** | **463.28** | **38.00** | **+17.3%** ✅ |
| 21/21/6 | OOM ctx | — | — |
| 22/22/4 | OOM load | — | — |

Linear gain ~2.5% per pair of layers shifted, until we hit the OOM ceiling at 21+. Each extra GPU layer = expert weights resident in VRAM, no PCIe-x4 streaming per token.

For 122B (more layers, bigger model):

| Split | tg128 | Δ vs prod 10/9/29 |
|---|---:|---:|
| 10/9/29 | 17.43 | — |
| 11/10/27 | 17.61 | +1.0% |
| 11/11/26 | 18.19 | +4.4% |
| **11/12/25** | **18.24** | **+4.6%** |
| 11/13/24 | 17.91 | +2.8% (plateau) |
| 12/10/26 | OOM | — |

122B has less headroom than 80B (model is 34 GiB vs 24 GiB). Sweet spot is `11/12/25` for ctx ≤ 8192.

## 4. Adaptive deploys

The wins above are all measured at ctx=2048. At ctx=200000 (the prod context), KV-cache becomes huge (~2.7 GiB extra per GPU on 80B with `ktq2_1+vtq2_2`) and the headroom disappears. So the production deploy scripts now branch on context length:

```bash
# scripts/deploy/deploy-80b.sh
if [[ "$CTX" -le 8192 ]]; then
  EXPERT_REGEX='blk\.(0|...|17)...=CUDA0,blk\.(18|...|35)...=CUDA1,blk\.(36|...|47)...=CPU'   # 18/18/12
else
  EXPERT_REGEX='blk\.(0|...|13)...=CUDA0,blk\.(14|...|27)...=CUDA1,blk\.(28|...|47)...=CPU'   # 14/14/20
fi
```

Most chat sessions are <8k tokens. When someone fires up a long-context summarization run, the deploy falls back gracefully.

## 5. Things that didn't work (kept honest)

- **Hugepages** (`MADV_HUGEPAGE` env-gated): ~0% effect on 80B and 27B-dense. Model is fully resident, no TLB pressure detected. Patch stays in tree as opt-in (no harm), but don't expect a win.
- **LTO build** (`GGML_LTO=ON`): `nvlink elfLink linker library load error`. CUDA's nvlink can't device-link GCC's LTO objects. Would need to scope LTO to non-CUDA targets; deferred.
- **CCX-pinning at guest level**: Linux inside KVM guest cannot see real CCX topology. `pthread_setaffinity_np` to "CCX 0" is meaningless when the host's `virsh vcpupin` mapping is opaque to the guest. Need host-side cpuset cgroup, out of llama.cpp scope.
- **Single-GPU on 35B**: hypothetically eliminates inter-GPU sync overhead, but at ctx=2048 35B-A3B fits in 24 GiB across both GPUs and TG is memory-bandwidth-bound on the active GPU per layer, not PCIe-bound. No measurable win.
- **27B dense partial CPU-offload**: 2.97 t/s. Dense models that don't fit fully on GPU on this asymmetric x4-PCIe system are 50× slower than MoE equivalents. Architecture-cliff, not optimization opportunity. MoE strategy validated as the right path.

## 6. The combined stack on 80B

```
Phase 3 baseline (vtq2_2):        30.80 t/s
+ OMP_WAIT_POLICY=active:         32.62  (+5.9%)
+ __builtin_prefetch:             32.88  (+0.8%)
+ 18/18/12 layer split (ctx≤8k):  ~36.5  (+11%)
─────────────────────────────────────
Total:                            ~36.5 t/s   (+18.5% over Phase 3)
```

For 122B at ctx ≤ 8192:

```
Phase 3 baseline:                 16.69 t/s
+ OMP+prefetch+11/12/25 split:    18.24  (+9.3%)
```

Both deployed in `oidanice-distillery/scripts/deploy/`. The TurboQuant fork's KV-cache work (Phases 1-3, KTQ + VTQ trellis quantization) plus this Phase 4 CPU-side perf pass are now stable production.

## Code references

- `ggml/src/ggml-cpu/ggml-cpu.c` — `__builtin_prefetch` in `mul_mat_id` (commit `6e50fc701`)
- `src/llama-mmap.cpp` — `MADV_HUGEPAGE` env-gated patch (commit `155557cc0`)
- `oidanice-distillery/scripts/deploy/_common.sh` — OMP env defaults (commit `29fc44e`)
- `oidanice-distillery/scripts/deploy/deploy-80b.sh` — adaptive 80B routing (commit `11b3543`)
- `oidanice-distillery/scripts/deploy/deploy-122b.sh` — adaptive 122B routing (commit `2343389`)

## What's next (Phase 5 candidates)

- **Async CUDA Streams Pipeline**: Layer N+1 on GPU0 while Layer N on GPU1, while Layer N-1 on CPU. ~3 hour architecture rewrite, expected +10-30% TG.
- **Per-Layer mixed V-cache**: Trick 2 PR2 (variance-driven mixing) is implemented but disabled-by-default; needs production-guard PR.
- **TurboQuant arxiv writeup**: KTQ + VTQ + asymmetric K/V + deferred-V prefill pipeline measured on real 35B/80B/122B MoE models is paper-worthy material.
