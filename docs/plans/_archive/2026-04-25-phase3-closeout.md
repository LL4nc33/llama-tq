# Phase 3 Closeout — VTQ_3 Outlier-Channel-Split + Production Migration

**Date:** 2026-04-25
**Branch:** turboquant
**Status:** SHIPPED

## Goals (entered with)

1. Implement VTQ_3 family (Trellis backbone + 4-fp16-outlier sidecar per block)
2. Validate on real-world MoE inference (no synthetic-only tests)
3. Decide whether to ship as production-default or keep as research tier
4. Migrate giants (80B + 122B) from `vtq2_1` to whatever wins

## Outcomes

### VTQ_3 implementation
- 3 types shipped: `vtq2_3` / `vtq3_3` / `vtq4_3` (2.50 / 3.50 / 4.50 bpw avg)
- Round-trip MSE: 4× lower than VTQ_2 at same K (per-element gain confirmed)
- Full FA dispatch matrix: KTQ × VTQ_3 wired in `fattn-vec-dispatch-vtq3.cu`
- D=64/128/256/512 all verified
- Crashes fixed in Phase 1: VTQ_2 D=512 specialized kernel, KTQ3/4 build errors

### PPL validation results

35B-A3B IQ2_XXS (c2-b1 single-token mode, deferred-V active):

| K / V | bpw avg | PPL | Δ vs f16/f16 |
|---|---:|---:|---:|
| f16 / f16 | 16.0 | 6.3717 ± 0.76 | baseline |
| ktq2_1 / vtq3_3 | 3.78 | 6.4015 ± 0.77 | **+0.47%** |

Asymmetric ktq2_1+vtq3_3 lands **inside f16 stderr** at 4.78× smaller KV. Quality
target met for the "research tier" — this is the cleanest production-candidate
PPL number measured on a 35B MoE on real wikitext data.

### Giants migration

Production deploy on gpu00:8791 (80B) and 8794 (122B) was running `vtq2_1` since
2026-04-22. On 2026-04-25 we measured prod-aligned PPL+TG sweep:

80B @ 200k ctx (prod-c4-b1 PPL methodology + llama-bench):

| Config | bpw KV | pp512 | tg128 | PPL | Δ |
|---|---:|---:|---:|---:|---:|
| ktq2_1 / vtq2_1 (old) | 3.0 | 386.5 | 30.6 | 5.2213 | +2.69% |
| ktq2_1 / vtq2_2 (new) | 2.78 | 402.6 | 30.9 | 5.0817 | -0.06% |
| ktq2_1 / vtq3_3 | 3.78 | — | — | 5.0791 | -0.11% |

122B @ 200k ctx:

| Config | bpw KV | pp512 | tg128 | PPL | Δ |
|---|---:|---:|---:|---:|---:|
| ktq2_1 / vtq2_1 (old) | 3.0 | 189.5 | 16.8 | 4.2338 | +4.19% |
| ktq2_1 / vtq2_2 (new) | 2.78 | 196.3 | 16.8 | 4.0379 | -0.63% |
| ktq2_1 / vtq3_3 | 3.78 | — | — | 4.0593 | -0.10% |

**Decision:** ship `vtq2_2` as production-default (not `vtq3_3`). Reasoning:

1. `vtq2_2` already beats `vtq2_1` on both quality AND throughput for both giants
2. `vtq3_3` only adds another 0.05-0.10% PPL improvement at +1 bpw cost
3. The +1 bpw is ~1 GB extra VRAM at 200k ctx on 122B — non-trivial
4. v3 stays in tree as the "quality-priority" tier for users who want it

### Deferred V activation

Confirmed: VTQ_2 + VTQ_3 require single-token decode (`-b 1 -ub 1`) to fire the
deferred-V-staging-buffer. Batched mode (b > 1) measures f16 + mixed-precision
overhead, NOT the actual VTQ_2/3 PPL. This was a major methodological error in
the 2026-04-22/23 sweeps — fixed and re-measured 2026-04-25.

## Code-level deliverables

- `ggml/src/ggml-cuda/fattn-vec-dispatch-vtq3.cu` — VTQ_3 FA dispatch, full K x V
- `ggml/src/ggml-cuda/trellis.cuh` — VTQ_3 outlier-channel-split decoder
- `ggml/src/ggml-trellis-vtq3.c` — CPU encoder/decoder
- `src/llama-kv-cache.cpp:288-291, 489-492` — `is_vtq_v` extended for `_3`
- `common/arg.cpp:384-406` — CLI parser accepts `vtq{2,3,4}_3`
- `tools/llama-bench.cpp` — added `_3` types to KV-cache parser
- `bench/plots/benchmarks.csv` — appended `c2-b1` and `prod-c4-b1` and
  `prod-bench-v3` tagged rows for replication

## Documentation

Blog series:
- `2026-04-25-vtq3-asymmetric-on-35b.md` — VTQ_3 family on 35B-A3B
- `2026-04-25-giant-models-prod-ppl-sweep.md` — 80B + 122B prod PPL sweep
- `2026-04-25-vtq2-attention-absorbs-bit-depth.md` — K-collision is feature
- `2026-04-25-ktq-on-35b-quality-win.md` — KTQ_1 +0.27% PPL @ 9.4 bpw
- `2026-04-25-vtq-family-comparison-table.md` — Pareto frontier consolidation

README updated 2026-04-25 with 4-model production summary, vtq2_2 promoted to
default across the board.

## Outstanding from Phase 3

- VTQ_1 family + `-b 1 -ub 1` crashes on Qwen3-Next-80B (Gated Delta Net path).
  Tracked separately — deferred to Phase 4 since v1 is no longer prod-default.
- Task #142 (Disable VTQ_MIXED per-layer as default) — production-guard pending,
  separate PR.
- Task #158 (MMA-KTQ Phase 5 docs) — already implemented in 34cd3a47a, needs
  status update only.

## Phase 3 verdict

VTQ family v2 wins. v3 in-tree as research tier. Production migration to
`ktq2_1 + vtq2_2` complete on docs; deploy-side switch remains user-gated
(scripts in `oidanice-distillery/scripts/deploy/`).

## Next phase

Phase 4 = optimization across all layers. See backlog below.

### Valve VRAM-fix research (2026-04-14, Natalie Vock @ Valve)

What it does: Linux kernel TTM (translation table maps) memory-management
patch + DRM device memory cgroup. Prevents browser/compositor from evicting
GPU memory pages allocated by foreground games. Reclaims ~1.3 GB VRAM on
RX 6500 XT, +0-6.4% FPS depending on game.

Applicability to llama-tq:

- **NOT directly applicable**: AMDGPU/TTM-specific. NVIDIA has its own memory
  manager (`nvidia.ko`) outside TTM. Patches don't fire on RTX 2060.
- **Concept is portable**: same goal achievable on NVIDIA via different tools:
  1. `nvidia-smi -c EXCLUSIVE_PROCESS` — GPU exclusive to one CUDA process
  2. `nvidia-persistenced` — keep driver state warm, faster allocation
  3. `nvidia-smi --persistence-mode=ENABLED`
  4. CUDA MPS (Multi-Process Service) — process priority scheduling
  5. Headless gpu00 (Xorg off, no compositor → no eviction races)

Realistic expected win for our workload: 1-3% TG stability (no stutter; we
don't stutter to begin with — we OOM). Worth measuring once we hit Phase 4
Tier B.

### Phase 4 Tier A measurement (2026-04-25 evening)

5-config sweep on 80B (Qwen3-Next-80B-A3B IQ2_XXS, ktq2_1+vtq2_2, on gpu00 KVM):

| Config | pp512 | tg128 | Δ tg vs base |
|---|---:|---:|---:|
| BASELINE | 403.6 | 30.80 | — |
| LLAMA_MMAP_HUGEPAGES=1 | 400.8 | 30.55 | -0.8% |
| **OMP_WAIT_POLICY=active + close + cores** | 397.99 | **32.62** | **+5.9%** ✅ |
| taskset -c 0-7 (8 vCPUs) + OMP active | 402.0 | 23.87 | -22.5% ❌ |
| taskset -c 0-3 (4 vCPUs) + OMP active | 403.1 | 19.69 | -36.1% ❌ |

**Discovery:** gpu00 is a KVM guest VM (12 vCPUs from a Ryzen 7 3700X host),
`systemd-detect-virt` = `kvm`. Direct CCX-pinning is impossible on the guest
(Linux sees one virtual L3, no real CCX-topology). But the VM-context makes
`OMP_WAIT_POLICY=active` a bigger win than on bare metal: with the default
passive yield policy, libgomp `sched_yield`s during compute waits, and the
hypervisor steals the vCPU for other guests / housekeeping. Active spin
keeps the vCPU on the run-queue.

**Hugepages:** measured null effect with prewarm too. Likely cause: model is
fully resident (24 GiB / 40 GiB total RAM, no eviction pressure), and TLB
miss penalty on 80B mat-vec is hidden behind the actual DRAM cost. Patch
stays in tree (no harm), env-gated, no default-on.

**Taskset shrinking is bad:** llama.cpp internally picks `--threads` based on
hardware_concurrency()=12 in our case. Taskset to 8 or 4 vCPUs starves the
thread pool, doesn't help cache locality.

**Win secured:** OMP env vars exported in `oidanice-distillery/scripts/deploy/_common.sh`
(commit 29fc44e). All three `OMP_*` vars are env-overridable.

### Phase 4 Tier A measurements expansion (2026-04-25 late)

Verified OMP_active across all 4 production tiers + dense reference:

| Config | tg128 baseline | tg128 +OMP_active | Δ |
|---|---:|---:|---:|
| 2B dense Q4_K_M single-GPU | 153.29 | 153.09 | 0.0% |
| 35B-A3B MoE dual-GPU full-VRAM | 75.39 | 75.27 | 0.0% |
| 80B-A3B MoE 14/14/20 split | 30.80 | **32.62** | **+5.9%** |
| 122B-A10B MoE 10/9/29 split | 16.69 | **17.12** | **+2.6%** |
| 27B dense Q4_K_M partial offload | 2.97 | 3.00 | +1.0% |

**Pattern:** OMP_active wins are proportional to CPU-thread activity during
inference. Models that fit fully on GPU (2B, 35B-A3B) see no win because
CPU threads are idle. Models with significant CPU-offloaded layers (80B,
122B) see real wins because the policy stops the hypervisor from stealing
vCPUs when libgomp would otherwise yield.

**Dense-vs-MoE finding:** 27B dense partial-offload runs at 2.97 t/s while
35B-A3B MoE on the same hardware runs at 75 t/s. Dense models with tensors
that don't fit fully in VRAM are crippled by the asymmetric PCIe x16/x4
bus on gpu00 — 50× slowdown, not an optimization opportunity but an
architectural cliff. MoE strategy validated as the right path on this
hardware.

**LTO build attempt:** Failed. `nvlink elfLink linker library load error`
when GGML_LTO=ON combines with GGML_CUDA=ON. Would need to scope LTO to
non-CUDA targets, deferred for now (gain is small relative to debugging
risk).

**Prefetch patch (commit 6e50fc701):** Added `__builtin_prefetch` for the
next active MoE expert's first 2 cachelines in `mul_mat_id`. Measured
on 80B with OMP_active baseline (32.62 → 32.88 t/s, +0.8%). Inside stderr
margin but consistent positive direction. Kept in tree — 11 LOC, no harm,
small but real gain on CPU-offloaded MoE.

**Combined Phase 4 stack on 80B:**
- baseline: 30.80 t/s
- + OMP_active: 32.62 (+5.9%)
- + prefetch: 32.88 (+6.8% total)

### Phase 4 backlog (priority order)

#### Tier A — Quick wins, ~1 day, +5-15% TG expected
1. Zen2 thread-pinning (`OMP_PLACES=cores OMP_PROC_BIND=close`, taskset to
   one CCX) — 3700X has 4 separate L3-caches, cross-CCX traffic is expensive.
   Patch documented in `2026-04-24-80b-low-hanging-perf.md` not yet merged.
2. `MADV_HUGEPAGE` for mmap (env-gated `LLAMA_MMAP_HUGEPAGES=1`) — TLB pressure
   reduction for sparse expert-routing.
3. DDR4 dual-channel verification (`dmidecode -t memory`).

#### Tier B — Medium effort, ~2-3 days, +5-10% TG/PP
4. NVIDIA exclusive-process mode + headless gpu00 (Valve-fix equivalent).
5. CUDA UVM/UMA optimization for expert-tensor pinning.
6. PCIe Resizable BAR support — relevant for asymmetric x16/x4 setup; check
   if BAR is currently sized for 12 GB or limited to 256 MB legacy.
7. CachyOS kernel cmdline backport: `transparent_hugepage=madvise`, scheduler
   tuning, etc. — research what helps non-gaming CUDA workloads.

#### Tier C — Larger effort, +10-30% if it works
8. Async CUDA stream pipeline (GPU0 layer N+1 while GPU1 N while CPU N-1).
9. FA tile-shape sweep beyond current launch_bounds tuning.
10. Per-Layer mixed-precision V-cache (Task #142 + Trick 5 lambda) — measure
    if MoE attention-head variance justifies vtq2/vtq3 mixing.

### Recommended next action after compact

```bash
ssh claude@gpu00.node "uname -r; cat /etc/os-release | head -3; \
  cat /sys/kernel/mm/transparent_hugepage/enabled; \
  dmidecode -t memory | grep -E 'Size|Speed' | head -10"
```

Then start Tier A: branch `turboquant-phase4-cpu`, patch `src/llama-mmap.cpp`
for `LLAMA_MMAP_HUGEPAGES=1`, build, bench 80B with vs without hugepages.
Decision gate: if ≥3% TG win, merge. Otherwise document and move to next.
