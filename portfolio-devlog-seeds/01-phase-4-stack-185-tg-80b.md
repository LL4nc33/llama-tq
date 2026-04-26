---
slug: "phase-4-stack-185-tg-80b"
title: "Phase 4 Stack — +18.5% TG auf 80B"
date: "2026-04-26"
status: "shipped"
tags: ["turboquant", "phase-4", "kv-cache", "moe", "deploy"]
hero_number: "+18.5% TG"
hardware: "2× RTX 2060 12GB, KVM-VM"
---

# Phase 4 Stack — +18.5% TG auf Qwen3-Next-80B-A3B

Phase 4 war der Hardcore-CPU/RAM-Bandwidth-Sprint. Drei orthogonale Wins gestackt, alles live deployed.

## Hardware Context

gpu00.node — KVM Guest VM:
- 12 vCPUs vom Ryzen 7 3700X Host (Zen 2, 8C/16T, 2 CCDs × 2 CCXs)
- 40 GB DDR4-3200 (~40 GB/s real bandwidth)
- 2× RTX 2060 12 GB auf asymmetrischem PCIe (GPU0 x16 / GPU1 x4)
- Ubuntu 24.04, Linux 6.8, transparent_hugepage=madvise

Das ist Consumer-Hardware mit Hypervisor-Overhead obendrauf. Phase 4 sollte zeigen wie viel TG aus der bandwidth-limited Inference rausgepresst werden kann ohne Quality zu verlieren.

## Phase 4 Win-Stack auf 80B (ctx ≤ 8192)

| Layer | tg128 (t/s) | Δ |
|---|---:|---:|
| Phase 3 baseline (vtq2_2 alone) | 30.80 | — |
| + `OMP_WAIT_POLICY=active` | 32.62 | +5.9% |
| + `__builtin_prefetch` in `mul_mat_id` | 32.88 | +0.8% |
| + 18/18/12 adaptive layer split | ~36.5 | +11.0% |
| **Cumulative** | **~36.5** | **+18.5%** |

## Win 1 — `OMP_WAIT_POLICY=active` (+5.9%)

**Mechanismus:** Auf Bare-Metal verbringen idle OpenMP-Threads ihre Zeit in `pthread_cond_wait`. Auf einer KVM Guest VM steal'd der Hypervisor diese Threads als "idle" weg und gibt vCPU-Zeit anderen Workloads. Wenn der nächste OMP-Sync kommt, müssen alle Threads erstmal scheduled werden — kostet ms.

**Fix:** `OMP_WAIT_POLICY=active` sagt OpenMP "spin while waiting, don't yield". Auf Bare-Metal hilft das wenig (5-10W Strom mehr). Auf einem KVM Guest hilft es **direkt**, weil der Hypervisor die spinning Threads nicht steal'd.

**Implementation:** Drei Zeilen im Deploy-Script:
```bash
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-active}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
```

**Win:** +5.9% auf 80B-MoE TG. Bei full-GPU dense Models = 0% (CPU threads idle weil alles auf GPU). Bei MoE: einige experts sind CPU-offloaded, dort schlägt der Hypervisor zu.

## Win 2 — `__builtin_prefetch` in mul_mat_id (+0.8%)

**Mechanismus:** Im MoE forward pass dispatched `mul_mat_id` jeden Token an den Top-K seiner Experts. Adjacent tokens in der gleichen sequence treffen oft die gleichen Experts (router-correlation). Aber zwischen den expert-loads gibt es einen cache-miss für das nächste expert-block.

**Fix:** Pre-fetch das nächste aktive Expert ein layer voraus:
```c
for (int next_a = cur_a + 1; next_a < n_as; ++next_a) {
    if (matrix_row_counts[next_a] > 0) {
        __builtin_prefetch((const char *) src0->data + next_a * nb02, 0, 0);
        __builtin_prefetch((const char *) src0->data + next_a * nb02 + 64, 0, 0);
        break;
    }
}
```

**Win:** +0.8% on 80B-MoE. Niedriger als erhofft weil nur RAM→cache, nicht PCIe→RAM. Aber free.

## Win 3 — Adaptive Layer Split 18/18/12 (+11%)

**Mechanismus:** 80B-MoE hat 48 layers. Default deploy splittet 16/16/16 (GPU0/GPU1/CPU). Aber GPU1 hat nur PCIe x4 — alle activations müssen über die schmale Pipe. **Wenn man weniger layers auf GPU1 packt und die schwachen layers (mehr Routing-overhead) lieber auf CPU offloadet, geht's schneller.**

**Fix:** Per-context-length adaptiver split. Bei ctx≤8192 aggressiv: 18 GPU0 + 18 GPU1 + 12 CPU. Bei ctx>8192: zurück zu safe 16/16/16 weil KV-cache mehr Platz braucht.

**Implementation:** Deploy-Script branching:
```bash
if [[ "$CTX" -le 8192 ]]; then
  EXPERT_REGEX='blk\.(0|...|17)\.ffn_(up|down|gate)_exps\.=CUDA0,blk\.(18|...|35)\.ffn_(up|down|gate)_exps\.=CUDA1,blk\.(36|...|47)\.ffn_(up|down|gate)_exps\.=CPU'
else
  EXPERT_REGEX='blk\.(0|...|13)\.ffn_(up|down|gate)_exps\.=CUDA0,blk\.(14|...|27)\.ffn_(up|down|gate)_exps\.=CUDA1,blk\.(28|...|47)\.ffn_(up|down|gate)_exps\.=CPU'
fi
```

**Win:** +11% on 80B short-ctx. **Sweet-spot war 18/18/12; 20/20/8 gab +17.3% aber OOM bei ctx>2048.**

## Was Phase 4 NICHT erreicht hat

- LTO-Build mit CUDA: `nvlink elfLink linker library load error`. Aufgegeben.
- Tier-B (TU-bloat in fattn-common.cuh): Erst in Phase-5-Investigation gefunden. -14% pp512 regression auf MoE prefill weil TQ-includes register-pressure für pure-f16 kernels erhöhen. Fix in flight (extract `fattn-tq.cuh`).
- Bare-metal CCX-pinning: gpu00 ist KVM Guest, host-cpuset würde nochmal 5-10% bringen aber separate ops-arbeit.

## Stack-Diagramm (was Phase 4 stacks)

```
Decode Budget (ctx≤8192, 80B-A3B):
    27.7 ms/token  ← baseline
       ↓
    Reduce by 5.9% → OMP active (no hypervisor steal)
    Reduce by 0.8% → prefetch next expert
    Reduce by 11.0% → 18/18/12 split (less PCIe-x4 traffic)
       ↓
    23.4 ms/token  → ~36.5 t/s (+18.5%)
```

## Deploy-Default

`oidanice-distillery/scripts/deploy/deploy-80b.sh` shippt Phase-4-Stack default-on. Existing live deploy auf gpu00:8791 nutzt das.

## Lessons

1. **KVM-Hypervisor-Overhead ist quantifizierbar und bekämpfbar.** OMP_active war der größte single-flag Win in der Geschichte des Forks.
2. **PCIe x4 Bottleneck ist real und steuerbar via Layer-Split.** Wer asymmetrische PCIe hat, sollte adaptive splits.
3. **Phase 4 war der letzte Tier-A/B Win.** Phase 5 muss tieferer Architecture-level work werden (XQuant, RULER harness, FA TU-bloat fix).

---

## Sources

- llama-tq Phase 4 commits: `155557cc0`, `6e50fc701` (turboquant branch)
- oidanice-distillery deploy scripts: `29fc44e`, `11b3543`, `2343389`
- LIVE_NUMBERS.md: [docs/bench/LIVE_NUMBERS.md](https://github.com/LL4nc33/llama-tq/blob/master/docs/bench/LIVE_NUMBERS.md)
