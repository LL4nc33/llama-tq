# Phase 5 — Pie / Turnip Port Spec

**Status:** spec, awaiting approval
**Date:** 2026-04-26
**Author:** Claude (auto)

## Goal

Architektur-Level TG-wins jenseits des Tier-A/B-Plateaus (+18.5% on 80B
already shipped). Ziel: ≥+30% TG auf 80B/122B vor merge, sonst kein ship.

## Inputs

- Pie (arxiv 2411.09317v1) — performance-transparent KV-cache swapping, FIFO
  queue, vLLM-based, **NVLink-assumption (419 GB/s CPU→GPU)**.
  Speedups: 1.9× vs vLLM throughput, 1.67× less GPU memory.
- Turnip (arxiv 2405.16283v3) — nondeterministic GPU runtime over memgraph,
  statische taskgraphs only, **kein token-by-token decode**.
  Speedups: 2.2× LoRA training vs ZeRO; inference nur first-token (prefill).

## Reality-Check test-box

| Pie/Turnip Annahme | test-box Realität | Verdict |
|---|---|---|
| NVLink CPU↔GPU (419 GB/s) | PCIe-x16 (GPU0) + PCIe-x4 (GPU1), DDR4-3200 (~40 GB/s real) | **10×–40× weniger Bandbreite** |
| GH200 96 GB HBM | 2× RTX 2060 12 GB | OK aber tight |
| Statische taskgraph | llama.cpp = dynamic, token-by-token decode | **Turnip nicht direkt portable** |
| KV-cache fits CPU RAM | 40 GB DDR4 — bei 200K ctx auf 80B knapp | borderline |
| vLLM page-block KV | llama.cpp unified KV-cache (no paging) | **Bigger refactor needed** |

## What's portable

### From Pie (high feasibility on test-box)

1. **FIFO swap-queue für CPU-offloaded experts** — wir offloaden ja bereits
   ffn_*_exps Layer auf CPU. Pie's FIFO-Logik (swap-in next-active-expert,
   swap-out coldest) ist agnostisch zur Bandwidth — funktioniert auch auf
   PCIe-x4, nur eben mit höherer queuing-latency.

2. **Mapping table mit timestamps** — light-weight, model-agnostisch, ergänzt
   unser bestehendes adaptive layer-split (`-ot` regex).

3. **Two-phase update** (allocate → confirm → publish pointer) — race-free
   layer migration, kein hard refactor in `ggml-cuda` nötig.

### NOT portable from Pie

- vLLM block manager — llama.cpp hat keine page-blocks
- Per-layer KV-cache resize at runtime — unser KV ist contiguous
- NVLink-bandwidth assumptions — auf PCIe-x4 ist effektiv ~3 GB/s nutzbar,
  damit Pie's "swap finishes before next use"-Garantie nicht haltbar bei
  großen experts (~500 MB pro layer)

### From Turnip (lower feasibility)

- **Memgraph-style streams für prefill** könnte gehen (statische graph),
  aber: prefill ist nicht unser Bottleneck (pp512 = 1005 t/s auf 35B).
  Kein Sinn dort zu investieren.

- **Async stream interleaving** im decode-path: theoretisch, aber Turnip's
  Premise (statische graph) gilt nicht. Müsste man von hand auf llama.cpp's
  `ggml_backend_sched` aufbauen — separater design exercise (Tier-D).

## Phase 5 Hauptkandidaten (priorisiert)

### A) Expert-Prefetch über PCIe (Pie-style, light)

**Idee:** im decode-path, bevor MoE-router den nächsten layer aktiviert,
prefetchen wir die top-k experts dieses layers über PCIe in GPU-RAM.

**Wo:** `ggml/src/ggml-cuda/ggml-cuda.cu` — neuer `cudaMemcpyAsync` mit
dedicated stream, getriggert von `mul_mat_id` dispatch ein layer voraus.

**Bench-gate:** ≥+30% TG auf 80B (ctx ≤ 8192). Falls nur +5-10% (PCIe-x4
bottleneck dominiert), shelf.

**Risiko:** mittel — könnte Backend-Sched-Race triggern.

### B) Async CUDA Stream Pipeline (Memgraph-light)

**Idee:** GPU0 layer N+1 prefetch || GPU1 layer N compute || CPU layer N-1
swap-out, alles auf separaten streams.

**Wo:** `ggml-cuda.cu` `ggml_cuda_compute_forward` + scheduler hook.

**Bench-gate:** ≥+15% TG. Niedrigere bar weil kompletter refactor.

**Risiko:** hoch — zwei concurrent streams auf 12 GB RTX 2060 = OOM-risk.

### C) Speculative Decoding (Task #149, spec exists)

Existing spec, kein Pie-Port. Draft-model-ansatz.

**Bench-gate:** ≥+50% effective TG auf 80B (acceptance ≥ 70%).

**Risiko:** niedrig — gut etabliert in vLLM/llama.cpp upstream.

## Recommended Roadmap

1. **Tag 1 (heute, post-approval):** Spike A (expert-prefetch) auf 80B.
   Single-file patch, env-gated. Bench-only branch.
2. **Gate 1:** Wenn A ≥ +30% TG → ship. Wenn < +30% → drop, gehe zu C.
3. **Tag 2-3:** Spec C (speculative decoding) implementieren falls A drops.
4. **Tag 4+:** B (async streams) nur wenn A oder C nicht reicht.

## Non-Goals Phase 5

- vLLM port (zu groß)
- Turnip memgraph runtime (incompat mit decode)
- Multi-GPU NVLink emulation (HW-impossible)
- KV-cache page allocator (separate Tier-D refactor)

## Verification

Pre-merge bench:
- `llama-bench` 35B/80B/122B mit `-p 512 -n 128`
- short-ctx (2048), mid-ctx (8192), long-ctx (32768)
- HellaSwag-200 nicht regressing > 1pp

## Open Questions

1. Hat llama.cpp `ggml_backend_sched` einen hook zum prefetch-trigger?
   → muss in code-recherche prüfen.
2. Funktioniert `cudaMemcpyAsync` reliable auf PCIe-x4 ohne stream-stall?
   → microbench bauen (~30 min).
3. Welche expert-size auf 80B? Wenn > 256 MB pro expert, dann ist prefetch-
   latency > compute-latency = no win.
   → measure first.

## Approval Gate

User-Freigabe nötig vor Spike A. Spec-only commit ok ohne approval.

## Microbench Result (2026-04-26)

PCIe `cudaMemcpyAsync` H2D, pinned host buffer, 20 iter avg:

| GPU | PCIe link | Bandwidth | 256MB time | 512MB time |
|---|---|---|---:|---:|
| GPU0 | x16 | **13.14 GB/s** | 20.4 ms | 40.9 ms |
| GPU1 | x4  | **1.44 GB/s** | 186.5 ms | 372.7 ms |

**Decode budget** auf 80B @ ~36 t/s = **27.7 ms/tok**.

**Verdict Spike A:**
- GPU1 (PCIe-x4) prefetch eines 256MB experts = **186 ms** = 6.7× über
  decode-budget. Pie's Kernannahme `tswap ≤ tcompute` fundamental verletzt.
- GPU0 (PCIe-x16) borderline: 20.4ms ≈ 74% of decode budget. Win-margin
  gegessen von Sched-Overhead. Theoretisch möglich, aber risk/reward schlecht.

**Decision:** Spike A SHELVED. Pivot zu **C (Speculative Decoding, #175)**.
Spec-decode hat keinen PCIe-bottleneck (draft model läuft komplett auf GPU0,
80B validiert in compute-bound batched verify) — passt zu test-box-Hardware.

## Spec-Decode Re-Eval (2026-04-26)

Existing spec `docs/plans/2026-04-23-speculative-decoding-spec.md` documents:

- Public benchmark (thc1006, RTX 3090, post-PR#19493, 2026-04-19) on
  Qwen3.6-35B-A3B with Qwen3.5-0.8B draft at **100% acceptance**: **-10.8%
  TG regression**, not speedup.
- Root cause: A3B routes 8-of-256 experts/token. Draft batch K=4-8 is below
  expert-saturation threshold (~94 tokens). Each draft token = more unique
  experts loaded = more memory traffic that cancels verification savings.
- test-box (2× RTX 2060 12GB asymmetric x16/x4) strictly weaker than RTX 3090
  single-GPU → expect ≥10% regression.

**Verdict C:** Spec-decode SHELVED for our primary target (Qwen3.6-35B-A3B).
Could work for dense Qwen3.6-27B (15.26 t/s baseline) — that's a separate
investigation, not Phase 5 priority.

## Phase 5 Outcome

**Both Tier-C/D architectural candidates blocked by test-box hardware reality:**
- Spike A blocked by PCIe-x4 (10× too slow for prefetch budget)
- Spike C blocked by A3B MoE expert-saturation pathology

**Pivot to Tier-E: code-quality + microopts that don't depend on PCIe/MoE.**

Open candidates:
1. CUDA Graphs for decode (Task #150 research done) — captured graph replay
   eliminates kernel-launch overhead. Estimated +5-10% TG.
2. mmvq quantized matmul vec on Turing sm_75 — Task #151 already done,
   re-verify if there's headroom.
3. Per-layer mixed-precision V-cache default-off (#142) — code hygiene.
4. Investigate dense Qwen3.6-27B path: spec-decode IS viable on dense
   (no MoE pathology). Could give +50-80% effective TG on the 27B target.

**Recommended next session:** Spec-decode auf dense **Qwen3.6-27B**, nicht
auf MoE 35B/80B/122B. Draft = Qwen3.5-0.6B oder 0.8B. Bench-gate ≥+50% TG.
