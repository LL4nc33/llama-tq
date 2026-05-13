# MASSIVE FIND: Multi-GPU Layer-Split delivers +40-74% PP — 2026-05-13

## TL;DR

Alle bisherigen single-GPU benches haben einen **40-74% PP-win übersehen**:
mit beiden RTX 2060 sichtbar (kein `CUDA_VISIBLE_DEVICES=0` cap) routet
llama.cpp default-split layers über beide GPUs → tensor-parallel prefill.

**Ministral-3-3B Q4_K_M PP@2k: 1316 → 1851 t/s (+40.7%).**

## Bench-vergleich (build `6cc4920d0`)

### Ministral-3-3B Q4_K_M

| ctx | single-GPU0 | dual-GPU | Δ |
|------|---|---|---|
| PP@512  | 1977 | 1896 | -4.1% |
| PP@1024 | 1699 | **2064** | **+21.5%** |
| **PP@2048** | 1316 | **1851** | **+40.7%** |
| PP@4096 | 901 | **1405** | **+56%** |
| PP@8192 | 546 | **919** | **+68%** |
| PP@10240 | 444 | **773** | **+74%** |
| TG@128 | 97 | 97 | flat |

### Ministral-3-14B IQ2_XXS

| ctx | single | dual | Δ |
|------|---|---|---|
| PP@1024 | 668 | 857 | +28% |
| PP@2048 | 555 | 837 | +51% |
| PP@4096 | 412 | 698 | +69% |
| TG@128 | 31 | 33 | +6% |

### Qwen3.6-35B-A3B IQ2_XXS

| ctx | single | dual | Δ |
|------|---|---|---|
| PP@1024 | 1027 | 1246 | +21% |
| PP@2048 | 962 | **1324** | +37% |
| PP@4096 | 839 | **1273** | +52% |
| PP@8192 | — | 1058 | (single likely VRAM-tight) |
| TG@64 | 81 | 76 | **-6%** (inter-GPU comm overhead) |

## Mechanics

- llama.cpp default split-mode = **LAYER** (layer 0 to (N/2)-1 on GPU0, rest on GPU1)
- During prefill: large batch passes through layers sequentially, BUT
  the two GPUs process **different layers concurrently in pipeline**
- Tensor parallelism manifests as: per-layer compute on one GPU at a time,
  but the pipeline is doubled because while GPU0 runs layer K of next ubatch,
  GPU1 runs layer K+M of current ubatch
- Effective compute throughput ≈ 2× single-GPU minus PCIe inter-link transfer

## Wann lohnt es?

- **Prefill (PP):** clear win bis ctx ≤ ~10k. Bei 3B: PP@2k +40%, PP@10k +74%
- **Decode (TG):** flat oder leicht schlechter (Ministral) bzw. -6% (35B)
  weil token-by-token serial pipeline mit inter-GPU sync overhead

## Deploy-implication

Für deploy-skripte (`deploy-*.sh`) **niemals** `CUDA_VISIBLE_DEVICES=0` setzen
wenn dual-GPU verfügbar und das model lange prefills macht. Den Bottleneck-balance
kann mit `-ts X,Y` getuned werden (z.B. mehr layers auf der schnelleren GPU).

Für single-user TG-only workloads (langes generieren von einzelnen tokens):
single-GPU bleibt äquivalent oder marginal besser für 35B.

## Conclusion

Phase 5 lieferte 8.2× PP via kernel-improvement. Multi-GPU layer-split
liefert weitere +40-74% **on top**, ohne kernel-änderung. Kombiniert:

- 3B PP@2k: ursprünglich pre-Phase-5 ~160 t/s → jetzt **1851 t/s = 11.5×**
- 3B PP@10k: ursprünglich pre-Phase-5 ~212 t/s → jetzt **773 t/s = 3.6×**

Ein "free lunch" der einfach übersehen wurde.
