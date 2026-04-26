---
slug: "phase-5-xquant-in-flight"
title: "Phase 5: XQuant Cross-Layer KV — 1.69 bpw target (WIP)"
date: "2026-04-26"
status: "wip"
tags: ["turboquant", "phase-5", "xquant", "cross-layer", "kv-cache"]
hero_number: "1.69 bpw target"
---

# Phase 5: XQuant Cross-Layer KV (in flight)

Aktueller deployed Stand auf llama-tq Fork: 2.78 bpw averaged KV cache (ktq2_1 + vtq2_2). Phase 5 versucht das auf 1.69 bpw zu drücken via Cross-Layer KV Reuse.

## Was XQuant macht

XQuant Paper (arxiv:2510.11236, EMNLP 2025) zeigt dass adjacent layers in einem LLM ähnliche K-vektoren haben — empirisch >80% der quantized positions in Layer L und L+1 differ nur um 1.

Lösung: pair adjacent layers. Layer 2k speichert quantized integer codes. Layer 2k+1 speichert nur seine eigenen scales und reused die codes von der dominanten layer beim dequant.

Plus einen data-free calibration parameter eta die die endpoint mapping relaxed.

## Stackability mit unserer Pipeline

Unser KTQ2_1 dequant pipeline endet mit per-block delta multiply. XQuant inserts genau dort: subordinate layer reads codes von layer l-1 buffer, applied dann eigene (z_hat, s_hat). Da Hadamard rotation deterministic per layer (Philox-seeded), ist cross-layer mathematisch sauber.

Critical: neuer ggml type GGML_TYPE_XKTQ2_1 mit nur 8 bytes per 8-token group (delta + zero-point), NICHT die 36 bytes von KTQ2_1.

## Layer-Pair Selection für Qwen3.6-35B-A3B (47 layers)

- Layers 0-3: keine XLC (boundary protection)
- Layers 4-45: 21 pairs (4,5), (6,7), ..., (44,45)
- Layer 46: keine XLC

## Memory Savings — Qwen3.6-35B-A3B at 200K ctx

| Config | K bpw | V bpw | Total KV |
|---|---:|---:|---:|
| f16 baseline | 16 | 16 | 78.1 GB |
| Today: KTQ2_1 + VTQ2_2 | 3.5 | 2.06 | 13.6 GB |
| Phase 5 v1: + XLC on K | 1.69 | 2.06 | 9.16 GB |
| Phase 5 v2: + XLC on V | 1.69 | 1.01 | 6.61 GB |

Concrete impact für unseren live deploy: 13.6 GB → 9.16 GB = -4.4 GB freed. Either ctx auf 290K, oder 3rd parallel slot.

## Phasen

| Phase | Scope | Status |
|---|---|---|
| 1 | ggml type + block struct + CPU round-trip | commit 559dc7809 |
| 2 | CUDA dequant kernels (XKTQ2_1, _3_1, _4_1) | spec ready |
| 3 | FA dispatch wiring (sibling tensor input) | spec ready |
| 4 | Layer-pairing logic in llama-kv-cache.cpp | spec ready |
| 5 | Calibration tool + 35B validation | spec ready |

## Was bisher passiert ist

Phase 1 foundation committed: GGML_TYPE_XKTQ2_1=57, block_xktq2_1 (8 bytes), CPU dequant ref, round-trip test, CLI parser entry. ~119 LOC.

Spec für Phase 2-5 fertig (~1165 LOC estimated total).

Phase 1 build läuft gerade auf gpu00.

## Risiken

- eta values transferiren nicht Mistral->Qwen3.6: per-model grid search auf gpu00
- Adjacent-layer similarity fails auf MoE expert layers: pre-flight correlation probe
- FA kernel register pressure: profile vor merge

Hard quality gate: PPL delta vs ktq2_1+vtq2_2 baseline ≤ +0.3% auf wikitext-2 64-chunk.

## Was als nächstes

1. Phase 1 build durchwarten + run round-trip test
2. Phase 2 CUDA kernels schreiben
3. Phase 3+4 FA dispatch wiring + pairing logic
4. Phase 5 calibration tool + 35B validation
5. Bench-gate run
6. Live deploy update

Realistisch: 2 Wochen wenn alles sauber läuft.

## Why this matters für den fork

Wenn Phase 5 hits target, geht der live deploy von 2.78 bpw → 1.69 bpw. Das wäre eindeutig SOTA für deployed open-source consumer-GPU inference — kein anderer Fork hat sub-2 bpw KV deployed.

## Was nach Phase 5 kommt

llama-tq goes into on-llama-tq — das hat bereits OpenWebUI tools+functions integriert. Endgame: TurboQuant + Tools = OidaNice Inference Engine.
</content>
