# Phase 4 deep-dive — upstream fusion-detection bereits da!

## TL;DR

Während der phase-4-skeleton-arbeit zwei kritische entdeckungen gemacht:

1. **Upstream llama.cpp hat bereits CUDA-side fusion-detection für `MUL_MAT_ID + MUL_MAT_ID + GLU`** (3 nodes → 1 fused kernel). Files: `ggml/src/ggml-cuda/ggml-cuda.cu:2186-2257` (`ggml_cuda_should_fuse_mul_mat`) + `:3419-3445` (`ggml_cuda_can_fuse`).

2. **Diese fusion ist DISABLED für split-buffers** (line 2253: `if (split) return false;`). Unser dual-GPU layer-split-mode triggert split-buffers → **fusion aktuell nicht aktiv auf gpu00 dual-GPU deploy.**

## Implication für phase 4 strategie

**KORREKTUR 22:25:** `ggml_backend_buft_is_cuda_split` trifft nur für `LLAMA_SPLIT_MODE_ROW`, NICHT für unseren `LLAMA_SPLIT_MODE_LAYER` (default).

Verified via `src/llama-model.cpp:589`: split-buffer-type wird nur erstellt wenn `split_mode == LLAMA_SPLIT_MODE_ROW`. Layer-split nutzt normale per-device cuda-buffer.

**Konsequenz:** fusion ist auf unserem dual-GPU layer-split deploy **bereits aktiv**! Der 3% gap zwischen single-GPU (1017) und dual-GPU (988) PP@4k ist NICHT durch fehlende fusion erklärt, sondern durch:
- PCIe sync overhead zwischen GPUs
- Inter-GPU memory-transfer per layer-boundary
- Möglicherweise scheduler-overhead

**Phase 4 strategie zurück zu original ik_llama PR #229 port** — oder ein anderer hebel.

## Concrete file:line pointers

**Fusion-detection pattern check:**
- `ggml/src/ggml-cuda/ggml-cuda.cu:2186` — `is_mul_mat_id` predicate
- `ggml/src/ggml-cuda/ggml-cuda.cu:2249` — split-buffer disabler (line 2253: `return false`)
- `ggml/src/ggml-cuda/ggml-cuda.cu:3422` — `mul_mat_id_glu_ops = { MUL_MAT_ID, MUL_MAT_ID, GLU }`
- `ggml/src/ggml-cuda/ggml-cuda.cu:3439-3445` — matcher + dispatch
- `ggml/src/ggml-cuda/ggml-cuda.cu:2287` — `dst->ne[2] != 1` disabler (für vec-fusion, decode-only)

**Graph-builder emits compatible pattern:**
- `src/llama-graph.cpp:1475` — `up = build_lora_mm_id(up_exps, ...)` → MUL_MAT_ID
- `src/llama-graph.cpp:1493` — `cur = build_lora_mm_id(gate_exps, ...)` → MUL_MAT_ID
- `src/llama-graph.cpp:1540` — `cur = ggml_swiglu_split(ctx0, cur, up)` → GLU

→ **Pattern matches.** Fusion sollte aktiv sein wenn nicht split-buffer.

## GGUF tensor verification

Verified via `gguf-py` on `/models/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`:

```
blk.0.ffn_gate_exps.weight     ← separate (not merged)
blk.0.ffn_up_exps.weight       ← separate
blk.0.ffn_down_exps.weight
blk.0.ffn_gate_inp.weight
blk.0.ffn_gate_inp_shexp.weight
blk.0.ffn_gate_shexp.weight    ← shared expert
blk.0.ffn_up_shexp.weight
blk.0.ffn_down_shexp.weight
```

Separate `ffn_gate_exps` + `ffn_up_exps` → fusion-pattern compatible. Kein `ffn_gate_up_exps` merged tensor.

## Bench-implication

Aktuelle stable PP@4k werte (build `ca9a68cf3`):
- f16/f16 dual-GPU: 1372
- ktq2_1+vtq2_1 dual-GPU: 988 (fusion DISABLED via split-buffer)
- ktq2_1+vtq2_1 single-GPU: 1017

Single-GPU ist 3% schneller (1017 vs 988) trotz weniger compute — vermutlich fusion-bonus.

## Phase 4 revidierter plan

**Schritt 4.0 (nach build):** Bench single-GPU mit fusion aktiv vs dual-GPU mit fusion disabled. Measurement bestätigt/widerlegt hypothese.

**Schritt 4.1:** Analyse warum split-buffer fusion disabled ist. Was sind die exakten technical limitations?

**Schritt 4.2:** Versuch fusion auf split-buffers zu enablen. Wenn jeder split-shard die volle fusion-chain hat (was bei layer-split der fall ist — pro layer wird ENTWEDER GPU0 ODER GPU1 verwendet, nicht beide), dann sollte fusion per-layer trivial sein.

**Schritt 4.3:** Falls 4.2 nicht trivial: alternative wege:
- per-layer-fusion: skip fusion-check für layer-split-mode
- pre-fusion in graph-builder (vor scheduler)

**Schritt 4.4:** Falls fusion-aktivierung nicht möglich auf dual-GPU: dann macht full ik_llama PR #229 port wieder sinn.

## Status

- ✅ Discovery dokumentiert
- 🔧 Build `bptwajf28` läuft (revert deployment), ~2h
- ⏳ Nach build: schritt 4.0 (bench measurement)

## Lehren bisher

1. **Upstream codebase studieren VOR neues op designen** — hätte uns 5-7 tage gespart wenn früher erkannt
2. **Split-buffer interaktion** ist ein wiederkehrendes thema (auch cuda graphs, jetzt fusion)
3. **Graph-pattern matching** ist mächtiger als ich dachte
