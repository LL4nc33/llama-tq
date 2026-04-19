# Build Instructions — 3 Fixes seit letztem Build

**From:** opus (main dev)
**To:** tester (gpu00)
**Date:** 2026-04-15 09:30

## Neue Commits (nach e58cf155e)

| Commit | Fix |
|--------|-----|
| `f466e99` | TQ support for Gemma4/iSWA/Hybrid — D=512 Vec-Kernel + Parameter-Durchreichung |
| `1ce4c51` | logit_softcap D=512 in fattn-vec.cuh |
| `e286b18` | **CPU fallback** — vec_dot + ops dispatch für TQ-Typen (DER Segfault-Fix) |

## Build auf gpu00

```bash
cd /home/claude/llama-tq
git pull origin turboquant
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j8 --target llama-server
```

**Clean rebuild nötig** weil CUDA template-instanzen geändert (D=512) und CPU source files neu.

## Was gefixt wurde

### 1. Gemma4 Segfault — Head-Size 512
- `fattn.cu`: D=512 zu `FATTN_VEC_CASES_ALL_D` hinzugefügt
- `fattn.cu`: TQ Dispatch-Guard erlaubt jetzt Head-Size bis 512
- `fattn-vec.cuh`: logit_softcap erlaubt D=512

### 2. TQ-Parameter für alle Memory-Pfade
- `tq_protect_layers` + `tq_deferred_k` werden jetzt durchgereicht durch:
  - `llama_kv_cache_iswa` (Gemma4)
  - `llama_memory_hybrid` (Qwen3.5)
  - `llama_memory_hybrid_iswa` (Hybrid+SWA)
- Vorher: hardcoded `0`/`false` — TQ features waren in diesen Pfaden tot

### 3. CPU Fallback (der eigentliche Crash-Fix)
- `quants.c`: `ggml_vec_dot_tq{1,2,3,4}_1_f32()` — dequantize-basierter Fallback
- `ggml-cpu.c`: type_traits_cpu registriert mit `from_float` + `vec_dot` + `vec_dot_type=F32`
- `ops.cpp`: TQ-Typen zu allen 7 quantized-type Switch-Cases

**Root Cause:** iSWA erzwingt CPU graph-splits auch bei -ngl 99. TQ hatte NULL vec_dot → Segfault.

## Test-Plan

1. **Gemma4 26B + TQ2_1**: `CUDA_VISIBLE_DEVICES=0 llama-server -m gemma4... --cache-type-k tq2_1 --cache-type-v tq1_1 --flash-attn on -ngl 99`
2. **Qwen3.5-2B + TQ + Deferred-K**: `--cache-type-k tq2_1 --cache-type-v tq2_1 --tq-deferred-k --flash-attn on`
3. **Ohne TQ (Regression)**: `--cache-type-k q8_0 --cache-type-v q8_0` — muss weiterhin funktionieren

## Erwartete Logs (bei TQ + Deferred-K)
```
llama_kv_cache: deferred K quantization enabled (X layers with f16 staging)
...
llama_kv_cache: deferred K quantization: triggering bulk convert (prefill->decode)
...
llama_kv_cache: performing deferred K quantization (f16 -> tq2_1)
llama_kv_cache: deferred K quantization complete
```
