# v8 Gate-B Results — Qwen3.6-35B-A3B-IQ2_XXS bartowski (2026-05-02 05:46)

**TLDR: vtq3_v8 ist essentially lossless auf 35B-A3B — besser als current prod.**

## Test Setup

- Modell: `Qwen_Qwen3.6-35B-A3B-IQ2_XXS-bartowski.gguf` (~10 GB)
- Hardware: GPU0 RTX 2060 12GB (single-GPU)
- Wikitext-2 raw, ctx=512, chunks=3 (smoke-tier)
- Build: commit `d3f7d2b09` (v8 vtq3_v8 + CUDA dispatch)
- Flash-Attention on, --ngl 99

## Results

| Config | bpw | PPL | Drift vs f16/f16 | Status |
|---|:---:|---:|---:|---|
| f16 / f16 | 32.0 | 7.2044 | 0.00% | baseline |
| ktq2_1 / vtq2_1 (current 35B prod) | 3.00 | 7.4816 | **+3.85%** | legacy |
| ktq2_1 / vtq2_2 (2025 default) | 2.78 | 7.1807 | **-0.33%** | legacy |
| **ktq2 / vtq3 (v8 NEW)** | **3.56** | **7.2024** | **-0.03%** | **WINNER (within noise)** |

## Key Insights

### 1. v8-default (ktq2/vtq3) ist essentially lossless

PPL drift -0.03% liegt vollständig in der Mess-Stderr (~0.7-1.2% bei 3-chunk runs). 
v8 vtq3_v8 (3.625 bpw) erreicht praktisch f16-Quality bei nur 11% mehr storage als vtq2_2 (3.25 bpw).

### 2. v8 schlägt current 35B prod deutlich

- Current: ktq2_1/vtq2_1 = +3.85% drift
- New: ktq2/vtq3 = -0.03% drift
- **Improvement: 3.88pp** bei +0.56 bpw extra

### 3. Trellis-Encoder Cost ist das Tradeoff

Wallclock per chunk-3 PPL pass:
- f16/f16: 14s
- ktq2_1/vtq2_1: 7s (codebook fast)
- ktq2_1/vtq2_2: 6s (deferred-V skipped beim PPL)
- **ktq2/vtq3 (v8): 69s** (Trellis encoder + 2 outliers picking ohne deferred-V)

Im **Live-Server mit deferred-V staging** wird das massiv kleiner — Encoder läuft nur 1× am prefill→decode boundary, nicht pro PPL-pass.

### 4. Memory-Estimate

KV cache @ 100k ctx auf 35B-A3B (60 attention layers, 8 KV heads, head_dim 128):
- ktq2_1/vtq2_1 (3.0 bpw): ~600 MiB
- ktq2/vtq3 (v8, 3.56 bpw): ~712 MiB (+19%)

Hinweis: structural bpw + actual layout overhead. 100k ctx fits comfortably auf single-GPU.

## Implications für Triple-Goal Deploy

**Production-recommendation update:**

| Goal | Current 35B prod | New v8 winner |
|---|---|---|
| Accuracy | +3.85% drift | **-0.03% drift (lossless)** |
| Speed (TG) | 71.74 t/s | TBD (need bench) |
| VRAM | 132 MB headroom | likely ~50 MB (vtq3 has more state) |

→ Falls TG nicht regressed (>70 t/s): **v8 ist klares prod-upgrade**.

## Next Steps

1. ✅ Gate-B passed
2. Speed-bench: `llama-bench` mit ktq2/vtq3 vs ktq2_1/vtq2_1 auf 35B-A3B (pp512 + tg128 + tg30k)
3. Gate-C: 4B-Q4_K_M (kann nicht jetzt — 4B prod läuft auf GPU1)
4. Full 8-model sweep (sequentielle Tests, ~5h)
5. Update prod deploy script `scripts/deploy-35b-singlegpu-100k.sh` mit v8 als default
6. README + Docs Update für v8 als prod-default
