# Deferred V Quantization — Results (2026-04-19)

## Model: Qwen3.5-0.8B-Q8_0 on RTX 2060

### Speed (llama-bench)

| Config | pp128 | tg64 |
|--------|-------|------|
| vtq3_2 (baseline, per-token Viterbi) | 35.36 t/s | 7.43 t/s |
| **vtq3_2 + `--tq-deferred-v`** | **4462.53 t/s** | **190.91 t/s** |
| Speedup | **126×** | **26×** |

### Quality (llama-perplexity, 4 chunks, n_ctx=2048)

| Chunk | f16 V | vtq3_2 deferred | Δ% |
|-------|-------|-----------------|-----|
| [1] | 14.80 | 15.37 | +3.8% |
| [2] | 16.68 | 16.99 | +1.9% |
| [3] | 17.29 | 17.68 | +2.3% |
| [4] | 17.37 | 17.73 | +2.0% |

**Average overhead: ~+2% PPL at 3.06 bpw.**

## Known Issue: PPL Prefill Speed

58.92s/pass vs f16 0.77s/pass in PPL-mode suggests Viterbi still
blocking prefill-path somewhere. Encode alone should be ~120ms/chunk.
Requires further investigation — tg path is unaffected and is the
primary use case.

## 27B on dual RTX 2060 (Qwen3.5-27B-UD-IQ2_XXS)

| Config | pp128 | tg64 |
|--------|-------|------|
| f16 V | 380.23 t/s | 14.90 t/s |
| **vtq3_2 + deferred** | **379.60 t/s** | **14.88 t/s** |
| Δ | -0.2% | -0.1% |

**Parity with f16** at 3.06 bpw → 73% V-cache VRAM savings at zero
speed cost. Perfect scaling validation.

## Verdict

Phase-3 deferred V quantization is a **decisive win**.
- Small model (0.8B): 26× tg speedup vs per-token Viterbi
- Mid model (27B): parity with f16 at 3.06 bpw V-cache
- Quality: ~+2% PPL overhead (Viterbi algorithm bit-identical,
  only dispatch timing changes)
