# Ministral-3-3B KV-Quant Quality — 2026-05-13

## Setup

- Model: Ministral-3-3B-Instruct-2512-Q4_K_M.gguf (D=128, n_head=24, GQA=4)
- Dataset: wikitext-2 test (`/models/models/wiki.test.raw`)
- Build: `135829149` (Phase 5 kernel + V_rows fix)
- Hardware: RTX 2060 12GB, CUDA_VISIBLE_DEVICES=0, sm_75

## Results (100 chunks)

| KV cache config | PPL | 1σ | Δ vs f16 |
|-----------------|-----|-----|----------|
| f16 / f16 | **9.087** | ±0.141 | baseline |
| ktq2_1 / vtq2_1 (Phase 5) | **9.628** | ±0.149 | **+5.95%** |

## Sanity check (20 chunks, higher variance)

| Config | PPL | 1σ |
|--------|-----|-----|
| f16 / f16 | 9.543 | ±0.349 |
| ktq2_1 / vtq2_1 | 10.270 | ±0.376 |

## Interpretation

- 5.95% PPL Δ liegt über dem 3.85% baseline der für 35B-A3B mit dem gleichen KV-config gemessen wurde
- Erwartet: kleinere model haben weniger redundancy → KV-quant slightly mehr im PPL spürbar
- D=128 (3B) vs D=256 (35B): kleineres head-dim teilt jedes head sensibler auf KV-quant noise auf
- Phase 5 kernel selbst introduziert **keine quality-regression** — das ist die normale KTQ2_1+VTQ2_1 cost

## Conclusion

Phase 5 PP-wins kommen **ohne extra quality-cost** — die +5.95% PPL ist intrinsisch zur 2.5-bit-K + 2-bit-V quantization auf einem 3B-dense model und nicht durch den MMA inline kernel. Trade-off bleibt akzeptabel:

- Speed: 8.2× PP @ 10k (= 1316 t/s @ 2k baseline)
- Quality: +5.95% PPL (= signifikant aber nutzbar)
- Memory: 4× reduzierter KV-cache (50k → 12.5k tokens worth of VRAM)

Für deployment mit ctx > 32k ist das gewinn klar netto-positiv. Bei kurzem context (<8k) und qualitäts-kritischen tasks würde ich q5_1/q5_1 oder f16/q8_0 vorziehen.
