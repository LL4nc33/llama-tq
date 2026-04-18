# llama-tq Devblog

Öffentliche Journey-Dokumentation von **TurboQuant Nano** — experimentelle
GPU-beschleunigte Quantization für llama.cpp.

Jeder Post beantwortet die W-Fragen: **Was? Warum? Wie? Wann? Wo? Wer? Wie viel?**
Wenn keine Messwerte → kein Post.

## Posts

- [2026-04-19 — Night Session Summary: Timeline + cherry-pick candidates](2026-04-19-night-session-summary.md)
- [2026-04-19 — V-Cache Pipeline Validated: 4× Encoder-Speedup, PPL measured](2026-04-19-v-cache-validation.md) ← **with real numbers**
- [2026-04-18 — TurboQuant V-Cache: Von CPU-Fallback zu nativem GPU-Dequant](2026-04-18-native-v-dequant.md)

## Branches

- `master` — stable, deployed in production (TQ v5)
- `trellis-v2-phase1` — VTQ_2 development (Trellis-based V-cache)
- `tqw-option-b` — TurboQuant Weight Quantization (experimental)

## Referenzen

- [TurboQuant Paper (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) — Google Research, ICLR 2026
- [QTIP Paper (arXiv:2406.11235)](https://arxiv.org/abs/2406.11235) — Trellis-Coded Quantization
- [Upstream llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT License, base fork
