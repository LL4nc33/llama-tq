# llama-tq devblog

Public journey documentation for **TurboQuant Nano** — experimental
GPU-accelerated quantization for llama.cpp.

Every post answers the basic questions: **What? Why? How? When? Where? Who? How much?**
No numbers → no post.

## Posts

- [2026-04-19 — Night session summary: timeline + cherry-pick candidates](2026-04-19-night-session-summary.md)
- [2026-04-19 — V-cache pipeline validated: 4× encoder speedup, PPL measured](2026-04-19-v-cache-validation.md) ← **with real numbers**
- [2026-04-18 — TurboQuant V-cache: from CPU fallback to native GPU dequant](2026-04-18-native-v-dequant.md)

## Branches

- `master` — stable, deployed on the author's hardware (TQ v5)
- `trellis-v2-phase1` — VTQ_2 development (trellis-based V-cache)
- `tqw-option-b` — TurboQuant weight quantization (experimental)

## References

- [TurboQuant paper (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) — Google Research, ICLR 2026
- [QTIP paper (arXiv:2406.11235)](https://arxiv.org/abs/2406.11235) — Trellis-Coded Quantization
- [Upstream llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT License, base fork
