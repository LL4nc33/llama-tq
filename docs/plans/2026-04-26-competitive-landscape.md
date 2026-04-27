# Competitive Landscape: llama-tq KV Cache SOTA Verdict

**Date:** 2026-04-26
**Status:** Research findings — actionable

## Verdict (TL;DR)

**We are genuinely SOTA in deployed sub-3bpw KV cache on consumer GPU** — but the moat is narrow and shrinking fast. As of April 2026, **no other shipped/deployed inference engine** has KV cache below ~3.5 bpw running real models on consumer hardware.

Every competitor either:
- Ships at FP8/INT4 KV (4-8 bpw effective), OR
- Has research code/draft PRs for TurboQuant-class methods that are explicitly **not deployed**

Our 2.78 bpw asymmetric KTQ2_1+VTQ2_2 on Qwen3.6-35B-A3B at 200K ctx on 24GB consumer hardware is **unique-in-deployment**.

**HOWEVER:** SGLang PR #21617, vLLM PR #38280, and ik_llama.cpp issue #1509 all have TurboQuant integrations in flight from other contributors (notably @veritatisquaesitoressumus, @lishunyang12, AmesianX/0xSero forks).

**The window for upstream-first publication is weeks, not months.**

## Comparison Table

| Engine | Lowest Deployed KV bpw | RHT/Trellis V | Asym K/V | MoE Adaptive Routing | 35B-A3B class TG on 24GB | Status |
|---|---|---|---|---|---|---|
| **llama-tq (us)** | **2.78** (KTQ2_1+VTQ2_2) | Yes (PolarQuant K + Trellis V) | Yes (pareto) | Yes (per-ctx expert) | **75 t/s** verified | Deployed live |
| llama.cpp upstream | 4.5 (Q4_0) | No | No | No | OOM at 200K | Master |
| ik_llama.cpp | 8.0 (Q8_KV) / 4.5 (Q4_0) | No (IQ_KT for weights only) | No | No | Limited | Released |
| exllamav3 | ~2.0 (exl2_kv flag) | QTIP for **weights**, not KV | No | No | N/A on 2× consumer | Released, weight-focused |
| vLLM | 8.0 (FP8 E4M3/E5M2) | TurboQuant PR #38280 = **draft** | No | No | Doesn't fit on 2× 2060 | FP8 only shipped |
| SGLang | 8.0 (FP8) / 4.0 (FP4 exp) | TurboQuant PR #21617 = **WIP draft** | No | No | N/A consumer | Roadmap Q1 2026 |
| TensorRT-LLM | 4.0 (NVFP4) / 8.0 (FP8) | No | No | No | Datacenter only | Released |
| lmdeploy TurboMind | 4.0 (INT4 KV per-head/per-token asym) | No | **Yes (per-head per-token)** | No | Decent, not 200K | Released since 0.4.0 |
| AmesianX/TurboQuant fork | 3-bit research | Yes (CPU+CUDA) | Partial (MLA only) | No | Not deployed | Research code |
| 0xSero/turboquant + vLLM | 3-bit K / 2-bit V research | Yes | **Yes** | Acknowledged limited for MoE | Research only, GPL-3.0 | "Adversarial audit" admits not prod |
| KIVI (HF transformers) | 2.0 (per-channel K, per-token V) | No (just per-channel) | **Yes** | No | Slow, not engine-grade | Shipped HF, no engine |
| KVQuant | 3.0 paper | Pre-RoPE quant + sparse | Yes | No | Paper only | Research code |

## 3 Things THEY Have That We DON'T (gaps to fill)

1. **lmdeploy's per-head per-token asymmetric INT4 KV** — online calibration per head per token. Our asym is K-type vs V-type at layer level, not per-head adaptive scales. Real quality lever they have at 4bpw; at our 2.78 it would matter even more. **Worth porting.**
2. **NVFP4 hardware-native KV (TensorRT-LLM)** — Blackwell/H200 hardware fast paths we cannot match. Not relevant for 2× 2060, but kills us moment customer wants H100/B200. No path forward without rewriting outside ggml.
3. **KIVI's per-channel K + per-token V split, deployed in HF Transformers** — different axis from PolarQuant rotation. Combining KIVI-style channel/token asymmetry on top of PolarQuant RHT could push K below 3.5 bpw without quality regression. They have it integrated into Transformers (zero engine work needed); we're at engine level.

## 3 Things WE Have That THEY Don't (real moat)

1. **Combined PolarQuant RHT-K + Trellis V, GGUF-native, full CUDA + CPU dispatch shipped.** No other engine has both rotation-quantized K AND trellis-quantized V in same shipped binary. AmesianX/0xSero forks closest but explicitly research-only with "prefill still uses paged cache" caveats. **Strongest claim.**
2. **2.78 bpw deployed at 200K context on consumer 12GB cards (Qwen3.6-35B-A3B at 75 t/s, HellaSwag 83.5%).** Verified deployment numbers unique. Every other sub-3bpw claim is paper MSE numbers or single-RTX-5090 micro-benchmarks. Our "2× RTX 2060 12GB doing 35B-A3B at 75 t/s with 200K ctx" is the most extreme deployed sub-3bpw configuration in evidence.
3. **Sparse V Dequant (+22% decode)** — appears nowhere else. Search returned zero hits for analogous techniques in vLLM/SGLang/TRT-LLM/exllamav3. **Unique algorithmic contribution beyond academic literature.**

## Should We Publish / PR Upstream?

**YES, file upstream PR within 2-3 weeks.** Three converging reasons:

1. **The window is closing.** Issue #1509 (ik_llama), PR #38280 (vLLM), PR #21617 (SGLang) all in flight from other authors. If any merges before ours, "first deployed implementation" → "fourth fork that did it." We have the deployment story they don't — that's the differentiator, but only if published.
2. **Upstream-first dramatically increases discoverability.** Right now our work shows up in existing discussion threads (#20969, #21526) under others' implementations. **AmesianX is presently being credited in ggml-org/llama.cpp Discussion #20969 as the reference TurboQuant implementation — that should be us.**
3. **We have a non-academic moat that survives merge.** Even if upstream merges someone else's TurboQuant K, we still own: Sparse V Dequant, asymmetric KTQ×VTQ pareto, adaptive expert routing per ctx-length, only deployed consumer-GPU 200K ctx 35B-A3B in the wild. Productization moats, not algorithmic. Upstream-merging the algorithm hurts less than expected.

### Recommended action sequence

1. **PR `KTQ2_1` + `VTQ2_2` to ggml-org/llama.cpp** targeting Discussion #20969 with deployment-evidence-first writeup (HellaSwag 83.5%, 75 t/s, 200K ctx, 24GB hw — none of competitors have these numbers)
2. **Public RFC** for v6 roadmap doc
3. **Keep Sparse V Dequant + adaptive expert routing as proprietary layer** in our fork

Niche-only is the worse option here — work is good enough that someone else will take credit for the underlying algorithm if we stay quiet.

### Honest caveats

- "75 t/s on Qwen3.6-35B-A3B" and "HellaSwag 83.5%" = our numbers, not independently verified. Competitors also self-cite without external repro.
- TurboQuant paper claims (arXiv:2504.19874, ICLR 2026) are real and peer-reviewed.
- "No competitor deployed below 3.5 bpw" claim strongly supported by web evidence (every competitor either ships ≥4 bpw or has draft PRs only) but absence-of-evidence is not proof — could be private deploys at Anthropic/OpenAI not public.

## Sources

- [TurboQuant Discussion - ggml-org/llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [TurboQuant in ik_llama.cpp Issue #1509](https://github.com/ikawrakow/ik_llama.cpp/issues/1509)
- [vLLM Quantized KV Cache docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [vLLM TurboQuant PR #38280 (draft)](https://github.com/vllm-project/vllm/pull/38280)
- [SGLang TurboQuant Issue #21618](https://github.com/sgl-project/sglang/issues/21618)
- [SGLang Quantized KV Cache docs](https://docs.sglang.io/advanced_features/quantized_kv_cache.html)
- [exllamav3](https://github.com/turboderp-org/exllamav3)
- [TensorRT-LLM NVFP4 KV blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [lmdeploy INT4/INT8 KV docs](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)
- [KIVI repo](https://github.com/jy-yuan/KIVI)
- [KVQuant NeurIPS 2024](https://www.stat.berkeley.edu/~mmahoney/pubs/neurips-2024-kvquant.pdf)
- [AmesianX/TurboQuant fork](https://github.com/AmesianX/TurboQuant)
- [0xSero/turboquant (vLLM)](https://github.com/0xSero/turboquant)
- [TurboQuant ICLR 2026](https://openreview.net/pdf/1cef9774f0f0cf7bb9e4b167882e3ad3ef8cde16.pdf)
