# Phase 6 Router Peakedness — Research Findings

## Executive Summary

Adaptive top-k MoE routing based on router-distribution shape (entropy or top-p cumulative mass) is well-supported in the literature, with reported compute reductions of **25-52%** at negligible quality loss. The two dominant control variables are (a) **entropy thresholds** and (b) **cumulative-probability (top-p) thresholds**. Re-normalization of kept experts (`p'[i] = p[i] / sum_kept`) is the standard scheme. Per-layer behavior is highly non-uniform: **early/lower layers want more experts, late layers concentrate on top-1**.

## 1. Empirical Router Peakedness Numbers

**Mixtral 8x7B (top-2 native):**
- Routing entropy is right-skewed; **~32% of tokens have H < 1.0** → could safely use k=1 (Adaptive-K whitepaper).
- Adaptive-K reports **52.5% compute reduction** on Mixtral with no measurable accuracy loss on standard benchmarks.

**Qwen1.5-MoE-A2.7B:** 32.4% compute reduction at iso-quality (Adaptive-K).

**OLMoE-1B-7B:** 24.7% compute reduction at iso-quality (Adaptive-K).

**Dynamic-k MoE (Huang et al., arXiv:2403.07652):** Trained from scratch with top-p routing, p=0.4. Average activated experts across tasks: **1.72-1.87** vs. fixed top-2. Achieved **+0.7% average accuracy with ~90% of params activated**, +2.0% on BBH. p=0.1-0.2 collapsed quality (too few experts).

## 2. DynaExq (correct ID: arXiv:2511.15015, not 2410.10456)

The originally cited paper (2410.10456 = "Ada-K Routing") was **withdrawn** Oct 2024 (coauthor disagreement). The successor work, DynaExq (Chu et al., Nov 2025), targets **Qwen3-MoE-30B/80B** explicitly:
- Estimates long-horizon expert hotness from router traces, keeps "hot" experts at higher precision, demotes cold experts.
- **Qwen3-80B: accuracy 73.09% → 77.57%** under iso-VRAM-budget vs. static PTQ.
- Up to **2.73x throughput** vs. offload/prefetch baselines at batch 32.
- Mechanism is precision-adaptive, not k-adaptive — but it confirms the "few experts dominate traffic" hypothesis on Qwen3 specifically.

## 3. Per-Layer Behavior

From Huang et al. (Dynamic-k):
- **Lowest layers activate up to 4 experts/token**.
- **Topmost layer reduces to ~1 expert/token**.
- This monotone pattern (more diversity early, more specialization late) was the most robust per-layer finding in the literature.

EAC-MoE (ACL 2025, Long 633) confirms expert-importance is **layer-dependent**, and naive uniform-k reduction "must be carefully managed to avoid significant performance loss" — recommends per-layer thresholds.

## 4. Quality Regressions Reported

| Method | Model | k reduction | Quality delta |
|---|---|---|---|
| Adaptive-K (entropy) | Mixtral 8x7B | -52.5% FLOPs | ~0% (claim, no PPL given) |
| Adaptive-K | Qwen1.5-MoE | -32.4% | ~0% |
| Dynamic-k (top-p=0.4) | trained MoE | ~10% param reduction | **+0.7% acc** |
| Top-p=0.1-0.2 | trained MoE | aggressive | **major collapse** |
| Ada-K (claimed, withdrawn) | various | -25% FLOPs | "no degradation" |

No clean published PPL/MMLU/HumanEval table for Qwen3 specifically with adaptive-k yet — DynaExq uses precision adaptation, not k-adaptation.

## 5. Re-normalization Scheme

The literature is unanimous: **`p'[i] = p[i] / Σ_{j∈kept} p[j]`** is the standard. EAC-MoE applies this. Dynamic-k (Huang) applies it. Adaptive-K applies it. No competing scheme reached production-paper status. Without re-normalization, output magnitude shrinks proportionally to dropped mass, biasing residual-stream norms.

## 6. Qwen3-Specific Notes

- Qwen3-MoE excludes shared experts (unlike Qwen2.5/DeepSeek-V3); pure top-k routing.
- Qwen3 uses **global-batch load balancing** → router distributions are flatter at training time, meaning peakedness at inference is genuine signal, not an aux-loss artifact.
- DeepSeek-V3-style sigmoid+normalization scoring **diminishes the gap between competing experts** (less peaked). Qwen3 uses softmax → expected to be **more peaked** than DeepSeek-V3 → favorable for adaptive-k.

## What This Means for Phase 6 Spec

**Recommended design:**

1. **Use top-p (cumulative mass), not entropy.** Top-p is monotonic, single-threshold, cheaper to compute, and matches our "DDR4-bandwidth" optimization goal directly (skip experts whose weights aren't worth fetching).

2. **Threshold: start at p=0.9, sweep 0.85-0.95.** p=0.85 is the literature aggressive-but-safe operating point. Below 0.7 expect quality collapse.

3. **Floor: keep ≥2 experts per layer, ≥4 in first 25% of layers.** Justified by Huang et al. lowest-layer-activates-4 finding. Hard floor prevents catastrophic single-expert routing on early layers.

4. **Re-normalize unconditionally:** `p'[i] = p[i] / sum_kept`. Non-negotiable per all sources.

5. **Profile expected k on Qwen3-Next-80B-A3B first** (smallest target). Hypothesis to verify on a 1k-token calibration set:
   - Top-1 mass: 35-50% (256-expert fine-grained routing is flatter than Mixtral's 8-expert)
   - Top-2 mass: 60-70%
   - Top-4 mass: 85-90%
   - Top-8 mass: 98%+
   - At p=0.9, average k probably **3-5**, not 1-2 (fine-grained routing is less peaked than coarse Mixtral).

6. **Per-layer threshold table** (post-calibration), not a single global p. Late-layer p=0.85, early-layer p=0.95.

7. **Quality gate:** PPL on wikitext-2 + a coding eval (HumanEval). Ship if PPL delta < 1% and HumanEval delta < 1pp at the chosen p.

8. **Bandwidth math sanity check:** if calibration shows avg k=4 instead of 8 on 80B-A3B, that's **~50% less DDR4 traffic per token** — directly translates to TG speedup in the bandwidth-bound regime. This is the actual win, not FLOPs.

## Sources

- [DynaExq: Dynamic Expert Quantization for Scalable MoE Inference (arXiv:2511.15015)](https://arxiv.org/abs/2511.15015)
- [Mixtral of Experts (arXiv:2401.04088)](https://arxiv.org/abs/2401.04088)
- [Harder Tasks Need More Experts: Dynamic Routing in MoE (arXiv:2403.07652)](https://arxiv.org/html/2403.07652v1)
- [EAC-MoE: Expert-Selection Aware Compressor (ACL 2025)](https://aclanthology.org/2025.acl-long.633.pdf)
- [Adaptive-K MoE Routing whitepaper](https://adaptive-k.vercel.app/whitepaper.html)
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/html/2505.09388v1)
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/pdf/2412.19437)
- [OLMoE paper summary](https://ritvik19.medium.com/papers-explained-270-olmoe-38832ff4f9bd)
- Note: arXiv:2410.10456 (Ada-K Routing) was withdrawn by authors Oct 2024.
