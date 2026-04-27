# Phase 6 — Innovator Synthesis

**Date:** 2026-04-27
**Inputs:** phase6-01 (literature), phase6-02 (codebase), phase6-03 (profiler), spec 2026-04-27-phase6.
**Purpose:** Surface conflicts, propose creative additions, rank risks, recommend sequencing.

---

## 1. Open Conflicts & Risks Between Research Streams

### 1.1 Peakedness assumption vs. Qwen3 reality (literature ↔ spec)
The motivation paragraph in the spec asserts "top-1 expert often carries 60-80%, top-2 95%+." That's **Mixtral 8-expert numbers**. Phase6-01 explicitly corrects this: 256-expert Qwen3 fine-grained routing is *flatter*, expected top-1 mass 35-50%, top-2 60-70%. The spec's headline "+50% TG" estimate (assuming avg k≈3-4) is plausible for the bandwidth-bound regime but **only if Qwen3 actually peaks**. If profiler returns mean_k≈5.5, the bandwidth win shrinks to ~30% and the implementation cost-benefit changes. **Action:** make the decision gate (Phase 6a → 6b) bidirectional — also abort if mean_k < 1.8 (suspiciously peaked → likely a profiler bug, not a free win).

### 1.2 Tensor naming gap (profiler ↔ codebase)
Phase6-03 assumes a `ffn_moe_probs-<il>` tensor name to regex-filter on. Phase6-02 enumerates `ffn_moe_logits` (L1296) and `probs` (L1311-1326) but **does not confirm a `ggml_set_name` is set on `probs`**. If absent, the callback never fires. Mitigation listed in 6-03 ("add one-line set_name") is correct but **must be done as a prerequisite PR**, not bundled into the profiler PR — it touches the hot graph path and deserves its own bench-gate.

### 1.3 Padding-mask wastes mmid compute (codebase ↔ spec)
Spec 6b chooses Strategy A (static-max + zero-weight mask) for v1, accepting that mmid kernels iterate all 8 slots and "waste compute on zero-weight slots." This is a real concern on **GPU full-VRAM deploys** where TG is *already* compute-bound on the dequant→FMA path, not bandwidth-bound. The spec's Iron-Law gate ("≤ -1% TG on 35B no-offload") may fail because the win (bandwidth) doesn't apply but the cost (zero-weight FMAs) does. Even an early-skip on weight==0 inside mmvq has a branch-predict cost. **Action:** add a CLI gate `--moe-topk-mode adaptive` defaults to **off**; recommend per-deploy enable. Don't ship adaptive as default.

### 1.4 Aggregation loop hazard (codebase finding, under-emphasized in spec)
Phase6-02 L1591-1612: the aggregation loop uses `hparams.n_expert_used` (the **hardcoded** value) as the upper bound, due to PR #14753 warmup-stability. Strategy A preserves this natively, but Strategy v2 (true dynamic k) would silently break warmup stability. **Action:** v2 must explicitly carry forward the warmup constant — not a future drop-in upgrade.

### 1.5 Multi-GPU split (mentioned but unsolved)
Spec risks-table line: "experts split GPU0/GPU1 may have different chosen k → compute k on host, broadcast." But profiler design and dispatcher design both assume single-stream graph-level k. Host-side k computation **breaks** the "everything stays on-device" property phase6-02 highlighted. There's no concrete design for the multi-GPU case. On gpu00 (asymmetric 2× RTX 2060 with x4 secondary), this is the actual deploy target.

---

## 2. Creative Additions Beyond the Spec

### 2.1 Predicted-k linear head (high creativity, medium feasibility)
Train a tiny `Linear(d_model → 1)` regressor offline against profiler-collected (input_embedding → optimal_k). At inference, run predictor *before* the router, branch on predicted k, skip softmax+sort entirely when prediction is high-confidence. **Caveat:** softmax+sort over 256 experts is ~10µs on GPU; the saving is real on CPU-offload but marginal on full-VRAM. **Verdict:** defer to v3 — only viable if profiler shows strong predictability (R² > 0.7).

### 2.2 Token-bucket k-cache (low creativity, high risk)
Hash input embedding → cache k-decision. **Reject:** embedding hashes have huge entropy (no repeat tokens at the residual-stream level after layer 1), and any KV-cache-style memoization breaks determinism guarantees. Not worth the bug surface.

### 2.3 Expert-locality L3 prefetch (high creativity, high payoff for CPU offload)
Profiler already collects per-token expert IDs. Aggregate per-document: which expert pairs co-fire? Use Markov chain on expert-id transitions to issue `__builtin_prefetch` on layer N+1's predicted hot experts during layer N's mmid. **DDR4 latency hide on CPU offload — could be a bigger win than adaptive k itself.** Costs nothing PPL-wise (it's pure prefetch). **Action:** add as Phase 6f (post-validation extension).

### 2.4 DynaExq integration: hot-experts at higher precision (high payoff, code-heavy)
Combine adaptive-k (skip cold experts) with DynaExq-style per-expert quant precision (hot experts at Q5/Q6, cold at IQ2/Q2). Profiler trace gives us hotness for free. Storage stays roughly iso-VRAM, but accuracy goes up by ~4pp per the DynaExq paper. **Verdict:** This is the actual phase 7 idea. Add as a stretch goal in the spec's "Future" section. Requires per-expert tensor metadata in GGUF — not trivial.

### 2.5 Layer-skip for low-entropy tokens (high creativity, medium feasibility)
If router entropy on layer L is below ε for a token, **skip layer L's MoE FFN entirely** (residual passes through). Justified by "depth-adaptive transformers" literature (CALM, DeeBERT). Saves whole-layer bandwidth not just expert count. **Risk:** changes layer-norm statistics downstream; needs re-eval. **Verdict:** parking-lot — too aggressive for v1.

### 2.6 Per-token weight quantization on the fly (creative but bad)
"At adaptive k=1, dequant the kept expert at higher precision since we have bandwidth budget." **Reject:** the weights are static on disk; runtime re-quant is a non-starter for GGUF.

### 2.7 Profile-guided gating loss bias (research-y)
Use profile data to identify experts that are *systematically* in tail positions and bias them out at quant time (set their gate weight to 0 in the GGUF). Static pruning informed by the same profiler. **Verdict:** nice — call this Phase 6g, can ride on Phase 6a profiler infrastructure with zero extra runtime cost.

---

## 3. Risk Ranking — Single Biggest Threat

**Ranked by (probability × impact):**

1. **🔴 mmid kernel slowdown on full-VRAM deploys** (Section 1.3). Probability HIGH — masking always pays the FMA cost. Impact HIGH — fork's prime directive is "no regressions" (`feedback_fork_only_improves`). Mitigation: keep adaptive-mode opt-in; add bench gate on 35B full-VRAM before merge.

2. **🟠 Profiler not converging on representative corpus.** Probability MEDIUM. The 32k-token wikitext set plus HumanEval is OK for baseline but Qwen3 deploys serve agentic + DE/EN code + long-context (200k!). A profile fitted to wikitext can mispredict k for the actual workload. Mitigation: also run profile on flattened `oidanice/` source dump as 6-03 already suggests, AND on a 32k-ctx slice — long-context routing peakedness differs from 512-ctx.

3. **🟠 Multi-GPU split disagreement** (Section 1.5). Probability MEDIUM (gpu00 is the deploy target). Impact HIGH (silent quality regression hard to detect). Mitigation: design the dynamic-mask op to be per-token-deterministic regardless of which GPU computes the router (router runs on GPU0, mask broadcast via existing tensor split).

4. **🟡 PPL regression on long-context.** Probability MEDIUM-LOW (truncation error compounds across 200k tokens). The literature p=0.85 numbers are 2k-ctx benchmarks. Mitigation: 32k+ PPL gate explicit.

5. **🟡 Tensor naming missing.** Probability LOW (one-line fix). Impact MEDIUM (blocks 6a entirely).

**Single biggest threat: #1 — mmid slowdown on full-VRAM.** This is the one that violates fork policy.

---

## 4. Recommended Sequencing Tweaks

The spec sequences as 6a → 6b → 6c → 6d. Recommend this revision:

### MVP: 6a + 6b-lite (combined, ~3 days)
- **Day 1:** Land the `ggml_set_name(probs, "ffn_moe_probs-<il>")` one-liner as its own PR with a smoke bench (35B PPL must be bit-identical).
- **Day 2:** Ship the profiler (Phase 6a) end-to-end on Qwen3.X-A3B small first, then 80B.
- **Day 3:** Decision gate. Look at *real* numbers before committing to the mask op.

### Decision branch
- **If mean_k @ τ=0.85 ∈ [2.5, 4.5] across 80B and 122B:** proceed to 6b (full implementation, ~2 days).
- **If mean_k > 5:** abort adaptive-k. Pivot to **expert-locality prefetch** (2.3) — same profiler data, different optimization.
- **If mean_k < 2:** suspicious; verify profiler before celebrating.

### Phase 6b sequencing change (the actual tweak)
Spec proposes: "build dynamic-mask op + plumb everything." Recommend instead:

1. **6b-1 (1 day):** add `--moe-topk-threshold` flag wired to a *static k override* that just clamps `n_expert_used` to e.g. 4. No new op, no graph rewrite. **This isolates "is the win bandwidth?" from "is the mask op correct?"**. If 6b-1 shows +30% TG at static k=4 with acceptable PPL, we have proof-of-bandwidth before any kernel work. If it doesn't, then adaptive-k (which is at best a smarter version of "use fewer experts on average") won't help either.

2. **6b-2 (2 days):** add the per-token mask op (the real spec content). Ship behind opt-in flag.

3. **6b-3 (1 day):** validation matrix. PPL + TG on 35B/80B/122B, both full-VRAM and offload.

This gives a **two-stage value delivery**: the static-k clamp alone may already be a +30% TG ship-it for offload deploys, and is reversible in 5 minutes. The adaptive op is the polish on top.

---

## Recommended Top Pick

**MVP path: Profiler (6a) + Static-k-clamp (6b-1) shipped together.** Two days of work, validates the entire bandwidth thesis, gives an immediate offload-deploy win, and de-risks the bigger mask-op investment. If the static clamp alone shows +30% TG with PPL within budget at k=4, ship it as a feature, then layer adaptive-k on top in a second PR. This honors `feedback_fork_only_improves` (each merge passes its own bench gate) and `feedback_build_batching` (don't burn 15-40min CUDA builds chasing op designs before knowing the bandwidth math holds).

**Alternative if profiler results disappoint:** pivot the same profiler infrastructure to expert-locality prefetch (2.3) — same calibration data, different lever, likely larger absolute win on DDR4-bound 80B/122B deploys.
