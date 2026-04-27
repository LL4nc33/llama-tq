# Phase 6 Brainstorm Summary — 2026-04-27

## ⚠️ STATUS UPDATE 2026-04-27 14:55

**Phase 6 (adaptive top-k) ABORTED.** Profiler ran on Qwen3.6-35B and Qwen3-Next-80B,
both miss the decision gate by ~30×: mean_k = 145.7 / 172.2 (gate < 5).

Root cause: Qwen3 uses `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT` — softmax
only on post-top-k weights, not on full 256/512-expert distribution. The
distribution is intentionally near-uniform; threshold-based k-reduction is
meaningless.

See `phase6-05-real-data-findings.md` for full analysis + pivot recommendations
(expert-locality L3 prefetch is the most-promising follow-up using the same
profiler infrastructure).

The Phase 6a profiler infrastructure ships as a reusable tool. Spec marked
ABORTED. Original brainstorm content below preserved for context.

---


## Kontext

Phase 6 = Adaptive top-k MoE routing. Ziel: +30-50% TG auf CPU-offload Deploys (80B/122B) durch Reduktion gelesener Expert-Weights pro Token, gesteuert über Router-Confidence-Threshold.

Vier Research-Streams synthetisiert (`phase6-01..04`).

## Must-Do (kritisch, in Reihenfolge)

### 1. ~~`ggml_set_name` one-liner PR~~ **NICHT NÖTIG**
- Verified 2026-04-27: `src/llama-graph.cpp:1327` ruft `cb(probs, "ffn_moe_probs", il)`, der callback in `src/llama-context.cpp:2417` mappt auf `ggml_format_name`. Tensors heißen schon `ffn_moe_probs-N`.
- Profiler-Regex funktioniert sofort.

### 2. **MVP — Profiler (6a) + Static-k-Clamp (6b-1) als Bundle**
- **6a Profiler** (~2 Tage): C++ extension von `llama-perplexity` mit `--log-router-stats`, plus Python `tools/profile-router.py` (mmap binary → NumPy histograms).
- **6b-1 Static-k-Clamp** (~1 Tag): nur ein CLI flag der `n_expert_used` clampt (z.B. zu 4). Keine neue ggml-op, keine Graph-Rewrite.
- Validation auf Qwen3.X-A3B (smoke), dann 80B-IQ2_XXS auf gpu00 mit CPU offload.

**Decision Gate (Profiler):**
- mean_k @ τ=0.85 ∈ [2.5, 4.5] → proceed to 6b-2 (full mask op)
- mean_k > 5 → abort adaptive-k, pivot to expert-locality prefetch (Idea 2.3)
- mean_k < 2 → suspicious, verify profiler

**Decision Gate (Static-k-clamp):**
- TG +30%@k=4 with PPL ≤ +0.5% → ship as standalone feature, layer adaptive on top later
- TG ≤ +10% → bandwidth-thesis falsch, abort phase 6

## Should-Do (high-impact follow-ups)

### 3. **Phase 6b-2: Per-token mask op** (nach MVP-Validation)
- `ggml_topk_dynamic_mask(probs, threshold, min_k, max_k)` zwischen `llama-graph.cpp:1374` und `:1387`.
- Static-max + zero-weight mask (Strategy A — preserves PR #14753 warmup hack).
- Behind opt-in flag `--moe-topk-mode adaptive` (default static).

### 4. **Phase 6f: Expert-locality L3 prefetch** (Innovator-Idea 2.3)
- Profiler-Daten (expert-id traces) als Markov-Chain → `__builtin_prefetch` auf next-layer hot experts während current-layer mmid.
- DDR4 latency hide auf CPU offload — könnte größerer Win sein als adaptive-k selbst.
- Zero PPL cost (pure prefetch).

### 5. **32k+ ctx PPL gate**
- Spec listet ≤ +0.5% wikitext-2 (512-ctx). Aber langer Context lässt truncation-error compound.
- Run RULER 32k oder long-form PPL bevor merge.

## Konflikte / Risk-Ranking

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| 🔴1 | mmid kernel slowdown auf full-VRAM Deploys (Strategy A pays FMA cost) | HIGH | HIGH | adaptive=opt-in, bench-gate auf 35B no-offload before merge |
| 🟠2 | Profiler nicht repräsentativ (wikitext != agentic 200k) | MEDIUM | HIGH | Auch oidanice/ flat dump + 32k-ctx slice |
| 🟠3 | Multi-GPU split disagreement (gpu00 = 2× RTX 2060) | MEDIUM | HIGH | Mask-op per-token deterministic, broadcast via existing tensor split |
| 🟡4 | PPL regression long-context | MEDIUM-LOW | MEDIUM | 32k+ PPL gate explicit |
| 🟢5 | ~~Tensor naming missing~~ | RESOLVED | - | Already named via cb→ggml_format_name |

## Konflikte zwischen Streams

- **Peakedness assumption**: Spec headline (60-80% top-1) basiert auf Mixtral 8-expert. Qwen3 256-expert ist flatter (top-1 35-50%). Decision-gate jetzt bidirektional (auch abort wenn mean_k < 1.8 = profiler bug).
- **Padding-mask wastes mmid**: Strategy A pays FMA cost auf full-VRAM. Iron-Law gate (-1% TG no-offload) könnte fail. → opt-in default off.
- **Aggregation loop hazard**: PR #14753 warmup-stability hack. Strategy A preserves natively, v2 würde silently break.

## Nice-to-Have (parking lot)

- **2.1 Predicted-k linear head**: nur viable wenn profiler R² > 0.7 zeigt
- **2.4 DynaExq integration** (hot experts higher precision): das ist eigentlich Phase 7
- **2.5 Layer-skip for low-entropy tokens**: zu aggressiv für v1
- **2.7 Profile-guided gating bias** (static prune tail experts at quant time): Phase 6g

## Rejected

- **2.2 Token-bucket k-cache**: huge entropy, breaks determinism
- **2.6 Per-token weight requant**: GGUF non-starter

## Empfohlener Top-Pick

**MVP: Profiler (6a) + Static-k-Clamp (6b-1) zusammen, ~3 Tage.**

Validates bandwidth-thesis, gives immediate offload-deploy win, de-risks bigger mask-op investment. Honors `feedback_fork_only_improves` (each merge passes own bench gate) und `feedback_build_batching` (don't burn CUDA builds chasing op designs before knowing math holds).

**Alternative wenn Profiler enttäuscht:** Pivot dieselbe Profiler-Infrastructure auf Expert-locality prefetch (2.3). Same calibration data, different lever, likely larger absolute win auf DDR4-bound 80B/122B.

## Next Action

Implement Phase 6a Profiler (C++ extension + Python analyzer), measure on Qwen3.X-A3B + Qwen3-Next-80B, decide based on real numbers.

## Files

- Spec: `/mnt/d/repos/llama-tq/docs/plans/2026-04-27-phase6-adaptive-topk-moe.md`
- Research: `phase6-01-router-peakedness.md`, `phase6-02-cpu-dispatch-path.md`, `phase6-03-profiler-design.md`
- Synthesis: `phase6-04-innovator-synthesis.md`
