# Phase 6 — Adaptive top-k MoE routing

**Date:** 2026-04-27
**Status:** Spec, not implemented
**Goal:** +30-50% TG on CPU-offload MoE deployments by reading fewer expert weights per token when router confidence is high.

---

## Motivation

llama-tq currently runs Qwen3-Next-80B and Qwen3.5-122B with N≈20-29 expert-FFN layers offloaded to CPU RAM. TG is bandwidth-bound on DDR4-3200 (~40 GB/s real) — each token reads `8 active experts × layer-bytes` worth of weights from CPU per layer. That's ~7-15 MB per CPU-layer per token at IQ2_XXS.

Architectural fact: MoE routers emit a softmax over 256 experts and llama.cpp picks the top-8. But the softmax distribution is **typically very peaked** — top-1 expert often carries 60-80% of the total weight, top-2 95%+. Reading all 8 experts when 1-2 carry the load is wasted bandwidth.

**Insight:** Router confidence is a free signal. Use it to dynamically choose how many experts to actually fetch.

## Mathematical formulation

For each token at each MoE layer:

1. Router emits logits `r ∈ R^256` (one per expert).
2. Apply softmax: `p = softmax(r)`.
3. Sort descending, take cumulative sum: `P_k = sum(p[:k])` for k=1..8.
4. Pick smallest k such that `P_k ≥ threshold` (e.g. 0.85 — covers 85% of router mass).
5. Re-normalize the k chosen experts: `p'[i] = p[i] / P_k`.
6. Forward only those k experts.

Per-token expert count k ∈ [1, 8] varies. Static k=8 reads 8× weights; adaptive averages ~3-4× weights for typical text → **~2× bandwidth reduction → ~+50% TG ceiling under bandwidth-bound regime**.

## Quality trade-off

The truncation error per token equals `1 - P_k`. At threshold=0.85 we drop ≤15% of router mass. After re-normalization the kept experts span the same simplex, so the layer output is a slightly biased estimate of the true softmax-weighted sum.

Empirical priors (from auto-research 2026-04-27, see `LEGION/brainstorming-llama-tq/phase6-01-router-peakedness.md`):
- Adaptive-K (whitepaper) — Mixtral-8x7B: -52.5% FLOPs at ~0% accuracy loss. Qwen1.5-MoE: -32.4%. OLMoE: -24.7%.
- Dynamic-k MoE (Huang et al., arXiv:2403.07652) — top-p=0.4 yields avg k=1.72-1.87 vs static top-2, +0.7% accuracy. p≤0.2 collapses quality.
- DynaExq (arXiv:2511.15015) — Qwen3-80B: 73.09% → 77.57% accuracy under iso-VRAM via per-expert precision adaptation. Confirms "few experts dominate traffic" on Qwen3 specifically.
- EAC-MoE (ACL 2025) — expert importance is layer-dependent; uniform-k reduction risks regressions; per-layer thresholds recommended.
- Per-layer pattern (Huang): lowest layers want up to 4 experts, top layers concentrate on ~1.

**Note**: arXiv:2410.10456 (Ada-K) was withdrawn Oct 2024. Replaced reference with DynaExq.

Threshold sweep target: PPL ≤ +0.5% vs static top-8 at threshold ≥ 0.85.

Qwen3 routing prior: 256-expert fine-grained routing is **less peaked** than Mixtral's 8-expert. Calibration target on Qwen3-Next-80B-A3B (1k-token sample): top-1 mass 35-50%, top-2 60-70%, top-4 85-90%, top-8 98%+. At p=0.9 expect avg k ≈ 3-5 (not 1-2). Bandwidth math: avg k=4 instead of 8 → ~50% less DDR4 traffic per token in CPU-offload regime.

## Implementation phases

### Phase 6a — Router confidence profiler (research)

**Host: `llama-perplexity`, NOT `llama-server`** (server's multi-slot batching corrupts per-token attribution).

Tools:
- C++ extension: new `common/router-profile.{h,cpp}` + 3 flags in `common/arg.cpp` (`LLAMA_EXAMPLE_PERPLEXITY` group)
  - `--log-router-stats <path>`
  - `--router-stats-tau <float>` (default 0.85)
  - `--router-stats-max-tokens <N>` (default 4096, cap dump size)
- Python analyzer: `tools/profile-router.py` (mmap binary dump → NumPy histograms)

Mechanism:
- Attach `ggml_backend_sched_eval_callback` (existing infra in `examples/eval-callback/eval-callback.cpp:56-57`).
- Filter on tensor name regex `^ffn_moe_probs-(\d+)$`. **VERIFIED 2026-04-27**: `src/llama-graph.cpp:1327` calls `cb(probs, "ffn_moe_probs", il)`, and `llama_context::graph_get_cb()` (`src/llama-context.cpp:2417`) maps that to `ggml_format_name(cur, "%s-%d", name, il)` → tensors are named `ffn_moe_probs-0..N-1` already. No prerequisite PR needed.
- Binary record format: `{token_idx u32, layer_idx u16, n_expert u16, probs[n_expert] f32}`. ~10× faster than JSONL for 100M rows.

Python analyzer output:
- `mean_k = k_tau.mean()` — global decision gate.
- `mean_k_per_layer` — find pathological layers.
- `p99_k`, `max_k` — tail behavior.
- Histograms: `k_tau` (1..n_expert), top-1 prob distribution.
- Heatmap: layer × k_tau bucket.
- Exit code 0 if `mean_k < 5 AND p99_k < n_expert/2`, else 1.

Test corpora:
- wikitext-2 (`--chunks 64`, ~32k tokens) for general density.
- HumanEval/MBPP concat for code-routing behavior.
- (Optional) flattened oidanice/ source dump for target-domain routing.

Decision gate: if `mean k @ threshold=0.85` is **< 5** across all 35B/80B/122B, proceed. If ≥ 5, abort.

Files: see `LEGION/brainstorming-llama-tq/phase6-03-profiler-design.md`.

### Phase 6b — CPU-side adaptive dispatcher (build)

**Critical insight from codebase research** (`LEGION/brainstorming-llama-tq/phase6-02-cpu-dispatch-path.md`):
top-k is **graph-level**, not runtime. `selected_experts` and `weights` are first-class ggml tensors built in `src/llama-graph.cpp:1374` via `ggml_argsort_top_k(selection_probs, n_expert_used)`. There is no host-side decision point — k is fixed at graph construction time.

**Strategy: Static-max + Mask (v1)** — least invasive, preserves PR #14753 warmup-stability hack:
- Keep `ne[1] = max_k = 8` at graph level (no kernel signature changes).
- Insert new fused op `ggml_topk_dynamic_mask(probs, threshold, min_k, max_k)` between L1374 and L1387.
- Tail slots (k+1..8) get **weight = 0** after re-normalization on kept slots.
- Existing `mul_mat_id` kernels run unchanged but waste compute on zero-weight slots (acceptable for v1).
- Optional kernel-side optimization (v1.5): peek weight at slot start, early-skip if weight == 0.

**Strategy v2 (deferred)** — true dynamic k via new `ggml_mul_mat_id_dyn` taking `k_per_token` I32 tensor; ~6 kernels (CPU + CUDA + Metal + Vulkan).

Add CLI flags:
- `--moe-topk-mode {static,adaptive}` (default static for safety)
- `--moe-topk-threshold FLOAT` (default 0.85, range 0.5-1.0)
- `--moe-topk-min INT` (default 2, floor — literature says ≥1 collapses early-layer quality)
- `--moe-topk-min-early INT` (default 4, floor for first 25% of layers — Huang et al.)
- `--moe-topk-max INT` (default 8, ceiling)

Landing points:
- `src/llama-graph.cpp:1374` — replace `ggml_argsort_top_k` with dynamic variant.
- `src/llama-cparams.h` + `common/arg.cpp` — flag plumbing.
- `ggml/include/ggml.h` — new op declaration.
- `ggml/src/ggml-cpu/ops.cpp` — new `ggml_compute_forward_topk_dynamic_mask`.

### Phase 6c — CUDA-side adaptive dispatcher

Files: `ggml/src/ggml-cuda/argsort.cu` (new dynamic top-k op), `ggml/src/ggml-cuda/mmid.cu` (optional weight-0 early-skip).

Edge cases:
- Backward-compat: when `mode=static` or `threshold ≥ 1.0`, fall back to static top-max (no behavior change).
- Mixed CPU+GPU offload: the masking happens once at graph level; both backends consume the same masked weights tensor — no host-side broadcast needed.
- Aggregation loop at `src/llama-graph.cpp:L1591-1612` uses `hparams.n_expert_used` as upper bound (PR #14753 hack); padding-approach preserves this natively.

### Phase 6d — Validation

- PPL gate: wikitext-2 64-chunk PPL must be ≤ +0.5% vs static top-8 at threshold=0.85. Run on 35B / 80B / 122B.
- TG gate: bench TG must improve ≥ 20% on the 80B-IQ2_XXS CPU-offload deploy at ctx=2048 tg256.
- Iron-Law gate: same TG/PPL test on full-VRAM deploys (35B no-offload) — must show ≤ -1% TG (we don't want this to regress non-offload paths).

### Phase 6e — Per-layer threshold tuning (stretch)

Some layers are peakier than others (early layers often router-uniform, mid layers very peaked). Allow `--moe-topk-threshold` to be a vector indexed by layer, or a CSV file `layer_idx,threshold`.

Auto-calibrate from Phase 6a's profile data.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Router output peakedness varies wildly across model families | Phase 6a profiles each model individually; threshold is per-deploy |
| PPL regression on long-context (truncation error compounds) | Run a 32k-ctx eval, not just 512-ctx wikitext |
| MoE training was tuned to top-8 expansion — top-1 inference may degrade specific capabilities | Quality eval on MMLU/HumanEval/RULER, not just PPL |
| CUDA dispatcher latency from extra sort+cumsum eats the win | Profile with nsys; if >2% overhead, fold into existing top-k kernel |
| Multi-GPU split breaks: experts split GPU0/GPU1 may have different chosen k | Compute k on host, broadcast to both GPUs before dispatch |

## Estimated effort

- Phase 6a (profiler): 1 day
- Phase 6b (CPU dispatch): 2 days
- Phase 6c (CUDA dispatch): 2-3 days
- Phase 6d (validation sweep): 1 day
- Phase 6e (per-layer): +1 day stretch

Total: ~1 week of focused work.

## Decision gate

Run Phase 6a profiler first. If `mean k @ threshold=0.85` is ≥ 5 across all 35B/80B/122B layers, abort and pivot to a different optimization (Phase-4-stack squeeze, async PCIe prefetch overlap, or expert-locality L3 pinning).

If profile shows mean k ≤ 4, proceed with full implementation.

## References

- Fedus et al. 2022 — "Switch Transformers" (top-1 inference works)
- Lepikhin et al. 2020 — "GShard" (capacity-aware routing)
- Huang et al. 2024 — "Harder Tasks Need More Experts: Dynamic Routing in MoE" (arXiv:2403.07652) — top-p training, per-layer k pattern
- Chu et al. 2025 — "DynaExq" (arXiv:2511.15015) — Qwen3-80B per-expert precision adaptation
- Adaptive-K MoE Routing whitepaper — entropy-based gating, -52% Mixtral, -32% Qwen1.5-MoE
- EAC-MoE (ACL 2025 Long 633) — layer-dependent expert importance
- Our own `LEGION/2026-04-23_2340_distillery_122b-deploy-complete-results.md` — bandwidth ceiling analysis confirming this is the right lever
- Auto-research findings: `LEGION/brainstorming-llama-tq/phase6-{01,02,03}-*.md`
