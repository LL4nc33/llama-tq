# Phase 6g — Profile-Guided Per-Expert Mixed Precision (PARKED, see Lever A first)

**Status:** PARKED 2026-04-27. Lever A (Phase 6f) takes priority.

This was researched 2026-04-27 alongside Lever A. Key finding: ggml does not
support mixed quant within a single physical tensor → Path A (full per-expert
split) is multi-month research, NOT a v1 feature.

Recommended order when Phase 6 work resumes:
1. Path C — `--tensor-type-file` regex per-layer override (no code change, ~1 week)
2. Path B — hot/cold expert bank split + 2-bank mmid dispatch (~2 weeks if Path C plateaus)
3. Path A — full per-expert split — only if Path B delivers and we want max compression

(Full spec preserved below for reference.)

---

# Phase 6g — Profile-Guided Per-Expert Mixed Precision (Static Prune Cold Experts)

**Date:** 2026-04-27
**Status:** Spec / Design Investigation
**Predecessor:** Phase 6a profiler

## 1. Motivation Recap

Profiler shows extreme cold-tail on Qwen3-MoE expert dispatch:

| Model | n_expert × n_layer | Cold (never top-8) | Top-10 dispatch |
|-------|--------------------|--------------------|-----------------|
| Qwen3.6-35B-A3B | 256 × 40 | 153/256 (60%) | 45.5% |
| Qwen3-Next-80B-A3B | 512 × 48 | 394/512 (77%) | 45.3% |

Adjacent-layer top-10 overlap = 0 → every layer has a private hot/cold partition. **Static** profile-guided per-expert quant is therefore a pure offline optimization with zero runtime cost — opposite to DynaExq, which re-quantizes at runtime.

**Target VRAM gains (estimated, Qwen3-Next-80B):**
- IQ2_XXS baseline: 26 GB → ~18 GB (-30%)
- TQ1_0 baseline: 20.5 GB → ~14-16 GB (cold experts dropped, hot stay TQ1)

## 2. Quant Pipeline — Where Decisions Land

### Entry points (verified)
- `tools/quantize/quantize.cpp:148-155` — `--tensor-type regex=type` and `--tensor-type-file`
- `src/llama-quant.cpp:184` — pattern compilation
- `src/llama-quant.cpp:686-698` — pattern application (regex against tensor name, replaces `new_type` on first match)
- `src/llama-quant.cpp:1247-1254` — 3D MoE quantize loop, reuses one `new_type` for all experts in a grouped tensor

**Blocker:** the 3D inner loop is per-expert-slice but applies the SAME quant type. There is no upstream mechanism for "expert 47 → IQ1_M while expert 48 → IQ2_XXS".

### Does ggml support mixed quant inside one physical tensor?
**No.** A `ggml_tensor` has exactly one `ggml_type`, one row size, one contiguous data block. Mixing quants would require new GGUF metadata, new tensor format, kernel rewrites.

## 3. Three Paths (recommendation: do Path C first)

### Path A — Full per-expert split via convert.py (DO NOT DO v1)
Split each `blk.N.ffn_*_exps.weight` into N individual 2D tensors. Maximum flexibility, but: GGUF format change, mmid kernel rewrite (pointer-array variant + per-slice type dispatch), every MoE arch needs the gather path. Multi-month effort. Rejected for v1.

### Path B — Hot/cold bank split (recommended for v2)
Permute experts within each layer's 3 banks (gate/up/down) so hot occupy low-index slots, cold high-index slots. Split into:
- `blk.N.ffn_*_exps_hot.weight` (shape `[d_ff, d_model, n_hot]`) → Q4_K or IQ3_XXS
- `blk.N.ffn_*_exps_cold.weight` (shape `[d_ff, d_model, n_cold]`) → IQ1_M

mmid kernel learns 2-bank dispatch: `if (eid < n_hot) bank = hot; else bank = cold;`. Router weights row-permuted to match. New GGUF KV `blk.N.ffn_expert_perm` (i32 array length n_expert).

Effort: ~2 weeks. Win: 25-30% VRAM reduction.

### Path C — Per-layer regex override only (recommended for v1, low-risk)
Use existing `--tensor-type-file` pipeline. Identify "fully cold" or "concentrated" LAYERS (rare; profiler shows layers 0-2 on 80B have especially flat dispatch). Downgrade those layers' expert banks to IQ1_M, raise hot mid-layers' attn_v/k to Q4_K.

Effort: ~1 week. Win: 5-15% VRAM reduction.

## 4. Implementation Phases

### Phase 6g.1 — Profiler aggregation (1 day)
- Phase 6f-1 profiler already dumps top-k expert IDs (DONE 2026-04-27).
- Extend `tools/profile-router.py --mode hotness` to also classify layers as "concentrated" vs "flat" by top-20 dispatch share.

### Phase 6g.2 — Path C baseline (2 days)
- Build per-layer `--tensor-type-file` from profiler JSON
- Quantize 35B + 80B with overrides
- Eval gate: wikitext-2 PPL Δ ≤ +1%, HumanEval pass@1 Δ ≤ -2pp, MMLU Δ ≤ -1pp
- **If Path C delivers ≥15% size reduction at gate-pass: ship and defer Path B.**

### Phase 6g.3 — Path B kernel work (1-2 weeks)
- convert_hf_to_gguf.py: emit hot/cold split + permuted router weights
- mmid kernel: 2-bank dispatch in CUDA + CPU + Metal
- Loader: gather both banks into `expert_bank_t { hot, cold, perm }` struct

### Phase 6g.4 — Eval + ship (3 days)

## 5. Decision Gate (ship 6g only if ALL hold on both 35B and 80B)

- wikitext-2 PPL: Δ ≤ +1.0%
- HumanEval pass@1: Δ ≥ -2 pp
- MMLU 5-shot: Δ ≥ -1 pp
- OOD probe set: no individual prompt regresses by >5% ROUGE-L vs baseline
- VRAM saved ≥ 20%

## 6. Concrete File Landing Points

- `src/llama-quant.cpp:184` — `tensor_type_patterns`
- `src/llama-quant.cpp:411-666` — `llama_tensor_get_type_impl` (Path B: handle `_exps_hot`/`_exps_cold`)
- `src/llama-quant.cpp:1247-1254` — 3D quantize loop (no change for Path B)
- `tools/quantize/quantize.cpp:148-155` — flag docs (add `--profile-guided-experts <profile.json>` for Path B)
- `convert_hf_to_gguf.py` — Qwen3MoE export; inject hot/cold split + perm key
- `ggml/src/ggml-cuda/mmid.cu` — 2-bank dispatch (Path B only)
- `ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_mul_mat_id` — CPU mirror

## 7. Honest Recommendation

**Do Lever A (Phase 6f hot-expert prefetch) first.** It's lower-risk and uses
the same profiler infrastructure. Phase 6g picks up after Phase 6f ships.

Within 6g: Path C first (free), only do Path B if it doesn't deliver.
