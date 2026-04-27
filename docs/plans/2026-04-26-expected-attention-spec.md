# Expected Attention KV Eviction — Port Spec

**Reference:** Hooper et al., arXiv:2510.00636v1 (Oct 2025)
**Status:** SPEC ONLY — no implementation
**Date:** 2026-04-26

## 1. Paper Summary

Score each cached KV pair by *expected* attention mass future queries will allocate to it, under a Gaussian prior over hidden states. Closed-form via Gaussian MGF — no sampling, training-free.

### 1.1 Mathematical formulation

Hidden states Gaussian: h ~ N(mu, Sigma). Queries inherit through linear head proj + RoPE rotation R_t:

```
q_t = R_t · W_Q · h_t,  q_t ~ N(mu_q_t, Sigma_q_t)
mu_q_t    = R_t · W_Q · mu
Sigma_q_t = R_t · W_Q · Sigma · W_Q^T · R_t^T
```

Average over horizon T → q_bar ~ N(mu_bar_q, Sigma_bar_q).

**Per-key expected pre-softmax score** (Gaussian MGF):
```
z_hat_i  = exp( mu_bar_q^T k_i / sqrt(d) + k_i^T Sigma_bar_q k_i / (2d) )
a_hat_i  = z_hat_i / Sum_j z_hat_j
score_i  = (a_hat_i + epsilon) · ||v_i||      # epsilon ~ 0.02
```

Keep top-(1-r)·N by score. r = compression ratio.

### 1.2 Statistics estimation

mu, Sigma estimated **online** from cached hidden states. Triggers:
- Once at end of prefill
- Every 512 decoded tokens

### 1.3 Per-token cost at prefill

Total per compression event: O(L · H · n_cache · d^2). For Qwen3.6-35B-A3B at 200K ctx ~10 ms — acceptable as one-time prefill op.

### 1.4 Reported results

- RULER 4K/16K: beats SnapKV/TOVA/KeyDiff at 75% & 90% compression
- LongBench: Pareto-optimal vs accuracy
- Needle-in-Haystack: lossless to 125K
- 50% compression lossless, up to 15 GB savings at 120K ctx

### 1.5 Caveats

- No empirical study with quantized KV (orthogonal but unverified)
- Gaussian assumption breaks on heavy-tailed activation outliers
- No principled compression-ratio selector (manual r)

## 2. Integration Into llama-tq

### 2.1 File layout

| File | LOC | Status |
|------|-----|--------|
| `src/llama-kv-eviction.h` | ~80 | NEW — enum, struct, public API |
| `src/llama-kv-eviction.cpp` | ~450 | NEW — mu/Sigma estimator, scorer, top-k pruner |
| `src/llama-kv-cache.h` | +25 | MOD — eviction_policy, ea state member, evict_n() |
| `src/llama-kv-cache.cpp` | +180 | MOD — find_slot() overflow hook, prefill→decode flush hook |
| `src/llama-kv-cache-iswa.cpp` | +60 | MOD — SWA-aware: only score non-window tail |
| `src/llama-cparams.h` | +5 | MOD — eviction params |
| `include/llama.h` | +15 | MOD — public enum + setter |
| `common/arg.cpp` | +30 | MOD — CLI flags |
| `tools/server/server.cpp` | +10 | MOD — wire CLI |
| `tests/test-kv-eviction.cpp` | ~250 | NEW — unit tests |
| `docs/expected-attention.md` | ~120 | NEW |

**Total:** ~1225 LOC.

### 2.2 Eviction policy enum

```cpp
enum llama_kv_eviction_policy {
    LLAMA_KV_EVICT_FIFO              = 0,  // current behaviour
    LLAMA_KV_EVICT_EXPECTED_ATTENTION = 1,
};
```

### 2.3 Integration: find_slot overflow

In `llama_kv_cache::find_slot()`, before failure return:
```cpp
if (eviction_policy == LLAMA_KV_EVICT_EXPECTED_ATTENTION && ea) {
    const uint32_t n_evict = std::max(n_pad, (uint32_t)(cells.get_used() * eviction_ratio));
    ea->run_compression(lctx, cells, n_evict, tq_protect_sinks, n_swa);
    for (auto idx : ea->last_evicted) cells.rm(idx);
    return find_slot_inner(ubatch);
}
```

### 2.4 Trigger hooks

1. End-of-prefill flush — co-located with deferred-K/V conversion
2. Decode every-N tokens (default 512) — post-graph-compute
3. On context-overflow — §2.3

### 2.5 SWA interaction

In `llama-kv-cache-iswa.cpp`, restrict scorable region to indices with `pos < latest_pos - n_swa`.

## 3. Stack interaction with KTQ2_1 + VTQ2_2

### 3.1 K-side

- **Score on dequantized k.** With KTQ2_1 dequant @ ~22% decode cost, one-time prefill compression adds ~10ms on 200K — invisible.
- **Storage stays KTQ.** After top-k, evicted cells `pos[i] = -1`. KTQ blocks abandoned in place, reclaimed by next find_slot.

### 3.2 V-side

- **Score on ||v_i||.** Use deferred-V f16 staging buffer during prefill (line 142-147). Compute v_norm BEFORE bulk-Viterbi conversion. Store v_norm[i] in expected_attention_state.
- **Storage:** evicted V blocks abandoned identically.

### 3.3 Stackability summary

| Combination | Status |
|---|---|
| EA + KTQ2_1 | OK |
| EA + VTQ2_2 | OK iff v_norm captured during deferred-V f16 stage |
| EA + KTQ + VTQ asymmetric (current) | OK |
| EA + boundary-protected first 4 layers | OK |
| EA + tq_protect_sinks | Required — always force-keep first N |

**Net storage saving stacked:** ~12× effective KV reduction → 200K → effective 800K @ same VRAM.

## 4. Memory layout

```cpp
struct expected_attention_state {
    std::vector<std::vector<float>> mu;       // [L][d]
    std::vector<std::vector<float>> sigma;    // [L][d*d] or [L][d] if diag
    bool diag_only = true;
    std::vector<std::vector<float>> v_norm;   // [L][kv_size]
    std::vector<float> last_score;
    std::vector<uint32_t> last_evicted;
    int decode_count = 0;
};
```

**Footprint** at 200K ctx, L=64, d=128: ~52 MB. Trivial vs multi-GB KV.

## 5. CLI flags

```
--kv-eviction-policy {fifo,expected-attention}   default: fifo
--kv-eviction-ratio  FLOAT (0.0-0.9)             default: 0.5
--kv-eviction-trigger N                          default: 512
--kv-eviction-diag-cov                           default: on
```

## 6. Bench-gate

| Benchmark | Threshold |
|---|---|
| RULER NIAH-single (Qwen3.6-35B-A3B, KTQ2_1+VTQ2_2+EA r=0.75) | ≥ 95% recall up to 800K eff ctx |
| RULER NIAH-multi-key | ≥ 90% |
| RULER VT | ≥ 85% |
| LongBench-E avg | within -1.0pt of EA-off baseline |
| PPL WikiText-103 4K (Qwen3.5-0.8B Q8 smoke) | within +0.5% of FIFO |
| Prefill latency overhead at 200K | ≤ 50 ms |
| Decode TG impact at 200K | ≤ 2% |

**Ablation:** EA off / r=0.5 / r=0.75 / r=0.9.

## 7. Risks

1. **Gaussian × outlier activations.** MoE experts can produce heavy-tailed states → mitigate via clipping at 99.5th percentile.
2. **Sigma degenerate at short prefill.** Fall back to FIFO when n_prefill < 256.
3. **deferred-V flush timing.** v_norm captured BEFORE bulk-Viterbi → assert + regression test.
4. **seq_pos invariants.** Eviction in middle creates pos gaps → audit all seq_pos[s] callers.
5. **SWA double-eviction.** Clamp r_eff = max(0, r - r_swa_implicit).
6. **Compression-ratio not lossless.** Default r=0.0; explicit opt-in. Document recommended r per model class.
7. **Concurrency with --parallel 2 slots.** ea per llama_kv_cache (per-context).

## 8. Implementation order

1. Phase A (~300 LOC): Skeleton + FIFO fallback path. No-op behind flag.
2. Phase B (~400 LOC): Scorer with diag-Sigma, f16 K/V only. Validate on Qwen3.5-0.8B smoke.
3. Phase C (~200 LOC): KTQ + VTQ stacking, deferred-flush hook for v_norm.
4. Phase D (~300 LOC): SWA integration, full Sigma option, CLI, docs, tests.
5. Phase E: Bench-gate run on test-box (sequential, never parallel).

## Sources

- [Expected Attention paper (arXiv 2510.00636v1)](https://arxiv.org/html/2510.00636v1)
