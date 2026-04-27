# Phase 6a Profiler Design — Router Confidence Capture

## Executive Summary

Build a router confidence profiler with two parts: (1) a small C++ extension to `llama-perplexity` (NOT a new binary, NOT llama-server) that attaches a `ggml_backend_sched_eval_callback` and dumps post-softmax router probabilities to a binary file, and (2) `tools/profile-router.py` that parses the dump, builds histograms and computes the mean-k @ τ=0.85 decision gate.

Reusing `llama-perplexity` gets us free wikitext chunking, deterministic batching, and existing `--cache-type-*` plumbing — perfect for a profiling harness.

## 1. Existing Patterns Found

- **Callback infrastructure already wired**: `common_params::cb_eval` + `cb_eval_user_data` (see `examples/eval-callback/eval-callback.cpp:56-57`). `base_callback_data` in `common/debug.h:25` shows the regex-filter pattern — directly reusable.
- **PR1 head profiler precedent**: `common/tq-profile.h` + `tq-profile.cpp` already define a JSON-on-disk profile schema (`profile`/`layer_stat`/`head_stat`) consumed via `--tq-profile` / `resolve_v_types()`. Phase 6a should follow the *same convention*: capture in a tool, persist as JSON, consume from a Python analyzer. Do NOT invent a parallel format.
- **Tools pattern**: All `tools/*` are C++ binaries (`tools/perplexity/CMakeLists.txt:1-4`). No Python helpers in `tools/` today — `profile-router.py` would be the first; place under `tools/` is fine but `scripts/` is the more conventional spot. **Recommend `tools/profile-router.py`** to match user's spec.

## 2. C++ Change — Extend `llama-perplexity`

**Why perplexity not server**: server has multi-slot batching, async streaming, KV-reuse — all of which corrupt per-token-per-layer attribution. Perplexity is single-stream, deterministic, already iterates wikitext chunks.

**New flag** (in `common/arg.cpp`, `LLAMA_EXAMPLE_PERPLEXITY` group):
```
--log-router-stats <path>     # binary dump path
--router-stats-tau <float>    # default 0.85
--router-stats-max-tokens <N> # default 4096 (cap I/O)
```

**Callback** (new file `common/router-profile.{h,cpp}`):
```cpp
struct router_capture {
    std::FILE * fp;
    int n_layer;
    int64_t token_idx = 0;
    float tau;
};
bool router_cb_eval(ggml_tensor * t, bool ask, void * ud);
```
The callback filters on tensor name regex `^ffn_moe_probs-(\d+)$` (post-softmax router weights — confirmed standard naming in `llama-graph.cpp` MoE build path; verify against `build_moe_ffn`). On `ask=true` return true only for matching nodes; on `ask=false`, copy `ggml_nbytes(t)` from device via `ggml_backend_tensor_get`, write a fixed record per token-row.

**Binary record** (little-endian, fixed-width — 10× faster than JSONL for ~100M rows):
```
struct router_record {
    uint32_t token_idx;
    uint16_t layer_idx;
    uint16_t n_expert;
    float    probs[n_expert];   // post-softmax, sums to ~1
};
```
Write a 32-byte header once: magic `"TQRP"`, version `1`, `n_layer`, `n_expert`, `tau`, model-hash. ~100 KB per token for a 128-expert model, ~4 GB for the full wikitext-2 test set — keep `--router-stats-max-tokens` cap.

## 3. Python Analyzer — `tools/profile-router.py`

```
usage: profile-router.py <dump.bin> [--tau 0.85] [--out report.json]
                                    [--plot-dir plots/]
```

Steps:
1. **Parse header**, mmap body, reshape to `(n_token, n_layer, n_expert)` via NumPy.
2. **Sort probs desc** along expert axis (`np.sort(...,axis=-1)[...,::-1]`).
3. **Compute k@τ per (token,layer)**: `k_tau = np.argmax(np.cumsum(sorted_probs, axis=-1) >= tau, axis=-1) + 1`.
4. **Aggregate**:
   - `mean_k = k_tau.mean()` — global decision-gate metric.
   - `mean_k_per_layer = k_tau.mean(axis=0)` — find pathological layers.
   - `p99_k`, `max_k` — tail behavior.
   - Histogram of `k_tau` (bins 1..n_expert).
   - Histogram of top-1 prob (confidence distribution).
5. **Decision gate** (exit code): `0` if `mean_k < 5` AND `p99_k < n_expert/2`, else `1`. Print verdict line:
   `GATE PASS: mean_k=3.2  p99_k=7  (threshold mean_k<5)`
6. **Plots** (matplotlib, optional): per-layer mean_k bar chart, top1-prob histogram, k_tau heatmap (layer × bucket).

JSON report mirrors the PR1 profile schema — extend `tq-profile.h::layer_stat` with optional `mean_k_tau` so the Phase 6 router-aware quantizer can consume it via the existing `load_profile()` path.

## 4. Test Corpora

- **Density**: existing `wikitext-2-raw/wiki.test.raw` (already used by perplexity). Run `--chunks 64` for ~32k tokens, plenty for histogram convergence.
- **Agentic / code**: no in-repo corpus. Recommend pulling `humaneval` prompts (164 prompts × ~200 tokens) or `mbpp` test split — concat to a flat `.txt` and feed via `-f`. Long-form: a flattened `oidanice/` source dump (~500k tokens) tests router behavior on the *target* deployment domain.

## 5. Files to Touch

- NEW: `common/router-profile.h`, `common/router-profile.cpp`
- EDIT: `common/arg.cpp` (3 flags under `LLAMA_EXAMPLE_PERPLEXITY`)
- EDIT: `common/common.h` (add `std::string router_stats_path; float router_stats_tau; int router_stats_max_tokens;` to `common_params`)
- EDIT: `tools/perplexity/perplexity.cpp:2010-2050` (set `params.cb_eval` before `common_init_from_params` if path is non-empty)
- EDIT: `tools/perplexity/CMakeLists.txt` (link new common sources — already inherited via `common` lib if added there)
- NEW: `tools/profile-router.py`

## 6. Verification

- **Tensor-name verification needed before coding**: grep `build_moe_ffn` in `src/llama-graph.cpp` for the exact `ggml_set_name` of the post-softmax tensor. If it's an unnamed intermediate, add `ggml_set_name(probs, format("ffn_moe_probs-%d", il))` to the MoE builder — one-line change, harmless to upstream.
- **Smoke**: `qwen3.5-0.8b-q8_0.gguf` is dense (no MoE) — useless. Use the smallest available MoE: Qwen3.X-A3B IQ2 on gpu00 with `--chunks 4 -c 512 --router-stats-max-tokens 512`. Expected runtime <30 s.

## Risks

- Router tensor not named in upstream graph builder → must add `ggml_set_name`, otherwise regex filter never fires.
- 128-expert × 4096-token dumps are ~2 GB — enforce `max_tokens` default.
- `ggml_backend_tensor_get` is sync per call; profile may slow inference 2-5×. Acceptable for offline run, NOT for server.

Relevant files:
- `/mnt/d/repos/llama-tq/common/debug.h`
- `/mnt/d/repos/llama-tq/examples/eval-callback/eval-callback.cpp`
- `/mnt/d/repos/llama-tq/common/tq-profile.h`
- `/mnt/d/repos/llama-tq/common/tq-profile.cpp`
- `/mnt/d/repos/llama-tq/tools/perplexity/perplexity.cpp`
- `/mnt/d/repos/llama-tq/common/arg.cpp`
- `/mnt/d/repos/llama-tq/src/llama-graph.cpp` (verify MoE tensor naming before coding)
