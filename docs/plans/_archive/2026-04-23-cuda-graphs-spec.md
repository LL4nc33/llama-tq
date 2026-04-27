# CUDA Graphs — Decode-Path Integration Spec

**Date:** 2026-04-23
**Branch:** phase2
**Target:** Qwen3.5-35B-A3B-IQ2_XS on 2× RTX 2060 (sm_75)
**Motivation:** Agent A profiling: ~40% launch-bound (~80k launches/s × ~5 µs = ~400 ms/s overhead). Graphs replay = 1 submit per step.

---

## 1. Status — Upstream Already Has It

**CUDA Graphs are implemented and ON by default upstream.** We inherit it on `phase2`.

- Build flag: `ggml/CMakeLists.txt:215` — `option(GGML_CUDA_GRAPHS ... ${GGML_CUDA_GRAPHS_DEFAULT})` with `DEFAULT=ON` (`CMakeLists.txt:141`).
- Compile define: `ggml/src/ggml-cuda/CMakeLists.txt:176` adds `GGML_CUDA_USE_GRAPHS`.
- Core struct: `common.cuh:1170` `struct ggml_cuda_graph` (cudaGraph_t + cudaGraphExec_t + node_properties cache).
- Per-context map: `common.cuh:1362-1396` `cuda_graphs` keyed by first node ptr → supports **multiple graphs per context**.
- Entry point: `ggml-cuda.cu:4116` `ggml_backend_cuda_graph_compute` (dispatch hook in `.graph_compute` at line 4457).
- Runtime disable: env `GGML_CUDA_DISABLE_GRAPHS=1` (`common.cuh:1195`).
- **Warmup required:** capture only after 2 consecutive identical calls (`ggml-cuda.cu:4136-4155`). Properties change → reset warmup → re-execute directly.

**Go/No-Go state:** likely already active on our prod build if compiled with defaults. **First action: verify at runtime.**

---

## 2. Applicability to Qwen3.5-A3B (MoE + FA)

### 2a. MoE / MUL_MAT_ID — **supported with bounds**

The well-known "MoE kills graphs" issue is handled in `ggml_cuda_graph_check_compability` (`ggml-cuda.cu:3045-3058`). Tag `[TAG_MUL_MAT_ID_CUDA_GRAPHS]`, ref PR #18958:

```cpp
if (node->op == GGML_OP_MUL_MAT_ID) {
    const int mmvq_mmid_max = get_mmvq_mmid_max_batch(node->src[0]->type, cc);
    if (!ggml_is_quantized(node->src[0]->type) || node->ne[2] > mmvq_mmid_max) {
        use_cuda_graph = false; // mul_mat_id needs stream sync here
    }
}
```

For **IQ2_XS on Turing+**: `get_mmvq_mmid_max_batch_turing_plus` (`mmvq.cu:130`) falls to default `MMVQ_MAX_BATCH_SIZE = 8`. During single-token TG, `ne[2]=1` (one expert-selection slot per token) → **graphs enabled**. During prompt processing/parallel>1, `ne[2]>8` → graphs disabled (expected, PP is compute-bound anyway).

Data-dependent expert routing is NOT a capture-time problem here: MMVQ launches a single fused kernel per MUL_MAT_ID whose **arguments don't change** between decode steps (same quant, same ne[2]=1, same pointers). Routing happens *inside* the kernel via gate indices read from an input tensor — tensor pointers are stable, tensor data can change. Graph capture only records launch args, not tensor contents.

### 2b. FlashAttention — supported, but ctx-sensitive

FA kernel launch params depend on ctx length. Upstream handles this via `cuda_graph_update_required` (`ggml-cuda.cu:3072-3103`): each step compares `node->ne[]`, src data ptrs, strides against cached `node_props`. If FA's K/V extent grew → properties_changed → warmup reset → re-execute directly 1–2 steps, then re-capture.

**Cost:** each context-length step that changes FA tile count forces a re-capture. For steady-state TG the graph stabilises within 2 steps. Re-capture cost ≈ one non-graph step, amortised over next 100+ tokens.

### 2c. Split buffers — blocker if row-split active

`ggml-cuda.cu:3038`: split buffers disable capture. Our 2× RTX 2060 deployment uses **layer split** (`--split-mode layer`, default), not row-split → split buffer not instantiated on attention tensors → graphs OK. Confirm with `nvidia-smi` layer distribution; if `--split-mode row` is ever used for TG benchmarking, graphs are off.

### 2d. Multi-GPU layer split

With 2 CUDA backends (one per device), each backend gets its own `graph_compute` call, each captures its own graph. Inter-GPU transfers happen between backend calls → no issue inside a single capture.

---

## 3. Turing sm_75 Compatibility

- CUDA Graphs available since CUDA 10 / sm_70 → sm_75 fully supported.
- `cudaGraphExecUpdate` (used at `ggml-cuda.cu:3110`) available since CUDA 10.2 → fine.
- No sm_75-specific limitations in the upstream code path; the `disable_due_to_gpu_arch` flag (`common.cuh:1184`) is set elsewhere only for known-broken arches (not Turing).
- Conditional graphs (CUDA 12.4+) NOT used — upstream uses capture+update pattern instead, which works on CUDA 11+.

---

## 4. Integration Points (file:line)

| Concern | Location |
|---|---|
| Build flag | `ggml/CMakeLists.txt:215`, `CMakeLists.txt:141` |
| Compile define | `ggml/src/ggml-cuda/CMakeLists.txt:175-176` |
| Capture eligibility | `ggml/src/ggml-cuda/ggml-cuda.cu:3026-3066` `ggml_cuda_graph_check_compability` |
| Prop-change detection | `ggml/src/ggml-cuda/ggml-cuda.cu:3072-3103` `ggml_cuda_graph_update_required` |
| Exec update | `ggml/src/ggml-cuda/ggml-cuda.cu:3105-3130` `ggml_cuda_graph_update_executable` |
| Capture/replay driver | `ggml/src/ggml-cuda/ggml-cuda.cu:3547` `ggml_cuda_graph_evaluate_and_capture` |
| Backend entry | `ggml/src/ggml-cuda/ggml-cuda.cu:4116` `ggml_backend_cuda_graph_compute` |
| Per-ctx graph cache | `ggml/src/ggml-cuda/common.cuh:1362-1396` |
| Graph struct + is_enabled | `ggml/src/ggml-cuda/common.cuh:1170-1199` |

No changes at `llama_decode` / `llama_build_graph` / `ggml_backend_graph_compute` layer needed — the ggml-cuda backend intercepts transparently.

---

## 5. Phased Plan

### Phase 1 — Verify (effort: 1–2 h)

1. Check build: `strings build/bin/llama-server | grep GGML_CUDA_USE_GRAPHS` or inspect CMake cache.
2. Runtime: rebuild with `GGML_DEBUG=1` (or `-DCMAKE_BUILD_TYPE=RelWithDebInfo` + un-`#ifdef NDEBUG`) and watch for `"CUDA graph warmup complete"` log.
3. Quick A/B: same build, compare `llama-bench` TG with and without `GGML_CUDA_DISABLE_GRAPHS=1`.
   - Expected: unset = faster. If no diff → graphs already off at capture-check stage; inspect which compatibility rule fails.

### Phase 2 — Measure on our workload (effort: 2–4 h)

1. `llama-bench -m Qwen3.5-A3B-IQ2_XS.gguf -ctk tq2_1 -p 0 -n 128 -r 5` with / without `GGML_CUDA_DISABLE_GRAPHS=1`.
2. Production server: 200K ctx slot — repeat at ctx depths {4k, 32k, 128k, 200k} to see FA re-capture cost at length.
3. Instrument: add counter for `cuda_graph_update_required` transitions in `ggml_backend_cuda_graph_compute` to measure re-capture frequency.
4. **Success criterion:** ≥ 15% TG uplift → worth keeping. Target from A's 40% launch-bound estimate: 25–40% realistic ceiling (Amdahl: can't remove kernel runtime, only launch overhead).

### Phase 3 — Adapt only if Phase 2 underwhelms (effort: 1–3 d)

Only if Phase 2 shows graphs disabled for unexpected reasons, or re-capture thrashing:

- **MoE ne[2] threshold:** if we hit `ne[2] > 8` for small parallel batches, investigate relaxing the bound (PR #18958 TODO noted). Not likely for pure TG.
- **TQ kernel compatibility:** verify our TQ2_1/TQ3_1/TQ4_1 dequant + FA paths don't use features that would break graph capture (dynamic parallelism, host callbacks). Our kernels are plain `__global__` launches → should be fine.
- **FA length buckets:** if FA re-capture is the dominant cost, pre-bucket contexts (pow-of-2 tiles) so graph updates happen only at bucket boundaries.

---

## 6. Risks

1. **Already on, already priced in.** If graphs are already active in our prod build, A's measured 67.65 tok/s is *with* them. "40% launch-bound" then refers to residual overhead that CUDA Graphs can't eliminate (kernel entry itself, not just driver submission). Phase 1 A/B is essential before investing further.
2. **Re-capture thrash.** Frequent ctx changes / prompt re-injection / parallel-slot reallocation → warmup reset each time → graphs net-neutral or worse. Monitor via Phase 2 counter.
3. **TQ + graph interaction unverified.** Our TQ kernels use cuRAND Philox state (`TQ v5 Philox 6r`). If state is per-stream and not per-launch, capture+replay is safe. If captured with a specific RNG offset, replayed output is deterministic (acceptable — dequant is deterministic by design). **Low risk, but worth confirming** the Philox counter advances correctly under replay.
4. **Split-mode row** (some benchmarks) — graphs off. Document as a known-disable, not a bug.

---

## 7. Go / No-Go

**GO for Phase 1 (verify) — immediately, ~1 h.** Cheap, answers the most important question: are we already benefiting?

- If **Phase 1 shows graphs ON and active** → report to A, recalibrate launch-bound model, close this spec.
- If **Phase 1 shows graphs OFF** for environmental reasons (e.g. env var set, non-default build, row-split) → fix and re-benchmark. Expect a big win.
- If **Phase 1 ON but Phase 2 shows small uplift** → 40% estimate was pessimistic; move on to other bottlenecks (kernel fusion, persistent kernels).
- Only consider Phase 3 if Phase 2 identifies a specific disable reason we can relax.

**No new kernel work, no llama-layer changes. This is a configuration + measurement task first, optimisation second.**

---

## References

- Upstream PR #18958 (MUL_MAT_ID graph support)
- CUDA Graphs API: capture/update/launch (CUDA 10+, sm_70+)
- `get_mmvq_mmid_max_batch_turing_plus` table: `ggml/src/ggml-cuda/mmvq.cu:130`
- Env knob: `GGML_CUDA_DISABLE_GRAPHS=1`
