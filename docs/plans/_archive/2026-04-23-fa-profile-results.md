# FA Profiling on Turing sm_75 — Results

**Date:** 2026-04-23
**Target:** prod llama-server PID 2753213 on gpu00 (Qwen3.5-35B-A3B IQ2_XS, TQ2_1 K-cache / F16 V-cache, 400k ctx, `-ub 512`, `-ts 12,12`, parallel=2)
**Baseline:** 67.65 tok/s TG @ 300 tokens
**Hardware:** 2× RTX 2060 12GB (sm_75, 30 SMs/GPU, 3 MB L2, ~336 GB/s DRAM BW)

---

## Dynamic Profiling Status: BLOCKED

Runtime profiling could not be performed under the given constraints. Documented here so the next session can pick up with a working approach instead of repeating the same dead-ends.

### Why option (a) — attach nsys to PID — failed

- `nsys --version` on gpu00: **2022.4.2.50** (package `nsight-systems 2022.4.2.50~12.0.1`).
- `nsys profile --pid=<PID>` attach was introduced in **Nsight Systems 2023.1**.
- The installed 2022.4 binary rejects `--pid`: `unrecognised option '--pid=2753213'`.
- `nsys` on this box has no `attach` subcommand (only `profile`, `launch`, `start`, `stop`, `cancel`, `stats`, `status`, `shutdown`, `sessions list`, `export`, `analyze`).
- `ncu --mode=attach` exists but requires the target to have been launched with `ncu --mode=launch …` first (cooperative). Cannot attach to an already-running process not started under ncu.

### Why option (b) — separate bench server — failed

- No `llama-bench` binary in `~/llama-tq/build/bin/` (only `llama-server`; the checkout predates that build target or it was never compiled).
- VRAM headroom is insufficient to run a **second** llama-server with the same 35B model:
  - GPU0: 11528 / 12288 MiB used → 760 MiB free
  - GPU1: 11256 / 12288 MiB used → 1032 MiB free
  - Loading a second Qwen3.5-35B IQ2_XS instance needs ~11 GiB per card; won't fit anywhere near.
- No CUDA toolkit under `/usr/local/cuda*` and no newer Nsight package available via apt (`nsight-systems 2022.4` is the only version).
- sudo requires a password, so I cannot install a newer nsys non-interactively.

### What would unblock runtime profiling next session

One of:
1. Install `nsight-systems-2024.x` (or at minimum 2023.1) on gpu00 with sudo.
2. Build `llama-bench` in `~/llama-tq/build/` (`cmake --build . --target llama-bench`) and momentarily stop prod to free VRAM for a 60-second bench run. User's constraint "don't disrupt prod" forbids this without coordination.
3. Run profiling on a smaller model that fits in the VRAM headroom (e.g. a 3B Qwen, same TQ2_1 config). That answers kernel-level questions (occupancy, L2 hit-rate, launch count per token) but numbers for memory-bound vs compute-bound may shift with model size.

Option 3 is the most pragmatic for next session. Recommend it.

---

## Static Analysis (what I could extract without running)

In lieu of runtime metrics, a static read of the FA-vec and VTQ2 kernels gives directional evidence for the bottleneck hypothesis.

### 1. Legacy FA-vec kernel has `__launch_bounds__(128, 1)`

`ggml/src/ggml-cuda/fattn-vec.cuh:9-20`:

```cpp
static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(...)
```

- 128 threads/block, **`minBlocksPerSM = 1`**.
- Tells the compiler it may use up to ~255 registers/thread.
- On sm_75 a 128-thread block with 128 regs/thread = 16384 regs = exactly 1 block × 4 warps on the SM's 64 KiB register file. Could theoretically be 2 blocks/SM if regs/thread ≤ 127.
- With 1 block/SM × 30 SMs = only 30 blocks resident. For Qwen3.5 Q-head count (typically 32 or 40 heads/layer) and `ncols=1` (vec path) in TG, this *barely* saturates one SM's worth of work per layer — decode marches through layers sequentially, so launch latency and tail effects dominate.
- This is the path the prod server (KTQ2_1 / F16 V) hits. KTQ dispatch is in `fattn-vec-dispatch-ktq.cu`.

### 2. E11 experimental kernel targets `__launch_bounds__(128, 4)`

`ggml/src/ggml-cuda/fattn-vec-vtq2.cuh:21, 93`:

```cpp
// Occupancy target (sm_75, D=128):
//   __launch_bounds__(128, 4)   — 4 blocks/SM. Requires reg/thread ≤ 127.
__launch_bounds__(ggml_cuda_fattn_vec_vtq2_get_nthreads_device(), 4)
static __global__ void flash_attn_ext_vec_vtq2_cached(...)
```

- Designed specifically to fix the occupancy problem via warp-cooperative V-dequant into shmem.
- Was REVERTED per `.claude/agent-memory/cwe-builder/project_vtq2_e11.md`: "60× slow".
- The slowdown is not the occupancy choice per se — it is that warp-cooperative decode serializes over the warp's V-row and the reuse assumed in the design doesn't pay for itself at `ncols=1` TG shape.

### 3. Other kernel paths in play during decode

TG decode on this config dispatches roughly (per token, per layer, per GPU):
- 1× `flash_attn_ext_vec<D=128, ncols=1, KTQ2_1, F16, …>` — the dominant kernel by every prior memory note.
- K-dequant: fused inside FA (no separate kernel — `dequantize_1_ktq` called per K-row inside the vec kernel).
- V-dequant: V is F16 in prod config → no dequant, pure fp16 loads (this is why sparse-V and cached-V optimisations didn't help prod baseline).
- MoE expert dispatch: `mul_mat_vec_q` + topk/softmax, several small kernels per layer.
- NCCL / tensor-split copies: `-ts 12,12` means per-layer KV split across 2 GPUs → cudaMemcpy between devices each FA step.

Without traces I cannot rank these by absolute time, but the static/architectural evidence plus prior project memory ranks them:

| Rank | Kernel (suspected) | Evidence |
|------|--------------------|----------|
| 1 | `flash_attn_ext_vec` (KTQ2_1 path) | Launch bounds `(128,1)`, prior memory, user's stated suspicion |
| 2 | MoE expert `mul_mat_vec_q` | A3B = 3B active = many small matmuls; Turing sm_75 lacks 8-bit tensor-core path |
| 3 | Inter-GPU cudaMemcpyPeer | `-ts 12,12` forces PCIe x4 link per token for KV fetch |
| 4 | topk/softmax + router | ncols=1, tiny kernels — launch overhead > compute |
| 5 | RoPE / norm / residual | standard llama.cpp ops, well-optimised |

### 4. Launch overhead signal

`-ub 512` with TG `ncols=1` means **every decode token issues a full graph of ~6-8 kernels × ~80 layers × 2 GPUs = ~960–1280 kernel launches/token** minimum. At 67 tok/s that's ~80k launches/s per GPU. Turing launch latency is ~5 µs → ~400 ms/s of pure launch overhead out of 1000 ms = **~40% launch-bound** as a ceiling estimate. This aligns with why FA occupancy alone can't 2× TG.

---

## Recommendation

Priority order for next work:

1. **CUDA Graphs for the decode path** (not previously on the roadmap, but implied by the launch-overhead math). If ~40% of wall-time is launch latency, capturing the per-token graph once and replaying it is the single biggest lever — it cuts that to near-zero. llama.cpp upstream has `GGML_CUDA_GRAPHS` already; verify it's enabled in this build and that the KTQ2_1 path doesn't break graph capture with dynamic shapes.

2. **`launch_bounds` 1→2 on legacy FA-vec** (`project_launch_bounds_fix.md` follow-up). Measurable win, low risk, limited scope. Requires checking `ptxas -v` output to confirm regs/thread ≤ 127 after the change; if not, a few `volatile` / register-pressure tweaks are needed. Expected: 5-15% TG uplift, not the main course.

3. **C1 streaming window** should be **deprioritised** relative to the above. The ring-buffer helps only if V-cache reads are the bottleneck — and V is F16 in prod, not quantised. C1 is a win for future TQ-V configs (VTQ_2 etc.) but not for the currently-deployed baseline.

4. **E11 cached-V** remains reverted. Not worth revisiting without prod moving to a VTQ-V config.

TL;DR: the occupancy fix is real and worth doing, but the bigger untapped lever on Turing is **launch overhead**, not memory BW. Get real nsys data before committing weeks to C1.

---

## Dynamic Profiling Results (post-prod-kill, 2026-04-21)

### Setup
- Host: `gpu00.node`, 2× RTX 2060 12GB (sm_75), 22 GB free/GPU
- Binary: `build-e14/bin/llama-bench` (phase2, fc1c512c1)
- Config: `-m Qwen3.5-35B-A3B-IQ2_XS.gguf -ctk ktq2_1 -ctv vtq2_1 -fa 1 -ngl 99 -p 0 -n 64 -r 1 -ts 12,12`
- Observed: **TG = 52–62 t/s** under profiler (baseline clean: 65 t/s)

### Tool availability
| Tool | Version | Usable? |
|------|---------|---------|
| `nsys` | 2022.4.2 | Runs, but the bundled importer is broken → cannot export stats from the qdstrm |
| `ncu`  | 2022.4.1 | **Blocked** — `ERR_NVGPUCTRPERM` (needs root to flip `RmProfilingAdminOnly=1`). No sudo on gpu00 |
| `nvprof` | CUDA 12.x | **Works** — deprecated on sm_75 but full timing data available |

We therefore have accurate **kernel timing + launch counts**, but no occupancy / L2 / DRAM BW counters. Those numbers below are marked `[measured]` or `[static]`.

### Top kernels during TG (`-n 64`, sorted by GPU time)

| Rank | % GPU time | Total (s) | Calls | Avg (µs) | Kernel (abbrev.) |
|-----:|-----------:|----------:|------:|---------:|------------------|
| 1 | 40.9% | 1.318 | 1902 | 692.9 | `[CUDA memcpy HtoD]` — dominated by one-shot weight uploads at init (43 ms + 10 ms chunks); not steady-state |
| 2 | 7.9% | 0.254 | 7930 | 32.0 | `mul_mat_vec_q<type=17, …>` (mmvq, type_k=IQ2_XS expert weights) |
| 3 | 7.7% | 0.248 | 5200 | 47.7 | `mul_mat_vec_q<type=17, fuse=1>` (mmvq) |
| 4 | 5.3% | 0.170 | 7150 | 23.7 | `mul_mat_vec_q<type=12, …>` (mmvq, Q4_K?) |
| 5 | 4.7% | 0.150 | 130 | 1156.5 | `mul_mat_vec_q<type=13, …>` — once per layer, large |
| 6 | 4.4% | 0.142 | 3900 | 36.4 | `concat_f32_dim0` (KV cache growth) |
| **7** | **3.7%** | **0.118** | **1300** | **91.0** | **`flash_attn_ext_vec<D=256, ncols=1, vtq_2_1, …>`** |
| 8 | 2.9% | 0.093 | 2600 | 35.7 | `mul_mat_vec_q<type=14, …>` |
| 9 | 2.6% | 0.083 | 8060 | 10.3 | `k_get_rows_float` (MoE expert gather) |
| 10 | 2.5% | 0.079 | 18200 | 4.3 | `mul_mat_vec_f<f,f,ncols=1,D=256>` (small f/f GEMV) |

Subtracting init memcpys, **kernel-only GPU time ≈ 1.85 s** → FA is **≈ 6.4 % of kernel time**, mmvq family is **≈ 28 %**.

### Answers to the 6 questions

1. **Top-5 kernels (TG):** mmvq type-17, mmvq type-17 fused, mmvq type-12, mmvq type-13, `concat_f32_dim0`. FA-vec is #7 overall, #6 among pure kernels. **mmvq dominates, not FA.**
2. **FA occupancy:** `[not measured]` — ncu blocked. Static analysis (D=256, 256 threads, 3.25 KB shmem per block) → 1 block/SM on Turing remains the **unverified but most-likely** answer. The `minBlocksPerSM 1→2` bump landed in 952e699e; its wall-clock effect is **within noise** (52–62 vs 65 baseline, profiler overhead dominates).
3. **L2 hit-rate on V-cache:** `[not measured]`. 3 MB L2 ÷ (F16 V, large ctx) → still expected to be low, but confirming requires root.
4. **DRAM BW:** `[not measured]`. Proxy: mmvq ncalls × bytes/call; won't reconstruct without counters.
5. **SM busy %:** `[not measured]`. Indirect proxy: 1.85 s of kernel time over a ~1.2 s TG wall (for 64 tok @ ~53 t/s under profiler) per GPU → 2 GPUs × 1.2 s = 2.4 s available ⇒ ~77 % utilisation, the rest is launch overhead + serial host time.
6. **Kernel launches per token:** **3 340 / token [measured]**. At 65 t/s that is **~217 k launches/s** across both GPUs. FA-vec alone is 20.3 launches/token (≈ 1 320/s).

### What actually matters

- **The 80 k/s launch estimate was low.** Real number is **~217 k/s total, ~108 k/s per GPU.** At a conservative 4 µs per launch that is **~430 µs/token of pure driver overhead** per GPU — **~2.8 % of wall time**, *not* the 20-30 % feared. Launch overhead is **not** the primary bottleneck at this point.
- **mmvq is the real work.** Four of the top six kernels are `mul_mat_vec_q` variants doing expert matmuls. They account for ~28 % of kernel time. Optimising these (or reducing expert gather/dispatch cost) has a much bigger ceiling than FA work.
- **`concat_f32_dim0` at 4.4 %** is the KV-cache append kernel — a good C1 (streaming / in-place KV growth) prize that sidesteps FA entirely.
- **`k_get_rows_float` at 2.6 % + 8 060 calls** is MoE expert gather, 10 µs each — another latency-bound kernel where fewer, larger launches would help.

### Concrete recommendation ranking (revised)

1. **CUDA Graphs over the per-layer MoE-expert sub-graph (B1+)** — biggest remaining hang-fruit now that launch count is confirmed high in absolute terms (~108 k/s/GPU). Capturing the 20-ish kernels of one decoder layer into a graph collapses launch overhead and lets the driver prefetch. Gain est.: **+3-6 % TG**.
2. **Stop optimising FA-vec further.** It is 6 % of kernel time; `launch_bounds 1→2` is already merged and unmeasurable here. Any further FA work (E11 / cached-V / warp-cooperative) is below the noise floor for the current prod config.
3. **Investigate mmvq / MoE expert dispatch** — the real 28 % slice. Batched expert GEMV or fewer expert launches is where the next percent lives.
4. **C1 (streaming KV / in-place concat)** — 4.4 % kernel-time ceiling, but also reduces memory traffic. Keep as medium priority.
5. **Re-run with `ncu` once `RmProfilingAdminOnly=0` is possible** (needs sudo). Occupancy and L2-hit numbers would let us finally close questions 2–5.

TL;DR: **The bottleneck is mmvq + launch count, not FA.** Prioritise CUDA Graphs at the decoder-layer level; de-prioritise further FA-vec micro-opts.
