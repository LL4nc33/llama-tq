# KTQ K-cache Autoresearch (2026-04-22)

Hypothesis-driven investigation of the KTQ PP/TG slowdown on RTX 2060 (sm_75).

## Refreshed baseline (HEAD 32e477b59, build-e14, Qwen3.5-35B-A3B IQ2_XS, ts=12,12)

| Config       | PP512 t/s | TG128 t/s | vs f16 PP | vs f16 TG |
|--------------|-----------|-----------|-----------|-----------|
| f16   K+V    | 880       | 72.0      |  1.00x    |  1.00x    |
| ktq2_1 K, f16 V | 97    | 55.9      |  0.11x    |  0.78x    |
| q4_0 K, f16 V   | 674   | 66.4      |  0.77x    |  0.92x    |

**Note:** User's starting table claimed KTQ2_1 TG=30.3. Current HEAD shows 55.9.
Agent B's prior perf commits (exp2f softmax + minBlocksPerSM=2) have already
recovered most of the TG path. Old table is stale.

## E1 — Does KTQ use DP4A?

**Hypothesis:** q4_0 fast because of `__dp4a`; KTQ does f32 FWHT serially.

**Verdict:** CONFIRMED, but not actionable as "port KTQ to DP4A".
- `vec_dot_fattn_vec_KQ_q4_0`: 1 `ggml_cuda_dp4a(v, u, 0)` per thread per K-row.
- `vec_dot_fattn_vec_KQ_ktq2_1`: warp-cooperative FWHT via 5 shuffle stages per
  K-block, repeated for D/32=4 blocks per K-row (D=128). ~25 shuffles per
  K-row vs. 1 DP4A op for q4_0.
- KTQ uses float codebook centroids in `__constant__` memory
  (`PQ_CUDA_CB_{1,2,3,4}BIT`). Cannot be replaced with int8 packed math
  without redesigning the quantization scheme — the centroid values are
  real-valued Lloyd-Max outputs, not uniform scaled ints.

## E2 — Codebook in `__constant__`?

**Verdict:** ALREADY DONE (turboquant.cuh:110-123). No change needed.

## E3 — Hoist FWHT(Q) outside K-row loop?

**Hypothesis:** FWHT is orthogonal → precompute once per Q.

**Verdict:** NOT POSSIBLE in the current formulation.
- v7 K·Q = `FWHT(D_s · Q) · c` where `D_s` are per-K-block sign flips from
  `K_tq[bi].sb[]`.
- `FWHT(D_s · Q) ≠ D_s · FWHT(Q)` (D_s and H don't commute).
- Alternatives (`(D_s · Q) · FWHT(c)`, `(H D_s H) · (H Q)`) all require
  either per-K-row FWHT on c or a full 32×32 dense matmul — no win.
- Math is already as tight as the v7 formulation allows.

## E4 — Thread-tile layout

**Verdict:** NOT THE PROBLEM.
- KTQ and q4_0 both use `nthreads_KQ = 32` (full warp per K-row) at D=128.
- Outer K-row loop iterates 32 times identically for both paths.
- Cost diff is purely per-iteration: 25 shuffles (KTQ) vs 1 DP4A (q4_0).

## E5 — PP vs TG asymmetry: root cause

**Verdict:** CONFIRMED — structural dispatch problem, not a vec-kernel bug.

In `fattn.cu:359-372`:
```
if (is_tq_k || is_tq_v || is_vtq_v) {
    ...
    return BEST_FATTN_KERNEL_VEC;
}
```

TQ K/V types **unconditionally** dispatch to the VEC kernel, regardless of
Q->ne[1] (batch size). f16/q4_0 with Q->ne[1] > 2 go to MMA_F16 (tensor cores).

The vec kernel supports only `cols_per_block ∈ {1, 2}`
(`fattn-vec.cuh:581-600`), so PP512 processes 512 Q tokens as 256 blocks of
2 columns each — massive underutilization of Q-side parallelism that
tensor-core MMA exploits. Hence the 9x PP gap is not algorithmic but
structural: KTQ never reaches the MMA path.

The TG gap (56 vs 72 for f16) is comparatively small — 78% of f16 — because
the vec kernel *is* the right fit for Q->ne[1]=1, and Agent B's prior perf
commits (exp2f, minBlocksPerSM=2) already tuned it well.

## Conclusion & recommended follow-up

The realistic wins for KTQ are **not** tunable in the vec-dot inner loop.
Real wins require one of:

1. **Add KTQ support to the MMA_F16 tile kernel.** The tile kernel already
   handles q4_0 at PP with tensor cores. Adding a KTQ dequant path (reuse
   `ktq_fattn_dequant_block_*` serial FWHT to materialize f16 tiles) would
   close most of the PP gap. High effort (new TU, careful dequant-on-tile
   integration), but the right axis of attack.

2. **On-the-fly K dequant to f16 during PP.** Allocate a scratch f16 KV
   tile per layer, dequant once, run f16 MMA FA. Memory overhead:
   `seq_len * n_heads * head_dim * 2 B`. Trades VRAM for speed — useful
   only if the KTQ VRAM savings aren't critical.

3. **Accept the trade-off.** KTQ is already a quality-first, decode-first
   format. PP is batched: compile-time KV gen dominates early and decode
   dominates overall serving. At the deployed target (Qwen3.5-35B,
   long context), decode throughput matters more than PP.

## Micro-wins not pursued (with reason)

- Raising `cols_per_block` to 4 in vec dispatch: static arrays
  `VKQ[ncols][...]`, `Q_f32[ncols][D/32]` would double register footprint.
  minBlocksPerSM=2 already caps regs at ~128/thread; going to 4 cols would
  force 1 block/SM, halving warp-level parallelism. Net: likely wash or
  regression. Not worth the risk without deeper register-budget analysis.

- Packing 2-bit indices as DP4A lookups via LUT-on-int8: codebook values
  are floats (real-valued Lloyd-Max). Converting to int8 would introduce
  quantization error on top of the 2-bit scheme — destroys the point of
  the TQ line.

## No code committed

No changes accepted during this loop. The baseline is the right shipping
configuration; further work belongs in a dedicated MMA-port branch.
