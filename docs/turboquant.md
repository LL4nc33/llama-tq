# TurboQuant — KTQ/VTQ KV Cache Quantization for CUDA

Status: 2026-05-02. **v8 unified type aliases shipped.** Short CLI names `ktq{1,2,3,4}` + `vtq{1,2,3,4}` map to the proven defaults; new `vtq3` (= `vtq3_v8`, enum 58) is a 3.625-bpw trellis-3bit + 2 outliers — essentially **lossless** on 35B-A3B (-0.03% PPL drift vs f16). Three V-cache families (v1, v2 Trellis, v3 Trellis+outlier-split) and one K-cache family (KTQ), freely composable.

**Recommended (35B-A3B):** `--cache-type-k ktq2 --cache-type-v vtq3` (v8 quality tier, lossless).
**Legacy default since 2026-04-25:** `--cache-type-k ktq2_1 --cache-type-v vtq2_2` (= `ktq2/vtq2` in v8).

## Overview

TurboQuant is the KV-cache quantization stack of the `llama-tq` fork. It compresses the KV cache via a Randomized Hadamard Transform (RHT) plus codebook or trellis quantization, enabling much longer contexts on the same VRAM with negligible perplexity loss.

Two type families exist, split by cache role:

- **KTQ** (K-cache) — per-block RHT (FWHT + per-block sign flip) + Lloyd-Max codebook. The Flash Attention kernel applies FWHT to Q once per tile and dots against codebook values, so K is never explicitly dequantized in the vec path. On CC ≥ 7.5 a tensor-core MMA-KTQ split-dequant path is wired for prefill.
- **VTQ** (V-cache) — three sub-families:
  - **v1** (codebook lookup): fixed D·H·D rotation applied once at the graph level via `self_v_rot`, then a flat codebook lookup per entry inside the FA inner loop. No FWHT inside FA, no per-block sign bits.
  - **v2 Trellis** (current default): group-level Viterbi trellis with shared state and shared scale, 16-state shift-register, 16-bit open-start state, inverse-Gaussian CDF code table.
  - **v3 Trellis + outlier-split**: same Viterbi backbone as v2, plus a 4-fp16-outliers-per-block sidecar that captures the largest absolute V values losslessly.

Reference: PolarQuant (Han et al., arXiv:2502.02617, ICLR 2026) and TurboQuant (Zandieh et al., arXiv:2504.19874).

## Quick Start

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server
```

Pick a tier by passing two cache-type flags. `-fa on` is required.

```bash
# Recommended: 2.78 bpw avg, +0.15% PPL, 83% smaller KV
./build/bin/llama-server -m model.gguf -fa on -ngl 99 \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2

# Aggressive: 4.0 bpw avg, +0.49% PPL (no v2 kernels needed)
./build/bin/llama-server -m model.gguf -fa on -ngl 99 \
    --cache-type-k ktq2_1 --cache-type-v vtq3_1

# Conservative: q8_0 K + VTQ V (no KTQ kernels needed)
./build/bin/llama-server -m model.gguf -fa on -ngl 99 \
    --cache-type-k q8_0 --cache-type-v vtq3_1

# Quality: 4.75 bpw avg
./build/bin/llama-server -m model.gguf -fa on -ngl 99 \
    --cache-type-k ktq4_1 --cache-type-v vtq3_1
```

`--cache-type-k` accepts the stock quants (`f16`, `q8_0`, `q4_0`, …) plus `ktq{1,2,3,4}_1`. `--cache-type-v` accepts the stock quants plus `vtq{1,2,3,4}_1` (v1), `vtq{2,3,4}_2` (v2 Trellis), and `vtq{2,3,4}_3` (v3 Trellis + outlier-split).

## Available Types

### K-cache — KTQ

Per-block RHT (FWHT + per-block sign) + Lloyd-Max codebook. Block stores normalization factor `d`, packed indices `qs`, and 32 precomputed sign bits in `sb[4]`.

| Type   | enum | Index bits | bpw | Block | Notes |
|--------|:---:|:---:|:---:|:---:|---|
| `ktq1_1` | 45 | 1 | 2.5 | 10 B | extreme K compression |
| `ktq2_1` | 42 | 2 | 3.5 | 14 B | **current default** |
| `ktq3_1` | 43 | 3 | 4.5 | 18 B | balanced |
| `ktq4_1` | 44 | 4 | 5.5 | 22 B | best KTQ quality |

### V-cache v1 — VTQ codebook (`vtq*_1`)

Pre-rotated via `self_v_rot` at graph level; FA dequant is `codebook[idx] * scale` only. No FWHT, no sign bits inside FA.

| Type   | enum | Index bits | bpw | Block | Notes |
|--------|:---:|:---:|:---:|:---:|---|
| `vtq1_1` | 46 | 1 | 1.5 | 6 B  | extreme VRAM, sharp quality drop |
| `vtq2_1` | 47 | 2 | 2.5 | 10 B | previous deployed default |
| `vtq3_1` | 48 | 3 | **4.0** | 16 B | 14 B index-payload + padding |
| `vtq4_1` | 49 | 4 | 4.5 | 18 B | smallest v1 codebook-fit error |

Note: `vtq3_1` is **4.0 bpw**, not 3.5 — block layout is `[d:2B] [qs:14B 3-bit packed across 16 bytes]`.

### V-cache v2 — Trellis (`vtq*_2`)

Group-level Viterbi trellis. 16-state shift register; 16-bit `start_state` per block plus 16-bit `d`. `QK_VTQ_TRELLIS=128` (since Task #143; halved from 256). Index-rate is the bare `K` bits/sample; structural bpw includes the 4-byte (`d`+`start_state`) overhead per 128-sample block.

| Type   | enum | Index bits | bpw (index-rate) | bpw (struct) | Block |
|--------|:---:|:---:|:---:|:---:|:---:|
| `vtq2_2` | 50 | 2 | 2.0 | 2.25 | 36 B |
| `vtq3_2` | 51 | 3 | 3.0 | 3.25 | 52 B |
| `vtq4_2` | 52 | 4 | 4.0 | 4.25 | 68 B |

K-collision is a feature: `vtq2_2 / vtq3_2 / vtq4_2` produce bit-identical PPL on the same model — per-element MSE drops 16× across K=2/3/4 but FA softmax averages it out (attention-absorbed). Pick `vtq2_2`.

### V-cache v3 — Trellis + Outlier-Channel-Split (`vtq*_3`)

Same Viterbi backbone as v2, plus 4 fp16 outliers per block (largest absolute V values, stored losslessly).

| Type   | enum | Index bits | bpw avg | Block |
|--------|:---:|:---:|:---:|:---:|
| `vtq2_3` | 54 | 2 | 3.0 | 48 B |
| `vtq3_3` | 55 | 3 | 4.0 | 64 B |
| `vtq4_3` | 56 | 4 | 5.0 | 80 B |

PPL impact on 35B-A3B at 3.78 bpw avg (`ktq2_1 + vtq3_3`): +0.47% vs f16/f16 — well below stderr.

### Reserved / dormant

| Type        | enum | Status |
|-------------|:---:|---|
| `vtq_mixed` | 53 | **Discarded** — dominated by `vtq3_1`, no CUDA path. Defined for ABI stability. |
| `xktq2_1`   | 57 | **Dormant** — XQuant subordinate, code-complete but yields 0 pairs on Qwen3.X-A3B (alternating Mamba/attention layers). For pure-transformer dense models. |

## Recommended Configurations

| Tier              | K        | V        | Avg bpw | PPL cost | Notes |
|-------------------|----------|----------|:---:|:---:|---|
| **Recommended (default)** | `ktq2_1` | `vtq2_2` | 2.78 | +0.15% | 83% smaller KV vs f16/f16; current default since 2026-04-25 |
| Aggressive        | `ktq2_1` | `vtq3_1` | 4.0  | +0.49% | If v2 kernels are not built |
| Conservative      | `q8_0`   | `vtq2_1` | 5.5  | low    | No KTQ kernels needed; mixes with stock K |
| Conservative-v3   | `q8_0`   | `vtq3_1` | 6.25 | +1.05% | Stock K + v1 V |
| Quality           | `ktq4_1` | `vtq3_1` | 4.75 | low    | Best KTQ + v1 V |
| Research          | `q8_0`   | `vtq4_2` | 6.03 | +0.44% | Highest-quality Trellis V |

PPL numbers are 35B-A3B 8-chunk wikitext, llama-tq vs upstream f16/f16 baseline (2026-04-27). Stderr ≈ 0.7–1.2% on 4–8 chunk runs.

## Memory Savings

KV-cache footprint at 32k ctx, 35B-A3B (Qwen3.5/3.6-A3B, 32 experts / 4 active, GQA):

| KV config            | bpw  | KV @ 32k |
|----------------------|:---:|:---:|
| `f16 / f16`          | 32.0 | 640 MiB |
| `q8_0 / q8_0`        | 17.0 | 340 MiB |
| `q4_0 / q4_0`        | 9.0  | 180 MiB |
| `ktq2_1 / vtq2_2` ⭐  | 2.78 | **115 MiB** |

A 35B-A3B at `ktq2_1 + vtq2_2` fits **~330k single-ctx** (or ~470k with `-ub 128`, or 2× 200k parallel slots) on 24 GB total VRAM.

## Benchmarks

All measured on the test box: Ryzen 7 3700X host (Zen 2, 8C/16T), KVM guest 12 vCPUs, 40 GB DDR4-3200, 2× RTX 2060 12 GB on asymmetric PCIe (GPU0 x16 / GPU1 x4). llama-tq commit `1a1d49ef5`; upstream baseline `0c6ee1cad` (ggerganov/llama.cpp master, 2026-04-27).

### 35B-A3B full-GPU (ctx=2048, `-fa 1 -ts 12,12`)

| Variant                          | pp512 | tg128 | tg1024 | PPL (8ch) | KV-bpw | KV @ 32k |
|----------------------------------|---:|---:|---:|---:|---:|---:|
| upstream f16/f16                 | 934  | 57.76 | 57.30 | 7.0810 | 32.0 | 640 MiB |
| llama-tq f16/f16                 | 1009 | 76.58 | 75.09 | 7.0863 | 32.0 | 640 MiB |
| llama-tq `ktq2_1 + vtq2_2` ⭐    | 1010 | 75.40 | 75.18 | 7.1814 | 2.78 | 115 MiB |
| Δ tq vs upstream (quant)         | +8% | +31% | +31% | +1.34% | −91% | −82% |

### 80B-A3B (Qwen3-Next, hybrid + CPU expert offload, ctx=2048)

| Variant            | pp512 | tg128 |
|--------------------|---:|---:|
| upstream f16/f16   | 378 | 31.21 |
| llama-tq f16/f16   | 404 | 31.50 |
| Δ                  | +7% | +1% |

### 122B-A10B (Qwen3.5-A10B, GQA(2) + CPU expert offload, ctx=2048)

| Variant            | pp512 | tg128 |
|--------------------|---:|---:|
| upstream f16/f16   | 178 | 15.65 |
| llama-tq f16/f16   | 187 | 17.00 |
| Δ                  | +5% | +9% |

Live numbers (TG, PPL, HellaSwag for all five deploy targets): [`docs/bench/LIVE_NUMBERS.md`](bench/LIVE_NUMBERS.md). Raw CSV: [`bench/plots/benchmarks.csv`](bench/plots/benchmarks.csv).

## How It Works

### KTQ — K-cache quantization

```
float[32] -> normalize -> per-block sign flip -> FWHT -> Lloyd-Max codebook -> norm correction -> pack
                              |                                                    |
                        [store in sb[4]]                              [store corrected d]
```

1. **Normalize:** `x_hat = x / ||x||`.
2. **Per-block sign flip:** Deterministic ±1 signs from Philox 2×32, 6-round counter-based PRNG (Salmon et al. 2011) keyed by block index.
3. **FWHT:** 32-point Fast Walsh-Hadamard Transform, scaled by `1/√32` (orthonormal, self-inverse).
4. **Lloyd-Max quantization:** Nearest-centroid scalar quantization. Codebooks are Lloyd-Max optimal for `Beta((d-1)/2, (d-1)/2) = Beta(15.5, 15.5)` at d=32 (the marginal of a unit vector coordinate after random rotation).
5. **Norm correction:** Store `||x|| / ||reconstruction||` instead of raw `||x||`. Cancels the systematic Lloyd-Max underestimation bias. ~1.2% PPL improvement at zero dequant-time cost.
6. **Sign bits:** Store the 32 precomputed signs in `sb[4]` (32 bits). Dequant reads bits, never re-runs Philox.

### KTQ dequant — Hadamard-domain dot product

Instead of inverse-FWHT on K, the FA kernel applies FWHT to Q once per tile and dots directly against codebook values:

```
score = norm * <K_dequant, Q>
      = norm * <sign · FWHT⁻¹(cb), Q>
      = norm * <cb, FWHT(sign · Q)>     // FWHT orthogonal: <Hx, y> = <x, Hᵀy>; H = Hᵀ when normalized
```

Shifts the 32-element FWHT from per-K-block (gather + butterfly in the hot loop) to a single per-Q FWHT amortized across all K blocks a Q tile attends to. No gathers, no branch divergence at dequant time.

### KTQ MMA path (CC ≥ 7.5)

Split-dequant for prefill: bulk K → f16 conversion, then the standard MMA-F16 tensor-core kernel runs unchanged. Active when prefill ≥ 8 tokens. TG falls back to VEC. Measured KTQ2_1 35B IQ2_XS: PP128 727 t/s (vs 431 f16), PP512 875 (parity), PP2048 868 (parity). Source: `ggml/src/ggml-cuda/fattn-mma-ktq.{cu,cuh}`.

### VTQ v1 — codebook lookup

```
float[d_head] -> R · v (graph-level via self_v_rot) -> per-block: normalize -> codebook -> norm correction -> pack
                                                                                           [no signs, no FWHT]
```

FA V-dequant reduces to:

```cuda
v_approx[j] = CB[q_j] * scale;   // ~8 thread-local registers, __forceinline__
```

After FA accumulates the weighted sum, a graph-level `Rᵀ · VKQ` matmul applies the inverse rotation. Eliminates the ~40-register staging buffer that caused FA accumulator corruption in the original combined-KTQ V path.

### VTQ v2 — Trellis

Group-level Viterbi trellis with shared state and shared scale, 16-state shift register, 16-bit open-start state, inverse-Gaussian CDF code table. Block layout: `[d:fp16] [start_state:u16] [packed indices: K · group_size / 8 bytes]`.

The Viterbi path optimizes globally over the block, adapting implicitly to the running model's V-distribution — every V-element leverages local statistics at the same average bpw.

**Encoder cost is non-trivial** (~22 ms/call), so VTQ_2 auto-enables **f16 staging during prefill** and runs the bulk Viterbi exactly once at the prefill→decode boundary. Startup logs: `deferred V quantization enabled (N layers with f16 staging)`. **PPL measurement requires `-b 1 -ub 1`** to fire the deferred-V trigger — batched runs measure f16 + mixed-precision overhead, not actual VTQ_2 PPL.

### VTQ v3 — Trellis + outlier-channel-split

Same v2 backbone plus 4 fp16 outliers per block (largest absolute V values). Round-trip MSE drops a further 4× vs v2 at +1 bpw average. PPL impact on 35B-A3B at 3.78 bpw avg (`ktq2_1 + vtq3_3`): +0.47% vs f16/f16.

### Deferred K/V quantization

KTQ K-cache suffers a repetition-loop pathology when quantized per-token during prefill (attention re-reads just-quantized rows; RHT round-trip noise accumulates; the model loops). f16 staging during prefill plus bulk-convert at the prefill→decode boundary avoids it. **Auto-enabled for any KTQ K-type and any VTQ v2/v3 V-type.**

### Attention-sink protection

The first 4 tokens stay f16 regardless of KV cache type. Standard practice for streaming attention; preserves the "always-attended" sink positions that quantization noise would otherwise corrupt.

### Sparse V dequant

In the FA V-accumulation loop, dequant is skipped for positions where attention weight < 1e-6. At 32k+ context, >90% of positions are skipped. +22% decode throughput. Works with both KTQ-V (legacy) and VTQ V-types.

### Shared codebooks (`PQ_CODEBOOK_*`)

Both KTQ and VTQ v1 use the same Lloyd-Max codebooks, optimal for `Beta(15.5, 15.5)`:

- 1-bit (2 centroids): `{−0.7979, +0.7979}` (= ±√(2/π))
- 2-bit (4 centroids): `{−1.4896, −0.4514, +0.4514, +1.4896}`
- 3-bit (8 centroids): `{−2.0719, −1.3150, −0.7453, −0.2424, +0.2424, +0.7453, +1.3150, +2.0719}`
- 4-bit (16 centroids): symmetric 16-point set, `±{0.1284, 0.3880, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326}`

All scaled by `1/√32`. CPU constants: `PQ_CODEBOOK_*BIT`. CUDA constants: `PQ_CUDA_CB_*BIT`.

## CUDA Implementation Details

| Feature                     | Description |
|-----------------------------|---|
| Hadamard-domain KQ          | FWHT on Q once per tile; dot against codebook values. 39% fewer warp shuffles per `vec_dot` call vs naive inverse-FWHT-on-K. |
| Warp-cooperative FWHT       | 32 lanes × 5 `__shfl_xor_sync` rounds (XOR masks 1/2/4/8/16); zero shared memory. |
| AVX2-FWHT-32                | Host-side path used by deferred-K bulk conversion. |
| MMA-KTQ split-dequant       | sm_75+ tensor-core path for KTQ K + f16 V on prefill ≥ 8 tokens. |
| Branchless sign × norm      | `(1 − 2·bit) · norm` replaces ternary; eliminates warp divergence. |
| Precomputed sign bits       | `sb[4]` stored at quantization; dequant reads bits, never re-runs Philox. |
| VTQ v1 V-dequant            | `codebook[idx] * scale`, `__forceinline__`, ~8 registers, no LMEM spills. |
| VTQ pre-rotation            | `self_v_rot` matmul before cache write + inverse `Rᵀ` matmul after FA. |
| Sparse V dequant            | Skip dequant for `attn_weight < 1e-6`; +22% decode at 32k+ ctx. |
| FA dispatch                 | KTQ1_1..KTQ4_1 (K), VTQ1_1..VTQ4_1 (v1), VTQ2_2..VTQ4_2 (v2), VTQ2_3..VTQ4_3 (v3). |
| SET_ROWS kernels            | `k_set_rows_ktq*` + `k_set_rows_vtq*`. |
| `head_dim` support          | KTQ: 64/128/256/512. VTQ v2: D=64/128/256/512 (verified live on Gemma4 D=256/512 SWA + full-attn, Qwen3.6-35B D=128, gpt-oss-20b D=64). |
| Compute capability          | Tested on CC 7.5 (Turing, RTX 2060). Builds for CC 6.0+ (anything with `__shfl_xor_sync`). MMA-KTQ requires CC 7.5+. Untested on Ampere/Ada/Hopper — sm_75-specific tuning is not necessarily optimal there. |

### Phase 4 perf stack (2026-04-26)

Cumulative +18.5% TG on 80B-A3B (30.80 → ~36.5 t/s at ctx ≤ 8192), +9.3% on 122B (16.69 → 18.24 t/s):

- `MADV_HUGEPAGE` on weight mmap regions
- `__builtin_prefetch` in `mul_mat_id` expert dispatch
- `OMP_WAIT_POLICY=active` (+5.9% on 80B partial-offload, +2.6% on 122B; 0% on full-GPU)
- Adaptive layer-split (80B: 18/18/12 default for ctx ≤ 8192, falls back to safe split for ctx ≥ 200000)
- P2P opt-in (was hurting on asymmetric x16/x4 PCIe)
- AVX2-FWHT-32 for host-side bulk K conversion

### Networking / API

Anthropic-compatible `/v1/messages` endpoint with prompt caching, `TCP_NODELAY`, gzip — used by Claude Code clients and external agents.

## Source Files

| File                                       | Description |
|--------------------------------------------|---|
| `ggml/include/ggml.h`                      | Type enums lines 389–449. KTQ1_1=45, KTQ2_1=42, KTQ3_1=43, KTQ4_1=44. VTQ1_1=46, VTQ2_1=47, VTQ3_1=48, VTQ4_1=49. VTQ2_2=50, VTQ3_2=51, VTQ4_2=52. VTQ_MIXED=53 (dormant). VTQ2_3=54, VTQ3_3=55, VTQ4_3=56. XKTQ2_1=57 (dormant). |
| `ggml/src/ggml-common.h`                   | Block structs: `block_ktq*` (with `sb[4]`), `block_vtq*_1`, `block_vtq*_2` (Trellis), `block_vtq*_3` (Trellis + outliers). |
| `ggml/src/ggml-cuda/turboquant.cuh`        | CUDA kernels: KTQ Philox, FWHT, quantize, dequant; VTQ v1 quantize/dequant. |
| `ggml/src/ggml-cuda/fattn-common.cuh`      | FA: `vec_dot_KQ_ktq*`, `dequantize_V_ktq*`, `dequantize_V_vtq*` (v1/v2/v3), Sparse-V guard. |
| `ggml/src/ggml-cuda/fattn-mma-ktq.{cu,cuh}` | MMA-KTQ tensor-core split-dequant path (CC ≥ 7.5). |
| `ggml/src/ggml-cuda/convert.cu`            | CUDA dequant dispatch (contiguous + NC) for KTQ + all VTQ families. |
| `ggml/src/ggml-quants.c`                   | CPU quantize/dequantize for KTQ + VTQ; shared `PQ_CODEBOOK_*` constants. |
| `common/arg.cpp`                           | CLI: `--cache-type-k`, `--cache-type-v` parser; accepts `ktq{1,2,3,4}_1`, `vtq{1,2,3,4}_1`, `vtq{2,3,4}_2`, `vtq{2,3,4}_3`. |
| `bench/plots/benchmarks.csv`               | Raw benchmark data (single source of truth). |
| `docs/bench/LIVE_NUMBERS.md`               | Current TG/PPL/HellaSwag for all five deploy targets. |

## Roadmap

### Active research

- **Correction Overlay Buffer** — designed, not implemented. Top-N lossless error patch on top of KTQ.
- **Phase 7 — imatrix-aware KTQ calibration** — proposed. Use importance matrix to bias the K-quant Lloyd-Max codebook.
- **`mmvq` IQ2_XS tuning on sm_75** — currently 28% of kernel time on 35B configs.

### Discarded after measurement

- **Speculative decoding on A3B MoE** — expert-saturation pathology makes it ineffective.
- **VTQ_MIXED** — dominated by `vtq3_1`, no CUDA path. Enum kept for ABI.
- **Calibrated outlier selection** (pre-v3 design) — marginal gain after RHT.
- **MMA-KTQ as default for all ctx** — regresses past ~512 tokens. Now short-ctx-prefill only.
- **XQuant on hybrid SSM/attention models** — Qwen3.X-A3B / Qwen3.6-27B alternate Mamba/attention layers, yielding 0 pairs. Code-complete and dormant; intended for pure-transformer dense models (Llama-3, Mistral, Gemma-2 family).

### Not on roadmap

- FA3 (requires sm_80+).
- Paged attention (scope mismatch).
- Multi-node inference.

## When *not* to use TurboQuant

- VRAM is not a constraint — upstream f16 KV is simpler and equally fast.
- Sub-50 ms/token latency at long ctx matters — VTQ V-dequant overhead grows with context length.
- Multi-node serving — this fork makes zero changes to llama.cpp's split logic.
- Ampere+ (CC ≥ 8.0) — untested; sm_75-specific tuning is not necessarily a good default.

## Known Issues

- **Ministral-3B with KTQ K**: produces gibberish output. Root cause is on the K-quant path (head-dim / GQA layout interaction); unresolved. Workaround: `--cache-type-k q8_0 --cache-type-v q8_0` for this model.
- **VTQ v1 on Qwen3-Next-80B with `-b 1 -ub 1`**: can crash via fused Gated Delta Net interaction. Tracked separately. Batched mode and other models unaffected.
- **`vtq3_1` is 4.0 bpw, not 3.5** — earlier doc revisions had this wrong. Block layout is `[d:2B][qs:14B 3-bit indices]` with internal padding.
- **`vtq2_2` index-rate 2.0 vs structural 2.25 bpw** — index-rate is bare quantizer width; structural includes the 16-bit `d` + 16-bit `start_state` overhead per 128-sample block. Older docs (pre Task #143) cite 2.06 / 132 B under the previous `QK_VTQ_TRELLIS=512` layout.
- **gpt-oss-20b head_dim=64**: hit the upstream `head_dim % blck_size` check (false-positive for VTQ_2, which quantizes along the sequence axis, not D). Fixed in commit `c818f6c84` (2026-04-27).

## References

This implementation is inspired by but deviates from the cited papers. KTQ uses RHT (FWHT + per-block random sign) + Lloyd-Max instead of QR rotation; VTQ v1 uses a fixed D·H·D rotation (specific to this fork); v2/v3 use a Viterbi trellis over an inverse-Gaussian CDF code table. QJL was used in v1–v4 and removed in v5.

| Paper | Authors | arXiv | Relevance |
|---|---|---|---|
| **PolarQuant: Quantizing KV Cache via Polar Coordinate Transformation** | Han, Kacham, Karbasi, Mirrokni, Zandieh | [2502.02617](https://arxiv.org/abs/2502.02617) (Feb 2025, ICLR 2026) | Primary inspiration for VTQ v1 D·H·D rotation. |
| **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** | Zandieh, Daliri, Hadian, Mirrokni | [2504.19874](https://arxiv.org/abs/2504.19874) (April 2025) | Random rotation + Lloyd-Max codebooks framework. |
| **QJL: 1-Bit Quantized JL Transform for KV Cache Quantization** | Zandieh, Daliri, Han | [2406.03482](https://arxiv.org/abs/2406.03482) (June 2024) | Used in v1–v4, removed in v5. |
| **XQuant: Cross-layer KV reuse** | (cf. 2510.11236) | [2510.11236](https://arxiv.org/abs/2510.11236) | Phase 5 dormant subordinate path. |
| **Parallel Random Numbers: As Easy as 1, 2, 3** | Salmon, Moraes, Dror, Shaw | SC 2011 | Philox 2×32 counter-based PRNG used for KTQ sign generation. |

## Version History

- **v1** (2026-04-06): Initial TurboQuant-inspired implementation + CPU reference.
- **v2** (2026-04-07): Warp-parallel FWHT, CUDA QJL attempt.
- **v3** (2026-04-08): Paper-compliant: stored `r_norm`, QJL on CUDA, Beta-exact codebooks.
- **v4** (2026-04-09): TQ4_1 (4-bit, 16 centroids), Sparse V Dequant, asymmetric K/V support.
- **v5** (2026-04-10): Precomputed sign bits in `sb[4]`, struct compaction (3.5/4.5/5.5 bpw), norm correction, Philox 6-round, QJL fully removed.
- **v6** (2026-04-13): Warp-cooperative FWHT in FA via `__shfl_xor_sync`, SET_ROWS kernel, FA dispatch registration, warp-cooperative V-dequant.
- **v7** (2026-04-14): Hadamard-domain KQ dot product, branchless sign × norm fusion. PP +13% on CC 6.1, TG +65% on CC 7.5 vs v6.
- **v7+VTQ split** (2026-04-16): KTQ/VTQ split. VTQ v1 types (`vtq{1..4}_1`) for V-cache — no FWHT, no sign bits, `__forceinline__` dequant. TQ renamed to KTQ. Shared `PQ_CODEBOOK` constants.
- **2026-04-23**: VTQ v2 Trellis family (`vtq{2,3,4}_2`) — group-Viterbi, inverse-Gaussian CDF table. VTQ v3 outlier-channel-split (`vtq{2,3,4}_3`) — 4 fp16 outliers per block.
- **2026-04-24**: MMA-KTQ tensor-core path live on CC ≥ 7.5 (Turing-tested). PP128 727 t/s vs 431 f16.
- **2026-04-25**: Default switched to `ktq2_1 + vtq2_2` (2.78 bpw avg, +0.15% PPL, 83% smaller KV) after vtq2_2 vs vtq2_1 sweep showed v2 wins or ties on PPL/pp/tg.
- **2026-04-26**: 80B-TQ1_0 deployed full-VRAM (54.93 t/s, +50% vs IQ2_XXS). Phase 4 perf stack: `MADV_HUGEPAGE`, `mul_mat_id` prefetch, `OMP_WAIT_POLICY=active`, adaptive layer-split (80B: 18/18/12), P2P opt-in, AVX2-FWHT-32. Cumulative +18.5% TG on 80B, +9.3% on 122B.
- **2026-04-27**: XQuant Phase 1–5 code-complete (XKTQ2_1, dormant on hybrid SSM). `--moe-pin-experts` opt-in (+3.3% TG on 80B-IQ2). gpt-oss-20b head_dim=64 fix (commit `c818f6c84`). Anthropic `/v1/messages` with prompt caching, `TCP_NODELAY`, gzip.
- **2026-04-28**: gpu00 model directory consolidated to `/home/lance/models/`. Doc rewrite — this file.
