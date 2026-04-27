# QTIP Weight Quantization — Port Spec

**Branch:** turboquant
**Paper:** Tseng et al., "QTIP: Quantization with Trellises and Incoherence Processing", ICLR 2025 (arXiv:2406.11235)
**Reference impl:** github.com/Cornell-RelaxML/qtip
**Date:** 2026-04-26
**Status:** SPEC ONLY

## 1. Executive Summary

QTIP combines **Trellis-Coded Quantization (TCQ)** with **Incoherence Processing (RHT)**. Bitshift trellis with state transitions = pure bit shifts → parallel decode without large LUTs. Port the **HYB(L=16, V=2, Q=9)** variant — its 2KiB codebook fits L1 on Turing sm_75; decode is 2 instructions/weight.

- **Quality target:** PPL(QTIP_2 @ Qwen3.6-35B-A3B) ≤ PPL(Q3_K_M) + 0.5
- **Speed target:** mmvq decode ≥ Q2_K throughput on 2× RTX 2060

## 2. Bitshift Trellis Algorithm

### 2.1 Trellis (L=16, K=2, V=2 → 2-bit weights)

- 2^L = 65536 states
- 2^(K·V) = 16 outgoing edges per state
- V=2 weights emitted per state

Transition: `next_state = ((state << K*V) | edge_label) & ((1<<L)-1)`. State at step t depends only on last L bits → parallel decode by sliding L-bit window.

### 2.2 HYB decode (hot loop, ~2 instructions/weight)

```
hash = (s*s + s) & 0xFFFF
idx  = (hash >> 7) & 0x1FF              # Q=9 bits → 512-entry LUT
v0,v1 = lut[idx]                        # fp16 pair
sign = (hash >> 15) & 1
if sign: v1 = -v1
emit (v0, v1)
```

### 2.3 Viterbi encoder (offline)

O(T · 2^L) per layer. **Tail-biting approximation:** rotate by T/2, two passes, drops to O(T·2^L).

### 2.4 BlockLDLQ wrapper

Per 16×16 weight block:
1. RHT: `W_tilde = V·S·W·S·V^T`
2. Hessian H_tilde from calibration
3. Iterative LDLQ with Cholesky of H_tilde
4. TCQ-quantize residual length T_x·T_y/V = 128

## 3. Reuse Map — Existing llama-tq Infra

| Component | Existing | Reuse | Mod needed |
|---|---|---|---|
| Group-Viterbi encoder (VTQ KV) | ggml-trellis.c | YES | Generalize MSE-on-residual → MSE-on-RHT-weight; add tail-biting |
| Shift-register decoder | ggml-trellis.h | YES | Swap PolarQuant LUT → HYB hash+LUT |
| RHT (Hadamard) | PolarQuant | YES | Add 2D RHT (rows AND cols) |
| Lloyd-Max scalar | shared | NO direct | Only for HYB LUT generation offline |
| Warp-FWHT shfl_xor_sync | KV cache | YES | Repackage as `rht_warp_fp16x16` |
| CUDA dispatch sm_75 | analog | YES | Add GGML_TYPE_QTIP_2 case |

**~70% reusable** — Viterbi engine, bit-shift decoder, warp-FWHT exist. New mass: encoder-for-weights + mmvq kernel + ggml type plumbing.

## 4. New Components — LOC

| File | Purpose | LOC |
|---|---|---|
| `ggml/src/ggml-qtip.h` | Public types, block layout | 80 |
| `ggml/src/ggml-qtip.c` | CPU dequant ref, HYB LUT, RHT 2D | 350 |
| `ggml/src/ggml-quants.c` (patch) | Register GGML_TYPE_QTIP_2, traits | +60 |
| `ggml/include/ggml.h` (patch) | Enum value | +5 |
| `src/llama-quant.cpp` (patch) | Pipeline integration, calibration hook | +180 |
| `tools/quantize/quantize.cpp` (patch) | CLI: --type qtip_2 | +40 |
| `ggml/src/qtip-encoder.cpp` | Viterbi + BlockLDLQ + tail-biting | 600 |
| `ggml/src/qtip-rht.cpp` | 2D RHT with persisted signs | 120 |
| `ggml/src/ggml-cuda/qtip-mmvq.cu` | sm_75 mmvq kernel | 450 |
| `ggml/src/ggml-cuda/qtip-dequant.cu` | Async dequant fallback | 180 |
| `tests/test-qtip-roundtrip.cpp` | Single-tensor PoC, MSE gate | 220 |
| `tests/test-qtip-ppl.sh` | Full-model PPL gate | 60 |
| **Total new** | | **~2345 LOC** |
| **Patches** | | **~285 LOC** |

## 5. Block Layout — block_qtip_2

```c
typedef struct {
    ggml_half d;            // 2 B   block fp16 scale
    uint16_t  s_row;        // 2 B   row-local RHT signs
    uint16_t  s_col;        // 2 B   col-local RHT signs
    uint8_t   pad[2];       // 2 B   align to 8
    uint8_t   tstream[64];  // 64 B  trellis bit-stream (2 bits/weight × 256)
} block_qtip_2;             // total: 72 B → 2.25 bpw
```

vs `block_q2_K` = 2.5625 bpw, `block_q3_K_M` ≈ 3.4 bpw. **QTIP_2 saves 12% over Q2_K and 34% over Q3_K_M.**

HYB LUT (2KiB, 512 fp16x2) lives once in `__constant__` memory.

## 6. mmvq Kernel Design — sm_75 Turing

Constraints: no native low-bit tensor cores on sm_75; 64KB shared mem/SM; 64K registers; 1024 threads/block max.

### 6.1 Kernel layout

```cuda
__constant__ __half2 qtip_hyb_lut[512];

template<int NCOLS>
__launch_bounds__(128, 4)
__global__ void mmvq_qtip_2(
    const block_qtip_2 * __restrict__ x,
    const half * __restrict__ y,
    float * __restrict__ dst,
    int ncols, int nrows)
{
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const block_qtip_2 * blk = &x[row * (ncols/256) + blockIdx.y];

    uint4 stream = *reinterpret_cast<const uint4*>(&blk->tstream[lane * 2]);

    half2 w[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t s   = (stream.x >> (i*4)) & 0xFFFF;
        uint32_t h   = s * s + s;
        uint32_t idx = (h >> 7) & 0x1FF;
        half2    v   = qtip_hyb_lut[idx];
        if (h & 0x8000) v.y = __hneg(v.y);
        w[i] = v;
    }

    // undo local RHT signs s_row, s_col

    half2 a[4];
    *reinterpret_cast<uint4*>(a) = *reinterpret_cast<const uint4*>(&y[blockIdx.y*256 + lane*8]);

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc += __low2float(__hmul2(w[i], a[i]));
        acc += __high2float(__hmul2(w[i], a[i]));
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) atomicAdd(&dst[row], acc * __half2float(blk->d));
}
```

### 6.2 Performance notes (sm_75)

- 2KiB constant-mem LUT → 1-cycle broadcast across warp
- HYB decode = INT pipe, overlaps FP pipe doing FMA
- Memory: 72B/256w = 0.281 B/w → bandwidth-bound at low batch (Q2_K profile)
- launch_bounds(128,4): 4 blocks × 128 threads = 512/SM, 50% occupancy on 2060
- No tensor-core abuse, pure CUDA cores

## 7. Quantizer Integration

```
llama-quantize --type qtip_2 \
    --calib calib_wikitext_512x4096.bin \
    --rht-block 16 \
    --tail-biting \
    in.gguf out.gguf
```

Pipeline in src/llama-quant.cpp: load calib acts → derive S_m,S_n signs → RHT 2D → BlockLDLQ over 16×16 tiles → Viterbi tail-biting → write block_qtip_2.

HYB LUT generation (one-time): `lloyd_max_2d_gaussian(num_entries=512, iters=200)` → persisted in GGUF metadata as `qtip.hyb_lut`.

## 8. Bench Gates

- **PPL gate:** PPL(QTIP_2) - PPL(Q3_K_M) < 0.5 on wikitext, Qwen3.6-35B-A3B
- **Decode-speed gate:** tg128(QTIP_2) ≥ tg128(Q2_K)
- **Single-tensor PoC:** cosine_sim ≥ 0.985, frob_err ≤ 0.04 · ||W||

## 9. Phased Plan

| Phase | Scope | Time | Gate |
|---|---|---|---|
| R | Research | DONE | spec |
| 1 | Single-Tensor PoC (CPU) | 1 wk | cosine ≥ 0.98 |
| 2 | Full-Model Quantizer | 1.5 wk | PPL gate on Qwen3.6-1.7B |
| 3 | CUDA mmvq Kernel sm_75 | 2 wk | speed gate on RTX 2060 |
| 4 | Integration | 1 wk | full 35B-A3B inference, 1000 prompts no NaN |
| 5 | 122B Stretch | open | 122B in 24GB @ Q3_K_M-equiv |

## 10. Five Specific Risks

1. **Tail-biting Viterbi numerical stability.** MoE expert layers ill-conditioned. Mitigation: damped LDLQ (H + λI), fallback non-tail-biting if cycle MSE > 1.5× linear.
2. **HYB LUT mismatch quantize/inference.** Mitigation: persist 2KiB LUT in GGUF metadata, version-tag, hash check, test-qtip-lut-determinism.cpp.
3. **sm_75 register pressure in mmvq.** Mitigation: __launch_bounds__(128,4), profile cuobjdump --dump-sass, split decode→sgemv if spill.
4. **RHT 2D sign-vector storage explosion.** Mitigation: derive signs from per-tensor 64-bit seed via xoshiro256.
5. **MoE expert quality cliff.** Rare experts → garbage Hessian → 2-bit collapse. Mitigation: per-expert gate-count check, fallback Q4_K for <128 calib tokens; budget +0.4 GB.

**Bonus risk:** Tail-biting + BlockLDLQ + RHT all interact. Reference impl performs gradient-based fine-tuning post-quant for ~0.2 PPL recovery. Plan Phase 2.5 fine-tune if Phase 4 misses gate.

## Sources

- [QTIP paper (arXiv:2406.11235)](https://arxiv.org/abs/2406.11235)
- [QTIP HTML](https://arxiv.org/html/2406.11235)
- [Cornell-RelaxML/qtip GitHub](https://github.com/Cornell-RelaxML/qtip)
