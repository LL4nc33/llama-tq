# Triton VTQ_2 Strategy A — Warp-Cooperative Block Cache

**Context:** VTQ_2 TG regression on D=256 heads (Qwen3.5-35B-A3B: 4.32 tok/s vs VTQ_1 66.5 tok/s). Root cause identified as L2-cache-thrashing in 256 KiB LUT random-access pattern. Strategy A from `trellis.cuh:222` (warp-shmem block cache) is the documented fix but requires invasive `fattn-vec.cuh` changes.

**This spec:** Implement Strategy A in Triton first for fast iteration (5s JIT vs 50min CUDA rebuild).

## Scope

Standalone Triton prototype on `gpu00.node:/home/claude/llama-tq/triton-strategy-a/`. No integration into llama.cpp yet. Goal: prove the speedup algorithmically and pick a block size / layout that works.

## Algorithm

Current decode-path (slow on D=256):
```
for each (query, sample) pair:
    state = vtq_state_at(s0, qs, sample_idx)    # O(1) arithmetic
    value = LUT[state] * d                        # 256 KiB random read
```

Strategy A (warp-cooperative):
```
# Once per block (256 samples):
shmem_values = __shared__ float[256]
for tid in 0..31:                                 # warp lanes
    for i = tid; i < 256; i += 32:
        state = vtq_state_at(s0, qs, i)
        shmem_values[i] = LUT[state] * d
__syncwarp()

# Then queries reuse shmem, 0 LUT access:
for (query, sample):
    value = shmem_values[sample]
```

**Expected speedup:** LUT-load amortization (256 queries × 256 samples → 256 loads instead of 65536). ~50-100× fewer LUT reads per block. Translates to ~10× decode speedup if LUT was the bottleneck (pessimistic assumption).

## Triton Implementation

`vtq_decode_strategy_a.py`:

```python
import triton
import triton.language as tl

@triton.jit
def vtq2_decode_block_strategy_a(
    qs_ptr,           # uint8 bit-packed K*256 bits per block
    d_ptr,            # fp16 scale per block
    s0_ptr,           # uint16 start_state per block
    out_ptr,          # fp16 output, N_blocks * 256 samples
    lut_ptr,          # 65536 float LUT (Gaussian inv-CDF)
    N_BLOCKS: tl.constexpr,
    K: tl.constexpr,
):
    # Each program = one warp = 32 threads = one block (256 samples)
    pid = tl.program_id(0)
    lane = tl.arange(0, 32)

    # Load block metadata
    d = tl.load(d_ptr + pid).to(tl.float32)
    s0 = tl.load(s0_ptr + pid).to(tl.uint32)
    cb_scale = 1.0 / tl.sqrt(256.0)
    ds = cb_scale * d

    # Cooperative decode: each lane handles 8 samples (256/32=8)
    for i_base in range(0, 256, 32):
        i = i_base + lane
        # Compute state(i+1) using vtq_state_at arithmetic
        # Read K bits at position (i+1)*K from [s0 || qs[...]]
        stream_bit = (i + 1) * K
        L = 16

        # Branchless: handle boundary-straddle case
        qs_bit = stream_bit - L
        byte = qs_bit // 8
        shift = qs_bit % 8
        b0 = tl.load(qs_ptr + pid * (K * 32) + byte)
        b1 = tl.load(qs_ptr + pid * (K * 32) + byte + 1)
        b2 = tl.load(qs_ptr + pid * (K * 32) + byte + 2)
        w = b0 | (b1 << 8) | (b2 << 16)
        state = (w >> shift) & 0xFFFF
        # (stream_bit < L case: use s0 bits)

        # LUT read + output
        val = tl.load(lut_ptr + state) * ds
        tl.store(out_ptr + pid * 256 + i, val)
```

Key Triton mechanisms:
- `tl.program_id(0)` = per-block index
- `tl.arange(0, 32)` = 32-lane warp
- `tl.load` with vector index = coalesced if consecutive

## Benchmark Plan

`bench_strategy_a.py`:

1. **Correctness:** decode 1024 blocks, compare bit-identical to CPU reference
2. **Throughput:** GB/s for pure decode (baseline from previous spike: 50.9 GB/s)
3. **Query simulation:** decode + random-access 256 queries × 256 samples = 65536 reads. Measure ms per block.

Expected Strategy A result:
- Decode alone: same or slightly slower than O(1) single-thread (because warp coordination overhead)
- Decode + 256 queries: **10× faster than O(1)** (queries hit shmem, no LUT)

If Strategy A wins on decode+queries but loses on pure decode, that's fine — the **real workload** is FA which does many queries per block.

## Fallback Plan

If Strategy A doesn't give speedup in Triton, the L2-thrashing theory is wrong and the real bottleneck is elsewhere (likely per-sample `vtq_state_at` arithmetic or fp16 store overhead). In that case:
- Bisect: turn off LUT load (use fixed value) — if still slow, it's not LUT
- Profile via `ncu` (nsight compute) on the CUDA kernel directly

## Done criteria

- Triton prototype runs
- Benchmark shows ≥5× speedup over single-thread decode on D=256-equivalent workload
- Clear recommendation: port to CUDA fattn-vec or keep as Triton experiment

## Timeline

- Spec + scaffold: 30min
- Triton kernel: 1-2h (Karpathy-autoresearch-loop: small changes, fast bench)
- Report: 30min

**Total: ~3h agent time. Compare: 1 CUDA rebuild = 50-75min × 3-5 iterations = 3-6h wall-clock.**
