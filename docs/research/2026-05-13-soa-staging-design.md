# VTQ SoA Staging — Path to f16-Parity — 2026-05-13

## Konvergente expert-empfehlung

Drei unabhängige agents haben den VTQ-V bottleneck analysiert:

| Agent | Empfehlung | Erwarteter recovery |
|-------|-----------|---------------------|
| Builder (kernel design) | 8-elem-per-thread refactor | -26% → -10 to -16% |
| Layout (memory access) | **SoA staging bei slot-load** | -26% → **-3 to -5%** |
| Reviewer | safe, kein bug | aktuelle commits validierbar |

**Klare empfehlung: SoA staging.** Reines kernel-rewrite hat **hard ceiling ~-10%** auf Turing (sm_75 hat kein cp.async für byte-strided, ldmatrix.sync braucht 128-bit alignment).

## Root cause: 10-byte struct ist nicht alignable

```cpp
typedef struct {
    ggml_half d;              // 2 bytes, offset 0
    uint8_t   qs[QK_VTQ / 4]; // 8 bytes, offset 2  ← UNALIGNED
} block_vtq2_1;  // total 10 bytes
```

Aufeinanderfolgende blocks starten bei offsets 0, 10, 20, 30, 40...
- Jedes 3. block straddet eine 32-byte sector boundary
- Compiler kann keine 128-byte coalesced loads erzeugen
- MMA tensor cores (ldmatrix.sync) brauchen 128-bit alignment → unmöglich

## Lösung: Struct-of-Arrays bei slot-load

**On-disk format bleibt unverändert** (10-byte struct):
- slot-save kompatibel
- VTQ-encode format stabil
- Backwards-compat zu existierenden checkpoints

**Beim ersten slot-load** wird in SoA umkopiert (1× linear, irrelevant kosten):

```cpp
// AoS (on-disk, current):
block_vtq2_1 cache[n_blocks];  // 10 bytes × n_blocks

// SoA (runtime, runtime-reformatted):
struct vtq2_1_soa {
    ggml_half * d;     // [n_blocks]      — 2-byte aligned, naturally
    uint8_t   * qs;    // [n_blocks * 8]  — 8-byte aligned (start of array)
};
```

**Wins:**
- `d[]` als kontinuierliches array → L1/Tex cache hit auf 16× scales pro 32B-cacheline
- `qs[]` als `uint64_t` load (8 bytes pro block) → single LDG.64 statt 8× byte gathers
- Naturally aligned → MMA-tensor-core kernels accessibel

## Implementation plan

### Phase 1: SoA runtime conversion (1-2 days)

1. Add `ggml_cuda_pool_alloc<ggml_half> d_buf` + `<uint8_t> qs_buf` allocated once at first VTQ-V FA call
2. Run reformat kernel: copy `block_vtq2_1[]` → `(d_buf, qs_buf)` SoA. 1× linear scan, ~1ms for 200k tokens
3. Modify `flash_attn_ext_f16_vtq_load_tile_V_vtq2_1` to take SoA inputs
4. Cache the SoA buffer between FA calls (only invalidate when KV cache grows)

### Phase 2: HND reordering for GQA (optional, +small win)

For GQA=8 (8 query heads share one KV head), reorder to `[blocks, kv_heads, page_size, head_dim]`
so all 8 query-heads read from same cache-line.

### Phase 3: New x4-output kernel using SoA

The 4-outputs-per-thread design from the builder-agent works perfectly with SoA:
- `__ldg<uint64_t>(&qs[ib*8])` is now aligned → single 8-byte load
- Each thread reads 1 byte from that uint64_t via shuffle/extract
- 4 codes per byte × 4 threads in warp = 16 outputs per warp issue cycle
- Pre-scaled codebook → save 1 FMUL per element

## Risk assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| SoA buffer doubles V-cache VRAM transiently during conversion | LOW | only during reformat call, deallocate after |
| Reformat kernel slows long-ctx use-case | LOW | cache invalidation only on cache-grow, not per token |
| Format drift between encode (AoS) and runtime (SoA) | MEDIUM | clear encapsulation: only `flash_attn_ext_f16_vtq*` sees SoA, all writes via on-disk AoS |
| Memory layout assumptions in slot-save/restore | MEDIUM | slot-save reads AoS (on-disk), slot-restore writes AoS, SoA invalidated on load |

## Sources

- [FlashInfer KV-Cache Layout](https://docs.flashinfer.ai/tutorials/kv_layout.html) — HND for GQA
- [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) — per-head scale strategy
- [GGUF Optimization deep dive](https://medium.com/@michael.hannecke/gguf-optimization-a-technical-deep-dive-for-practitioners-ce84c8987944) — alignment

## Decision

Phase 1 implementation worth pursuing IF current commits land cleanly. SoA staging is the principled fix that closes the f16 gap; multi-warp/pre-scaled CB are incremental at best (current commits).
