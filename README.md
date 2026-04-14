# llama-tq

**TurboQuant v7 for llama.cpp** -- KV cache quantization via PolarQuant (3.5-5.5 bpw)

Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) implementing [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026). **78% less KV-Cache VRAM** vs f16, enabling 4x longer context in the same memory.

## What Makes This Fork Different

| Feature | llama-tq (this) | Other TQ forks |
|---------|:---:|:---:|
| **Hadamard-domain KQ dot product** | yes | -- |
| Warp-cooperative FWHT in Flash Attention | yes | -- |
| Branchless sign x norm fusion | yes | -- |
| Precomputed sign bits (zero Philox at dequant) | yes | some |
| Sparse V Dequant (+22% decode) | yes | some |
| CUDA (CC 6.1+) | **yes** | yes |
| Metal (Apple Silicon) | -- | yes (TheTom) |

**Key innovation:** Instead of inverse-FWHT on K (expensive, per-block in the inner loop), we apply FWHT to Q once and dot directly against codebook values. This exploits FWHT orthogonality to eliminate all gather shuffles and branch divergence -- **39% fewer warp shuffles** per vec_dot call.

## Quick Start

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server

# Maximum compression (3.5 bpw) -- 384K context on 12GB VRAM
./build/bin/llama-server -m model.gguf \
    --cache-type-k tq2_1 --cache-type-v tq2_1 \
    -fa on -ngl 99
```

### Available Types

| Type | bpw | VRAM vs f16 | Use Case |
|------|-----|-------------|----------|
| `tq2_1` | 3.5 | **-78%** | Maximum compression, long context |
| `tq3_1` | 4.5 | -72% | Balanced quality/compression |
| `tq4_1` | 5.5 | -66% | Best quality, better PPL than q4_0 |

## Benchmarks

### CC 7.5 (12 GB) -- Qwen3.5-35B-A3B IQ2_XS

| KV Cache | bpw | pp512 | tg32 | TG vs q4_0 |
|----------|-----|:---:|:---:|:---:|
| q4_0 | 4.5 | 838 | 69.5 | -- |
| **tq2_1 v7** | 3.5 | 634 | 63.4 | -8.8% |

8.8% TG penalty with 78% less VRAM. v7 is **+65% faster** than v6 on this GPU.

### CC 6.1 (6 GB) -- Ministral 3B IQ2_M

| KV Cache | bpw | pp128 | tg32 | TG vs q4_0 |
|----------|-----|:---:|:---:|:---:|
| q4_0 | 4.5 | 798 | 43.0 | -- |
| **tq2_1 v7** | 3.5 | 211 | 33.3 | -22.6% |

### NIAH Retrieval (@sztlink, RTX 4090)

q8_0-K + turbo3-V, Qwen3-30B-A3B, 65K context: **25/25 = 100%** retrieval across all depths and context lengths (4K-49K).

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/turboquant.md](docs/turboquant.md) | Technical deep-dive, CUDA implementation, benchmarks, roadmap |
| [docs/DELTA.md](docs/DELTA.md) | Complete file-by-file delta from upstream llama.cpp |

## Related Projects

| Project | Focus | Hardware |
|---------|-------|----------|
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Reference community fork | Metal + CUDA + CPU |
| [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | Trellis-constrained quantization | CUDA (RTX 3090) |
| [Madreag/turbo3-cuda](https://github.com/Madreag/turbo3-cuda) | Blackwell (SM120) support | CUDA (RTX 5090) |
| [Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) | Community coordination thread | -- |

## Paper

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Google Research, ICLR 2026 -- https://arxiv.org/abs/2504.19874

## License

MIT (inherited from llama.cpp)
