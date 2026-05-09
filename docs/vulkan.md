# Vulkan Backend

`llama-tq` ships with the upstream `ggml-vulkan` backend enabled, plus a
small set of Turing-specific tunings that close most of the historical
Vulkan-vs-CUDA TG gap on `sm_75` hardware.

## Status

- **Standard quants** (Q4_K_M, IQ-types, F16, etc.): fully supported via
  upstream-maintained shaders.
- **TurboQuant KV cache** (KTQ1-4 / VTQ1-4): **CUDA-only**. Vulkan builds
  silently fall back to F16 KV. A GLSL port is a multi-week effort and is
  not on the immediate roadmap.
- **Multi-GPU**: works through Vulkan device selection (`--main-gpu`,
  `--tensor-split`).
- **Turing TG tuning**: `rm_kq=4` is enabled by default on `NVIDIA_TURING`
  devices (sm_75 — RTX 2060/2070/2080 family). See *Tuning* below.

## When Vulkan helps

| GPU class                       | Typical CUDA vs Vulkan |
|---------------------------------|------------------------|
| RTX 4090 / 5090 (Ada/Blackwell) | CUDA usually leads by a few percent |
| RTX 3090 (Ampere)               | CUDA leads by ~10% on TG |
| **RTX 2060/2070 (Turing)**      | CUDA leads by ~22% on IQ2_XXS TG (after tuning); PP at parity |
| RX 7900 XTX (RDNA3)             | Vulkan dominates (no CUDA path at all) |
| Intel Arc                       | Vulkan only |
| Apple Silicon (MoltenVK)        | Vulkan only — Metal backend usually preferred |

The Turing CUDA edge is structural — CUDA's IQ-quant mat-vec path has
multi-year tuning that the Vulkan path doesn't have a direct equivalent
for. TG on IQ-quants is memory-bandwidth-bound (~80% SoL); further closing
the gap requires a DP4A MMVQ shader for IQ2 (sign-flip makes packed-int-dot
non-trivial).

## Measured (2x RTX 2060, llama-tq vulkan-turing-rmkq, May 2026)

LunarG SDK 1.4.313 + shaderc 2025.2, vulkan1.3 SPIR-V target, single GPU
(`-mg 0 -fa 1`), `rm_kq=4` Turing default on:

| Model                            | Test    | CUDA     | Vulkan   | Δ      |
|----------------------------------|---------|----------|----------|--------|
| Qwen3.6-35B-A3B-IQ2_XXS          | pp512   | 1160 t/s | 1185 t/s | **+2%** |
| Qwen3.6-35B-A3B-IQ2_XXS          | tg128   | 80 t/s   | 62 t/s   | -23%    |
| Gemma-4-26B-A4B-IQ4_XS           | tg128   | ~67 t/s  | 55 t/s   | -18%    |
| Ministral-3-3B-Q4_K_M            | pp1024  | 4048 t/s | 3798 t/s | -6%     |
| Ministral-3-3B-Q4_K_M            | tg128   | 114 t/s  | 105 t/s  | -8%     |

Pre-tuning baseline on Qwen3.6-35B-A3B-IQ2_XXS tg128 was 51.5 t/s —
the Turing `rm_kq=4` default closes ~37% of the original CUDA gap.

## Build prerequisites

For best performance, install the LunarG Vulkan SDK rather than distro
shader tooling. Ubuntu 24.04 ships shaderc 2023.8 which lacks
`coopmat2`, `int_dot_product`, and `bf16` shader features. The LunarG
SDK ships shaderc 2025.2 / glslang 15.3:

```bash
curl -fsSL https://packages.lunarg.com/lunarg-signing-key-pub.asc \
  | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/lunarg.gpg
curl -fsSL https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list \
  | sudo tee /etc/apt/sources.list.d/lunarg-vulkan-noble.list
sudo apt-get update
sudo apt-get install -y --allow-downgrades shaderc vulkan-sdk
```

Distro-only fallback (slower path):

```bash
sudo apt-get install -y libvulkan-dev glslc spirv-tools spirv-headers
```

Arch:

```bash
sudo pacman -S vulkan-devel shaderc spirv-tools spirv-headers
```

macOS (Apple Silicon, via MoltenVK):

```bash
brew install molten-vk shaderc spirv-headers
```

## Building

```bash
./scripts/build-vulkan.sh
```

Produces `build-vulkan/bin/llama-{server,bench,cli}`. The script keeps
the Vulkan build in a separate directory so a CUDA build can live in
`build/` side by side.

Custom configuration:

```bash
BUILD_DIR=build-vk-debug BUILD_TYPE=Debug ./scripts/build-vulkan.sh
JOBS=8 ./scripts/build-vulkan.sh llama-server
```

## Running

Vulkan binaries auto-detect available devices on startup. With LunarG SDK
and the modern shaderc, expect a Turing device line like:

```
ggml_vulkan: 0 = NVIDIA GeForce RTX 2060 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
```

`bf16: 1`, `int dot: 1`, and `matrix cores: NV_coopmat2` are all
dependent on the modern shaderc — if any of them say `0` or
`KHR_coopmat`, the shaderc on the host is too old.

To restrict to one GPU:

```bash
./build-vulkan/bin/llama-bench -m model.gguf -ngl 99 -mg 0 -p 256 -n 64
```

## Tuning (Turing-specific)

| Env var                  | Default | Effect |
|--------------------------|---------|--------|
| `GGML_VK_TURING_RMKQ4`   | `1`     | Use `rm_kq=4` on Turing (rows-per-workgroup for K-/IQ-quants). +20% TG on IQ2_XXS, +5% on IQ4_XS, neutral elsewhere. Set to `0` to fall back to upstream's `rm_kq=2`. |
| `GGML_VK_RM_KQ`          | (heur)  | Force a specific `rm_kq` value (1-8). For sweep benches. |
| `GGML_VK_RM_IQ`          | (heur)  | Force a specific `rm_iq` value (1-16). For sweep benches. |

The defaults work; the knobs exist for repro of A-B benches and for
sites where the heuristic might not match the device.

## Caveats

- The TurboQuant cache-type flags (`--cache-type-k ktq2`, etc.) silently
  fall back to F16 under Vulkan today. If you depend on aggressive KV
  quantisation for VRAM headroom, stay on CUDA.
- TG on IQ-quants is memory-bandwidth-bound at this point; further gains
  require a DP4A MMVQ shader for IQ2_XXS (sign-flip makes packed-int-dot
  non-trivial — see `mul_mat_vecq_funcs.glsl` for the IQ1_S/M reference).
- On heterogeneous systems (NVIDIA + AMD + Intel in one box) Vulkan will
  enumerate every visible device. Use `-mg`/`-ts` to keep things
  deterministic.

## Verified on

- 2× RTX 2060 12 GB (Turing, sm_75) — primary test box
- shaderc 2025.2 (LunarG SDK 1.4.313)
- vulkan1.3 + SPIR-V 1.6

CUDA-vendor reports (Ada, Blackwell) and AMD/Intel reports welcome via
GitHub issues.
