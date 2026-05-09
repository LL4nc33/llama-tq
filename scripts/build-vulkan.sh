#!/usr/bin/env bash
# Build llama-tq with the Vulkan backend.
#
# Vulkan extends the supported GPU vendors beyond CUDA (NVIDIA-only):
#   - AMD (Radeon, MI-series via amdgpu/RADV)
#   - Intel (Arc, integrated graphics via ANV)
#   - Apple Silicon (via MoltenVK)
#   - NVIDIA (via the proprietary Vulkan ICD)
#
# Caveat: the TurboQuant KV cache types (KTQ*/VTQ*) are CUDA-only.
# Vulkan builds silently fall back to F16 KV; CUDA remains the primary
# path for KV-quant deployments. See docs/vulkan.md for the full status
# matrix and tuning knobs.
#
# Required system packages (Ubuntu/Debian, distro shaderc — for best perf
# install the LunarG SDK instead, see docs/vulkan.md):
#   sudo apt-get install -y libvulkan-dev glslc spirv-tools spirv-headers
#
# Usage:
#   ./scripts/build-vulkan.sh                 # release build into build-vulkan/
#   BUILD_DIR=build-vk-debug BUILD_TYPE=Debug ./scripts/build-vulkan.sh
#   JOBS=8 ./scripts/build-vulkan.sh          # override parallel job count

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build-vulkan}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
TARGETS=("${@:-llama-server llama-bench llama-cli}")

cd "$REPO_ROOT"

echo "==> Configuring Vulkan build in $BUILD_DIR ($BUILD_TYPE, -j$JOBS)"
cmake -B "$BUILD_DIR" \
  -DGGML_VULKAN=ON \
  -DGGML_CUDA=OFF \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "==> Building targets: ${TARGETS[*]}"
# shellcheck disable=SC2086
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$JOBS" -t ${TARGETS[*]}

echo "==> Done. Binaries are in $BUILD_DIR/bin/"
ls -la "$BUILD_DIR/bin/" | grep -E 'llama-(server|bench|cli)$' || true
