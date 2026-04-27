// MoE Expert Pinning — Tier-S Phase A (Option C, minimal pinned-RAM-only)
//
// Registers existing host-resident memory regions (e.g. mmap'd MoE expert
// tensor pages) as pinned via cudaHostRegister, enabling async PCIe DMA.
// This is the smallest behavior-changing slice of the two-tier expert cache:
// no slot ring, no eviction, no stream changes — just turn pageable host
// pages into pinned pages so the existing copy path can DMA without staging.
//
// See: docs/plans/2026-04-28-two-tier-expert-cache.md (Phase A, Option C).

#include "moe-pin.cuh"

#include <cuda_runtime.h>

#include <vector>
#include <utility>
#include <stdio.h>

// Tracks ranges we have successfully registered, so we can unregister at
// process shutdown and so callers can be idempotent (we skip overlapping
// re-registers within the exact same range).
static std::vector<std::pair<void *, size_t>> g_pinned_ranges;

extern "C" int ggml_cuda_pin_host_range(void * addr, size_t bytes) {
    if (addr == nullptr || bytes == 0) {
        return -1;
    }

    // Idempotency: skip exact duplicates (some loaders may re-iterate tensors).
    for (const auto & r : g_pinned_ranges) {
        if (r.first == addr && r.second == bytes) {
            return 0;
        }
    }

    // Try with READ_ONLY (preferred — matches mmap PROT_READ semantics and
    // gives the driver permission to keep the pages immutable for DMA).
    cudaError_t err = cudaHostRegister(addr, bytes,
                                       cudaHostRegisterPortable | cudaHostRegisterReadOnly);

    if (err != cudaSuccess) {
        // Clear sticky error; try fallback without READ_ONLY.
        (void) cudaGetLastError();
        err = cudaHostRegister(addr, bytes, cudaHostRegisterPortable);
    }

    if (err != cudaSuccess) {
        // Clear and warn — best-effort: caller continues without pin.
        (void) cudaGetLastError();
        fprintf(stderr,
                "ggml_cuda_pin_host_range: cudaHostRegister failed (%s) for "
                "%p..%p (%.1f MiB) — continuing without pin\n",
                cudaGetErrorString(err),
                addr, (char *) addr + bytes,
                (double) bytes / (1024.0 * 1024.0));
        return -1;
    }

    g_pinned_ranges.emplace_back(addr, bytes);
    return 0;
}

extern "C" void ggml_cuda_unpin_all(void) {
    for (auto & r : g_pinned_ranges) {
        cudaError_t err = cudaHostUnregister(r.first);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
        }
    }
    g_pinned_ranges.clear();
}
