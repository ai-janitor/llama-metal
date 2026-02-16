#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Backend-agnostic device profile
// Classifies GPU capabilities along two axes:
//   1. Memory path: unified vs discrete, bandwidth tiers
//   2. Compute tier: SIMD/matrix hardware, threadgroup limits

enum ggml_gpu_vendor {
    GGML_GPU_VENDOR_UNKNOWN = 0,
    GGML_GPU_VENDOR_APPLE   = 1,
    GGML_GPU_VENDOR_AMD     = 2,
    GGML_GPU_VENDOR_INTEL   = 3,
    GGML_GPU_VENDOR_NVIDIA  = 4,
};

enum ggml_gpu_connection {
    GGML_GPU_CONNECTION_INTEGRATED = 0,  // on-package, unified memory
    GGML_GPU_CONNECTION_INTERNAL   = 1,  // discrete, PCIe/TB internal
    GGML_GPU_CONNECTION_EXTERNAL   = 2,  // discrete, TB external enclosure
    GGML_GPU_CONNECTION_NETWORK    = 3,  // remote GPU over network
};

struct ggml_device_profile {
    // Identity
    enum ggml_gpu_vendor vendor;
    char name[64];

    // Memory path
    bool     shared_memory;       // true = unified/UMA, false = discrete
    uint64_t vram_size;           // bytes
    uint64_t local_bandwidth;     // bytes/sec (GPU-local, 0 if unknown)
    uint64_t transfer_bandwidth;  // bytes/sec (host<->GPU, 0 if unknown)

    // Compute tier
    bool     has_matrix_hw;       // hardware matrix multiply (simdgroup_mm, tensor cores)
    bool     has_simd_reduction;  // SIMD shuffle/reduction intrinsics
    uint32_t compute_units;       // SMs/CUs (0 if unknown)
    uint32_t simd_width;          // warp/wavefront size
    uint32_t max_threads_per_threadgroup;
    uint32_t shared_mem_size;     // threadgroup/shared memory per block (bytes)
    uint32_t max_threadgroups_per_dispatch; // max threadgroups per dispatch for timeout-prone ops (0 = unlimited)
};

#ifdef __cplusplus
}
#endif
