#pragma once

// KV cache device router for heterogeneous GPU splits.
//
// On dual-GPU Macs (AMD dGPU + Intel iGPU), the dGPU's PCIe bandwidth
// bottlenecks KV cache attention at high context lengths. The iGPU sits
// on the memory controller with direct system RAM access (25-40 GB/s).
// This router pins KV attention ops to the iGPU while weight matmuls
// stay on the dGPU — the "Memory Scout" pattern.
//
// Uses ggml_backend_sched_set_tensor_backend() to override per-tensor
// backend assignment. The scheduler auto-inserts copies for query vectors
// crossing the PCIe bridge (negligible overhead for small Q tensors).

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>

// Compression tiers for KV cache on iGPU
enum kv_compression_type {
    KV_COMPRESSION_NONE   = 0,  // f16 KV cache, no compression
    KV_COMPRESSION_Q4_0   = 1,  // 4-bit quantization (75% reduction, Tier 1)
    KV_COMPRESSION_Q8_0   = 2,  // 8-bit quantization (50% reduction, conservative)
    KV_COMPRESSION_SPARSE = 3,  // 4-bit + head pruning (90% target, Tier 1+2)
};

struct kv_compression_config {
    kv_compression_type type;              // which compression tier
    float               sparsity_threshold; // attention score below which heads are pruned (Tier 2)
};

struct llama_kv_router_params {
    bool                    enabled;        // toggle (default: false)
    int32_t                 kv_device_idx;  // backend index for KV ops (e.g., MTL1)
    kv_compression_config   compression;    // KV cache compression config
};

// Scan graph, pin KV attention ops to kv_backend
void llama_kv_router_apply(
    ggml_backend_sched_t   sched,
    struct ggml_cgraph   * graph,
    ggml_backend_t         kv_backend
);
