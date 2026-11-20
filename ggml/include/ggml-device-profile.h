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

// Per-shape matvec row of a card: a K-quant weight of this type and shape (ne00 and ne01 -1 = every shape) runs nr0 rows per simdgroup.
// kind 0 = single matvec, 1 = grouped launch of such members; prep 1 = only with headers prepared at load;
// cols 1 = one activation column, 2-4 = a decode batch of that many columns in one launch, in loop form "form".
// card = the card's Metal device name, compared whole.
struct ggml_device_mv_row {
    const char * card;
    int32_t  kind;
    int32_t  type; // ggml_type
    int64_t  ne00;
    int64_t  ne01;
    int32_t  nr0;
    int32_t  prep;
    int32_t  cols;
    int32_t  form; // bit 0 = weights held (activations reloaded per row), bit 1 = weights kept packed
    uint32_t wave;        // the wave and the threadgroup memory the card reported when the row was measured:
    uint32_t group_bytes; // the row is used only while the card reports both
};

struct ggml_device_profile {
    // Identity
    enum ggml_gpu_vendor vendor;
    char name[64]; // the device's name; 63 bytes are kept, so a name of 63 bytes here may be a longer one cut short (table rows hold at most 62)

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

    // Cache hierarchy (0 = unknown).
    // These come from a hardware lookup table, not Metal API queries.
    // Used to tune activation reuse (nr0), tiling, and prefetch strategies.
    uint32_t l1_cache_per_cu;     // L1 data cache per CU/SM, bytes (e.g., 16384 for GCN)
    uint32_t l2_cache_total;      // L2 cache total, bytes (e.g., 786432 for D700)

    // Attention gate copy, sigmoid, output copy and multiply: 1 runs them as one launch, 0 one launch each (from the hw row).
    uint32_t attn_gate_fuse;

    // conv_state launch also writes split_l2 of its q and k heads: 1 on, 0 off (from the hw row).
    uint32_t conv_l2_fold;

    // Residual add, rms norm of its sum and the weight multiply: 1 runs them as one launch, 0 one launch each (from the hw row).
    uint32_t add_norm_fuse;

    // Command buffers after the first, encoded and committed in order on the calling thread (from the hw row); 0 keeps one buffer encoded as before.
    uint32_t cb_seq;

    // Discrete card: 1 commits each input upload without waiting for it (the graph runs after it on the same queue), 0 waits (from the hw row).
    uint32_t set_nowait;

    // Discrete card: 1 keeps one logits staging buffer per context and reuses it every token, 0 makes a new one per read (from the hw row).
    uint32_t staging_keep;

    // Per-type column count where K-quants take mul_mv_ext on this card (generated rows); none = 4.

    // Bit i set = lane capability i is reached and correct on this card (generated capability rows); 0 = base lanes only.
    uint32_t lane_ok;
};

#ifdef __cplusplus
}
#endif
