// Generated hardware rows (input SHA-256 b949e40e2ab6). Do not edit.
#pragma once

#include <stdint.h>

// -1 or "" = unknown
struct ggml_metal_hw_row {
    uint32_t     pci_vendor;
    uint32_t     pci_device;
    const char * chip;
    const char * arch_family;
    int32_t      simd_width;
    int32_t      compute_units;
    int32_t      l1_cache_per_cu; // bytes
    int32_t      l2_cache_total;  // bytes
    int64_t      local_bw;        // bytes/s
    int32_t      int_dot;
    int32_t      simd_sum_w64_ok;
    int32_t      mv_reduce;       // matvec row-sum tail: 0 serial, 1 tree
    int32_t      f_scalar_w64;    // f16 and bf16 scalar matvec at width 64: 1 on, 0 off
    int32_t      mv_group;        // Q8_0 matvecs of one activation in one launch: 1 on, 0 off
    int32_t      attn_gate_fuse;  // attention gate copy, sigmoid, copy, multiply in one launch: 1 on, 0 off
    int32_t      conv_l2_fold;    // conv_state also writes the split_l2 norms of q and k: 1 on, 0 off
    int32_t      add_norm_fuse;   // residual add, rms norm of its sum, weight multiply in one launch: 1 on, 0 off
    int32_t      cb_seq;          // command buffers after the first, encoded in order on the calling thread: 1-8
    int32_t      q8_load_bytes;   // Q8_0 matvec weight load width in bytes: 1 or 4 (aligned words)
    int32_t      q8_rows8_min;    // smallest row count of a single Q8_0 matvec run at 8 whole rows per simdgroup
    int32_t      q8_act_load_bytes; // Q8_0 matvec activation load width in bytes: 4 or 16 (two aligned loads per lane)
    int32_t      set_nowait;      // input uploads committed without waiting for each: 1 on, 0 off
    int32_t      staging_keep;    // one logits staging buffer kept per context: 1 on, 0 off
    int32_t      mv_group_red;    // grouped K-quant matvec stores every row before the row sums: 1 on, 0 off
};

static const struct ggml_metal_hw_row ggml_metal_hw_rows[] = {
    { 0x1002, 0x6798, "FirePro D700 (GCN 1.0, Tahiti)", "gcn1", 64, 32, 16384, 786432, 263000000000LL, -1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 16384, 16, 1, 1, 1 },
    { 0x1002, 0x7340, "Radeon Pro 5500M (RDNA 1, Navi 14)", "rdna1", 32, 24, 16384, 4194304, 192000000000LL, -1, -1, -1, 0, 1, 1, 0, 0, 4, 1, -1, -1, 1, 1, -1 },
    { 0x1002, 0x6860, "Radeon Pro Vega 64 (GCN 5, Vega 10)", "gcn5", 64, -1, 16384, 4194304, -1LL, -1, 1, 0, 1, 1, 1, 1, 1, 4, 4, -1, -1, 1, 1, -1 },
};

static const int ggml_metal_hw_rows_n = (int) (sizeof(ggml_metal_hw_rows) / sizeof(ggml_metal_hw_rows[0]));

// Per-shape matvec rows (input SHA-256 215baa62ef82). See ggml_device_mv_row in ggml-device-profile.h.
// { pci_vendor, pci_device, kind (0 single, 1 group), ggml_type, ne00, ne01, nr0, prep, cols, form }; the last row is the end mark
static const struct ggml_device_mv_row ggml_metal_hw_mv_rows[] = {
    { 0x1002, 0x6798, 0, 14, 2560, 248320, 8, 0, 1, 0 },
    { 0x1002, 0x6798, 0, 12, 2560, 4096, 8, 1, 1, 0 },
    { 0x1002, 0x6798, 0, 12, 2560, 4096, 8, 0, 1, 0 },
    { 0x1002, 0x6798, 0, 12, 9216, 2560, 4, 1, 1, 0 },
    { 0x1002, 0x6798, 0, 12, 4096, 2560, 2, 1, 1, 0 },
    { 0x1002, 0x6798, 1, 12, 2560, 9216, 8, 1, 1, 0 },
    { 0x1002, 0x6798, 0, 13, 2560, 8192, 8, 0, 1, 0 },
    { 0x1002, 0x6798, 0, 13, 4096, 2560, 8, 0, 1, 0 },
    { 0x1002, 0x6798, 0, 12, 2560, 9216, 4, 1, 2, 4 },
    { 0x1002, 0x6798, 0, 12, 2560, 8192, 4, 0, 2, 4 },
    { 0x1002, 0x6798, 0, 12, 4096, 2560, 2, 1, 2, 0 },
    { 0x1002, 0x6798, 0, 12, 9216, 2560, 2, 0, 2, 2 },
    { 0x1002, 0x6798, 0, 12, 2560, 4096, 4, 1, 2, 4 },
    { 0x1002, 0x6798, 0, 12, 2560, 1024, 2, 0, 2, 2 },
    { 0x1002, 0x6798, 0, 14, 2560, 8192, 4, 0, 2, 4 },
    { 0x1002, 0x6798, 0, 14, 9216, 2560, 2, 0, 2, 2 },
    { 0x1002, 0x6798, 0, 14, 2560, 1024, 2, 0, 2, 0 },
    { 0x1002, 0x6798, 0, 14, 2560, 248320, 4, 0, 2, 4 },
    { 0x1002, 0x6798, 0, 12, 2560, 9216, 4, 1, 3, 4 },
    { 0x1002, 0x6798, 0, 12, 2560, 4096, 4, 1, 3, 4 },
    { 0x1002, 0x6798, 0, 12, 2560, 1024, 2, 0, 3, 4 },
    { 0x1002, 0x6798, 0, 12, 2560, 8192, 4, 0, 3, 4 },
    { 0x1002, 0x6798, 0, 12, 4096, 2560, 4, 1, 3, 2 },
    { 0x1002, 0x6798, 0, 12, 9216, 2560, 4, 1, 3, 2 },
    { 0x1002, 0x6798, 0, 14, 2560, 8192, 2, 0, 3, 4 },
    { 0x1002, 0x6798, 0, 14, 9216, 2560, 2, 0, 3, 2 },
    { 0x1002, 0x6798, 0, 14, 2560, 1024, 2, 0, 3, 2 },
    { 0x1002, 0x6798, 0, 14, 2560, 248320, 4, 0, 3, 4 },
    { 0x1002, 0x6798, 0, 12, 2560, 9216, 2, 1, 4, 0 },
    { 0x1002, 0x6798, 0, 12, 2560, 8192, 1, 0, 4, 1 },
    { 0x1002, 0x6798, 0, 12, 4096, 2560, 2, 1, 4, 2 },
    { 0x1002, 0x6798, 0, 12, 9216, 2560, 2, 1, 4, 2 },
    { 0x1002, 0x6798, 0, 12, 2560, 4096, 2, 1, 4, 2 },
    { 0x1002, 0x6798, 0, 12, 2560, 1024, 1, 0, 4, 1 },
    { 0x1002, 0x6798, 0, 14, 2560, 8192, 2, 0, 4, 3 },
    { 0x1002, 0x6798, 0, 14, 9216, 2560, 2, 0, 4, 3 },
    { 0x1002, 0x6798, 0, 14, 2560, 1024, 1, 0, 4, 1 },
    { 0x1002, 0x6798, 0, 14, 2560, 248320, 1, 0, 4, 2 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

static const int ggml_metal_hw_mv_rows_n = (int) (sizeof(ggml_metal_hw_mv_rows) / sizeof(ggml_metal_hw_mv_rows[0])) - 1;

// Minimum column count for mul_mv_ext by K-quant type (input SHA-256 60e0f5884801).
// { pci_vendor, pci_device, ggml_type, ext_min }; the last row is the end mark
static const struct ggml_device_mv_ext_min ggml_metal_hw_mv_ext_min[] = {
    { 0x1002, 0x6798, 12, 5 },
    { 0x1002, 0x6798, 14, 5 },
    { 0, 0, 0, 0 },
};

static const int ggml_metal_hw_mv_ext_min_n = (int) (sizeof(ggml_metal_hw_mv_ext_min) / sizeof(ggml_metal_hw_mv_ext_min[0])) - 1;
