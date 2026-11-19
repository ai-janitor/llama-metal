#pragma once

// Values the flash attention router computes once and the routes read.
// Tensor sizes are not here: each route takes them from the op again.
#include "../ops/ggml-metal-ops-internal.h"

struct fa_shared {
    float scale; // already divided by the soft cap when there is one
    float max_bias;
    float logit_softcap;
    bool  has_mask;
    bool  has_sinks;
    bool  has_bias;
    bool  has_scap;
    int32_t n_head_log2;
    float m0;
    float m1;
    ggml_metal_buffer_id bid_src0;
    ggml_metal_buffer_id bid_src1;
    ggml_metal_buffer_id bid_src2;
    ggml_metal_buffer_id bid_src3; // src0 when there is no mask
    ggml_metal_buffer_id bid_src4; // src0 when there are no sinks
    ggml_metal_buffer_id bid_dst;
    ggml_metal_buffer_id bid_pad;
    ggml_metal_buffer_id bid_blk;
    ggml_metal_buffer_id bid_tmp;
};

// Each route encodes the op and returns 1.
int ggml_metal_fa_route_intel_tiled(ggml_metal_op_t ctx, int idx, const struct fa_shared & s);
int ggml_metal_fa_route_scalar     (ggml_metal_op_t ctx, int idx, const struct fa_shared & s);
int ggml_metal_fa_route_matrix     (ggml_metal_op_t ctx, int idx, const struct fa_shared & s);
int ggml_metal_fa_route_vec        (ggml_metal_op_t ctx, int idx, const struct fa_shared & s);

// Pre-passes: pad the K/V tail to a multiple of ncpsg; mark the mask blocks that are fully masked.
void ggml_metal_fa_encode_pad(ggml_metal_op_t ctx, int idx, const struct fa_shared & s, int ncpsg);
void ggml_metal_fa_encode_blk(ggml_metal_op_t ctx, int idx, const struct fa_shared & s, int nqptg, int ncpsg);
