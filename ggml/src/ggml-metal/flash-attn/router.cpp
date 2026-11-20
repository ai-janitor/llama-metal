#include "../ops/ggml-metal-ops-internal.h"
#include "ggml-metal-probe.h"
#include "fa-shared.h"
#include "routes.h"
GGML_METAL_FAMILY(flash_attn)

int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    // K type != V type runs on the scalar kernels only (see use_scalar below)
    GGML_ASSERT(op->src[1]->type == op->src[2]->type || !profile->has_matrix_hw);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16);
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= op->src[0]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask to be at least n_queries big");

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale,         ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;
    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(ne01 < 65536);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_src3 = has_mask  ? ggml_metal_get_buffer_id(op->src[3]) : bid_src0;
    ggml_metal_buffer_id bid_src4 = has_sinks ? ggml_metal_get_buffer_id(op->src[4]) : bid_src0;

    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    const struct fa_shared s = {
        scale, max_bias, logit_softcap,
        has_mask, has_sinks, has_bias, has_scap,
        n_head_log2, m0, m1,
        bid_src0, bid_src1, bid_src2, bid_src3, bid_src4,
        bid_dst, bid_pad, bid_blk, bid_tmp,
    };

    // The rows of attention.tsv decide the route, in file order. Why each row is there is its note cell.
    // GGML_METAL_FA_SCALAR set keeps the Intel tiled route out.
    const bool force_scalar = getenv("GGML_METAL_FA_SCALAR") != nullptr;
    const bool f_matrix  = profile->has_matrix_hw;
    const bool f_intel   = profile->vendor == GGML_GPU_VENDOR_INTEL;
    const bool f_width32 = profile->simd_width == 32;
    const bool f_small   = ggml_metal_op_flash_attn_ext_use_vec(op);
    const bool f_kv_same = op->src[1]->type == op->src[2]->type;
    const bool f_fp8     = op->src[1]->type == GGML_TYPE_F8_E4M3;

    for (const struct fa_route_row & row : fa_route_rows) {
        if (!fa_route_row_matches(&row, f_matrix, f_intel, f_width32, f_small, f_kv_same, f_fp8, ne00)) {
            continue;
        }
        switch (row.route) {
            case FA_ROUTE_INTEL_TILED:
                if (!force_scalar) {
                    return ggml_metal_fa_route_intel_tiled(ctx, idx, s);
                }
                break; // the next matching row takes it
            case FA_ROUTE_SCALAR:
                return ggml_metal_fa_route_scalar(ctx, idx, s);
            case FA_ROUTE_MATRIX:
                return ggml_metal_fa_route_matrix(ctx, idx, s);
            case FA_ROUTE_VEC:
                return ggml_metal_fa_route_vec(ctx, idx, s);
        }
    }

    GGML_ABORT("flash attention: no row of attention.tsv matches");
}
