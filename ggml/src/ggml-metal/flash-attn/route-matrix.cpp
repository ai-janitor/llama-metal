#include "../ops/ggml-metal-ops-internal.h"
#include "fa-shared.h"

// Matrix route: the simdgroup matrix kernel, with the pad and mask-block pre-passes.
int ggml_metal_fa_route_matrix(ggml_metal_op_t ctx, int idx, const struct fa_shared & s) {
    ggml_tensor * op = ctx->node(idx);
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

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

    const bool has_mask = s.has_mask;
    const bool has_sinks = s.has_sinks;
    const bool has_bias = s.has_bias;
    const bool has_scap = s.has_scap;
    const float scale = s.scale;
    const float max_bias = s.max_bias;
    const float m0 = s.m0;
    const float m1 = s.m1;
    const int32_t n_head_log2 = s.n_head_log2;
    const float logit_softcap = s.logit_softcap;
    const ggml_metal_buffer_id bid_src0 = s.bid_src0;
    const ggml_metal_buffer_id bid_src1 = s.bid_src1;
    const ggml_metal_buffer_id bid_src2 = s.bid_src2;
    const ggml_metal_buffer_id bid_src3 = s.bid_src3;
    const ggml_metal_buffer_id bid_src4 = s.bid_src4;
    const ggml_metal_buffer_id bid_dst = s.bid_dst;
    const ggml_metal_buffer_id bid_pad = s.bid_pad;
    const ggml_metal_buffer_id bid_blk = s.bid_blk;

    // half8x8 kernel
    const int nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
    const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

    GGML_ASSERT(nqptg <= 32);
    GGML_ASSERT(nqptg  % 8  == 0);
    GGML_ASSERT(ncpsg  % 32 == 0);

    bool need_sync = false;

    const bool has_kvpad = ne11 % ncpsg != 0;

    if (has_kvpad) {
        ggml_metal_fa_encode_pad(ctx, idx, s, ncpsg);

        need_sync = true;
    }

    if (has_mask) {
        ggml_metal_fa_encode_blk(ctx, idx, s, nqptg, ncpsg);

        need_sync = true;
    }

    if (need_sync) {
        ggml_metal_op_concurrency_reset(ctx, idx);
    }

    const int is_q = ggml_is_quantized(op->src[1]->type) ? 1 : 0;

    // 2*(2*ncpsg)
    // ncpsg soft_max values + ncpsg mask values
    //
    // 16*32*(nsg)
    // the shared memory needed for the simdgroups to load the KV cache
    // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
    //
#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(ne00 + 2*GGML_PAD(ne20, 64) + 2*(2*ncpsg)) + is_q*(16*32*(nsg)))*(sizeof(float)/2), 16))

    //int64_t nsgmax = 4;
    //
    //if (is_q) {
    //    nsgmax = 2;
    //    while (true) {
    //        const size_t smem = FATTN_SMEM(nsgmax);
    //        if (smem > props_dev->max_theadgroup_memory_size) {
    //            break;
    //        }
    //        nsgmax *= 2;
    //    }
    //    nsgmax /= 2;
    //}

    // simdgroups per threadgroup (a.k.a. warps)
    //nsg = ne01 <= nqptg ? MAX(4, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32))) : 4;
    int32_t nsg = ne00 >= 512 ? 8 : 4;

    const size_t smem = FATTN_SMEM(nsg);

    ggml_metal_kargs_flash_attn_ext args = {
        /*.ne01          =*/ ne01,
        /*.ne02          =*/ ne02,
        /*.ne03          =*/ ne03,
        /*.nb01          =*/ nb01,
        /*.nb02          =*/ nb02,
        /*.nb03          =*/ nb03,
        /*.ne11          =*/ ne11,
        /*.ne_12_2       =*/ ne12,
        /*.ne_12_3       =*/ ne13,
        /*.ns10          =*/ int32_t(nb11/nb10),
        /*.nb11          =*/ nb11,
        /*.nb12          =*/ nb12,
        /*.nb13          =*/ nb13,
        /*.ns20          =*/ int32_t(nb21/nb20),
        /*.nb21          =*/ nb21,
        /*.nb22          =*/ nb22,
        /*.nb23          =*/ nb23,
        /*.ne31          =*/ ne31,
        /*.ne32          =*/ ne32,
        /*.ne33          =*/ ne33,
        /*.nb31          =*/ nb31,
        /*.nb32          =*/ nb32,
        /*.nb33          =*/ nb33,
        /*.ne1           =*/ ne1,
        /*.ne2           =*/ ne2,
        /*.ne3           =*/ ne3,
        /*.scale         =*/ scale,
        /*.max_bias      =*/ max_bias,
        /*.m0            =*/ m0,
        /*.m1            =*/ m1,
        /*.n_head_log2   =*/ n_head_log2,
        /*.logit_softcap =*/ logit_softcap,
    };

    auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
    ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
    ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
    ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
    ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  8);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
#undef FATTN_SMEM

    return 1;
}
