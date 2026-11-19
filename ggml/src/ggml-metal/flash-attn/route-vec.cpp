#include "../ops/ggml-metal-ops-internal.h"
#include "fa-shared.h"

// Vec route: small batches; workgroups write partial results that a reduce kernel folds.
int ggml_metal_fa_route_vec(ggml_metal_op_t ctx, int idx, const struct fa_shared & s) {
    ggml_tensor * op = ctx->node(idx);
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

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
    const ggml_metal_buffer_id bid_tmp = s.bid_tmp;

    // half4x4 kernel
    const int nqptg = OP_FLASH_ATTN_EXT_VEC_NQPSG; // queries per threadgroup
    const int ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
    const int nhptg = 1;                           // heads per threadgroup

    GGML_ASSERT(nqptg <= 32);
    GGML_ASSERT(nqptg  % 1  == 0);
    GGML_ASSERT(ncpsg  % 32 == 0);

    bool need_sync = false;

    const bool has_kvpad = ne11 % ncpsg != 0;

    if (has_kvpad) {
        ggml_metal_fa_encode_pad(ctx, idx, s, ncpsg);

        need_sync = true;
    }

    if (need_sync) {
        ggml_metal_op_concurrency_reset(ctx, idx);
    }

    // note: for simplicity assume the K is larger or equal than V
    GGML_ASSERT(ne10 >= ne20);

    // ne00 + 2*ncpsg*(nsg)
    // for each query, we load it as f16 in shared memory (ne00)
    // and store the soft_max values and the mask
    //
    // ne20*(nsg)
    // each simdgroup has a full f32 head vector in shared mem to accumulate results
    //
#define FATTN_SMEM(nsg) (GGML_PAD(((GGML_PAD(ne00, 128) + 4*ncpsg + 2*GGML_PAD(ne20, 128))*(nsg))*(sizeof(float)/2), 16))

    int64_t nsg = 1;

    // workgroups
    // each workgroup handles nsg*nkpsg cache values
    int32_t nwg = 1;
    if (false) {
        // for small KV caches, we could launch a single workgroup and write the results directly to dst/
        // however, this does not lead to significant improvement, so disabled
        nwg = 1;
        nsg = 4;
    } else {
        nwg = 32;
        nsg = 1;
        while (2*nwg*nsg*ncpsg < ne11 && nsg < 4) {
            nsg *= 2;
        }
    }

    ggml_metal_kargs_flash_attn_ext_vec args = {
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

    auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, nwg);

    GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
    ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
    ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);

    const size_t smem = FATTN_SMEM(nsg);

    //printf("smem: %zu, max: %zu, nsg = %d, nsgmax = %d\n", smem, props_dev->max_theadgroup_memory_size, (int) nsg, (int) nsgmax);
    GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

    if (nwg == 1) {
        assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) == 0);

        // using 1 workgroup -> write the result directly into dst
        ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);
    } else {
        // sanity checks
        assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

        GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
        GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

        // write the results from each workgroup into a temp buffer
        ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
        ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);

        // sync the 2 kernels
        ggml_metal_op_concurrency_reset(ctx, idx);

        // reduce the results from the workgroups
        {
            const int32_t nrows = ne1*ne2*ne3;

            ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                nrows,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
        }
    }
#undef FATTN_SMEM

    return 1;
}
