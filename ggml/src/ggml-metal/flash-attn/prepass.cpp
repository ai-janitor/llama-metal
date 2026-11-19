#include "../ops/ggml-metal-ops-internal.h"
#include "fa-shared.h"

// Pad the K/V tail (and the mask) to a multiple of ncpsg cache values.
void ggml_metal_fa_encode_pad(ggml_metal_op_t ctx, int idx, const struct fa_shared & s, int ncpsg) {
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
    const ggml_metal_buffer_id bid_src1 = s.bid_src1;
    const ggml_metal_buffer_id bid_src2 = s.bid_src2;
    const ggml_metal_buffer_id bid_src3 = s.bid_src3;
    const ggml_metal_buffer_id bid_pad = s.bid_pad;

    assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

    ggml_metal_kargs_flash_attn_ext_pad args0 = {
        /*.ne11    =*/ne11,
        /*.ne_12_2 =*/ne12,
        /*.ne_12_3 =*/ne13,
        /*.nb11    =*/nb11,
        /*.nb12    =*/nb12,
        /*.nb13    =*/nb13,
        /*.nb21    =*/nb21,
        /*.nb22    =*/nb22,
        /*.nb23    =*/nb23,
        /*.ne31    =*/ne31,
        /*.ne32    =*/ne32,
        /*.ne33    =*/ne33,
        /*.nb31    =*/nb31,
        /*.nb32    =*/nb32,
        /*.nb33    =*/nb33,
    };

    auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

    ggml_metal_encoder_set_pipeline(enc, pipeline0);
    ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
    ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

    assert(ne12 == ne22);
    assert(ne13 == ne23);

    ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);
}

// Mark the (query block, cache block) pairs the mask rules out.
void ggml_metal_fa_encode_blk(ggml_metal_op_t ctx, int idx, const struct fa_shared & s, int nqptg, int ncpsg) {
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

    const ggml_metal_buffer_id bid_src3 = s.bid_src3;
    const ggml_metal_buffer_id bid_blk = s.bid_blk;

    assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

    ggml_metal_kargs_flash_attn_ext_blk args0 = {
        /*.ne01 =*/ ne01,
        /*.ne30 =*/ ne30,
        /*.ne31 =*/ ne31,
        /*.ne32 =*/ ne32,
        /*.ne33 =*/ ne33,
        /*.nb31 =*/ nb31,
        /*.nb32 =*/ nb32,
        /*.nb33 =*/ nb33,
    };

    auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

    ggml_metal_encoder_set_pipeline(enc, pipeline0);
    ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

    const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
    const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

    ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);
}
