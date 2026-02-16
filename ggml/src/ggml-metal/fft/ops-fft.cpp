#include "../ops/ggml-metal-ops-internal.h"

int ggml_metal_op_irfft(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_irfft args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1
    };

    auto pipeline = ggml_metal_library_get_pipeline_irfft(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_fold(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t n_out = ((const int32_t *)(op->op_params))[0];
    const int32_t n_hop = ((const int32_t *)(op->op_params))[1];
    const int32_t n_pad = ((const int32_t *)(op->op_params))[2];

    ggml_metal_kargs_fold args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.ne10  =*/ ne10,
        /*.nb10  =*/ nb10,
        /*.ne0   =*/ ne0,
        /*.nb0   =*/ nb0,
        /*.n_hop =*/ n_hop,
        /*.n_pad =*/ n_pad
    };

    auto pipeline = ggml_metal_library_get_pipeline_fold(lib, op);

    // Output length might be large, need multiple threadgroups if > 1024
    const int nth = std::min(1024, ne0);
    const int n_tg = (ne0 + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, nth, 1, 1);

    return 1;
}
