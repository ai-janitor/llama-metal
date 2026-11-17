// GCN fallback dispatch: for AMD GPUs with 64-wide wavefronts, use dedicated kernels
// that avoid all simd intrinsics. Pure threadgroup memory + barrier reduction.
// Used for f32/f16/bf16 types where the scalar mul_mv shmem_reduce is broken at NW=64.
#include "dispatch.h"

int dispatch_mul_mat_gcn(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);

    // GCN fallback only applies to 64-wide wavefront GPUs (AMD GCN / D700)
    if (profile->simd_width != 64) {
        return 0;
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_gcn(ctx->lib, op->src[0]->type, op->src[1]->type);
    if (!pipeline.pipeline) {
        return 0;
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00, /*.ne01 =*/ ne01, /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00, /*.nb01 =*/ nb01, /*.nb02 =*/ nb02, /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10, /*.ne11 =*/ ne11, /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10, /*.nb11 =*/ nb11, /*.nb12 =*/ nb12, /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0, /*.ne1  =*/ ne1, /*.nr0  =*/ 1,
        /*.r2   =*/ r2, /*.r3   =*/ r3,
        /*.tg_x_offset =*/ 0, /*.tg_y_offset =*/ 0,
        /*.n_tg_x =*/ 0,
    };

    ggml_metal_encoder_t enc = ctx->enc;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);

    // shape attribution for profile=3
    ggml_metal_encoder_set_dispatch_shape(enc, ne00, ne01);
    // One TG per row, one simdgroup, NW=64
    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne11, ne12 * ne13, 64, 1, 1);

    return 1;
}
