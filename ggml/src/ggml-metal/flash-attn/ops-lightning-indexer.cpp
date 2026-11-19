#include "../ops/ggml-metal-ops-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(flash_attn) int ggml_metal_op_lightning_indexer(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * w = op->src[2];
    const ggml_tensor * m = op->src[3];

    int kv_code = FA_SCALAR_KV_F16;
    switch (k->type) {
        case GGML_TYPE_F16:    kv_code = FA_SCALAR_KV_F16;    break;
        case GGML_TYPE_F32:    kv_code = FA_SCALAR_KV_F32;    break;
        case GGML_TYPE_BF16:   kv_code = FA_SCALAR_KV_BF16;   break;
        case GGML_TYPE_Q8_0:   kv_code = FA_SCALAR_KV_Q8_0;   break;
        case GGML_TYPE_Q4_0:   kv_code = FA_SCALAR_KV_Q4_0;   break;
        case GGML_TYPE_Q4_1:   kv_code = FA_SCALAR_KV_Q4_1;   break;
        case GGML_TYPE_Q5_0:   kv_code = FA_SCALAR_KV_Q5_0;   break;
        case GGML_TYPE_Q5_1:   kv_code = FA_SCALAR_KV_Q5_1;   break;
        case GGML_TYPE_IQ4_NL: kv_code = FA_SCALAR_KV_IQ4_NL; break;
        default: GGML_ABORT("unsupported K type for the lightning indexer");
    }

    ggml_metal_kargs_lightning_indexer args = {
        /*.n_kv     =*/ (int32_t) k->ne[2],
        /*.n_batch  =*/ (int32_t) q->ne[2],
        /*.n_embd   =*/ (int32_t) q->ne[0],
        /*.n_head   =*/ (int32_t) q->ne[1],
        /*.mask_ne3 =*/ (int32_t) m->ne[3],
        /*.pad0     =*/ 0,
        /*.nb1      =*/ op->nb[1],
        /*.nb3      =*/ op->nb[3],
        /*.nbq1     =*/ q->nb[1],
        /*.nbq2     =*/ q->nb[2],
        /*.nbq3     =*/ q->nb[3],
        /*.nbk2     =*/ k->nb[2],
        /*.nbk3     =*/ k->nb[3],
        /*.nbw1     =*/ w->nb[1],
        /*.nbw3     =*/ w->nb[3],
        /*.nbm1     =*/ m->nb[1],
        /*.nbm3     =*/ m->nb[3],
    };

    auto pipeline = ggml_metal_library_get_pipeline_lightning_indexer(ctx->lib, kv_code);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(q),  1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(k),  2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w),  3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(m),  4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);

    ggml_metal_encoder_dispatch_threadgroups(enc, (k->ne[2] + 31)/32, q->ne[2], q->ne[3], 32, 1, 1);

    return 1;
}
