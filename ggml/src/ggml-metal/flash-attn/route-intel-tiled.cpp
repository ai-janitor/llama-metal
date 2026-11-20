#include "../ops/ggml-metal-ops-internal.h"
#include "fa-shared.h"

// Intel tiled route: K/V tiled into threadgroup memory. Its row of attention.tsv names the head size it has kernels for.
int ggml_metal_fa_route_intel_tiled(ggml_metal_op_t ctx, int idx, const struct fa_shared & s) {
    ggml_tensor * op = ctx->node(idx);
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;
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

    const bool has_sinks = s.has_sinks;
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

    const int dk = ne00;
    const int dv = ne00;
    const ggml_type kv_type = op->src[1]->type;

    const char * kernel_name = nullptr;

    // Select tiled kernel by head size and K/V type (dk64 only for now)
    if (kv_type == GGML_TYPE_Q8_0) {
        if (dk == 64 && dv == 64) {
            kernel_name = "kernel_flash_attn_ext_tiled_q8_0_dk64_dv64";
        }
    } else if (kv_type == GGML_TYPE_Q4_0) {
        if (dk == 64 && dv == 64) {
            kernel_name = "kernel_flash_attn_ext_tiled_q4_0_dk64_dv64";
        }
    } else {
        // F16
        if (dk == 64 && dv == 64) {
            kernel_name = "kernel_flash_attn_ext_tiled_dk64_dv64";
        }
    }

    if (kernel_name) {
        auto pipeline = ggml_metal_library_compile_pipeline(lib, kernel_name, kernel_name, NULL);
        GGML_ASSERT(pipeline.pipeline && "Failed to compile Intel tiled FA pipeline");

        constexpr int TG_SIZE = 128;
        constexpr int Bc = 32;

        // Shared memory: Q(DK floats) + K_tile(Bc*DK halfs) + V_tile(Bc*DV halfs) + Scores(Bc floats) + Reduce(TG_SIZE floats)
        const size_t smem = dk * sizeof(float)
                          + Bc * dk * sizeof(short)  // half = 2 bytes = sizeof(short)
                          + Bc * dv * sizeof(short)
                          + Bc * sizeof(float)
                          + TG_SIZE * sizeof(float);

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01         =*/ (int32_t) ne01,
            /*.ne02         =*/ (int32_t) ne02,
            /*.ne03         =*/ (int32_t) ne03,
            /*.nb01         =*/ (uint64_t) nb01,
            /*.nb02         =*/ (uint64_t) nb02,
            /*.nb03         =*/ (uint64_t) nb03,
            /*.ne11         =*/ (int32_t) ne11,
            /*.ne_12_2      =*/ (int32_t) ne12,
            /*.ne_12_3      =*/ (int32_t) ne13,
            /*.ns10         =*/ (int32_t) (has_sinks ? 1 : 0),
            /*.nb11         =*/ (uint64_t) nb11,
            /*.nb12         =*/ (uint64_t) nb12,
            /*.nb13         =*/ (uint64_t) nb13,
            /*.ns20         =*/ 0,
            /*.nb21         =*/ (uint64_t) nb21,
            /*.nb22         =*/ (uint64_t) nb22,
            /*.nb23         =*/ (uint64_t) nb23,
            /*.ne31         =*/ (int32_t) ne31,
            /*.ne32         =*/ (int32_t) ne32,
            /*.ne33         =*/ (int32_t) ne33,
            /*.nb31         =*/ (uint64_t) nb31,
            /*.nb32         =*/ (uint64_t) nb32,
            /*.nb33         =*/ (uint64_t) nb33,
            /*.ne1          =*/ (int32_t) ne1,
            /*.ne2          =*/ (int32_t) ne2,
            /*.ne3          =*/ (int32_t) ne3,
            /*.scale        =*/ scale,
            /*.max_bias     =*/ max_bias,
            /*.m0           =*/ m0,
            /*.m1           =*/ m1,
            /*.n_head_log2  =*/ (int32_t) n_head_log2,
            /*.logit_softcap=*/ logit_softcap,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 6);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        // Chunked dispatch for Intel timeout avoidance (same as scalar)
        const int32_t total_tg = ne01 * ne02 * ne03;
        const int32_t chunk = (int32_t)profile->max_threadgroups_per_dispatch;

        if (chunk > 0 && total_tg > chunk) {
            const int32_t chunk_queries = chunk / (ne02 * ne03);
            if (chunk_queries == 0) {
                GGML_ASSERT(false && "FA chunk size too small for head*batch count");
            }

            for (int32_t q_offset = 0; q_offset < ne01; q_offset += chunk_queries) {
                const int32_t q_count = ((q_offset + chunk_queries) < ne01) ? chunk_queries : (ne01 - q_offset);

                uint32_t q_offset_u32 = (uint32_t)q_offset;
                ggml_metal_encoder_set_bytes(enc, &q_offset_u32, sizeof(q_offset_u32), 7);
                ggml_metal_encoder_dispatch_threadgroups(enc, q_count, ne02, ne03, TG_SIZE, 1, 1);
            }
        } else {
            uint32_t q_offset = 0;
            ggml_metal_encoder_set_bytes(enc, &q_offset, sizeof(q_offset), 7);
            ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, TG_SIZE, 1, 1);
        }

        return 1;
    }

    GGML_ABORT("flash attention: the Intel tiled route has no kernel for this head size; see the head cell of attention.tsv");
}
