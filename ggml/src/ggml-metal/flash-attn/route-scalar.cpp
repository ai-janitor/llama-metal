#include "../ops/ggml-metal-ops-internal.h"
#include "fa-shared.h"
#include "launch.h"

// Scalar route: cards without simdgroup matrices, every K/V format pair.
int ggml_metal_fa_route_scalar(ggml_metal_op_t ctx, int idx, const struct fa_shared & s) {
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

    const int dk = ne00; // head size of Q and K
    const int dv = ne0;  // head size of V and of the result
    const ggml_type kv_type = op->src[1]->type;
    const ggml_type v_type  = op->src[2]->type;

    auto fmt_code = [](ggml_type t) -> int {
        switch (t) {
            case GGML_TYPE_F16:    return FA_SCALAR_KV_F16;
            case GGML_TYPE_F32:    return FA_SCALAR_KV_F32;
            case GGML_TYPE_BF16:   return FA_SCALAR_KV_BF16;
            case GGML_TYPE_Q8_0:   return FA_SCALAR_KV_Q8_0;
            case GGML_TYPE_Q4_0:   return FA_SCALAR_KV_Q4_0;
            case GGML_TYPE_Q4_1:   return FA_SCALAR_KV_Q4_1;
            case GGML_TYPE_Q5_0:   return FA_SCALAR_KV_Q5_0;
            case GGML_TYPE_Q5_1:   return FA_SCALAR_KV_Q5_1;
            case GGML_TYPE_IQ4_NL: return FA_SCALAR_KV_IQ4_NL;
            case GGML_TYPE_F8_E4M3: return FA_SCALAR_KV_F8_E4M3;
            default: GGML_ABORT("unsupported K/V type for SCALAR FA");
        }
    };

    const int kv_code = fmt_code(kv_type);
    const int v_code  = fmt_code(v_type);

    // The typed kernels serve F16 at K == V (not 512) and Q8_0 / Q4_0 at 64, 128, 256;
    // every other head pair and format takes the kernel with the format constant.
    const bool typed_f16 = kv_type == v_type && kv_type == GGML_TYPE_F16 && dk == dv && dk != 512;
    const bool typed_q   = kv_type == v_type && (kv_type == GGML_TYPE_Q8_0 || kv_type == GGML_TYPE_Q4_0) && dk == dv &&
                           (dk == 64 || dk == 128 || dk == 256);

    char kernel_name[128];
    char pipeline_name[160];
    if (typed_f16) {
        snprintf(kernel_name, sizeof(kernel_name), "kernel_flash_attn_ext_scalar_dk%d_dv%d", dk, dv);
    } else if (typed_q) {
        snprintf(kernel_name, sizeof(kernel_name), "kernel_flash_attn_ext_scalar_%s_dk%d_dv%d", ggml_type_name(kv_type), dk, dv);
    } else {
        snprintf(kernel_name, sizeof(kernel_name), "kernel_flash_attn_ext_scalar_kv_dk%d_dv%d", dk, dv);
    }
    snprintf(pipeline_name, sizeof(pipeline_name), "%s_kv=%d_v=%d", kernel_name, kv_code, v_code);

    auto pipeline = ggml_metal_library_get_pipeline(lib, pipeline_name);
    if (!pipeline.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int32(cv, kv_code, FC_FLASH_ATTN_EXT_SCALAR + 0);
        ggml_metal_cv_set_int32(cv, v_code,  FC_FLASH_ATTN_EXT_SCALAR + 1);
        pipeline = ggml_metal_library_compile_pipeline(lib, kernel_name, pipeline_name, cv);
        ggml_metal_cv_free(cv);
    }
    GGML_ASSERT(pipeline.pipeline && "Failed to compile SCALAR FA pipeline");

    // Query SIMD width from compiled pipeline
    // NOTE: Intel Gen9 may report 32 here but run with 8-32 at runtime.
    // The kernel handles this dynamically via [[threads_per_simdgroup]].
    const int BD = ggml_metal_pipeline_thread_execution_width(pipeline);

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

    // MLX-style: 1 query per threadgroup, BN simdgroups x BD threads
    // BD comes from pipeline's threadExecutionWidth, but Intel Gen9 may run with
    // smaller BD at runtime (8 instead of 32), yielding more simdgroups (BN).
    // Shared memory must accommodate worst case: BD=8 → BN=tg_size/8.
    // simdgroups per query: the row's number, held to the threads the compiled kernel takes
    const int BN = std::min<int>(fa_launch_pick(FA_LAUNCH_SCALAR, profile->name, (int32_t) profile->simd_width).simdgroups,
                                 (int) ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) / BD);
    const int tg_size = BN * BD;
    const int BN_max = tg_size / 8;  // worst case: Intel Gen9 with BD=8
    const size_t smem = (2 * BN_max + tg_size) * sizeof(float);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    // Chunked dispatch for Intel iGPU (avoid command buffer timeout)
    // Chunk along X dimension (queries) only, keeping heads and batches full
    const int32_t total_tg = ne01 * ne02 * ne03;
    const int32_t chunk = (int32_t)profile->max_threadgroups_per_dispatch;

    if (chunk > 0 && total_tg > chunk) {
        // Calculate chunk size along query dimension
        const int32_t chunk_queries = chunk / (ne02 * ne03);
        if (chunk_queries == 0) {
            GGML_ASSERT(false && "FA chunk size too small for head*batch count");
        }

        // Chunked dispatch: process queries in chunks
        for (int32_t q_offset = 0; q_offset < ne01; q_offset += chunk_queries) {
            const int32_t q_count = ((q_offset + chunk_queries) < ne01) ? chunk_queries : (ne01 - q_offset);

            uint32_t q_offset_u32 = (uint32_t)q_offset;
            ggml_metal_encoder_set_bytes(enc, &q_offset_u32, sizeof(q_offset_u32), 7);
            ggml_metal_encoder_dispatch_threadgroups(enc, q_count, ne02, ne03, BN * BD, 1, 1);
        }
    } else {
        // Single dispatch (AMD, Apple, or small Intel dispatch)
        uint32_t q_offset = 0;
        ggml_metal_encoder_set_bytes(enc, &q_offset, sizeof(q_offset), 7);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, BN * BD, 1, 1);
    }

    return 1;
}
