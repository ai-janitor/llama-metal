#include "../ops/ggml-metal-ops-internal.h"

bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) && (ne00 % 32 == 0);
}

size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);

    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (false) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11*ne12*ne13 +
                nb21*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } else {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11*ne12*ne13 +
                nb21*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_blk(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    if (!has_mask) {
        return res;
    }

    const bool is_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    const int nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPSG : OP_FLASH_ATTN_EXT_NQPSG;
    const int ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    const int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    const int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += GGML_PAD(ggml_type_size(GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (true) {
        const int64_t nwg = 32;
        const int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += ggml_type_size(GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));
    }

    return res;
}

int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
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
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

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

    // TILED kernel for Intel Gen9 iGPUs — tiles K/V into shared memory for cooperative reuse
    // Requires DK % 16 == 0 (D_SPLIT=16). Falls through to scalar for DK=40, DK=72, etc.
    const bool force_scalar = getenv("GGML_METAL_FA_SCALAR") != nullptr;
    const bool use_intel_tiled = !profile->has_matrix_hw
        && profile->vendor == GGML_GPU_VENDOR_INTEL
        && (ne00 % 16 == 0)
        && !force_scalar;

    if (use_intel_tiled) {
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
        // Fall through to scalar for unsupported head sizes
    }

    // SCALAR fallback for GPUs without simdgroup matrix multiply support
    // Route ALL batch sizes through scalar on Intel — vec kernel compiles to
    // th_width=16 on Intel iGPU (register pressure) but assumes th_width=32 (BUG-005)
    // For AMD (th_width=32), use vec for small batch (ne01 < 20) as designed
    const bool use_scalar = !profile->has_matrix_hw &&
        (profile->vendor == GGML_GPU_VENDOR_INTEL || !ggml_metal_op_flash_attn_ext_use_vec(op));
    if (use_scalar) {
        const int dk = ne00;
        const int dv = ne00;
        const ggml_type kv_type = op->src[1]->type;

        const char * kernel_name = nullptr;

        // Select kernel based on head size and K/V type
        if (kv_type == GGML_TYPE_Q8_0) {
            if (dk == 64 && dv == 64) {
                kernel_name = "kernel_flash_attn_ext_scalar_q8_0_dk64_dv64";
            } else if (dk == 128 && dv == 128) {
                kernel_name = "kernel_flash_attn_ext_scalar_q8_0_dk128_dv128";
            } else if (dk == 256 && dv == 256) {
                kernel_name = "kernel_flash_attn_ext_scalar_q8_0_dk256_dv256";
            } else {
                GGML_ASSERT(false && "Unsupported head size for SCALAR FA Q8_0");
            }
        } else if (kv_type == GGML_TYPE_Q4_0) {
            if (dk == 64 && dv == 64) {
                kernel_name = "kernel_flash_attn_ext_scalar_q4_0_dk64_dv64";
            } else if (dk == 128 && dv == 128) {
                kernel_name = "kernel_flash_attn_ext_scalar_q4_0_dk128_dv128";
            } else if (dk == 256 && dv == 256) {
                kernel_name = "kernel_flash_attn_ext_scalar_q4_0_dk256_dv256";
            } else {
                GGML_ASSERT(false && "Unsupported head size for SCALAR FA Q4_0");
            }
        } else {
            // F16 variants (all head sizes)
            if (dk == 32 && dv == 32) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk32_dv32";
            } else if (dk == 40 && dv == 40) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk40_dv40";
            } else if (dk == 48 && dv == 48) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk48_dv48";
            } else if (dk == 64 && dv == 64) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk64_dv64";
            } else if (dk == 72 && dv == 72) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk72_dv72";
            } else if (dk == 80 && dv == 80) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk80_dv80";
            } else if (dk == 96 && dv == 96) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk96_dv96";
            } else if (dk == 112 && dv == 112) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk112_dv112";
            } else if (dk == 128 && dv == 128) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk128_dv128";
            } else if (dk == 192 && dv == 192) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk192_dv192";
            } else if (dk == 256 && dv == 256) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk256_dv256";
            } else if (dk == 576 && dv == 576) {
                kernel_name = "kernel_flash_attn_ext_scalar_dk576_dv576";
            } else {
                GGML_ASSERT(false && "Unsupported head size for SCALAR FA F16");
            }
        }

        auto pipeline = ggml_metal_library_compile_pipeline(lib, kernel_name, kernel_name, NULL);
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
        const int BN = 4;
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

    if (!ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;

        if (has_kvpad) {
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

            need_sync = true;
        }

        if (has_mask) {
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

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
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
    } else {
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

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
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
            ggml_metal_op_concurrency_reset(ctx);

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
    }

    return 1;
}
