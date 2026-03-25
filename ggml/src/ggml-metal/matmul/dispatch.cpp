// Host-side dispatch: selects kernel pattern (scalar/tiled/mm) and type variant for MUL_MAT.
// Encodes Metal command buffer with the chosen kernel, args, and threadgroup config.
#include "../ops/ggml-metal-ops-internal.h"

// Maps ggml_type to the suffix used in tiled kernel host_name instantiations.
// Each entry corresponds to a `template [[host_name("kernel_mul_mat_tiled_<suffix>")]]`
// line in kernel-tiled.metal. Adding a new tiled type = add one entry here + one
// template instantiation there.
static const char * tiled_type_suffix(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:     return "f16";
        case GGML_TYPE_F32:     return "f32";
        case GGML_TYPE_BF16:    return "bf16";
        case GGML_TYPE_MXFP4:   return "mxfp4";
        case GGML_TYPE_Q8_0:    return "q8_0";
        case GGML_TYPE_Q4_0:    return "q4_0";
        case GGML_TYPE_Q4_1:    return "q4_1";
        case GGML_TYPE_Q5_0:    return "q5_0";
        case GGML_TYPE_Q5_1:    return "q5_1";
        case GGML_TYPE_IQ4_NL:  return "iq4_nl";
        case GGML_TYPE_Q4_K:    return "q4_K";
        case GGML_TYPE_Q5_K:    return "q5_K";
        case GGML_TYPE_Q6_K:    return "q6_K";
        case GGML_TYPE_Q2_K:    return "q2_K";
        case GGML_TYPE_Q3_K:    return "q3_K";
        case GGML_TYPE_IQ4_XS:  return "iq4_xs";
        case GGML_TYPE_IQ2_XXS: return "iq2_xxs";
        case GGML_TYPE_IQ2_XS:  return "iq2_xs";
        case GGML_TYPE_IQ2_S:   return "iq2_s";
        case GGML_TYPE_IQ3_XXS: return "iq3_xxs";
        case GGML_TYPE_IQ3_S:   return "iq3_s";
        case GGML_TYPE_IQ1_S:   return "iq1_s";
        case GGML_TYPE_IQ1_M:   return "iq1_m";
        default:                return nullptr;
    }
}

int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const int ne11_mm_min = 8;

    bool dispatched = false;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    if (!dispatched && op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && (ne11 <= ne11_mm_min || (!profile->has_matrix_hw && op->src[0]->type != GGML_TYPE_F16 && op->src[0]->type != GGML_TYPE_MXFP4)))
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           false) && (ne11 >= 4 && (ne11 <= 8 || !profile->has_matrix_hw))
         )
        )
       ) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const int nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        const int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be someting cleverer
        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                r1ptg = 4; break;
        };

        const bool use_shmem_reduce_ext = (profile->vendor == GGML_GPU_VENDOR_INTEL);
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op->src[0]->type, op->src[1]->type, nsg, nxpsg, r1ptg, use_shmem_reduce_ext);

        // Vendor verification — fall back to mul_mv if unverified
        bool vendor_ok = true;
        if (pipeline.pipeline) {
            const uint8_t verified = ggml_metal_pipeline_get_verified_vendors(pipeline.pipeline);
            const uint8_t vendor_bit = ggml_metal_vendor_to_verified_bit(profile->vendor);
            if (!(verified & vendor_bit)) {
                GGML_LOG_WARN("%s: mul_mv_ext NOT VERIFIED on vendor %d — falling back to mul_mv\n", __func__, profile->vendor);
                vendor_ok = false;
            }
        }

        if (vendor_ok) {
            ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
            /*.tg_x_offset =*/ 0,
            /*.tg_y_offset =*/ 0,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        if (use_shmem_reduce_ext) {
            const size_t smem = nsg * 32 * sizeof(float);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        }

        const int32_t ext_grid_x = ((ne01 + r0ptg - 1) / r0ptg);
        const int32_t ext_grid_y = ((ne11 + r1ptg - 1) / r1ptg);
        const int32_t ext_grid_z = ne12 * ne13;

        const int32_t ext_total_tg = ext_grid_x * ext_grid_y * ext_grid_z;
        const int32_t ext_chunk = (int32_t)profile->max_threadgroups_per_dispatch / 16;

        if (ext_chunk > 0 && ext_total_tg > ext_chunk) {
            // 2D chunked dispatch: chunk along X first, then Y if needed
            const int32_t chunk_x = ext_chunk / (ext_grid_y * ext_grid_z);

            if (chunk_x > 0) {
                for (int32_t x_off = 0; x_off < ext_grid_x; x_off += chunk_x) {
                    const int32_t x_count = ((x_off + chunk_x) < ext_grid_x) ? chunk_x : (ext_grid_x - x_off);
                    args.tg_x_offset = x_off;
                    args.tg_y_offset = 0;
                    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                    ggml_metal_encoder_dispatch_threadgroups(enc, x_count, ext_grid_y, ext_grid_z, 32, nsg, 1);
                }
            } else {
                // grid_y*grid_z alone exceeds chunk — need Y chunking too
                const int32_t chunk_y = ext_chunk / ext_grid_z;
                const int32_t eff_chunk_y = (chunk_y > 0) ? chunk_y : 1;
                for (int32_t x_off = 0; x_off < ext_grid_x; x_off++) {
                    for (int32_t y_off = 0; y_off < ext_grid_y; y_off += eff_chunk_y) {
                        const int32_t y_count = ((y_off + eff_chunk_y) < ext_grid_y) ? eff_chunk_y : (ext_grid_y - y_off);
                        args.tg_x_offset = x_off;
                        args.tg_y_offset = y_off;
                        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                        ggml_metal_encoder_dispatch_threadgroups(enc, 1, y_count, ext_grid_z, 32, nsg, 1);
                    }
                }
            }
        } else {
            ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, ext_grid_x, ext_grid_y, ext_grid_z, 32, nsg, 1);
        }
            dispatched = true;
        }
    }

    if (!dispatched &&
        !ggml_is_transposed(op->src[0]) &&
        !ggml_is_transposed(op->src[1]) &&
        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
        profile->has_matrix_hw && ne00 >= 64 && ne11 > ne11_mm_min) {
        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
            /*.tg_x_offset =*/ 0,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        const size_t smem = pipeline.smem;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + 31)/32), ((ne01 + 63)/64), ne12*ne13, 128, 1, 1);
        dispatched = true;
    }

    if (!dispatched &&
        !ggml_is_transposed(op->src[0]) &&
        !ggml_is_transposed(op->src[1]) &&
        !profile->has_matrix_hw &&
        ne01 >= 64 &&  // need at least one full BM=64 tile — tiny M wastes tile compute and risks timeout
        ne11 > ne11_mm_min &&
        tiled_type_suffix(op->src[0]->type) != nullptr &&
        op->src[1]->type == GGML_TYPE_F32) {

        // Tiled matmul for non-simdgroup_mm GPUs (AMD Radeon, Intel UHD)
        // Kernel name built from type suffix — matches template instantiations in kernel-tiled.metal
        //
        // INTEL: f16 tiled is MANDATORY — scalar mul_mv fallback causes GPU watchdog timeout (BUG-011).
        // Do NOT remove GGML_METAL_VERIFIED_INTEL for f16. The f16 tiled kernel is faster than
        // scalar and stays within the ~5s Metal watchdog budget. Scalar does not.
        // mxfp4 tiled times out on Intel — stays AMD-only, falls back to scalar.
        char kernel_name[64];
        snprintf(kernel_name, sizeof(kernel_name), "kernel_mul_mat_tiled_%s", tiled_type_suffix(op->src[0]->type));

        auto pipeline = ggml_metal_library_compile_pipeline(lib, kernel_name, kernel_name, NULL);
        if (pipeline.pipeline) {
            ggml_metal_pipeline_set_min_family(pipeline.pipeline, GGML_METAL_FAMILY_COMMON1); // no simd intrinsics needed

            uint8_t verified = GGML_METAL_VERIFIED_AMD;
            if (op->src[0]->type != GGML_TYPE_MXFP4) {
                verified |= GGML_METAL_VERIFIED_INTEL;  // f16 tiled OK on Intel, mxfp4 tiled is not
            }
            ggml_metal_pipeline_set_verified_vendors(pipeline.pipeline, verified);
        }

        // Vendor verification
        bool vendor_ok = false;
        if (pipeline.pipeline) {
            const uint8_t verified = ggml_metal_pipeline_get_verified_vendors(pipeline.pipeline);
            const uint8_t vendor_bit = ggml_metal_vendor_to_verified_bit(profile->vendor);
            if (verified & vendor_bit) {
                vendor_ok = true;
            } else {
                GGML_LOG_WARN("%s: mul_mat_tiled NOT VERIFIED on vendor %d — falling back to mul_mv\n", __func__, profile->vendor);
            }
        }

        if (vendor_ok) {

        const int BM = 64, BN = 64, BK = 32;
        const int SHMEM_STRIDE = BK/2 + 1;  // 32/2 + 1 = 17

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
            /*.tg_x_offset =*/ 0,
        };

        const size_t smem = 2 * BM * SHMEM_STRIDE * 2 * sizeof(float);  // ~17KB (float2 = 2 floats)

        const int32_t tiled_grid_x = ((ne01 + BM - 1)/BM);
        const int32_t tiled_grid_y = ((ne11 + BN - 1)/BN);
        const int32_t tiled_grid_z = ne12*ne13;
        const int32_t tiled_total_tg = tiled_grid_x * tiled_grid_y * tiled_grid_z;
        // Tiled TGs are heavy (64×64 output tile with full K reduction) — use 1/64 of FA limit
        const int32_t tiled_chunk = (int32_t)profile->max_threadgroups_per_dispatch / 64;

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (tiled_chunk > 0 && tiled_total_tg > tiled_chunk) {
            const int32_t chunk_x = tiled_chunk / (tiled_grid_y * tiled_grid_z);
            if (chunk_x > 0) {
                for (int32_t x_off = 0; x_off < tiled_grid_x; x_off += chunk_x) {
                    const int32_t x_count = ((x_off + chunk_x) < tiled_grid_x) ? chunk_x : (tiled_grid_x - x_off);

                    args.tg_x_offset = x_off;
                    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                    ggml_metal_encoder_dispatch_threadgroups(enc, x_count, tiled_grid_y, tiled_grid_z, 128, 1, 1);
                }
            } else {
                for (int32_t x_off = 0; x_off < tiled_grid_x; x_off++) {

                    args.tg_x_offset = x_off;
                    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                    ggml_metal_encoder_dispatch_threadgroups(enc, 1, tiled_grid_y, tiled_grid_z, 128, 1, 1);
                }
            }
        } else {

            ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, tiled_grid_x, tiled_grid_y, tiled_grid_z, 128, 1, 1);
        }
            dispatched = true;
        }
    }

    if (!dispatched) {
        const bool use_shmem_reduce = (profile->vendor == GGML_GPU_VENDOR_INTEL);
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op, use_shmem_reduce);

        // mul_mv is the last resort — if it's not verified for this vendor, crash
        // rather than silently producing wrong results
        if (pipeline.pipeline) {
            const uint8_t verified = ggml_metal_pipeline_get_verified_vendors(pipeline.pipeline);
            const uint8_t vendor_bit = ggml_metal_vendor_to_verified_bit(profile->vendor);
            if (!(verified & vendor_bit)) {
                GGML_ABORT("%s: mul_mv NOT VERIFIED on vendor %d — no fallback available. "
                           "Use supports_op to route this op to CPU instead.\n", __func__, profile->vendor);
            }
        }

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nr0  =*/ nr0,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
            /*.tg_x_offset =*/ 0,
            /*.tg_y_offset =*/ 0,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        const int32_t grid_x = (op->src[0]->type == GGML_TYPE_F32 ||
                                op->src[0]->type == GGML_TYPE_F16 ||
                                op->src[0]->type == GGML_TYPE_BF16 ||
                                op->src[0]->type == GGML_TYPE_Q8_0)
                               ? ((ne01 + nr0 - 1) / nr0)
                               : ((ne01 + nr0*nsg - 1) / (nr0*nsg));
        const int32_t grid_y = ((ne11 + nr1 - 1) / nr1);
        const int32_t grid_z = ne12 * ne13;

        const int32_t total_tg = grid_x * grid_y * grid_z;
        const int32_t chunk = (int32_t)profile->max_threadgroups_per_dispatch / 16;

        if (chunk > 0 && total_tg > chunk) {
            // 2D chunked dispatch: chunk along X first, then Y if needed
            const int32_t chunk_x = chunk / (grid_y * grid_z);

            if (chunk_x > 0) {
                for (int32_t x_off = 0; x_off < grid_x; x_off += chunk_x) {
                    const int32_t x_count = ((x_off + chunk_x) < grid_x) ? chunk_x : (grid_x - x_off);
                    args.tg_x_offset = x_off;
                    args.tg_y_offset = 0;
                    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                    ggml_metal_encoder_dispatch_threadgroups(enc, x_count, grid_y, grid_z, 32, nsg, 1);
                }
            } else {
                // grid_y*grid_z alone exceeds chunk — need Y chunking too
                const int32_t chunk_y = chunk / grid_z;
                const int32_t eff_chunk_y = (chunk_y > 0) ? chunk_y : 1;
                for (int32_t x_off = 0; x_off < grid_x; x_off++) {
                    for (int32_t y_off = 0; y_off < grid_y; y_off += eff_chunk_y) {
                        const int32_t y_count = ((y_off + eff_chunk_y) < grid_y) ? eff_chunk_y : (grid_y - y_off);
                        args.tg_x_offset = x_off;
                        args.tg_y_offset = y_off;
                        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                        ggml_metal_encoder_dispatch_threadgroups(enc, 1, y_count, grid_z, 32, nsg, 1);
                    }
                }
            }
        } else {

            ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);
        }
    }

    return 1;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert

    return ggml_type_size(GGML_TYPE_I32)*ne02;
}

size_t ggml_metal_op_mul_mat_id_extra_ids(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    return ggml_type_size(GGML_TYPE_I32)*ne02*ne21;
}

int ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, int idx) {
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
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    // ne20 = n_used_experts
    // ne21 = n_rows (batch size)
    const int ne21_mm_id_min = 32;

    if (profile->has_matrix_hw && ne00 >= 64 && (ne21 >= ne21_mm_id_min)) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne02,
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(ne02 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        // this barrier is always needed because the next kernel has to wait for the id maps to be computed
        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne21  =*/ ne21, // n_tokens
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);

            const size_t smem = pipeline.smem;

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne21 + 31)/32, (ne01 + 63)/64, ne02, 128, 1, 1);
        }
    } else {
        const bool use_shmem_reduce_id = (profile->vendor == GGML_GPU_VENDOR_INTEL);
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op, use_shmem_reduce_id);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
        };

        if (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
    }

    return 1;
}
