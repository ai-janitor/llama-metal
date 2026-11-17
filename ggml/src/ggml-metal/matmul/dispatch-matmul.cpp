// Matrix-matrix multiplication dispatch — simdgroup_mm (Apple Silicon) and tiled (AMD/Intel).
// Extracted from dispatch.cpp to keep each dispatch strategy in its own file.
#include "dispatch.h"

#include <cstdlib>
#include <cstring>

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

// P17.12/P17.13: GGML_METAL_TILED_FP16_MUL=1 (exact string) requests the Q4_K packed-fp16 multiply
// stage kernel (kernel_mul_mat_tiled_q4_K_v3, fp32 accumulation). Absent, empty, "0" or any other
// value is false. P17.13: read at EVERY dispatch, never cached in a static, so an activation harness
// that sets and restores the variable per row (fresh context) sees its own value; the earlier
// process-wide latch pinned the first row's answer onto every later row. To undo: restore the
// P17.12 version of this function and tiled_fp16_mul_trace from git history.
static bool tiled_fp16_mul_requested() {
    const char * value = std::getenv("GGML_METAL_TILED_FP16_MUL");
    return value != nullptr && strcmp(value, "1") == 0;
}

// Diagnostics are opt-in: GGML_METAL_ACTIVATION_TRACE exactly "1" (same literal-"1" flag the GDN
// route trace uses, ssm/ops-ssm.cpp). Default quiet. Read per call, no suppression state.
static bool tiled_fp16_mul_trace_enabled() {
    const char * value = std::getenv("GGML_METAL_ACTIVATION_TRACE");
    return value != nullptr && strcmp(value, "1") == 0;
}

// One line per ACTUAL route taken while the request flag is set (no once/set/lock: a runner counts
// every row). effective=1 only after the v3 pipeline was submitted; effective=0 = requested but the
// default tiled kernel (or mul_mv fallback) ran instead. Field layout is unchanged from P17.12.
bool ggml_metal_tiled_fp16_mul_admitted(ggml_metal_op_t ctx, int idx) {
    if (!tiled_fp16_mul_requested()) return false;

    const ggml_tensor * op = ctx->node(idx);
    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);

    return op->src[0]->type == GGML_TYPE_Q4_K &&
        profile->vendor == GGML_GPU_VENDOR_AMD &&
        !profile->has_matrix_hw &&
        op->src[1]->type == GGML_TYPE_F32 &&
        !ggml_is_transposed(op->src[0]) &&
        !ggml_is_transposed(op->src[1]) &&
        op->src[0]->ne[1] >= 64 &&           // same BM=64 tile guard as the tiled path
        op->src[1]->ne[1] > NE11_MM_MIN;     // same long-prompt threshold as the tiled path
}

static void tiled_fp16_mul_trace(bool effective, const char * kernel, int64_t ne11, int64_t batch, const char * why) {
    if (!tiled_fp16_mul_trace_enabled()) return;
    GGML_LOG_INFO("ggml-metal: tiled fp16 mul requested=1 effective=%d kernel=%s ne11=%lld batch=%lld (%s)\n",
                  effective ? 1 : 0, kernel, (long long) ne11, (long long) batch, why);
}

int dispatch_mul_mat_matmul(ggml_metal_op_t ctx, int idx) {
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

    // =========================================================================
    // simdgroup_mm path — Apple Silicon (M1+/A14+) with hardware matrix support.
    // =========================================================================
    if (!ggml_is_transposed(op->src[0]) &&
        !ggml_is_transposed(op->src[1]) &&
        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
        profile->has_matrix_hw && ne00 >= 64 && ne11 > NE11_MM_MIN) {
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
        return 1;  // dispatched
    }

    // =========================================================================
    // Tiled matmul path — for non-simdgroup_mm GPUs (AMD Radeon, Intel UHD).
    // =========================================================================
    if (!ggml_is_transposed(op->src[0]) &&
        !ggml_is_transposed(op->src[1]) &&
        !profile->has_matrix_hw &&
        ne01 >= 64 &&  // need at least one full BM=64 tile — tiny M wastes tile compute and risks timeout
        ne11 > NE11_MM_MIN &&
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

        // P17.12: opt-in Q4_K packed-fp16 multiply entry, AMD only. The distinct kernel name keeps
        // pipeline cache identity separate from the default entry; nothing else changes.
        const char * fp16_mul_env = std::getenv("GGML_METAL_TILED_FP16_MUL");
        if (fp16_mul_env != nullptr && fp16_mul_env[0] != '\0' && strcmp(fp16_mul_env, "0") != 0 &&
            strcmp(fp16_mul_env, "1") != 0 && tiled_fp16_mul_trace_enabled()) {
            // malformed value stays off; reported per call, only under the diagnostic flag
            GGML_LOG_WARN("ggml-metal: GGML_METAL_TILED_FP16_MUL='%s' is invalid; using the default tiled kernel\n", fp16_mul_env);
        }
        // P17.14: identical admission to the ext strategy's step-aside (shared helper).
        const bool fp16_mul_wanted = ggml_metal_tiled_fp16_mul_admitted(ctx, idx);
        const int64_t fp16_mul_batch = (int64_t) ne12 * ne13;
        bool fp16_mul_effective = false;
        ggml_metal_pipeline_with_params pipeline = {};
        if (fp16_mul_wanted) {
            char v3_name[64];
            snprintf(v3_name, sizeof(v3_name), "%s_v3", kernel_name);
            pipeline = ggml_metal_library_compile_pipeline(lib, v3_name, v3_name, NULL);
            if (pipeline.pipeline) {
                snprintf(kernel_name, sizeof(kernel_name), "%s", v3_name);
                fp16_mul_effective = true;
            } else {
                // compile/pipeline failure: ordinary kernel below, logged as requested-not-effective
                tiled_fp16_mul_trace(false, v3_name, ne11, fp16_mul_batch, "v3 pipeline unavailable, using default tiled kernel");
            }
        }
        if (!pipeline.pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, kernel_name, kernel_name, NULL);
        }
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
                if (fp16_mul_wanted) {
                    tiled_fp16_mul_trace(false, kernel_name, ne11, fp16_mul_batch, "vendor not verified, falling back to mul_mv");
                }
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

        // P17.13: positive route evidence AFTER the dispatches above were submitted, and never for a
        // zero grid (nothing was dispatched then). Positive only when the v3 pipeline is the one used.
        if (fp16_mul_wanted && tiled_total_tg > 0) {
            tiled_fp16_mul_trace(fp16_mul_effective, kernel_name, ne11, fp16_mul_batch,
                                 fp16_mul_effective ? "dispatched packed-fp16 multiply" : "dispatched default tiled kernel");
        }
            return 1;  // dispatched
        }
    }

    return 0;  // neither path handled it
}
