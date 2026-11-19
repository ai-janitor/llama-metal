// Matrix-matrix multiplication dispatch — simdgroup_mm (Apple Silicon) and tiled (AMD/Intel).
// Extracted from dispatch.cpp to keep each dispatch strategy in its own file.
#include "dispatch.h"
#include "formats.h"
#include "act-copy/act-copy.h"
#include "lanes/select.h"
#include "ggml-metal-probe.h"
#include <cstdlib>
#include <cstring>
GGML_METAL_FAMILY(matmul)
// The suffix of a type's kernel_mul_mat_tiled_<suffix> entry in kernel-tiled.metal, or nullptr when it has none.
// A new tiled type is the tiled cell of its row in formats.tsv and one template instantiation there.
static const char * tiled_type_suffix(enum ggml_type type) {
    const struct mv_format_row * row = ggml_metal_mv_format(type);
    return row != nullptr && row->tiled ? ggml_type_name(type) : nullptr;
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
    // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
    // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
    if (ggml_metal_mm_route_takes(profile->has_matrix_hw, op)) {
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
    if (ggml_metal_batch_route(ctx, idx) == MV_BATCH_TILED) {

        // Tiled matmul for non-simdgroup_mm GPUs (AMD Radeon, Intel UHD)
        // Kernel name built from type suffix — matches template instantiations in kernel-tiled.metal
        //
        // INTEL: f16 tiled is MANDATORY — scalar mul_mv fallback causes GPU watchdog timeout (BUG-011).
        // Do NOT remove GGML_METAL_VERIFIED_INTEL for f16. The f16 tiled kernel is faster than
        // scalar and stays within the ~5s Metal watchdog budget. Scalar does not.
        // mxfp4 tiled times out on Intel — stays AMD-only, falls back to scalar.
        char kernel_name[64];
        snprintf(kernel_name, sizeof(kernel_name), "kernel_mul_mat_tiled_%s", tiled_type_suffix(op->src[0]->type));

        // Activations read from a copy at the model's precision when this card's row says so. A missing entry for
        // the weight type leaves the op on the f32 kernel.
        ggml_metal_pipeline_with_params pipeline = {};
        struct ggml_metal_act_plan act = ggml_metal_act_copy_plan(ctx, idx, GGML_METAL_LANE_FAMILY_MATMUL, GGML_METAL_LANE_ROUTE_MM_TILED, op->src[1], (ne01 + 63)/64);
        if (act.applies) {
            char act_name[64];
            snprintf(act_name, sizeof(act_name), "%s_a%s", kernel_name, ggml_type_name(act.type));
            pipeline = ggml_metal_library_compile_pipeline(lib, act_name, act_name, NULL);
            if (pipeline.pipeline) {
                snprintf(kernel_name, sizeof(kernel_name), "%s", act_name);
            } else {
                act.applies = 0;
            }
        }
        if (!pipeline.pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, kernel_name, kernel_name, NULL);
        }
        if (pipeline.pipeline) {
            ggml_metal_pipeline_set_min_family(pipeline.pipeline, GGML_METAL_FAMILY_COMMON1); // no simd intrinsics needed

            ggml_metal_pipeline_set_verified_vendors(pipeline.pipeline, ggml_metal_tiled_verified_vendors(op->src[0]->type));
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

        if (act.applies) {
            ggml_metal_act_copy_encode(ctx, idx, &act, op->src[1]);
            args.nb10 = act.nb[0];
            args.nb11 = act.nb[1];
            args.nb12 = act.nb[2];
            args.nb13 = act.nb[3];
        }

        const size_t smem = 2 * BM * SHMEM_STRIDE * 2 * sizeof(float);  // ~17KB (float2 = 2 floats)

        const int32_t tiled_grid_x = ((ne01 + BM - 1)/BM);
        const int32_t tiled_grid_y = ((ne11 + BN - 1)/BN);
        const int32_t tiled_grid_z = ne12*ne13;
        const int32_t tiled_total_tg = tiled_grid_x * tiled_grid_y * tiled_grid_z;
        // Tiled TGs are heavy (64×64 output tile with full K reduction) — use 1/64 of FA limit
        const int32_t tiled_chunk = (int32_t)profile->max_threadgroups_per_dispatch / 64;

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, act.applies ? act.bid : ggml_metal_get_buffer_id(op->src[1]), 2);
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
            return 1;  // dispatched
        }
    }

    return 0;  // neither path handled it
}
