// Ext mul_mv dispatch path — small-batch matrix-vector kernels for batch sizes 2..8.
// Extracted from dispatch.cpp to keep each dispatch strategy in its own file.
#include "dispatch.h"

int dispatch_mul_mat_ext(ggml_metal_op_t ctx, int idx) {
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

    // P17.14: opt-in packed-fp16 tiled prefill. The K-quant predicate below (ne11 >= 4 &&
    // (ne11 <= 8 || !has_matrix_hw)) would take long AMD Q4_K prompts before the tiled strategy runs,
    // so step aside - before any pipeline work - only when the shared admission says the tiled V3
    // route will take this op. Flag off (default) -> helper false -> ext unchanged.
    if (ggml_metal_tiled_fp16_mul_admitted(ctx, idx)) {
        return 0;
    }

    // =========================================================================
    // Small-batch matrix-vector ext kernels (mul_mv_ext).
    // =========================================================================
    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    //
    // More efficient than both mul_mv (batch=1 only, no batch tiling) and
    // mul_mm (designed for large batches, has tile overhead for small batches).
    // The ext kernels tile both the row (r0) and batch (r1) dimensions.
    if (op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
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
           false) && (ne11 >= 2 && (ne11 <= NE11_MM_MIN || (!profile->has_matrix_hw && op->src[0]->type != GGML_TYPE_F16 && op->src[0]->type != GGML_TYPE_MXFP4)))
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           // NOTE: ne11 >= 4 is intentional for kquant types, NOT a bug.
           // Tested 2026-03-28: changing to ne11 >= 2 regresses pp2 from 39.4 to 28.9 (-27%).
           // The ext kernel uses generic dequant functions (dequantize_q4_K per 16 elements)
           // which are much slower than the specialized kquant scalar kernel's hand-optimized
           // bitmask/scale unpacking. At batch=2-3, the scalar kernel runs 2-3 independent
           // matvecs which is faster than ext's weight-sharing with slow dequant.
           // The ext kernel only wins at batch >= 4 where weight reuse overcomes the dequant cost.
           false) && (ne11 >= 4 && (ne11 <= 8 || !profile->has_matrix_hw))
         )
        )
       ) {
        // Ext kernel configuration:
        //   nsg   — simdgroups per threadgroup
        //   nxpsg — threads along the K dimension per simdgroup (more = fewer reads,
        //           but must divide ne00; larger K benefits from more threads)
        //   r1ptg — src1 rows (batch elements) per threadgroup (chosen to divide ne11
        //           evenly, reducing wasted threads in the last threadgroup)
        //
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        const int nsg    = 2;

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;  // large K + tiny batch → maximize K parallelism
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;   // default for K divisible by 128
        } else {
            nxpsg = 4;   // fallback for smaller K
        }

        int16_t r1ptg  = 4;                 // num src1 rows per threadgroup
        // Chosen to evenly divide ne11 where possible, reducing tail waste.
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

        // simd_sum and simd_shuffle_down produce wrong results when the hardware
        // SIMD width differs from 32 (Intel NW<32, AMD GCN NW=64).
        // Enable shmem-based reduction for any non-32-wide GPU.
        const bool use_shmem_reduce_ext = (profile->simd_width != 32);
        const bool src1_trans = ggml_is_transposed(op->src[1]);
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op->src[0]->type, op->src[1]->type, nsg, nxpsg, r1ptg, use_shmem_reduce_ext, src1_trans);

        // Compute nypsg and r0ptg AFTER pipeline compilation using the actual SIMD width.
        // The adaptive recompile may change FC_SIMD_WIDTH (e.g., Intel: 32→16 or 32→8).
        const int ext_simd_w =
            (pipeline.pipeline ? ggml_metal_pipeline_thread_execution_width(pipeline) : 32);
        const int16_t nypsg  = ext_simd_w/nxpsg;  // num threads along col per simdgroup
        const int16_t r0ptg  = nypsg*nsg;          // num src0 rows per threadgroup

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
            const size_t smem = nsg * ext_simd_w * sizeof(float);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        }

        const int32_t ext_grid_x = ((ne01 + r0ptg - 1) / r0ptg);
        const int32_t ext_grid_y = ((ne11 + r1ptg - 1) / r1ptg);
        const int32_t ext_grid_z = ne12 * ne13;

        const int32_t ext_total_tg = ext_grid_x * ext_grid_y * ext_grid_z;

        // Chunked dispatch for GPUs with limited max_threadgroups_per_dispatch.
        // AMD GCN hardware has a small max grid size (~65536 TGs); if we exceed
        // it, we split the grid into multiple dispatches with offset parameters.
        // The /16 factor gives headroom to avoid hitting the exact limit.
        const int32_t ext_chunk = (int32_t)profile->max_threadgroups_per_dispatch / 16;

        if (ext_chunk > 0 && ext_total_tg > ext_chunk) {
            // 2D chunked dispatch: chunk along X (rows) first, then Y (batch) if needed
            const int32_t chunk_x = ext_chunk / (ext_grid_y * ext_grid_z);

            if (chunk_x > 0) {
                for (int32_t x_off = 0; x_off < ext_grid_x; x_off += chunk_x) {
                    const int32_t x_count = ((x_off + chunk_x) < ext_grid_x) ? chunk_x : (ext_grid_x - x_off);
                    args.tg_x_offset = x_off;
                    args.tg_y_offset = 0;
                    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                    ggml_metal_encoder_dispatch_threadgroups(enc, x_count, ext_grid_y, ext_grid_z, ext_simd_w, nsg, 1);
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
                        ggml_metal_encoder_dispatch_threadgroups(enc, 1, y_count, ext_grid_z, ext_simd_w, nsg, 1);
                    }
                }
            }
        } else {
            ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, ext_grid_x, ext_grid_y, ext_grid_z, ext_simd_w, nsg, 1);
        }
            return 1;  // dispatched
        }
    }

    return 0;  // not handled
}
