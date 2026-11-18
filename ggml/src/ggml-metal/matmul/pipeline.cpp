// Pipeline registration for matmul kernel variants.
// Maps (ggml_type, op) → Metal pipeline state objects at context init time.
#include "../ggml-metal-device.h"
#include "../ggml-metal-impl.h"
#include "../ggml-impl.h"
#include "tuning.h"
#include "dispatch.h"
#include "lane-policy.h"

#include <cassert>
#include <memory>
#include <string>

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, ggml_type tsrc0, ggml_type tsrc1, int nsg, int nxpsg, int r1ptg, bool use_shmem_reduce, bool src1_trans) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d_sr=%d_st=%d", base, nsg, nxpsg, use_shmem_reduce ? 1 : 0, src1_trans ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,   FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, nxpsg, FC_MUL_MV + 1);
        ggml_metal_cv_set_bool(cv, use_shmem_reduce, FC_MUL_MV + 2);
        // src1_trans: compile-time specialization for transposed src1.
        // When false (normal path), the compiler dead-code-eliminates all strided
        // access code — zero overhead. When true, enables ks1-strided B-loading.
        ggml_metal_cv_set_bool(cv, src1_trans, FC_MUL_MV + 3);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        if (res.pipeline) {
            ggml_metal_pipeline_set_min_family(res.pipeline, GGML_METAL_FAMILY_COMMON2); // mul_mv_ext needs simd_reduction
            if (use_shmem_reduce) {
                // shmem_reduce path verified on Intel — BUG-008/BUG-009
                ggml_metal_pipeline_set_verified_vendors(res.pipeline, GGML_METAL_VERIFIED_APPLE | GGML_METAL_VERIFIED_AMD | GGML_METAL_VERIFIED_INTEL);
            } else {
                ggml_metal_pipeline_set_verified_vendors(res.pipeline, GGML_METAL_VERIFIED_APPLE | GGML_METAL_VERIFIED_AMD | GGML_METAL_VERIFIED_INTEL);
            }
        }

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const bool bc_inp = op->src[0]->ne[0] % 32 != 0;
    const bool bc_out = op->ne[0] % 64 != 0 || op->ne[1] % 32 != 0;

    snprintf(base, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_bco=%d", base, bc_inp, bc_out);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_set_bool(cv, bc_out, FC_MUL_MM + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        if (res.pipeline) {
            ggml_metal_pipeline_set_min_family(res.pipeline, GGML_METAL_FAMILY_APPLE7); // mul_mm needs simdgroup_matrix_multiply
            ggml_metal_pipeline_set_verified_vendors(res.pipeline, GGML_METAL_VERIFIED_APPLE);
        }

        ggml_metal_cv_free(cv);
    }

    // when the output size is not multiple of 64x32, we need extra smem to prevent out-of-bounds writes
    res.smem = bc_out ? 8192 : 4096 + 2048;

    return res;
}

int ggml_metal_q8_load_bytes(const struct ggml_device_profile * hw, const ggml_tensor * src0) {
    if (!hw || hw->q8_load_bytes != 4 || src0->type != GGML_TYPE_Q8_0 || src0->nb[1] % 4 != 0) {
        return 1;
    }
    // A word read can reach 4 bytes past the last weight: keep that inside the buffer.
    const struct ggml_metal_buffer_id bid = ggml_metal_get_buffer_id(src0);
    if (bid.offs % 4 != 0 || bid.offs + ggml_nbytes(src0) + 4 > ggml_metal_buffer_id_length(bid)) {
        return 1;
    }
    return 4;
}

int ggml_metal_q8_act_load_bytes(const struct ggml_device_profile * hw, const ggml_tensor * src1) {
    if (!hw || hw->q8_act_load_bytes != 16 || src1->type != GGML_TYPE_F32 || src1->nb[1] % 16 != 0) {
        return 4;
    }
    // A 16-byte load needs a 16-byte aligned address: the lane's eight values start at a multiple of 32 bytes in the row.
    return ggml_metal_get_buffer_id(src1).offs % 16 == 0 ? 16 : 4;
}

// kq_prep: the Q4_K entry point that reads block headers prepared at load (the caller binds them at 5); only for Q4_K x F32.
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, const ggml_tensor * op, bool use_shmem_reduce, bool src1_trans, int nr0_override, bool use_residual, bool kq_prep, int ncols, int col_form) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int simd_width = ggml_metal_library_get_simd_width(lib);
    const struct ggml_device_profile * hw = ggml_metal_library_get_device_profile(lib);

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;
    const int16_t selected_lane = ggml_metal_select_matvec_lane(
        lib, tsrc0, GGML_METAL_MATVEC_ROUTE_SCALAR);

    const char * suffix = "";

    // --- Device-gated mat-vec tuning ---
    // nsg and nr0 come from ggml_metal_get_mv_tuning(), which selects values
    // based on hardware traits (vendor, simd_width, matrix_hw).
    // D700 (GCN legacy): nsg=2, nr0=4 — validated 2026-03-29.
    // Other devices: nsg=2, nr0=2 — safe upstream defaults.
    // Per-type smem requirements are still handled in the switch below.
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                // Float types use their own nsg/nr0 logic (not device-gated yet)
                if (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } else {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    // The shared-memory reduce (sr=1) sums nsg*NW partials per row.
                    smem = (size_t)(use_shmem_reduce ? nsg : 1)*simd_width*sizeof(float)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
            } break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
            {
                struct mv_tuning t = ggml_metal_get_mv_tuning(hw, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
            } break;
        case GGML_TYPE_Q8_0:
            {
                struct mv_tuning t = ggml_metal_get_mv_tuning(hw, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
                // shmem_reduce path uses nsg*NW as reduction group_size in
                // helper_mv_reduce_and_write_shmem, needing nsg*NW*NR0 floats.
                // Non-shmem path (NW=32) uses NW*NR0 floats for simdgroup partials.
                smem = use_shmem_reduce
                    ? (size_t)nsg*simd_width*sizeof(float)*nr0
                    : (size_t)simd_width*sizeof(float)*nr0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                struct mv_tuning t = ggml_metal_get_mv_tuning(hw, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
                smem = (size_t)simd_width*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            {
                struct mv_tuning t = ggml_metal_get_mv_tuning(hw, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                struct mv_tuning t = ggml_metal_get_mv_tuning(hw, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                struct mv_tuning t = ggml_metal_get_mv_tuning(hw, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                struct mv_tuning t = ggml_metal_get_mv_tuning(hw, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                struct mv_tuning t = ggml_metal_get_mv_tuning(hw, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            {
                struct mv_tuning t = ggml_metal_get_mv_tuning(hw, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
            } break;
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            {
                struct mv_tuning t = ggml_metal_get_mv_tuning(hw, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
                smem = (size_t)simd_width*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    // Apply NR0 override from model/hardware profile tuning (dispatch-scalar.cpp).
    // Applies to types whose kernels read the FC_mul_mv_nr0 function constant:
    //   - basic quant types (q4_0, q4_1, q5_0, q5_1) via BASIC_MV_ENTRY macro
    //   - q8_0 via dynamic dispatch in kernel_mul_mv_q8_0_f32
    // All other types (k-quants, IQ, mxfp4, f16/f32/bf16) use hardcoded
    // N_R0_* template parameters — overriding nr0 here would mismatch dispatch
    // grid dimensions vs kernel-compiled NR0, producing corrupted output.
    // Apply NR0 override ONLY for basic quant types (q4_0..q5_1).
    // q8_0 is excluded: its kernel_mul_mv_q8_0_f32_impl uses a different structure
    // (4 threads per quant block, separate row pointers) that produces wrong results
    // when NR0 doesn't match the compiled template parameter. The dynamic dispatch
    // added to the kernel entry point routes FC_mul_mv_nr0 to template instantiations,
    // but the dispatch grid + shmem sizing assumes the pipeline nr0 matches.
    // q8_0 keeps its compiled N_R0_Q8_0 (ggml-metal-impl.h); the rows override does not apply to it.
    const bool supports_nr0_override = (tsrc0 == GGML_TYPE_Q4_0 || tsrc0 == GGML_TYPE_Q4_1 ||
                                        tsrc0 == GGML_TYPE_Q5_0 || tsrc0 == GGML_TYPE_Q5_1 ||
                                        tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q5_K || tsrc0 == GGML_TYPE_Q6_K);
    if (nr0_override > 0 && supports_nr0_override) {
        nr0 = nr0_override;
        if (tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q5_K || tsrc0 == GGML_TYPE_Q6_K) {
            smem = use_shmem_reduce
                ? (size_t)nsg*simd_width*sizeof(float)*nr0
                : (size_t)simd_width*sizeof(float)*nr0;
        }
    }

    if (ncols > 0) {
        smem *= (size_t) ncols;
    }
    snprintf(base, 256, "kernel_mul_mv_%s%s%s_%s%s", ggml_type_name(tsrc0), kq_prep ? "_prep" : "", ncols > 0 ? "_cols" : "", ggml_type_name(tsrc1), suffix);
    // Include nr0 in the pipeline cache key so different NR0 values produce separate pipelines.
    // The residual epilogue changes buffer 3's destination and activates buffer 4,
    // so it must select a distinct cached function-constant specialization.
    // The hw row's mv_reduce picks the shared-memory reduce tail (1 = tree); other paths keep 0.
    const int16_t mv_tail = (use_shmem_reduce && hw && hw->mv_reduce == 1) ? 1 : 0;
    snprintf(name, 256, "%s_nsg=%d_sr=%d_st=%d_nr0=%d_lane=%d_residual=%d_tail=%d", base, nsg, use_shmem_reduce ? 1 : 0, src1_trans ? 1 : 0, nr0, selected_lane, use_residual ? 1 : 0, mv_tail);
    if (ncols > 0) {
        const size_t used = strlen(name);
        snprintf(name + used, 256 - used, "_nc=%d_cf=%d", ncols, col_form);
    }
    const int q8_load = (tsrc1 == GGML_TYPE_F32 && !src1_trans) ? ggml_metal_q8_load_bytes(hw, op->src[0]) : 1;
    if (q8_load != 1) {
        const size_t used = strlen(name);
        snprintf(name + used, 256 - used, "_q8load=%d", q8_load);
    }
    const int q8_act = (op->src[0]->type == GGML_TYPE_Q8_0 && !src1_trans) ? ggml_metal_q8_act_load_bytes(hw, op->src[1]) : 4;
    if (q8_act != 4) {
        const size_t used = strlen(name);
        snprintf(name + used, 256 - used, "_q8act=%d", q8_act);
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_set_bool(cv, use_shmem_reduce, FC_MUL_MV + 2);
        ggml_metal_cv_set_bool(cv, src1_trans, FC_MUL_MV + 3);
        // Set dynamic NR0 function constant — used by basic quant kernels (q4_0..q5_1).
        // K-quant kernels still use template<int nr0> and ignore this FC slot.
        ggml_metal_cv_set_int16(cv, (int16_t)nr0, FC_MUL_MV + 4);
        ggml_metal_cv_set_int16(cv, selected_lane, FC_MATVEC_LANE);
        // Only the scalar Q4_K entry reads FC606; every other kernel ignores it.
        ggml_metal_cv_set_bool(cv, use_residual, FC_MUL_MV_RESIDUAL);
        ggml_metal_cv_set_int16(cv, mv_tail, FC_MUL_MV + 7);
        if (ncols > 0) {
            ggml_metal_cv_set_int16(cv, (int16_t) ncols, FC_MUL_MV + 12);
            ggml_metal_cv_set_int16(cv, (int16_t) col_form, FC_MUL_MV + 13);
        }
        if (q8_load != 1) {
            ggml_metal_cv_set_int16(cv, (int16_t) q8_load, FC_MUL_MV + 10);
        }
        if (q8_act != 4) {
            ggml_metal_cv_set_int16(cv, (int16_t) q8_act, FC_MUL_MV + 11);
        }

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        if (res.pipeline) {
            if (use_shmem_reduce) {
                // shmem_reduce path verified on Intel — BUG-008
                ggml_metal_pipeline_set_verified_vendors(res.pipeline, GGML_METAL_VERIFIED_APPLE | GGML_METAL_VERIFIED_AMD | GGML_METAL_VERIFIED_INTEL);
            } else {
                ggml_metal_pipeline_set_verified_vendors(res.pipeline, GGML_METAL_VERIFIED_APPLE | GGML_METAL_VERIFIED_AMD | GGML_METAL_VERIFIED_INTEL);
            }
        }

        ggml_metal_cv_free(cv);
    }

    // After compilation, adjust smem to match the pipeline's actual SIMD width.
    // The adaptive recompile in compile_pipeline may change FC_SIMD_WIDTH from the
    // probed value (e.g., Intel: probe=32 but kernel compiles to th_width=8 or 16).
    // Smem formulas using simd_width must be recomputed with the actual value.
    const int pw = res.pipeline ? ggml_metal_pipeline_thread_execution_width(res) : simd_width;
    if (pw != simd_width) {
        // Recompute only for types whose smem depends on simd_width.
        // Types with fixed smem (IQ2/IQ3 lookup tables) are unaffected.
        switch (tsrc0) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
                if (ne00 >= 32) { smem = (size_t)(use_shmem_reduce ? nsg : 1) * pw * sizeof(float) * nr0; }
                break;
            case GGML_TYPE_Q8_0:
                smem = use_shmem_reduce
                    ? (size_t)nsg*pw*sizeof(float)*nr0
                    : (size_t)pw*sizeof(float)*nr0;
                break;
            case GGML_TYPE_MXFP4:
                if (!use_shmem_reduce && pw < 32) {
                    // At NW<32 (Intel iGPU), simd_sum and simd_shuffle_down are
                    // unreliable. Recompile mxfp4 with shmem_reduce=true so the
                    // kernel uses the barrier-based reduction path that is known to
                    // work on Intel (the same path used by the ext kernel at nxpsg=16).
                    // This gives a fresh pipeline with name _sr=1 and correct smem.
                    char sr1_name[256];
                    snprintf(sr1_name, 256, "%s_nsg=%d_sr=1", base, nsg);
                    ggml_metal_pipeline_with_params sr1 = ggml_metal_library_get_pipeline(lib, sr1_name);
                    if (!sr1.pipeline) {
                        ggml_metal_cv_t cv2 = ggml_metal_cv_init();
                        ggml_metal_cv_set_int16(cv2, nsg, FC_MUL_MV + 0);
                        ggml_metal_cv_set_bool  (cv2, true, FC_MUL_MV + 2); // sr=1
                        sr1 = ggml_metal_library_compile_pipeline(lib, base, sr1_name, cv2);
                        if (sr1.pipeline) {
                            ggml_metal_pipeline_set_verified_vendors(sr1.pipeline,
                                GGML_METAL_VERIFIED_APPLE | GGML_METAL_VERIFIED_AMD | GGML_METAL_VERIFIED_INTEL);
                        }
                        ggml_metal_cv_free(cv2);
                    }
                    if (sr1.pipeline) {
                        res = sr1;
                        smem = (size_t)pw * sizeof(float); // lookup table base
                        // shmem_reduce needs nsg*pw*nr0 floats; enforce at line 285
                        use_shmem_reduce = true;
                        break;
                    }
                }
                smem = (size_t)pw * sizeof(float);
                break;
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
                smem = (size_t)pw * sizeof(float);
                break;
            default:
                break; // fixed smem, no change
        }
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = use_shmem_reduce ? std::max(smem, (size_t)(nsg * pw * nr0 * sizeof(float))) : smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, int ne02, int ne20) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s_ne02=%d", base, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = (size_t) ne02*ne20*sizeof(uint16_t);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const bool bc_inp = op->src[0]->ne[0] % 32 != 0;

    snprintf(base, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d", base, bc_inp);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        if (res.pipeline) {
            ggml_metal_pipeline_set_min_family(res.pipeline, GGML_METAL_FAMILY_APPLE7); // same as mul_mm
            ggml_metal_pipeline_set_verified_vendors(res.pipeline, GGML_METAL_VERIFIED_APPLE);
        }

        ggml_metal_cv_free(cv);
    }

    res.smem = 8192;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, const ggml_tensor * op, bool use_shmem_reduce) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int simd_width = ggml_metal_library_get_simd_width(lib);

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const char * suffix = "";

        // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = (size_t)simd_width*sizeof(float)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                // shmem_reduce path uses nsg*NW as reduction group_size in
                // helper_mv_reduce_and_write_shmem, needing nsg*NW*NR0 floats.
                // Non-shmem path (NW=32) uses NW*NR0 floats for simdgroup partials.
                smem = use_shmem_reduce
                    ? (size_t)nsg*simd_width*sizeof(float)*nr0
                    : (size_t)simd_width*sizeof(float)*nr0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = (size_t)simd_width*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } break;
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = (size_t)simd_width*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = (size_t)simd_width*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_sr=%d", base, nsg, use_shmem_reduce ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_set_bool(cv, use_shmem_reduce, FC_MUL_MV + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        if (res.pipeline) {
            if (use_shmem_reduce) {
                ggml_metal_pipeline_set_verified_vendors(res.pipeline, GGML_METAL_VERIFIED_APPLE | GGML_METAL_VERIFIED_AMD | GGML_METAL_VERIFIED_INTEL);
            } else {
                ggml_metal_pipeline_set_verified_vendors(res.pipeline, GGML_METAL_VERIFIED_APPLE | GGML_METAL_VERIFIED_AMD | GGML_METAL_VERIFIED_INTEL);
            }
        }

        ggml_metal_cv_free(cv);
    }

    const int pw_id = res.pipeline ? ggml_metal_pipeline_thread_execution_width(res) : simd_width;
    if (pw_id != simd_width) {
        const ggml_type tsrc0_id = op->src[0]->type;
        switch (tsrc0_id) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
                smem = (size_t)pw_id * sizeof(float) * nr0;
                break;
            case GGML_TYPE_Q8_0:
                smem = (size_t)pw_id * sizeof(float) * nr0;
                break;
            case GGML_TYPE_MXFP4:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
                smem = (size_t)pw_id * sizeof(float);
                break;
            default:
                break;
        }
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = use_shmem_reduce ? std::max(smem, (size_t)(nsg * pw_id * sizeof(float))) : smem;

    return res;
}

// GCN-safe pipeline: pure threadgroup-memory reduction, no simd intrinsics.
// For AMD GPUs with 64-wide wavefronts where simd_sum/simd_shuffle_down are unreliable.
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_gcn(ggml_metal_library_t lib, ggml_type tsrc0, ggml_type tsrc1) {
    char base[256];
    char name[256];

    const char * type_name_0 = nullptr;
    switch (tsrc0) {
        case GGML_TYPE_F32:    type_name_0 = "f32";  break;
        case GGML_TYPE_F16:    type_name_0 = "f16";  break;
        case GGML_TYPE_Q4_0:   type_name_0 = "q4_0"; break;
        case GGML_TYPE_Q4_1:   type_name_0 = "q4_1"; break;
        case GGML_TYPE_Q5_0:   type_name_0 = "q5_0"; break;
        case GGML_TYPE_Q5_1:   type_name_0 = "q5_1"; break;
        case GGML_TYPE_Q8_0:   type_name_0 = "q8_0"; break;
        case GGML_TYPE_Q4_K:   type_name_0 = "q4_K"; break;
        case GGML_TYPE_Q3_K:   type_name_0 = "q3_K"; break;
        case GGML_TYPE_Q5_K:   type_name_0 = "q5_K"; break;
        case GGML_TYPE_Q6_K:   type_name_0 = "q6_K"; break;
        default:               return {}; // unsupported — caller falls back to scalar
    }

    const char * type_name_1 = nullptr;
    switch (tsrc1) {
        case GGML_TYPE_F32:    type_name_1 = "f32";  break;
        case GGML_TYPE_F16:    type_name_1 = "f16";  break;
        default:               return {}; // unsupported
    }

    snprintf(base, 256, "kernel_mul_mv_gcn_%s_%s", type_name_0, type_name_1);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
        if (res.pipeline) {
            ggml_metal_pipeline_set_min_family(res.pipeline, GGML_METAL_FAMILY_COMMON1);
            ggml_metal_pipeline_set_verified_vendors(res.pipeline, GGML_METAL_VERIFIED_AMD);
        }
    }

    // 64 threads per threadgroup, each needs one float for reduction
    res.smem = 64 * sizeof(float);
    res.nr0  = 1;  // one output row per threadgroup
    res.nr1  = 1;
    res.nsg  = 1;

    return res;
}
