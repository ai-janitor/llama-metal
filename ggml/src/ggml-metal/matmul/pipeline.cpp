// Pipeline registration for matmul kernel variants.
// Maps (ggml_type, op) → Metal pipeline state objects at context init time.
#include "../ggml-metal-device.h"
#include "../ggml-metal-impl.h"
#include "../ggml-impl.h"

#include <cassert>
#include <memory>
#include <string>

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, ggml_type tsrc0, ggml_type tsrc1, int nsg, int nxpsg, int r1ptg, bool use_shmem_reduce) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d_sr=%d", base, nsg, nxpsg, use_shmem_reduce ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,   FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, nxpsg, FC_MUL_MV + 1);
        ggml_metal_cv_set_bool(cv, use_shmem_reduce, FC_MUL_MV + 2);

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

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, const ggml_tensor * op, bool use_shmem_reduce) {
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
                if (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } else {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    smem = (size_t)simd_width*sizeof(float)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
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
                smem = (size_t)simd_width*sizeof(float)*N_R0_Q8_0;
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
                GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(base, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_sr=%d", base, nsg, use_shmem_reduce ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_set_bool(cv, use_shmem_reduce, FC_MUL_MV + 2);

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
                if (ne00 >= 32) { smem = (size_t)pw * sizeof(float) * nr0; }
                break;
            case GGML_TYPE_Q8_0:
                smem = (size_t)pw * sizeof(float) * nr0;
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
                smem = (size_t)simd_width*sizeof(float)*N_R0_Q8_0;
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
