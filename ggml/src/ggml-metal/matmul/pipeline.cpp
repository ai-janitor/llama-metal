// Pipeline registration for matmul kernel variants.
// Maps (ggml_type, op) → Metal pipeline state objects at context init time.
#include "../ggml-metal-device.h"
#include "../ggml-metal-impl.h"
#include "../ggml-impl.h"
// The table header checks its rows against the constants of ggml-metal-impl.h in this file.
#define GGML_METAL_MV_FORMATS_CHECK
#include "tuning.h"
#include "dispatch.h"
#include "matvec-plan.h"
#include "mv-launch.h"
#include "../lanes/select.h"
#include "ggml-metal-probe.h"
#include <cassert>
#include <memory>
#include <string>
GGML_METAL_FAMILY(matmul)
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, ggml_type tsrc0, ggml_type tsrc1, int nsg, int nxpsg, int r1ptg, bool use_shmem_reduce, bool src1_trans) { GGML_METAL_ZONE_BEGIN(pipeline);
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

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, const ggml_tensor * op) { GGML_METAL_ZONE_BEGIN(pipeline);
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

int ggml_metal_q8_load_bytes(int pref, const ggml_tensor * src0) {
    if (pref != 4 || src0->type != GGML_TYPE_Q8_0 || src0->nb[1] % 4 != 0) {
        return 1;
    }
    // A word read can reach 4 bytes past the last weight: keep that inside the buffer.
    const struct ggml_metal_buffer_id bid = ggml_metal_get_buffer_id(src0);
    if (bid.offs % 4 != 0 || bid.offs + ggml_nbytes(src0) + 4 > ggml_metal_buffer_id_length(bid)) {
        return 1;
    }
    return 4;
}

int ggml_metal_q8_act_load_bytes(int pref, const ggml_tensor * src1) {
    if (pref != 16 || src1->type != GGML_TYPE_F32 || src1->nb[1] % 16 != 0) {
        return 4;
    }
    // A 16-byte load needs a 16-byte aligned address: the lane's eight values start at a multiple of 32 bytes in the row.
    return ggml_metal_get_buffer_id(src1).offs % 16 == 0 ? 16 : 4;
}

// The row of a block-quantized format with a kernel_mul_mv_<type>_f32 entry point; float types have none here.
static const struct mv_format_row * mv_format_row_of(enum ggml_type type) {
    const struct mv_format_row * row = ggml_metal_mv_format(type);
    return row && row->nsg > 0 ? row : NULL;
}

// Threadgroup memory of a format's matvec kernel at this geometry.
static size_t mv_format_smem(const struct mv_format_row * row, int simd_width, int nsg, int nr0, bool use_shmem_reduce) {
    switch (row->smem) {
        case MV_SMEM_NONE:
            return 0;
        case MV_SMEM_BYTES:
            return (size_t) row->bytes;
        case MV_SMEM_LANE_FLOATS:
            return (size_t)simd_width*sizeof(float);
        case MV_SMEM_ROW_SUMS:
            return use_shmem_reduce
                ? (size_t)nsg*simd_width*sizeof(float)*nr0
                : (size_t)simd_width*sizeof(float)*nr0;
    }
    return 0;
}

// plan: the matvec work plan of this multiply (matmul/matvec-plan.h). Its prep = the Q4_K entry point that reads block
// headers prepared at load (the caller binds them at 5). name_out (256 bytes, or NULL) receives the name of the pipeline returned.
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, const ggml_tensor * op, const struct ggml_metal_matvec_plan * plan, bool src1_trans, bool use_residual, char * name_out) { GGML_METAL_ZONE_BEGIN(pipeline);
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int simd_width = ggml_metal_library_get_simd_width(lib);
    const struct ggml_device_profile * hw = ggml_metal_library_get_device_profile(lib);

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;
    const int16_t selected_lane = ggml_metal_select_lane_id(
        hw, GGML_METAL_LANE_FAMILY_MATMUL, GGML_METAL_LANE_ROUTE_MV_SCALAR, tsrc0, tsrc1);

    if (plan == nullptr || !plan->ok) {
        GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
        GGML_ABORT("not implemented");
    }

    // Simdgroups, rows, reduce, columns and prep come from the plan; the threadgroup memory is sized here.
    bool use_shmem_reduce = plan->reduce != 0;
    const bool kq_prep  = plan->prep == 1;
    const int  ncols    = plan->cols > 1 ? plan->cols : 0;
    const int  col_form = plan->form;
    const int  nsg = plan->nsg; // number of simdgroups
    const int  nr0 = plan->nr0; // number of src0 rows per simdgroup
    const int  nr1 = 1;         // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const char * suffix = "";

    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                if (ggml_metal_mv_float_short(ggml_metal_mv_format(tsrc0), ne00)) {
                    suffix = "_short";
                } else {
                    // The shared-memory reduce (sr=1) sums nsg*NW partials per row.
                    smem = (size_t)(use_shmem_reduce ? nsg : 1)*simd_width*sizeof(float)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
            } break;
        default:
            {
                const struct mv_format_row * row = mv_format_row_of(tsrc0);
                if (!row) {
                    GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
                    GGML_ABORT("not implemented");
                }
                // sized with the format's own row count; a K-quant whose rows were set is sized again below
                smem = mv_format_smem(row, simd_width, nsg, row->nr0, use_shmem_reduce);
            }
    };

    // A row count that a row or the cache rule set (plan->nr0_set) reaches only the kernels that read it: the basic
    // quants through the function constant, the K-quants with shape rows through their entry's instances. The rule
    // sets it for no other type.
    if (plan->nr0_set && (tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q5_K || tsrc0 == GGML_TYPE_Q6_K)) {
        smem = use_shmem_reduce
            ? (size_t)nsg*simd_width*sizeof(float)*nr0
            : (size_t)simd_width*sizeof(float)*nr0;
    }

    if (ncols > 0) {
        smem *= (size_t) ncols;
    }
    snprintf(base, 256, "kernel_mul_mv_%s%s%s_%s%s", ggml_type_name(tsrc0), kq_prep ? "_prep" : "", ncols > 0 ? "_cols" : "", ggml_type_name(tsrc1), suffix);
    // Include nr0 in the pipeline cache key so different NR0 values produce separate pipelines.
    // The residual epilogue changes buffer 3's destination and activates buffer 4,
    // so it must select a distinct cached function-constant specialization.
    // The hw row's mv_reduce picks the shared-memory reduce tail (1 = tree); other paths keep 0.
    const int16_t mv_tail = plan->reduce == 2 ? 1 : 0;
    snprintf(name, 256, "%s_nsg=%d_sr=%d_st=%d_nr0=%d_lane=%d_residual=%d_tail=%d", base, nsg, use_shmem_reduce ? 1 : 0, src1_trans ? 1 : 0, nr0, selected_lane, use_residual ? 1 : 0, mv_tail);
    if (ncols > 0) {
        const size_t used = strlen(name);
        snprintf(name + used, 256 - used, "_nc=%d_cf=%d", ncols, col_form);
    }
    const int q8_load = (tsrc1 == GGML_TYPE_F32 && !src1_trans) ? ggml_metal_q8_load_bytes(plan->q8_load_pref, op->src[0]) : 1;
    if (q8_load != 1) {
        const size_t used = strlen(name);
        snprintf(name + used, 256 - used, "_q8load=%d", q8_load);
    }
    const int q8_act = (op->src[0]->type == GGML_TYPE_Q8_0 && !src1_trans) ? ggml_metal_q8_act_load_bytes(plan->q8_act_pref, op->src[1]) : 4;
    if (q8_act != 4) {
        const size_t used = strlen(name);
        snprintf(name + used, 256 - used, "_q8act=%d", q8_act);
    }
    if (name_out != nullptr) {
        snprintf(name_out, 256, "%s", name);
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
        // Only the kinds sized by the width change; a fixed table and the short float kernel (no threadgroup memory) do not.
        const struct mv_format_row * row = ggml_metal_mv_format(tsrc0);
        const bool float_short = ggml_metal_mv_float_short(row, ne00);
        if (row != nullptr && row->reduce_recompile && !use_shmem_reduce && pw < 32) {
            // Below width 32 (Intel iGPU) simd_sum and simd_shuffle_down are unreliable: compile the kernel again
            // with the shared-memory reduce, the barrier-based path the ext kernel uses there.
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
                use_shmem_reduce = true;
                if (name_out != nullptr) {
                    snprintf(name_out, 256, "%s", sr1_name);
                }
            }
        }
        if (row != nullptr && !float_short && (row->smem == MV_SMEM_LANE_FLOATS || row->smem == MV_SMEM_ROW_SUMS)) {
            smem = mv_format_smem(row, pw, nsg, nr0, use_shmem_reduce);
        }
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = use_shmem_reduce ? std::max(smem, (size_t)(nsg * pw * nr0 * sizeof(float))) : smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, int ne02, int ne20) { GGML_METAL_ZONE_BEGIN(pipeline);
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

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, const ggml_tensor * op) { GGML_METAL_ZONE_BEGIN(pipeline);
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

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, const ggml_tensor * op, bool use_shmem_reduce) { GGML_METAL_ZONE_BEGIN(pipeline);
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
                const struct ggml_metal_launch launch = ggml_metal_mv_launch_of(ggml_metal_library_get_device_profile(lib), GGML_METAL_MV_LAUNCH_ID_DENSE);
                nsg = (int) std::min<int64_t>(launch.nsg, ((int64_t) ne00 + launch.units - 1) / launch.units);
                nr0 = 2;
                nr1 = 1;
                smem = (size_t)simd_width*sizeof(float)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } break;
        default:
            {
                const struct mv_format_row * row = mv_format_row_of(tsrc0);
                if (!row) {
                    GGML_LOG_ERROR("Asserting on type %d\n", (int)op->src[2]->type);
                    GGML_ABORT("not implemented");
                }
                const struct mv_tuning t = ggml_metal_get_mv_tuning(nullptr, tsrc0);
                nsg = t.nsg;
                nr0 = t.nr0;
                smem = mv_format_smem(row, simd_width, nsg, nr0, use_shmem_reduce);
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
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_gcn(ggml_metal_library_t lib, ggml_type tsrc0, ggml_type tsrc1) { GGML_METAL_ZONE_BEGIN(pipeline);
    char base[256];
    char name[256];

    const struct mv_format_row * row0 = ggml_metal_mv_format(tsrc0);
    if (row0 == nullptr || !row0->w64_fallback) {
        return {}; // unsupported — caller falls back to scalar
    }
    const char * type_name_0 = ggml_type_name(tsrc0);

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
