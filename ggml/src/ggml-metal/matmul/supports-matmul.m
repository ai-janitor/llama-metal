#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
#include "matmul/formats.h"
GGML_METAL_FAMILY(matmul) // Admit only weight types with existing scalar matvec and get_rows entries.
// INTENT/PDL: reject missing kernels before dispatch, preserving CPU fallback.
bool ggml_metal_has_matvec_type(enum ggml_type type) {
    const struct mv_format_row * row = ggml_metal_mv_format(type);
    return row != NULL && row->matvec;
}

// Shared by MUL_MAT, MUL_MAT_ID.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_mul_mat(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    // a BF16 or F16 activation runs only with a weight type whose row has a kernel that reads it
    const enum ggml_type act = ggml_metal_src(op, 1)->type;
    if (act == GGML_TYPE_BF16 || act == GGML_TYPE_F16) {
        const struct mv_format_row * row = ggml_metal_mv_format(ggml_metal_src(op, 0)->type);
        if (row == NULL || (act == GGML_TYPE_BF16 ? !row->act_bf16 : row->act_f16 == MV_ACT_NO)) {
            return false;
        }
        if (act == GGML_TYPE_F16 && row->act_f16 == MV_ACT_MM && !ggml_metal_mm_route_takes(dev->profile.has_matrix_hw, op)) {
            return false;
        }
    }
    // fp8 weights run on the matvec template only, with f32 activations (no simdgroup matrix kernel reads them)
    if (ggml_metal_src(op, 0)->type == GGML_TYPE_F8_E4M3 && (ggml_metal_src(op, 1)->type != GGML_TYPE_F32 || dev->profile.has_matrix_hw)) {
        return false;
    }
    return has_simdgroup_reduction && ggml_metal_has_matvec_type(ggml_metal_src(op, 0)->type);
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_matmul[] = {
    { GGML_OP_MUL_MAT, 2, ggml_metal_supports_mul_mat },
    { GGML_OP_MUL_MAT_ID, 3, ggml_metal_supports_mul_mat },
};

struct ggml_metal_supports_table ggml_metal_supports_table_matmul(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_matmul, (int) (sizeof(ggml_metal_supports_rows_matmul)/sizeof(ggml_metal_supports_rows_matmul[0])) };
    return t;
}
