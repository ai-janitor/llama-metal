#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(misc) static bool ggml_metal_supports_count_equal(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction &&
        ggml_metal_src(op, 0)->type == GGML_TYPE_I32 &&
        ggml_metal_src(op, 1)->type == GGML_TYPE_I32 &&
        op->type == GGML_TYPE_I64;
}

static bool ggml_metal_supports_argmax(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction;
}

// Shared by OPT_STEP_ADAMW, OPT_STEP_SGD.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_opt_step(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction;
}

static bool ggml_metal_supports_dsv4_hc_comb(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction &&
        ggml_metal_src(op, 0)->type == GGML_TYPE_F32 &&
        ggml_metal_src(op, 1)->type == GGML_TYPE_F32 &&
        ggml_metal_src(op, 2)->type == GGML_TYPE_F32 &&
        op->type         == GGML_TYPE_F32 &&
        ggml_metal_src(op, 0)->ne[0] == 24 &&
        ggml_metal_src(op, 1)->ne[0] >= 3 &&
        ggml_metal_src(op, 2)->ne[0] == 24 &&
        ggml_is_contiguous_rows(ggml_metal_src(op, 0)) &&
        ggml_is_contiguous_rows(ggml_metal_src(op, 1)) &&
        ggml_is_contiguous_rows(ggml_metal_src(op, 2));
}

static bool ggml_metal_supports_dsv4_hc_pre(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction &&
        ggml_metal_src(op, 0)->type == GGML_TYPE_F32 &&
        ggml_metal_src(op, 1)->type == GGML_TYPE_F32 &&
        op->type         == GGML_TYPE_F32 &&
        ggml_is_contiguous_rows(ggml_metal_src(op, 0)) &&
        ggml_is_contiguous_rows(ggml_metal_src(op, 1));
}

static bool ggml_metal_supports_dsv4_hc_post(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction &&
        ggml_metal_src(op, 0)->type == GGML_TYPE_F32 &&
        ggml_metal_src(op, 1)->type == GGML_TYPE_F32 &&
        ggml_metal_src(op, 2)->type == GGML_TYPE_F32 &&
        (ggml_metal_src(op, 3) == NULL || ggml_metal_src(op, 3)->type == GGML_TYPE_F32) &&
        op->type         == GGML_TYPE_F32 &&
        ggml_metal_src(op, 1)->ne[1] == 4 &&
        ggml_metal_src(op, 2)->ne[0] == 4 &&
        (ggml_metal_src(op, 3) == NULL || (ggml_metal_src(op, 3)->ne[0] == 4 && ggml_metal_src(op, 3)->ne[1] == 4)) &&
        ggml_is_contiguous_rows(ggml_metal_src(op, 0)) &&
        ggml_is_contiguous_rows(ggml_metal_src(op, 1)) &&
        ggml_is_contiguous_rows(ggml_metal_src(op, 2)) &&
        (ggml_metal_src(op, 3) == NULL || ggml_is_contiguous_rows(ggml_metal_src(op, 3)));
}

static bool ggml_metal_supports_top_k_mask(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    return (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) &&
        ggml_metal_src(op, 0)->type == op->type &&
        ggml_metal_src(op, 0)->nb[0] == ggml_type_size(ggml_metal_src(op, 0)->type) &&
        op->nb[0] == ggml_type_size(op->type) &&
        ggml_metal_src(op, 1)->type == GGML_TYPE_I32 &&
        (ggml_metal_src(op, 2) == NULL || ggml_metal_src(op, 2)->type == GGML_TYPE_F32) &&
        (ggml_metal_src(op, 3) == NULL || ggml_metal_src(op, 3)->type == GGML_TYPE_I32) &&
        (ggml_metal_src(op, 4) == NULL || ggml_metal_src(op, 4)->type == GGML_TYPE_I32);
}

static bool ggml_metal_supports_out_prod(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    return op->type == GGML_TYPE_F32 && ggml_metal_src(op, 0)->type == GGML_TYPE_F32 && ggml_metal_src(op, 1)->type == GGML_TYPE_F32;
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_misc[] = {
    { GGML_OP_COUNT_EQUAL, 2, ggml_metal_supports_count_equal },
    { GGML_OP_ARGMAX, 1, ggml_metal_supports_argmax },
    { GGML_OP_OPT_STEP_ADAMW, 5, ggml_metal_supports_opt_step },
    { GGML_OP_OPT_STEP_SGD, 3, ggml_metal_supports_opt_step },
    { GGML_OP_DSV4_HC_COMB, 3, ggml_metal_supports_dsv4_hc_comb },
    { GGML_OP_DSV4_HC_PRE, 2, ggml_metal_supports_dsv4_hc_pre },
    { GGML_OP_DSV4_HC_POST, 4, ggml_metal_supports_dsv4_hc_post },
    { GGML_OP_TOP_K_MASK, 5, ggml_metal_supports_top_k_mask },
    { GGML_OP_OUT_PROD, 2, ggml_metal_supports_out_prod },
};

struct ggml_metal_supports_table ggml_metal_supports_table_misc(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_misc, (int) (sizeof(ggml_metal_supports_rows_misc)/sizeof(ggml_metal_supports_rows_misc[0])) };
    return t;
}
