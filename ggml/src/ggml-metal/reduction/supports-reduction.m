#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(reduction) static bool ggml_metal_supports_sum(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction && ggml_is_contiguous(ggml_metal_src(op, 0));
}

static bool ggml_metal_supports_sum_rows(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    // SUM_ROWS now handles non-contiguous (transposed) src0 via nb00 stride.
    // This eliminates ggml_cont(ggml_transpose(...)) before sum_rows in SSM
    // delta-net autoregressive path — the "sum_cols" workaround pattern.
    return has_simdgroup_reduction;
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_reduction[] = {
    { GGML_OP_SUM, 1, ggml_metal_supports_sum },
    { GGML_OP_SUM_ROWS, 1, ggml_metal_supports_sum_rows },
};

struct ggml_metal_supports_table ggml_metal_supports_table_reduction(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_reduction, (int) (sizeof(ggml_metal_supports_rows_reduction)/sizeof(ggml_metal_supports_rows_reduction[0])) };
    return t;
}
