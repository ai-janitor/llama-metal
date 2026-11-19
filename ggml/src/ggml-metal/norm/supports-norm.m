#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(norm) // Shared by NORM, RMS_NORM.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_norm(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction && ggml_is_contiguous_rows(ggml_metal_src(op, 0)) && ggml_metal_src(op, 0)->type == GGML_TYPE_F32;
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_norm[] = {
    { GGML_OP_NORM, 1, ggml_metal_supports_norm },
    { GGML_OP_RMS_NORM, 1, ggml_metal_supports_norm },
};

struct ggml_metal_supports_table ggml_metal_supports_table_norm(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_norm, (int) (sizeof(ggml_metal_supports_rows_norm)/sizeof(ggml_metal_supports_rows_norm[0])) };
    return t;
}
