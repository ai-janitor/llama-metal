#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(spatial) static bool ggml_metal_supports_tri(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    return ggml_is_contiguous_rows(ggml_metal_src(op, 0));
}

static bool ggml_metal_supports_upscale(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    return ggml_metal_src(op, 0)->type == GGML_TYPE_F32 && op->op_params[0] == GGML_SCALE_MODE_NEAREST && !(op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS);
}

static bool ggml_metal_supports_pad(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    // TODO: add circular padding support for metal, see https://github.com/ggml-org/llama.cpp/pull/16985
    if (ggml_get_op_params_i32(op, 8) != 0) {
        return false;
    }

    return ggml_metal_src(op, 0)->type == GGML_TYPE_F32;
}

// Shared by ARGSORT, TOP_K, ARANGE.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_sort_ops(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(op);
    GGML_UNUSED(dev);
    return true;
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_spatial[] = {
    { GGML_OP_TRI, 1, ggml_metal_supports_tri },
    { GGML_OP_UPSCALE, 1, ggml_metal_supports_upscale },
    { GGML_OP_PAD, 1, ggml_metal_supports_pad },
    { GGML_OP_ARGSORT, 1, ggml_metal_supports_sort_ops },
    { GGML_OP_TOP_K, 1, ggml_metal_supports_sort_ops },
    { GGML_OP_ARANGE, 0, ggml_metal_supports_sort_ops },
};

struct ggml_metal_supports_table ggml_metal_supports_table_spatial(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_spatial, (int) (sizeof(ggml_metal_supports_rows_spatial)/sizeof(ggml_metal_supports_rows_spatial[0])) };
    return t;
}
