#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(conv) static bool ggml_metal_supports_conv_transpose_2d(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    // the kernel has no batch index
    return ggml_metal_src(op, 1)->ne[3] == 1 &&
        ggml_is_contiguous(ggml_metal_src(op, 0)) && ggml_is_contiguous(ggml_metal_src(op, 1)) &&
        (ggml_metal_src(op, 0)->type == GGML_TYPE_F16 || ggml_metal_src(op, 0)->type == GGML_TYPE_F32) &&
        ggml_metal_src(op, 1)->type == GGML_TYPE_F32 &&
        op->type == GGML_TYPE_F32;
}

static bool ggml_metal_supports_im2col(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    if (!(ggml_is_contiguous(ggml_metal_src(op, 1)) && ggml_metal_src(op, 1)->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32))) {
        return false;
    }
    if (!dev->library) {
        return false;
    }
    // One threadgroup holds a whole kernel window; a larger window runs on the CPU.
    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;
    const int64_t KH = is_2D ? ggml_metal_src(op, 0)->ne[1] : 1;
    const int64_t KW = ggml_metal_src(op, 0)->ne[0];
    struct ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline_im2col(dev->library, op);
    return pipeline.pipeline && KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
}

static bool ggml_metal_supports_conv_2d(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    return ggml_is_contiguous(ggml_metal_src(op, 0)) &&
           ggml_metal_src(op, 1)->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           (ggml_metal_src(op, 0)->type == GGML_TYPE_F16 || ggml_metal_src(op, 0)->type == GGML_TYPE_F32);
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_conv[] = {
    { GGML_OP_CONV_TRANSPOSE_2D, 2, ggml_metal_supports_conv_transpose_2d },
    { GGML_OP_IM2COL, 2, ggml_metal_supports_im2col },
    { GGML_OP_CONV_2D, 2, ggml_metal_supports_conv_2d },
};

struct ggml_metal_supports_table ggml_metal_supports_table_conv(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_conv, (int) (sizeof(ggml_metal_supports_rows_conv)/sizeof(ggml_metal_supports_rows_conv[0])) };
    return t;
}
