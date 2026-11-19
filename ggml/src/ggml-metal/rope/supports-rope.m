#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(rope) // Shared by ROPE, ROPE_BACK.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_rope(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(op);
    GGML_UNUSED(dev);
    return true;
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_rope[] = {
    { GGML_OP_ROPE, 3, ggml_metal_supports_rope },
    { GGML_OP_ROPE_BACK, 3, ggml_metal_supports_rope },
};

struct ggml_metal_supports_table ggml_metal_supports_table_rope(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_rope, (int) (sizeof(ggml_metal_supports_rows_rope)/sizeof(ggml_metal_supports_rows_rope[0])) };
    return t;
}
