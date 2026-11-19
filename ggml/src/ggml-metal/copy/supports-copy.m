#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(copy) static bool ggml_metal_supports_pool_1d(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    return ggml_is_contiguous(ggml_metal_src(op, 0)) && ggml_metal_src(op, 0)->type == GGML_TYPE_F32;
}

static bool ggml_metal_supports_pool_2d(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    return ggml_metal_src(op, 0)->type == GGML_TYPE_F32;
}

// Shared by SET, CPY, DUP, CONT.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_copy_types(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    {
        switch (ggml_metal_src(op, 0)->type) {
            case GGML_TYPE_F32:
                switch (op->type) {
                   case GGML_TYPE_F32:
                   case GGML_TYPE_F16:
                   case GGML_TYPE_BF16:
                   case GGML_TYPE_Q8_0:
                   case GGML_TYPE_Q4_0:
                   case GGML_TYPE_Q4_1:
                   case GGML_TYPE_Q5_0:
                   case GGML_TYPE_Q5_1:
                   case GGML_TYPE_IQ4_NL:
                   case GGML_TYPE_Q1_0:
                   case GGML_TYPE_Q2_0:
                   case GGML_TYPE_TQ1_0:
                   case GGML_TYPE_TQ2_0:
                   case GGML_TYPE_NVFP4:
                   case GGML_TYPE_I32:
                        return true;
                   default:
                        return false;
                }
            case GGML_TYPE_F16:
                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                        return true;
                    default:
                        return false;
                }
            case GGML_TYPE_BF16:
                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q1_0:
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:
            case GGML_TYPE_NVFP4:
                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                        return true;
                    default:
                        return false;
                }
            case GGML_TYPE_I32:
                return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32;
            default:
                return false;
        };
    }
}

static bool ggml_metal_supports_get_rows(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    return ggml_metal_src(op, 0)->type == GGML_TYPE_I32 || ggml_metal_has_matvec_type(ggml_metal_src(op, 0)->type);
}

static bool ggml_metal_supports_set_rows(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    {
        if (ggml_metal_src(op, 0)->type != GGML_TYPE_F32) {
            return false;
        }

        switch (op->type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
            case GGML_TYPE_F8_E4M3:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_Q1_0:
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:
            case GGML_TYPE_NVFP4:
                return true;
            default:
                return false;
        };
    }
}

static bool ggml_metal_supports_diag(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(op);
    GGML_UNUSED(dev);
    return true;
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_copy[] = {
    { GGML_OP_POOL_1D, 1, ggml_metal_supports_pool_1d },
    { GGML_OP_POOL_2D, 1, ggml_metal_supports_pool_2d },
    { GGML_OP_SET, 2, ggml_metal_supports_copy_types },
    { GGML_OP_CPY, 2, ggml_metal_supports_copy_types },
    { GGML_OP_DUP, 1, ggml_metal_supports_copy_types },
    { GGML_OP_CONT, 1, ggml_metal_supports_copy_types },
    { GGML_OP_GET_ROWS, 2, ggml_metal_supports_get_rows },
    { GGML_OP_SET_ROWS, 3, ggml_metal_supports_set_rows },
    { GGML_OP_DIAG, 1, ggml_metal_supports_diag },
};

struct ggml_metal_supports_table ggml_metal_supports_table_copy(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_copy, (int) (sizeof(ggml_metal_supports_rows_copy)/sizeof(ggml_metal_supports_rows_copy[0])) };
    return t;
}
