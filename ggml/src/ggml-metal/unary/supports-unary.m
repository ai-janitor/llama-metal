#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(unary) // Shared by SCALE, FILL, CLAMP, SQR, SQRT, SIN, COS, LOG.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_unary_float(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    return ggml_is_contiguous_rows(ggml_metal_src(op, 0)) && (ggml_metal_src(op, 0)->type == GGML_TYPE_F32 || ggml_metal_src(op, 0)->type == GGML_TYPE_F16);
}

static bool ggml_metal_supports_unary(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_ELU:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_SGN:
        case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_HARDSWISH:
        case GGML_UNARY_OP_HARDSIGMOID:
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_EXPM1:
        case GGML_UNARY_OP_FLOOR:
        case GGML_UNARY_OP_CEIL:
        case GGML_UNARY_OP_ROUND:
        case GGML_UNARY_OP_TRUNC:
        case GGML_UNARY_OP_XIELU:
            return ggml_is_contiguous_rows(ggml_metal_src(op, 0)) && (ggml_metal_src(op, 0)->type == GGML_TYPE_F32 || ggml_metal_src(op, 0)->type == GGML_TYPE_F16);
        default:
            return false;
    }
}

static bool ggml_metal_supports_glu(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    switch (ggml_get_glu_op(op)) {
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
        case GGML_GLU_OP_SWIGLU_CLAMP:
            return ggml_is_contiguous_1(ggml_metal_src(op, 0)) && ggml_metal_src(op, 0)->type == GGML_TYPE_F32;
       default:
            return false;
    }
}

// Shared by ADD, SUB, MUL, DIV, ADD_ID.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_binary(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    return ggml_is_contiguous_rows(ggml_metal_src(op, 0)) && ggml_is_contiguous_rows(ggml_metal_src(op, 1)) && ggml_metal_src(op, 0)->type == GGML_TYPE_F32;
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_unary[] = {
    { GGML_OP_SCALE, 1, ggml_metal_supports_unary_float },
    { GGML_OP_FILL, 1, ggml_metal_supports_unary_float },
    { GGML_OP_CLAMP, 1, ggml_metal_supports_unary_float },
    { GGML_OP_SQR, 1, ggml_metal_supports_unary_float },
    { GGML_OP_SQRT, 1, ggml_metal_supports_unary_float },
    { GGML_OP_SIN, 1, ggml_metal_supports_unary_float },
    { GGML_OP_COS, 1, ggml_metal_supports_unary_float },
    { GGML_OP_LOG, 1, ggml_metal_supports_unary_float },
    { GGML_OP_UNARY, 1, ggml_metal_supports_unary },
    { GGML_OP_GLU, 2, ggml_metal_supports_glu },
    { GGML_OP_ADD, 2, ggml_metal_supports_binary },
    { GGML_OP_SUB, 2, ggml_metal_supports_binary },
    { GGML_OP_MUL, 2, ggml_metal_supports_binary },
    { GGML_OP_DIV, 2, ggml_metal_supports_binary },
    { GGML_OP_ADD_ID, 3, ggml_metal_supports_binary },
};

struct ggml_metal_supports_table ggml_metal_supports_table_unary(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_unary, (int) (sizeof(ggml_metal_supports_rows_unary)/sizeof(ggml_metal_supports_rows_unary[0])) };
    return t;
}
