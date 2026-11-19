#import "device/ggml-metal-device-internal.h"

// Ops whose BF16 kernels exist over 16-bit storage (no native bfloat type needed).
static bool ggml_metal_bf16_u16_has_op(const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_CONCAT:
        case GGML_OP_REPEAT:
        case GGML_OP_FLASH_ATTN_EXT:
            return true;
        default:
            return false;
    }
}

// Shared by NONE, RESHAPE, VIEW, TRANSPOSE, PERMUTE.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_layout_ops(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(op);
    GGML_UNUSED(dev);
    return true;
}

static bool ggml_metal_supports_concat(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (ggml_metal_src(op, 0)->type != ggml_metal_src(op, 1)->type || ggml_metal_src(op, 0)->type != op->type) {
        return false;
    }
    switch (op->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static bool ggml_metal_supports_repeat(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    switch (ggml_metal_src(op, 0)->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I16:
            return true;
        default:
            return false;
    }
}

static bool ggml_metal_supports_acc(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(op);
    GGML_UNUSED(dev);
    // the kernel gives wrong results for strided placements
    return false;
}

static bool ggml_metal_supports_conv_transpose_1d(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(op);
    GGML_UNUSED(dev);
    return true;
}

// Shared by CUMSUM, MEAN, SOFT_MAX, GROUP_NORM, L2_NORM.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_reduce_contiguous_rows(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction && ggml_is_contiguous_rows(ggml_metal_src(op, 0));
}

// Shared by PAD_REFLECT_1D, IRFFT, FOLD, TIMESTEP_EMBEDDING, LEAKY_RELU.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_src0_f32(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_metal_src(op, 0)->type == GGML_TYPE_F32;
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_core[] = {
    { GGML_OP_NONE, 0, ggml_metal_supports_layout_ops },
    { GGML_OP_RESHAPE, 1, ggml_metal_supports_layout_ops },
    { GGML_OP_VIEW, 1, ggml_metal_supports_layout_ops },
    { GGML_OP_TRANSPOSE, 1, ggml_metal_supports_layout_ops },
    { GGML_OP_PERMUTE, 1, ggml_metal_supports_layout_ops },
    { GGML_OP_CONCAT, 2, ggml_metal_supports_concat },
    { GGML_OP_REPEAT, 1, ggml_metal_supports_repeat },
    { GGML_OP_ACC, 2, ggml_metal_supports_acc },
    { GGML_OP_CONV_TRANSPOSE_1D, 2, ggml_metal_supports_conv_transpose_1d },
    { GGML_OP_CUMSUM, 1, ggml_metal_supports_reduce_contiguous_rows },
    { GGML_OP_MEAN, 1, ggml_metal_supports_reduce_contiguous_rows },
    { GGML_OP_SOFT_MAX, 2, ggml_metal_supports_reduce_contiguous_rows },
    { GGML_OP_GROUP_NORM, 1, ggml_metal_supports_reduce_contiguous_rows },
    { GGML_OP_L2_NORM, 1, ggml_metal_supports_reduce_contiguous_rows },
    { GGML_OP_PAD_REFLECT_1D, 1, ggml_metal_supports_src0_f32 },
    { GGML_OP_IRFFT, 1, ggml_metal_supports_src0_f32 },
    { GGML_OP_FOLD, 2, ggml_metal_supports_src0_f32 },
    { GGML_OP_TIMESTEP_EMBEDDING, 1, ggml_metal_supports_src0_f32 },
    { GGML_OP_LEAKY_RELU, 1, ggml_metal_supports_src0_f32 },
};

// every op's row, by op; built once from the family tables
static const struct ggml_metal_supports_row * ggml_metal_supports_by_op[GGML_OP_COUNT];

static void ggml_metal_supports_index_add(const struct ggml_metal_supports_row * rows, int n) {
    for (int i = 0; i < n; ++i) {
        GGML_ASSERT(ggml_metal_supports_by_op[rows[i].op] == NULL);
        ggml_metal_supports_by_op[rows[i].op] = &rows[i];
    }
}

static void ggml_metal_supports_index_init(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        ggml_metal_supports_index_add(ggml_metal_supports_rows_core, (int) (sizeof(ggml_metal_supports_rows_core)/sizeof(ggml_metal_supports_rows_core[0])));
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_conv(); ggml_metal_supports_index_add(t.rows, t.n); }
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_copy(); ggml_metal_supports_index_add(t.rows, t.n); }
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_flash_attn(); ggml_metal_supports_index_add(t.rows, t.n); }
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_matmul(); ggml_metal_supports_index_add(t.rows, t.n); }
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_misc(); ggml_metal_supports_index_add(t.rows, t.n); }
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_norm(); ggml_metal_supports_index_add(t.rows, t.n); }
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_reduction(); ggml_metal_supports_index_add(t.rows, t.n); }
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_rope(); ggml_metal_supports_index_add(t.rows, t.n); }
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_spatial(); ggml_metal_supports_index_add(t.rows, t.n); }
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_ssm(); ggml_metal_supports_index_add(t.rows, t.n); }
        { const struct ggml_metal_supports_table t = ggml_metal_supports_table_unary(); ggml_metal_supports_index_add(t.rows, t.n); }
    });
}

// Input i of op, for a predicate; i must be within the input count of the op's row.
const struct ggml_tensor * ggml_metal_src(const struct ggml_tensor * op, int i) {
    const struct ggml_metal_supports_row * row = ggml_metal_supports_by_op[op->op];
    GGML_ASSERT(row != NULL && i >= 0 && i < row->n_src);
    return op->src[i];
}

bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    const bool has_bfloat = dev->props.has_bfloat;

    if (!has_bfloat) {
        bool has_bf16_operand = op->type == GGML_TYPE_BF16;

        for (size_t i = 0, n = 3; i < n; ++i) {
            if (op->src[i] != NULL && op->src[i]->type == GGML_TYPE_BF16) {
                has_bf16_operand = true;
            }
        }

        if (has_bf16_operand && !(dev->props.has_bf16_u16 && ggml_metal_bf16_u16_has_op(op))) {
            return false;
        }
    }

    ggml_metal_supports_index_init();

    const struct ggml_metal_supports_row * row = ggml_metal_supports_by_op[op->op];
    return row != NULL && row->fn(dev, op);
}
