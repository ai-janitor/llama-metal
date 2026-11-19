#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(ssm) static bool ggml_metal_supports_ssm_conv(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction;
}

static bool ggml_metal_supports_ssm_scan(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    // the kernel writes the final state only; rollback snapshots (K > 1) go to the CPU
    const int64_t n_state = ggml_metal_src(op, 0)->ne[0]*ggml_metal_src(op, 0)->ne[1]*ggml_metal_src(op, 0)->ne[2]*ggml_metal_src(op, 6)->ne[0];
    return has_simdgroup_reduction && ggml_nelements(op) == ggml_nelements(ggml_metal_src(op, 1)) + n_state;
}

static bool ggml_metal_supports_gated_delta_net(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    const int width = dev->profile.simd_width;
    const int size = ggml_metal_src(op, 2)->ne[0];
    const int threads = size < width ? size : width;
    if (!has_simdgroup_reduction || threads <= 0 || size % threads != 0) {
        return false;
    }
    const int nsg = size / threads;
    return (nsg == 1 || nsg == 2 || nsg == 4) &&
        threads * nsg <= (int) dev->profile.max_threads_per_threadgroup;
}

// Shared by RWKV_WKV6, RWKV_WKV7.
// An edit for one of them first gives that op its own function.
static bool ggml_metal_supports_rwkv_wkv(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(op);
    GGML_UNUSED(dev);
    return true;
}

static bool ggml_metal_supports_gated_linear_attn(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    // one thread per state column, the column is kept in a 128-entry thread array
    const int64_t head_size = ggml_metal_src(op, 0)->ne[0];
    return ggml_metal_src(op, 0)->type == GGML_TYPE_F32 && head_size <= 128 &&
        head_size <= (int64_t) dev->profile.max_threads_per_threadgroup;
}

static bool ggml_metal_supports_solve_tri(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(op);
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    return has_simdgroup_reduction;
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_ssm[] = {
    { GGML_OP_SSM_CONV, 3, ggml_metal_supports_ssm_conv },
    { GGML_OP_SSM_SCAN, 7, ggml_metal_supports_ssm_scan },
    { GGML_OP_GATED_DELTA_NET, 6, ggml_metal_supports_gated_delta_net },
    { GGML_OP_RWKV_WKV6, 6, ggml_metal_supports_rwkv_wkv },
    { GGML_OP_RWKV_WKV7, 7, ggml_metal_supports_rwkv_wkv },
    { GGML_OP_GATED_LINEAR_ATTN, 5, ggml_metal_supports_gated_linear_attn },
    { GGML_OP_SOLVE_TRI, 2, ggml_metal_supports_solve_tri },
};

struct ggml_metal_supports_table ggml_metal_supports_table_ssm(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_ssm, (int) (sizeof(ggml_metal_supports_rows_ssm)/sizeof(ggml_metal_supports_rows_ssm[0])) };
    return t;
}
