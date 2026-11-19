#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(flash_attn) // K/V element formats the scalar flash attention kernels read
static bool ggml_metal_fa_scalar_format(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_F8_E4M3:
            return true;
        default:
            return false;
    }
}

static bool ggml_metal_supports_flash_attn_ext(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    const bool has_simdgroup_mm        = dev->profile.has_matrix_hw;
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;

    // no kernel reads the K/V row selection of a shared cache
    if (ggml_metal_src(op, 5)) {
        return false;
    }
    // sparse-mask attention (kv-max hint set) gives wrong rows on the simdgroup path of 32-wide cards
    if (((const int32_t *) op->op_params)[4] != 0 && dev->profile.simd_width == 32 && !dev->profile.has_matrix_hw) {
        return false;
    }
    const int64_t hsk = ggml_metal_src(op, 0)->ne[0]; // head size of Q and K
    const int64_t hsv = op->ne[0];         // head size of V and of the result
    // for new head sizes, add checks here
    if (hsk != 32 && hsk != 40 && hsk != 48 && hsk != 64 && hsk != 72 && hsk != 80 && hsk != 96 &&
        hsk != 112 && hsk != 128 && hsk != 192 && hsk != 256 && hsk != 320 && hsk != 512 && hsk != 576) {
        return false;
    }
    // K head != V head: kernels exist for these pairs only (320 has no K == V kernel)
    const bool pair_old = (hsk == 192 && hsv == 128) || (hsk == 576 && hsv == 512);
    const bool pair_new = (hsk == 96 && hsv == 64) || (hsk == 128 && hsv == 96) || (hsk == 320 && hsv == 256);
    if ((hsk != hsv || hsk == 320) && !pair_old && !pair_new) {
        return false;
    }
    if (ggml_metal_src(op, 1)->type != ggml_metal_src(op, 2)->type || ggml_metal_src(op, 1)->type == GGML_TYPE_F8_E4M3) {
        // K type != V type, or fp8: the scalar kernels only, one format constant each
        return !has_simdgroup_mm && has_simdgroup_reduction &&
            ggml_metal_fa_scalar_format(ggml_metal_src(op, 1)->type) && ggml_metal_fa_scalar_format(ggml_metal_src(op, 2)->type);
    }
    // the vec kernels run only where a simdgroup is 32 threads wide;
    // every other card without simdgroup matrices takes the scalar kernels
    const bool takes_vec = ggml_metal_op_flash_attn_ext_use_vec(op) &&
        (has_simdgroup_mm ||
         (dev->profile.vendor != GGML_GPU_VENDOR_INTEL && dev->profile.simd_width == 32));
    if (takes_vec) {
        // vec kernels only need simdgroup_reduction, not simdgroup_mm
        switch (ggml_metal_src(op, 1)->type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
                return has_simdgroup_reduction;
            default:
                return false;
        }
    }
    if (has_simdgroup_mm) {
        // the simdgroup matrix kernels have no 512 head and none of the newer pairs
        return hsk != 512 && !pair_new;
    }
    if (!has_simdgroup_reduction) {
        return false;
    }
    // scalar kernels: one per head pair, the K/V format is a function constant
    switch (ggml_metal_src(op, 1)->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

static bool ggml_metal_supports_lightning_indexer(ggml_metal_device_t dev, const struct ggml_tensor * op) { GGML_METAL_ZONE_BEGIN_OP(supports, op->op);
    GGML_UNUSED(dev);
    // scalar kernel: one thread per (K row, token), the K format as a function constant
    if (op->type != GGML_TYPE_F32 || ggml_metal_src(op, 0)->type != GGML_TYPE_F32 ||
        ggml_metal_src(op, 2)->type != GGML_TYPE_F32 || ggml_metal_src(op, 3)->type != GGML_TYPE_F16) {
        return false;
    }
    switch (ggml_metal_src(op, 1)->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

// op, inputs a predicate may read, predicate
static const struct ggml_metal_supports_row ggml_metal_supports_rows_flash_attn[] = {
    { GGML_OP_FLASH_ATTN_EXT, 6, ggml_metal_supports_flash_attn_ext },
    { GGML_OP_LIGHTNING_INDEXER, 4, ggml_metal_supports_lightning_indexer },
};

struct ggml_metal_supports_table ggml_metal_supports_table_flash_attn(void) {
    struct ggml_metal_supports_table t = { ggml_metal_supports_rows_flash_attn, (int) (sizeof(ggml_metal_supports_rows_flash_attn)/sizeof(ggml_metal_supports_rows_flash_attn[0])) };
    return t;
}
