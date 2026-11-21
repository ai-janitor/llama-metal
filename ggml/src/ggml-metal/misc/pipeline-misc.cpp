#include "../ggml-metal-device.h"
#include "../ggml-metal-impl.h"
#include "ggml-impl.h"
#include "ggml-metal-probe.h"
#include "device/launch-plan.h"
#include <cassert>
#include <memory>
#include <string>
GGML_METAL_FAMILY(misc)
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw(ggml_metal_library_t lib, const ggml_tensor * op) { GGML_METAL_ZONE_BEGIN(pipeline);
    assert(op->op == GGML_OP_OPT_STEP_ADAMW);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd(ggml_metal_library_t lib, const ggml_tensor * op) { GGML_METAL_ZONE_BEGIN(pipeline);
    assert(op->op == GGML_OP_OPT_STEP_SGD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, const ggml_tensor *  op) { GGML_METAL_ZONE_BEGIN(pipeline);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_count_equal(ggml_metal_library_t lib, const ggml_tensor *  op) { GGML_METAL_ZONE_BEGIN(pipeline);
    assert(op->op == GGML_OP_COUNT_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    char base[256];
    char name[256];

    int nsg = 1;
    const struct ggml_metal_launch launch = ggml_metal_launch_pick(GGML_METAL_LAUNCH_COUNT_EQUAL, ggml_metal_library_get_device_profile(lib)->name, ggml_metal_library_get_simd_width(lib));
    while (launch.start*nsg < ne00 && nsg < launch.nsg) {
        nsg *= 2;
    }

    snprintf(base, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_COUNT_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = (size_t) launch.mem_units * sizeof(int32_t);
    res.nsg  = nsg;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_mask(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    fill,
        bool    has_score,
        bool    has_blocks,
        bool    has_extra,
        bool    vec) { GGML_METAL_ZONE_BEGIN(pipeline);
    GGML_ASSERT(op->op == GGML_OP_TOP_K_MASK);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_top_k_mask%s_%s", fill ? "_fill" : "", ggml_type_name(op->type));

    ggml_metal_pipeline_with_params res;

    ggml_metal_cv_t cv = ggml_metal_cv_init();

    if (fill) {
        snprintf(name, 256, "%s_vec=%d", base, vec ? 1 : 0);

        ggml_metal_cv_set_bool(cv, vec, FC_TOP_K_MASK + 3);
    } else {
        snprintf(name, 256, "%s_sc=%d_bl=%d_ex=%d", base, has_score ? 1 : 0, has_blocks ? 1 : 0, has_extra ? 1 : 0);

        ggml_metal_cv_set_bool(cv, has_score,  FC_TOP_K_MASK + 0);
        ggml_metal_cv_set_bool(cv, has_blocks, FC_TOP_K_MASK + 1);
        ggml_metal_cv_set_bool(cv, has_extra,  FC_TOP_K_MASK + 2);
    }

    res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
    }

    ggml_metal_cv_free(cv);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc(ggml_metal_library_t lib, const ggml_tensor * op) { GGML_METAL_ZONE_BEGIN(pipeline);
    char name[256];
    const char * base = nullptr;

    switch (op->op) {
        case GGML_OP_DSV4_HC_COMB:
            base = "kernel_dsv4_hc_comb_f32";
            snprintf(name, 256, "%s", base);
            break;
        case GGML_OP_DSV4_HC_PRE:
            base = ggml_get_op_params_i32(op, 1) != 0 ? "kernel_dsv4_hc_pre_gated_f32" : "kernel_dsv4_hc_pre_f32";
            snprintf(name, 256, "%s_n_hc=%d", base, (int) op->src[0]->ne[1]);
            break;
        case GGML_OP_DSV4_HC_POST:
            base = op->src[3] ? "kernel_dsv4_hc_post_f32" : "kernel_dsv4_hc_post_nocomb_f32";
            snprintf(name, 256, "%s", base);
            break;
        default:
            GGML_ABORT("fatal error");
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = nullptr;

        if (op->op == GGML_OP_DSV4_HC_PRE) {
            cv = ggml_metal_cv_init();
            ggml_metal_cv_set_int32(cv, (int32_t) op->src[0]->ne[1], FC_DSV4_HC + 0);
        }

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        if (cv) {
            ggml_metal_cv_free(cv);
        }
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_out_prod(ggml_metal_library_t lib) { GGML_METAL_ZONE_BEGIN(pipeline);
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_out_prod_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }

    return res;
}
