#include "../ggml-metal-device.h"
#include "../ggml-metal-impl.h"
#include "../ggml-impl.h"

#include <cassert>
#include <memory>
#include <string>

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_SCALE:      op_num = OP_UNARY_NUM_SCALE;      break;
        case GGML_OP_FILL:       op_num = OP_UNARY_NUM_FILL;       break;
        case GGML_OP_CLAMP:      op_num = OP_UNARY_NUM_CLAMP;      break;
        case GGML_OP_SQR:        op_num = OP_UNARY_NUM_SQR;        break;
        case GGML_OP_SQRT:       op_num = OP_UNARY_NUM_SQRT;       break;
        case GGML_OP_SIN:        op_num = OP_UNARY_NUM_SIN;        break;
        case GGML_OP_COS:        op_num = OP_UNARY_NUM_COS;        break;
        case GGML_OP_LOG:        op_num = OP_UNARY_NUM_LOG;        break;
        case GGML_OP_LEAKY_RELU: op_num = OP_UNARY_NUM_LEAKY_RELU; break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:        op_num = OP_UNARY_NUM_TANH;        break;
                case GGML_UNARY_OP_RELU:        op_num = OP_UNARY_NUM_RELU;        break;
                case GGML_UNARY_OP_SIGMOID:     op_num = OP_UNARY_NUM_SIGMOID;     break;
                case GGML_UNARY_OP_GELU:        op_num = OP_UNARY_NUM_GELU;        break;
                case GGML_UNARY_OP_GELU_ERF:    op_num = OP_UNARY_NUM_GELU_ERF;    break;
                case GGML_UNARY_OP_GELU_QUICK:  op_num = OP_UNARY_NUM_GELU_QUICK;  break;
                case GGML_UNARY_OP_SILU:        op_num = OP_UNARY_NUM_SILU;        break;
                case GGML_UNARY_OP_ELU:         op_num = OP_UNARY_NUM_ELU;         break;
                case GGML_UNARY_OP_NEG:         op_num = OP_UNARY_NUM_NEG;         break;
                case GGML_UNARY_OP_ABS:         op_num = OP_UNARY_NUM_ABS;         break;
                case GGML_UNARY_OP_SGN:         op_num = OP_UNARY_NUM_SGN;         break;
                case GGML_UNARY_OP_STEP:        op_num = OP_UNARY_NUM_STEP;        break;
                case GGML_UNARY_OP_HARDSWISH:   op_num = OP_UNARY_NUM_HARDSWISH;   break;
                case GGML_UNARY_OP_HARDSIGMOID: op_num = OP_UNARY_NUM_HARDSIGMOID; break;
                case GGML_UNARY_OP_EXP:         op_num = OP_UNARY_NUM_EXP;         break;
                case GGML_UNARY_OP_SOFTPLUS:    op_num = OP_UNARY_NUM_SOFTPLUS;    break;
                case GGML_UNARY_OP_EXPM1:       op_num = OP_UNARY_NUM_EXPM1;       break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;
    const bool is_cnt = ggml_is_contiguous(op->src[0]) && ggml_nelements(op) < 32768;

    snprintf(base, 256, "kernel_unary_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_cnt=%d", base, op_num, is_cnt);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_UNARY + 0);
        ggml_metal_cv_set_bool (cv, is_cnt, FC_UNARY + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_cnt;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:        op_str = "reglu";        break;
                case GGML_GLU_OP_GEGLU:        op_str = "geglu";        break;
                case GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       break;
                case GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   break;
                case GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    break;
                case GGML_GLU_OP_GEGLU_QUICK:  op_str = "geglu_quick";  break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, const ggml_tensor * op, int32_t n_fuse) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t1_str = ggml_type_name(op->src[1]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    const bool is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d", base, op_num, n_fuse, is_rb);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, n_fuse, FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, is_rb,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_rb;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one(ggml_metal_library_t lib, ggml_op op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s", "f32", "f32", "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", base, op_num, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, false,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}
