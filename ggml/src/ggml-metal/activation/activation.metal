#include "00-common.metal"

// One expression per gated-linear unit: the output of the value x0 gated by x1.
struct glu_reglu {
    static float apply(float x0, float x1, constant ggml_metal_kargs_glu &) {
        return x0*x1*(x0 > 0.0f);
    }
};

struct glu_geglu {
    static float apply(float x0, float x1, constant ggml_metal_kargs_glu &) {
        const float gelu = 0.5f*x0*(1.0f + precise::tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));

        return gelu*x1;
    }
};

struct glu_swiglu {
    static float apply(float x0, float x1, constant ggml_metal_kargs_glu &) {
        const float silu = x0 / (1.0f + exp(-x0));

        return silu*x1;
    }
};

struct glu_swiglu_oai {
    static float apply(float x0, float x1, constant ggml_metal_kargs_glu & args) {
        x0 = min(x0, args.limit);
        x1 = max(min(x1, args.limit), -args.limit);

        float out_glu = x0 / (1.0f + exp(-x0 * args.alpha));
        out_glu = out_glu * (1.0f + x1);

        return out_glu;
    }
};

struct glu_swiglu_clamp {
    static float apply(float x0, float x1, constant ggml_metal_kargs_glu & args) {
        const float gate = min(x0, args.limit);
        const float up   = clamp(x1, -args.limit, args.limit);

        return gate / (1.0f + exp(-gate)) * up;
    }
};

struct glu_geglu_erf {
    static float apply(float x0, float x1, constant ggml_metal_kargs_glu &) {
        const float gelu_erf = 0.5f*x0*(1.0f+erf_approx<float>(x0*SQRT_2_INV));

        return gelu_erf*x1;
    }
};

struct glu_geglu_quick {
    static float apply(float x0, float x1, constant ggml_metal_kargs_glu &) {
        const float gelu_quick = x0*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x0)));

        return gelu_quick*x1;
    }
};

template<typename Expr>
kernel void kernel_glu(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]],
        uint3    ntgs[[threadgroups_per_grid]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        dst_row[i0] = Expr::apply(x0, x1, args);
    }
}

typedef decltype(kernel_glu<glu_reglu>) kernel_glu_t;

template [[host_name("kernel_reglu_f32")]]        kernel kernel_glu_t kernel_glu<glu_reglu>;
template [[host_name("kernel_geglu_f32")]]        kernel kernel_glu_t kernel_glu<glu_geglu>;
template [[host_name("kernel_swiglu_f32")]]       kernel kernel_glu_t kernel_glu<glu_swiglu>;
template [[host_name("kernel_swiglu_oai_f32")]]   kernel kernel_glu_t kernel_glu<glu_swiglu_oai>;
template [[host_name("kernel_swiglu_clamp_f32")]] kernel kernel_glu_t kernel_glu<glu_swiglu_clamp>;
template [[host_name("kernel_geglu_erf_f32")]]    kernel kernel_glu_t kernel_glu<glu_geglu_erf>;
template [[host_name("kernel_geglu_quick_f32")]]  kernel kernel_glu_t kernel_glu<glu_geglu_quick>;
