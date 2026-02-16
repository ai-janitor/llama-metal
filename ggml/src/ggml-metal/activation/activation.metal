#include "00-common.metal"

kernel void kernel_reglu_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig.x*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig.x*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig.x*args.nb1);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        dst_row[i0] = x0*x1*(x0 > 0.0f);
    }
}

kernel void kernel_geglu_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig.x*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig.x*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig.x*args.nb1);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu = 0.5f*x0*(1.0f + precise::tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));

        dst_row[i0] = gelu*x1;
    }
}

kernel void kernel_swiglu_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig.x*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig.x*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig.x*args.nb1);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = silu*x1;
    }
}

kernel void kernel_swiglu_oai_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig.x*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig.x*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig.x*args.nb1);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        float x0 = src0_row[i0];
        float x1 = src1_row[i0];

        x0 = min(x0, args.limit);
        x1 = max(min(x1, args.limit), -args.limit);

        float out_glu = x0 / (1.0f + exp(-x0 * args.alpha));
        out_glu = out_glu * (1.0f + x1);

        dst_row[i0] = out_glu;
    }
}

kernel void kernel_geglu_erf_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig.x*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig.x*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig.x*args.nb1);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_erf = 0.5f*x0*(1.0f+erf_approx<float>(x0*SQRT_2_INV));

        dst_row[i0] = gelu_erf*x1;
    }
}

kernel void kernel_geglu_quick_f32(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig.x*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig.x*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig.x*args.nb1);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_quick = x0*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x0)));

        dst_row[i0] = gelu_quick*x1;
    }
}
