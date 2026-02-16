#include "00-common.metal"

template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.ne00; ) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;

template [[host_name("kernel_cpy_f32_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   float>;
template [[host_name("kernel_cpy_f32_f16")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   half>;
template [[host_name("kernel_cpy_f32_i32")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   int32_t>;
template [[host_name("kernel_cpy_i32_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<int32_t, float>;
template [[host_name("kernel_cpy_i32_i32")]]   kernel kernel_cpy_t kernel_cpy_t_t<int32_t, int32_t>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_cpy_f32_bf16")]]  kernel kernel_cpy_t kernel_cpy_t_t<float,   bfloat>;
#endif
template [[host_name("kernel_cpy_f16_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<half,    float>;
template [[host_name("kernel_cpy_f16_f16")]]   kernel kernel_cpy_t kernel_cpy_t_t<half,    half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_cpy_bf16_f32")]]  kernel kernel_cpy_t kernel_cpy_t_t<bfloat,  float>;
template [[host_name("kernel_cpy_bf16_bf16")]] kernel kernel_cpy_t kernel_cpy_t_t<bfloat,  bfloat>;
#endif

template<short QK,
         typename block_q,
         void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_cpy_f32_q(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK;

    device block_q * dst_data = (device block_q *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.nk0; ) {
        device const float * src = (device const float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + (i00*QK)*args.nb00);

        quantize_func(src, dst_data[i00]);

        break;
    }
}

typedef decltype(kernel_cpy_f32_q<QK8_0,  block_q8_0,  quantize_q8_0>)  cpy_f_q_t;

template [[host_name("kernel_cpy_f32_q8_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK8_0,  block_q8_0,   quantize_q8_0>;
template [[host_name("kernel_cpy_f32_q4_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK4_0,  block_q4_0,   quantize_q4_0>;
template [[host_name("kernel_cpy_f32_q4_1")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK4_1,  block_q4_1,   quantize_q4_1>;
template [[host_name("kernel_cpy_f32_q5_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK5_0,  block_q5_0,   quantize_q5_0>;
template [[host_name("kernel_cpy_f32_q5_1")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK5_1,  block_q5_1,   quantize_q5_1>;
template [[host_name("kernel_cpy_f32_iq4_nl")]] kernel cpy_f_q_t kernel_cpy_f32_q<QK4_NL, block_iq4_nl, quantize_iq4_nl>;

template<typename T4x4, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_cpy_q_f32(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device const block_q * src_data = (device const block_q *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T4x4    * dst_data = (device       T4x4    *)(dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.nk0; ) {
        T4x4 temp;
        dequantize_func(src_data + i00/nl, i00%nl, temp);
        dst_data[i00] = temp;

        break;
    }
}

typedef decltype(kernel_cpy_q_f32<float4x4, block_q4_0, 2, dequantize_q4_0>) cpy_q_f_t;

template [[host_name("kernel_cpy_q4_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_cpy_q4_1_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_cpy_q5_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_cpy_q5_1_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_cpy_q8_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q8_0, 2, dequantize_q8_0>;

template [[host_name("kernel_cpy_q4_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_cpy_q4_1_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_cpy_q5_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_cpy_q5_1_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_cpy_q8_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q8_0, 2, dequantize_q8_0>;

kernel void kernel_concat(
    constant ggml_metal_kargs_concat & args,
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    int o[4] = {0, 0, 0, 0};
    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));

    device const float * x;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            x = (device const float *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);
        } else {
            x = (device const float *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);
        }

        device float * y = (device float *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

        *y = *x;
    }
}

