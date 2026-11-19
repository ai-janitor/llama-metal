#include "00-common.metal"

// INTENT: preserve destination element strides for reshaped/permuted copies.
// PDL: map the source row, then advance each destination element by nb0.
// Reference: ggml-org/llama.cpp 4e5116f47, kernels/quantize.metal:29-33 (MIT).
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

    if (i01 >= args.ne01) {
        return;
    }

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device char * dst_data = dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0;

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.ne00; ) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        *(device T1 *)(dst_data + i00*args.nb0) = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;

template [[host_name("kernel_cpy_f32_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   float>;
template [[host_name("kernel_cpy_f32_f16")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   half>;
template [[host_name("kernel_cpy_f32_i32")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   int32_t>;
template [[host_name("kernel_cpy_i32_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<int32_t, float>;
template [[host_name("kernel_cpy_i32_i32")]]   kernel kernel_cpy_t kernel_cpy_t_t<int32_t, int32_t>;
#if defined(GGML_METAL_BF16_ANY)
template [[host_name("kernel_cpy_f32_bf16")]]  kernel kernel_cpy_t kernel_cpy_t_t<float,   GGML_BF16_T>;
#endif
template [[host_name("kernel_cpy_f16_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<half,    float>;
template [[host_name("kernel_cpy_f16_f16")]]   kernel kernel_cpy_t kernel_cpy_t_t<half,    half>;
#if defined(GGML_METAL_BF16_ANY)
template [[host_name("kernel_cpy_bf16_f32")]]  kernel kernel_cpy_t kernel_cpy_t_t<GGML_BF16_T, float>;
template [[host_name("kernel_cpy_bf16_bf16")]] kernel kernel_cpy_t kernel_cpy_t_t<GGML_BF16_T, GGML_BF16_T>;
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
template [[host_name("kernel_cpy_f32_q1_0")]] kernel cpy_f_q_t kernel_cpy_f32_q<QK1_0, block_q1_0, quantize_q1_0>;
template [[host_name("kernel_cpy_f32_q2_0")]] kernel cpy_f_q_t kernel_cpy_f32_q<QK2_0, block_q2_0, quantize_q2_0>;
template [[host_name("kernel_cpy_f32_tq1_0")]] kernel cpy_f_q_t kernel_cpy_f32_q<QK_K, block_tq1_0, quantize_tq1_0>;
template [[host_name("kernel_cpy_f32_tq2_0")]] kernel cpy_f_q_t kernel_cpy_f32_q<QK_K, block_tq2_0, quantize_tq2_0>;
template [[host_name("kernel_cpy_f32_nvfp4")]] kernel cpy_f_q_t kernel_cpy_f32_q<QK_NVFP4, block_nvfp4, quantize_nvfp4>;

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

template [[host_name("kernel_cpy_q1_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q1_0, 8, dequantize_q1_0>;
template [[host_name("kernel_cpy_q2_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q2_0, 4, dequantize_q2_0>;
template [[host_name("kernel_cpy_tq1_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_tq1_0, 16, dequantize_tq1_0>;
template [[host_name("kernel_cpy_tq2_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_tq2_0, 16, dequantize_tq2_0>;
template [[host_name("kernel_cpy_nvfp4_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_nvfp4, 4, dequantize_nvfp4>;
template [[host_name("kernel_cpy_q1_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q1_0, 8, dequantize_q1_0>;
template [[host_name("kernel_cpy_q2_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q2_0, 4, dequantize_q2_0>;
template [[host_name("kernel_cpy_tq1_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_tq1_0, 16, dequantize_tq1_0>;
template [[host_name("kernel_cpy_tq2_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_tq2_0, 16, dequantize_tq2_0>;
template [[host_name("kernel_cpy_nvfp4_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_nvfp4, 4, dequantize_nvfp4>;

// For quantized types the row sizes in args are in blocks (nb0 is the block size).
template<typename T>
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
    // For narrow rows, consecutive lane spans own consecutive output rows.
    // Derive the row count from the dispatched width, not a hardware assumption.
    const bool pack_rows = args.ne0 < 32;
    const int rows_per_group = pack_rows ? max(1, int(ntg.x)/args.ne0) : 1;
    const int row_in_group = pack_rows ? int(tpitg.x)/args.ne0 : 0;
    if (row_in_group >= rows_per_group) return; // unused tail lanes when ne0 is not a divisor
    const int i1 = tgpig.x*rows_per_group + row_in_group;
    if (i1 >= args.ne1) return; // partial final group

    int o[4] = {0, 0, 0, 0};
    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));

    device const T * x;

    const int first_column = pack_rows ? int(tpitg.x)%args.ne0 : int(tpitg.x);
    for (int i0 = first_column; i0 < args.ne0; i0 += ntg.x) {
        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            x = (device const T *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);
        } else {
            x = (device const T *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);
        }

        device T * y = (device T *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

        // blocks whose size is not a multiple of 4 bytes are copied as 16-bit words
        if (sizeof(T) > 8 && sizeof(T) % 4 != 0) {
            for (uint k = 0; k < sizeof(T)/2; ++k) {
                ((device ushort *) y)[k] = ((device const ushort *) x)[k];
            }
        } else {
            *y = *x;
        }
    }
}

typedef decltype(kernel_concat<float>) kernel_concat_t;

// BF16 is copied as its 16-bit storage.
template [[host_name("kernel_concat_f32")]]  kernel kernel_concat_t kernel_concat<float>;
template [[host_name("kernel_concat_f16")]]  kernel kernel_concat_t kernel_concat<half>;
template [[host_name("kernel_concat_bf16")]] kernel kernel_concat_t kernel_concat<ushort>;
template [[host_name("kernel_concat_i8")]]   kernel kernel_concat_t kernel_concat<char>;
template [[host_name("kernel_concat_i16")]]  kernel kernel_concat_t kernel_concat<short>;
template [[host_name("kernel_concat_i32")]]  kernel kernel_concat_t kernel_concat<int>;
template [[host_name("kernel_concat_i64")]]  kernel kernel_concat_t kernel_concat<long>;
template [[host_name("kernel_concat_q4_0")]] kernel kernel_concat_t kernel_concat<block_q4_0>;
template [[host_name("kernel_concat_q4_1")]] kernel kernel_concat_t kernel_concat<block_q4_1>;
template [[host_name("kernel_concat_q5_0")]] kernel kernel_concat_t kernel_concat<block_q5_0>;
template [[host_name("kernel_concat_q5_1")]] kernel kernel_concat_t kernel_concat<block_q5_1>;
template [[host_name("kernel_concat_q8_0")]] kernel kernel_concat_t kernel_concat<block_q8_0>;

