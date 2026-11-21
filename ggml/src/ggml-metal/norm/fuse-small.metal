// Combine small elementwise and norm chains with their original F32 boundaries.
#include "00-common.metal"

// Preserve the stored F32 value between formerly separate kernels.
static inline float small_f32_boundary(float value) {
    volatile float rounded = value;
    return rounded;
}

static inline float4 small_f32_boundary(float4 value) {
    return float4(small_f32_boundary(value.x), small_f32_boundary(value.y),
                  small_f32_boundary(value.z), small_f32_boundary(value.w));
}

template <typename T>
kernel void kernel_small_gate_g_impl(
        constant ggml_metal_kargs_small_gate & args,
        device const char * src,
        device const char * bias,
        device const char * multiplier,
        device char * dst,
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 tid [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int i1 = group.x % args.ne1;
    const int i2 = (group.x / args.ne1) % args.ne2;
    const int i3 = group.x / (args.ne1 * args.ne2);
    device const T * x = (device const T *) (src + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device const T * b = (device const T *) bias;
    device const T * a = (device const T *) multiplier;
    device T * y = (device T *) (dst + i1*args.nb1 + i2*args.nb2 + i3*args.nb3);
    for (int i = tid.x; i < args.ne0; i += ntg.x) {
        const T xv = x[i];
        const T bv = b[i];
        const T av = a[i];
        const T v = small_f32_boundary(xv + bv);
        const T softplus = small_f32_boundary(select(log(1 + exp(v)), v, v > 20));
        y[i] = softplus * av;
    }
}

typedef decltype(kernel_small_gate_g_impl<float>) kernel_small_gate_g_t;
template [[host_name("kernel_small_gate_g_f32")]] kernel kernel_small_gate_g_t kernel_small_gate_g_impl<float>;

constant short FC_small_attn_gate_e [[function_constant(FC_SMALL_ATTN_GATE + 0)]];

// Attention output gate: kqv row times the sigmoid of the gate half of the q projection, both read in place.
// One threadgroup; a thread owns E elements and reads them all before the first write, so the result may overlap
// either input.
template <typename T, short E>
static inline void small_attn_gate_impl(
        constant ggml_metal_kargs_small_gate & args,
        device const char * gate,
        device const char * kqv,
        device char * dst,
        ushort3 tid,
        ushort3 ntg) {
    T s[E];
    T xv[E];
    FOR_UNROLL (short e = 0; e < E; ++e) {
        const int i  = tid.x + e*ntg.x;
        const int i1 = i / args.ne0;
        const int i0 = i - i1*args.ne0;
        T gv = T(0);
        xv[e] = T(0);
        if (i1 < args.ne1) {
            gv    = ((device const T *) (gate + i1*args.nb01))[i0];
            xv[e] = ((device const T *) (kqv  + i1*args.nb1))[i0];
        }
        s[e] = small_f32_boundary(1 / (1 + exp(-gv)));
    }
    threadgroup_barrier(mem_flags::mem_none);
    FOR_UNROLL (short e = 0; e < E; ++e) {
        const int i  = tid.x + e*ntg.x;
        const int i1 = i / args.ne0;
        const int i0 = i - i1*args.ne0;
        if (i1 < args.ne1) {
            ((device T *) (dst + i1*args.nb1))[i0] = xv[e] * s[e];
        }
    }
}

[[host_name("kernel_small_attn_gate_f32_4")]]
kernel void kernel_small_attn_gate_f32_4(
        constant ggml_metal_kargs_small_gate & args,
        device const char * gate,
        device const char * kqv,
        device char * dst,
        ushort3 tid [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    if (FC_small_attn_gate_e == 2) {
        small_attn_gate_impl<float4, 2>(args, gate, kqv, dst, tid, ntg);
    } else {
        small_attn_gate_impl<float4, 1>(args, gate, kqv, dst, tid, ntg);
    }
}

template <typename T, bool GATED>
kernel void kernel_small_norm_impl(
        constant ggml_metal_kargs_small_norm & args,
        device const char * src0,
        device const char * src1,
        device const char * gate,
        device char * dst0,
        device char * dst1,
        threadgroup float * shmem [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int first_rows = args.ne[0][0] * args.ne[0][1] * args.ne[0][2];
    const bool second = !GATED && group.x >= (uint) first_rows;
    const int which = second ? 1 : 0;
    const int row = group.x - (second ? first_rows : 0);
    const int i1 = row % args.ne[which][0];
    const int i2 = (row / args.ne[which][0]) % args.ne[which][1];
    const int i3 = row / (args.ne[which][0] * args.ne[which][1]);
    constant ggml_metal_kargs_norm & a = args.norm[which];
    device const char * src = second ? src1 : src0;
    device char * dst = second ? dst1 : dst0;
    device const T * x = (device const T *) (src + i1*a.nbf1[0] + i2*a.nbf2[0] + i3*a.nbf3[0]);
    device T * y = (device T *) (dst + i1*a.nb1 + i2*a.nb2 + i3*a.nb3);

    const T xv = x[tid.x];
    T wv(0.0f);
    T gv(0.0f);
    if (GATED) {
        wv = ((device const T *) src1)[tid.x];
        gv = ((device const T *) (gate + i1*args.nbg1 + i2*args.nbg2 + i3*args.nbg3))[tid.x];
    }

    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }
    float sum = 0.0f;
    sum += dot(xv, xv);
    const ushort threads = ntg.x;
    sum = tgmem_tree_sum<true>(sum, tid.x, threads, shmem);
    const float mean = sum / a.ne00;
    const float scale = 1.0f / sqrt(mean + a.eps);
    if (GATED) {
        const T weighted = small_f32_boundary((xv * scale) * wv);
        const T silu = small_f32_boundary(gv / (1 + exp(-gv)));
        y[tid.x] = weighted * silu;
    } else {
        y[tid.x] = (xv * scale) * a.scale;
    }
}

typedef decltype(kernel_small_norm_impl<float, true>) kernel_small_norm_t;
template [[host_name("kernel_small_gated_rms_norm_f32")]] kernel kernel_small_norm_t kernel_small_norm_impl<float, true>;
template [[host_name("kernel_small_gated_rms_norm_f32_4")]] kernel kernel_small_norm_t kernel_small_norm_impl<float4, true>;
template [[host_name("kernel_small_split_l2_f32")]] kernel kernel_small_norm_t kernel_small_norm_impl<float, false>;
template [[host_name("kernel_small_split_l2_f32_4")]] kernel kernel_small_norm_t kernel_small_norm_impl<float4, false>;

// Residual add, the rms norm of its sum and the weight multiply in one launch; the sum is written too.
// One threadgroup, one float4 per thread: every thread reads its inputs before the first barrier and
// writes after the last one, so a result that shares memory with an input is safe.
kernel void kernel_small_add_norm_f32_4(
        constant ggml_metal_kargs_norm & args,
        device const float4 * src0,
        device const float4 * src1,
        device const float4 * weight,
        device float4 * dst_sum,
        device float4 * dst,
        threadgroup float * shmem [[threadgroup(0)]],
        ushort3 tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const float4 xv = small_f32_boundary(src0[tid.x] + src1[tid.x]);

    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }
    float sum = 0.0f;
    sum += dot(xv, xv);
    const ushort threads = ntg.x;
    sum = tgmem_tree_sum<true>(sum, tid.x, threads, shmem);
    const float mean = sum / args.ne00;
    const float scale = 1.0f / sqrt(mean + args.eps);
    dst_sum[tid.x] = xv;
    dst[tid.x] = (xv * scale) * weight[tid.x];
}
