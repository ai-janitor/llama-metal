#include "00-common.metal"

// Lane helpers of the row softmax: a float row is read one value per step, a float4 row four values.
static inline int   lanes(float)  { return 1; }
static inline int   lanes(float4) { return 4; }
static inline float  upd_max(float  cur, float  x) { return MAX(cur, x); }
static inline float4 upd_max(float4 cur, float4 x) { return fmax(cur, x); }
static inline float fold_max(float  l) { return l; }
static inline float fold_max(float4 l) { return MAX(MAX(l[0], l[1]), MAX(l[2], l[3])); }
static inline float fold_sum(float  l) { return l; }
static inline float fold_sum(float4 l) { return l[0] + l[1] + l[2] + l[3]; }

// V = float or float4 (the lane), T = the mask element of that lane.
template<typename V, typename T>
kernel void kernel_soft_max(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort simd_width[[threads_per_simdgroup]],
        ushort3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const     V * psrc0 =                (device const V *    ) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device           V * pdst  =                (device       V *    ) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    // ALiBi
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    V lmaxv = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00/lanes(V()); i00 += tptg.x) {
        lmaxv = upd_max(lmaxv, psrc0[i00]*args.scale + (V)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = fold_max(lmaxv);

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (tptg.x > simd_width) {
        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ushort nsg = (tptg.x + simd_width - 1) / simd_width;
        float best = -INFINITY;
        for (ushort i = tiisg; i < nsg; i += simd_width) {
            best = MAX(best, buf[i]);
        }
        max_val = simd_max(best);
    }

    // parallel sum
    V lsumv = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00/lanes(V()); i00 += tptg.x) {
        const V exp_psrc = exp((psrc0[i00]*args.scale + (V)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsumv += exp_psrc;
        pdst[i00] = exp_psrc;
    }

    const float lsum = fold_sum(lsumv);

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > simd_width) {
        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ushort nsg = (tptg.x + simd_width - 1) / simd_width;
        float total = 0.0f;
        for (ushort i = tiisg; i < nsg; i += simd_width) {
            total += buf[i];
        }
        sum = simd_sum(total);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00/lanes(V()); i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float,  float>)  kernel_soft_max_t;
typedef decltype(kernel_soft_max<float4, float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<float,  half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float,  float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max<float4, half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max<float4, float4>;
