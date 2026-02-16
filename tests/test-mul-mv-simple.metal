// Minimal standalone mul_mv kernel for testing
// Extracted from mul-mv-ext.metal - just the simple f32_f32 short variant

#include <metal_stdlib>
using namespace metal;

// Kernel args struct
typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mv;

// Simple matrix-vector multiply kernel
// Each thread computes one output element (one dot product)
kernel void kernel_mul_mv_f32_f32_simple(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {

    const int r0 = tgpig.x*32 + tiisg;  // row index
    const int r1 = tgpig.y;             // vector index (usually 0)
    const int im = tgpig.z;             // batch index (usually 0)

    if (r0 >= args.ne01) {
        return;
    }

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const float * x = (device const float *) (src0 + offset0);
    device const float * y = (device const float *) (src1 + offset1);

    float res = 0.0f;

    for (int i = 0; i < args.ne00; ++i) {
        res += x[i] * y[i];
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;
    dst_f32[(uint64_t)r1*args.ne0 + r0] = res;
}
