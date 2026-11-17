// GCN-safe matmul kernel for AMD GPUs with 64-wide wavefronts (e.g., FirePro D700).
//
// The standard scalar kernels use simd_sum/simd_shuffle_down which operate on all 64
// hardware lanes regardless of the FC_SIMD_WIDTH function constant. This kernel avoids
// ALL simdgroup intrinsics — uses only threadgroup memory + barrier for reduction.
//
// Each threadgroup computes one output element: dst[r0, r1].
// All 64 threads in the wavefront split the K dimension, accumulate partial sums,
// then reduce via shared memory.

#include "dequant.metal"

// tgmem_reduce_sum is now defined in 00-common.metal

// =============================================================================
// Dense (non-quantized) GCN kernels: f32, f16, bf16.
// =============================================================================
// Each threadgroup computes one output element: dst[r0, r1] = dot(src0_row, src1_row).
// All 64 threads in the wavefront split the K dimension (ne00 elements),
// each accumulating a partial sum, then reduce via shared memory.
// This is the simplest pattern — no dequantization, no scale unpacking.

// ---- f32 x f32 → f32 ----

kernel void kernel_mul_mv_gcn_f32_f32(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],   // weight matrix (f32)
        device const char * src1 [[buffer(2)]],   // input activation (f32)
        device       char * dst  [[buffer(3)]],   // output (f32)
        threadgroup float  * shmem [[threadgroup(0)]],  // shared memory for reduction (64 floats)
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntpg  [[threads_per_threadgroup]],
        uint   tidx  [[thread_index_in_threadgroup]]) {

    // Grid position → output element coordinates
    const int r0 = (int)tgpig.x + args.tg_x_offset;  // weight row index (output row)
    const int r1 = (int)tgpig.y + args.tg_y_offset;  // src1 row (0 for matvec)
    const int im = (int)tgpig.z;                       // attention heads × batch

    // GQA broadcast indices
    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    if (r0 >= args.ne01 || r1 >= args.ne11) return;

    // Byte offsets into weight and activation buffers (handles GQA broadcast)
    const uint64_t offset0 = (uint64_t)r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = (uint64_t)r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const float * x = (device const float *)(src0 + offset0);
    device const float * y = (device const float *)(src1 + offset1);

    // K-dimension split: each thread sums elements at stride ntpg.x (= 64 on GCN)
    float sum = 0.0f;
    for (int i = (int)tidx; i < args.ne00; i += (int)ntpg.x) {
        sum += x[i] * y[i];
    }

    // Reduce across all 64 threads via shared memory (simd_sum broken at NW=64)
    sum = tgmem_reduce_sum(sum, tidx, ntpg.x, shmem);

    // Thread 0 writes the final result
    if (tidx == 0) {
        device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
        dst_f32[r0] = sum;
    }
}

// ---- f16 x f32 → f32 ----

kernel void kernel_mul_mv_gcn_f16_f32(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntpg  [[threads_per_threadgroup]],
        uint   tidx  [[thread_index_in_threadgroup]]) {

    const int r0 = (int)tgpig.x + args.tg_x_offset;
    const int r1 = (int)tgpig.y + args.tg_y_offset;
    const int im = (int)tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    if (r0 >= args.ne01 || r1 >= args.ne11) return;

    const uint64_t offset0 = (uint64_t)r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = (uint64_t)r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const half  * x = (device const half  *)(src0 + offset0);
    device const float * y = (device const float *)(src1 + offset1);

    float sum = 0.0f;
    for (int i = (int)tidx; i < args.ne00; i += (int)ntpg.x) {
        sum += (float)x[i] * y[i];
    }

    sum = tgmem_reduce_sum(sum, tidx, ntpg.x, shmem);

    if (tidx == 0) {
        device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
        dst_f32[r0] = sum;
    }
}

// ---- f16 x f16 → f32 ----

kernel void kernel_mul_mv_gcn_f16_f16(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntpg  [[threads_per_threadgroup]],
        uint   tidx  [[thread_index_in_threadgroup]]) {

    const int r0 = (int)tgpig.x + args.tg_x_offset;
    const int r1 = (int)tgpig.y + args.tg_y_offset;
    const int im = (int)tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    if (r0 >= args.ne01 || r1 >= args.ne11) return;

    const uint64_t offset0 = (uint64_t)r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = (uint64_t)r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const half * x = (device const half *)(src0 + offset0);
    device const half * y = (device const half *)(src1 + offset1);

    float sum = 0.0f;
    for (int i = (int)tidx; i < args.ne00; i += (int)ntpg.x) {
        sum += (float)x[i] * (float)y[i];
    }

    sum = tgmem_reduce_sum(sum, tidx, ntpg.x, shmem);

    if (tidx == 0) {
        device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
        dst_f32[r0] = sum;
    }
}

// f32 weights x f16 input, reached via the width-64 float fallback
kernel void kernel_mul_mv_gcn_f32_f16(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntpg  [[threads_per_threadgroup]],
        uint   tidx  [[thread_index_in_threadgroup]]) {

    const int r0 = (int)tgpig.x + args.tg_x_offset;
    const int r1 = (int)tgpig.y + args.tg_y_offset;
    const int im = (int)tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    if (r0 >= args.ne01 || r1 >= args.ne11) return;

    const uint64_t offset0 = (uint64_t)r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = (uint64_t)r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const float * x = (device const float *)(src0 + offset0);
    device const half * y = (device const half *)(src1 + offset1);

    float sum = 0.0f;
    for (int i = (int)tidx; i < args.ne00; i += (int)ntpg.x) {
        sum += (float)x[i] * (float)y[i];
    }

    sum = tgmem_reduce_sum(sum, tidx, ntpg.x, shmem);

    if (tidx == 0) {
        device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
        dst_f32[r0] = sum;
    }
}

// =============================================================================
// Basic quant GCN kernels: q4_0, q4_1, q5_0, q5_1, q8_0.
// =============================================================================
// Same 1-threadgroup-per-row pattern as the dense kernels, but each thread
// dequantizes whole blocks (QK4_0/QK4_1/etc. elements) instead of single floats.
// The dequant math is inlined for each type rather than using a template.

// ---- q4_0 x f32 → f32 ----
// q4_0: 4-bit unsigned quants centered at 8. dequant = d * (q - 8).
// Block size: QK4_0=32 elements. Each byte holds 2 quants (low/high nibble).

kernel void kernel_mul_mv_gcn_q4_0_f32(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntpg  [[threads_per_threadgroup]],
        uint   tidx  [[thread_index_in_threadgroup]]) {

    const int r0 = (int)tgpig.x + args.tg_x_offset;
    const int r1 = (int)tgpig.y + args.tg_y_offset;
    const int im = (int)tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    if (r0 >= args.ne01 || r1 >= args.ne11) return;

    const uint64_t offset0 = (uint64_t)r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = (uint64_t)r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_0 * x = (device const block_q4_0 *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    const int nb = args.ne00 / QK4_0;  // number of quant blocks per row

    // Each thread processes blocks at stride 64 (ntpg.x = 64 on GCN)
    float sum = 0.0f;
    for (int ib = (int)tidx; ib < nb; ib += (int)ntpg.x) {
        device const block_q4_0 * xb = x + ib;
        device const float      * yb = y + ib * QK4_0;
        const float d = xb->d;           // per-block scale
        device const uint8_t * qs = xb->qs;  // packed 4-bit quants

        // Dequant and dot-product: loop over all QK4_0 elements in this block
        for (int j = 0; j < QK4_0; ++j) {
            const int idx = j / 2;           // byte index (2 quants per byte)
            const int shift = (j % 2) * 4;   // 0 for low nibble, 4 for high nibble
            const float xi = d * (((qs[idx] >> shift) & 0xF) - 8);  // dequant: scale * (q - 8)
            sum += xi * yb[j];
        }
    }

    sum = tgmem_reduce_sum(sum, tidx, ntpg.x, shmem);

    if (tidx == 0) {
        device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
        dst_f32[r0] = sum;
    }
}

// ---- q4_1 x f32 → f32 ----
// q4_1: 4-bit unsigned quants with min value. dequant = d * q + m.
// Same as q4_0 but uses an additive min (m) instead of centering at 8.

kernel void kernel_mul_mv_gcn_q4_1_f32(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntpg  [[threads_per_threadgroup]],
        uint   tidx  [[thread_index_in_threadgroup]]) {

    const int r0 = (int)tgpig.x + args.tg_x_offset;
    const int r1 = (int)tgpig.y + args.tg_y_offset;
    const int im = (int)tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    if (r0 >= args.ne01 || r1 >= args.ne11) return;

    const uint64_t offset0 = (uint64_t)r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = (uint64_t)r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_1 * x = (device const block_q4_1 *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    const int nb = args.ne00 / QK4_1;

    float sum = 0.0f;
    for (int ib = (int)tidx; ib < nb; ib += (int)ntpg.x) {
        device const block_q4_1 * xb = x + ib;
        device const float      * yb = y + ib * QK4_1;
        const float d = xb->d;
        const float m = xb->m;
        device const uint8_t * qs = xb->qs;

        for (int j = 0; j < QK4_1; ++j) {
            const int idx = j / 2;
            const int shift = (j % 2) * 4;
            const float xi = d * ((qs[idx] >> shift) & 0xF) + m;
            sum += xi * yb[j];
        }
    }

    sum = tgmem_reduce_sum(sum, tidx, ntpg.x, shmem);

    if (tidx == 0) {
        device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
        dst_f32[r0] = sum;
    }
}

// ---- q5_0 x f32 → f32 ----
// q5_0: 5-bit unsigned quants centered at 16. dequant = d * (q - 16).
// Low 4 bits in qs[], 5th bit in qh (32-bit field, one bit per element).

kernel void kernel_mul_mv_gcn_q5_0_f32(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntpg  [[threads_per_threadgroup]],
        uint   tidx  [[thread_index_in_threadgroup]]) {

    const int r0 = (int)tgpig.x + args.tg_x_offset;
    const int r1 = (int)tgpig.y + args.tg_y_offset;
    const int im = (int)tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    if (r0 >= args.ne01 || r1 >= args.ne11) return;

    const uint64_t offset0 = (uint64_t)r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = (uint64_t)r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q5_0 * x = (device const block_q5_0 *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    const int nb = args.ne00 / QK5_0;

    float sum = 0.0f;
    for (int ib = (int)tidx; ib < nb; ib += (int)ntpg.x) {
        device const block_q5_0 * xb = x + ib;
        device const float      * yb = y + ib * QK5_0;
        const float d = xb->d;
        device const uint8_t * qs = xb->qs;
        const uint32_t qh = *((device const uint32_t *)xb->qh);

        for (int j = 0; j < QK5_0; ++j) {
            const int idx = j / 2;
            const int shift = (j % 2) * 4;
            const int hi_bit = (qh >> j) & 1;
            const float xi = d * ((((qs[idx] >> shift) & 0xF) | (hi_bit << 4)) - 16);
            sum += xi * yb[j];
        }
    }

    sum = tgmem_reduce_sum(sum, tidx, ntpg.x, shmem);

    if (tidx == 0) {
        device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
        dst_f32[r0] = sum;
    }
}

// ---- q5_1 x f32 → f32 ----
// q5_1: 5-bit unsigned quants with min value. dequant = d * q + m.
// Same as q5_0 but uses additive min instead of centering at 16.

kernel void kernel_mul_mv_gcn_q5_1_f32(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntpg  [[threads_per_threadgroup]],
        uint   tidx  [[thread_index_in_threadgroup]]) {

    const int r0 = (int)tgpig.x + args.tg_x_offset;
    const int r1 = (int)tgpig.y + args.tg_y_offset;
    const int im = (int)tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    if (r0 >= args.ne01 || r1 >= args.ne11) return;

    const uint64_t offset0 = (uint64_t)r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = (uint64_t)r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q5_1 * x = (device const block_q5_1 *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    const int nb = args.ne00 / QK5_1;

    float sum = 0.0f;
    for (int ib = (int)tidx; ib < nb; ib += (int)ntpg.x) {
        device const block_q5_1 * xb = x + ib;
        device const float      * yb = y + ib * QK5_1;
        const float d = xb->d;
        const float m = xb->m;
        device const uint8_t * qs = xb->qs;
        const uint32_t qh = *((device const uint32_t *)xb->qh);

        for (int j = 0; j < QK5_1; ++j) {
            const int idx = j / 2;
            const int shift = (j % 2) * 4;
            const int hi_bit = (qh >> j) & 1;
            const float xi = d * (((qs[idx] >> shift) & 0xF) | (hi_bit << 4)) + m;
            sum += xi * yb[j];
        }
    }

    sum = tgmem_reduce_sum(sum, tidx, ntpg.x, shmem);

    if (tidx == 0) {
        device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
        dst_f32[r0] = sum;
    }
}

// ---- q8_0 x f32 → f32 ----
// q8_0: 8-bit signed quants. dequant = d * q (simplest quant format).
// Block size: QK8_0=32 elements. One int8 per element, one scale per block.

kernel void kernel_mul_mv_gcn_q8_0_f32(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntpg  [[threads_per_threadgroup]],
        uint   tidx  [[thread_index_in_threadgroup]]) {

    const int r0 = (int)tgpig.x + args.tg_x_offset;
    const int r1 = (int)tgpig.y + args.tg_y_offset;
    const int im = (int)tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    if (r0 >= args.ne01 || r1 >= args.ne11) return;

    const uint64_t offset0 = (uint64_t)r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = (uint64_t)r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q8_0 * x = (device const block_q8_0 *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    const int nb = args.ne00 / QK8_0;

    float sum = 0.0f;
    for (int ib = (int)tidx; ib < nb; ib += (int)ntpg.x) {
        device const block_q8_0 * xb = x + ib;
        device const float      * yb = y + ib * QK8_0;
        const float d = xb->d;

        for (int j = 0; j < QK8_0; ++j) {
            sum += d * xb->qs[j] * yb[j];
        }
    }

    sum = tgmem_reduce_sum(sum, tidx, ntpg.x, shmem);

    if (tidx == 0) {
        device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
        dst_f32[r0] = sum;
    }
}

// =============================================================================
// K-quant GCN kernels: q3_K, q4_K, q5_K, q6_K.
// =============================================================================
// These use a generic template that delegates to the per-type dequantize_*()
// functions from dequant.metal.
// Each call dequantizes 16 elements (float4x4). QK_K=256 → 16 calls per block.
//
// Key difference from the scalar k-quant kernels: the scalar kernels have
// hand-optimized per-type inner loops with activation reuse (NR0 rows).
// These GCN kernels are simpler — one threadgroup per row, no activation reuse.
// They exist as a fallback for when the scalar kernels can't work (e.g.,
// GCN shmem_reduce issues with certain types). The grouped-mv kernel
// (kernel-grouped-mv.metal) is the primary fast path for k-quants on GCN.

// Preamble shared across all k-quant GCN kernels: grid → coordinates + offsets.
#define GCN_KQUANT_PREAMBLE \
    const int r0 = (int)tgpig.x + args.tg_x_offset; /* weight row index */ \
    const int r1 = (int)tgpig.y + args.tg_y_offset; /* src1 row index */ \
    const int im = (int)tgpig.z; /* heads × batch */ \
    const uint i12 = im % args.ne12; \
    const uint i13 = im / args.ne12; \
    if (r0 >= args.ne01 || r1 >= args.ne11) return; \
    const uint64_t offset0 = (uint64_t)r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03; \
    const uint64_t offset1 = (uint64_t)r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

template<typename block_type, void (*deq_fn)(device const block_type *, short, thread float4x4 &)>
void kernel_mul_mv_gcn_kquant_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup float  * shmem,
        uint3  tgpig,
        uint3  ntpg,
        uint   tidx) {

    GCN_KQUANT_PREAMBLE

    device const block_type * x = (device const block_type *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    const int nb = args.ne00 / QK_K;  // superblocks per row

    // Work is split at sub-block granularity (16 dequant groups per superblock)
    // rather than at superblock granularity. This gives nb*16 work items instead
    // of nb, keeping all 64 GCN threads busy even when K is small.
    // Example: K=1024 → nb=4 → only 4 items at block granularity (60 threads idle).
    //          At sub-block granularity: 4*16=64 items → all threads active.
    const int n_groups = nb * 16;

    float sum = 0.0f;
    for (int ig = (int)tidx; ig < n_groups; ig += (int)ntpg.x) {
        const int ib = ig / 16;           // which superblock
        const short il = (short)(ig % 16); // which 16-element dequant group within the block

        device const block_type * xb = x + ib;
        device const float      * yb = y + ib * QK_K + il * 16;

        // Dequantize 16 elements into a 4×4 float matrix
        float4x4 reg;
        deq_fn(xb, il, reg);

        // Dot product: dequantized weights × activation
        for (short r = 0; r < 4; ++r) {
            for (short c = 0; c < 4; ++c) {
                sum += reg[r][c] * yb[r * 4 + c];
            }
        }
    }

    // Reduce and write (same tgmem pattern as dense kernels)
    sum = tgmem_reduce_sum(sum, tidx, ntpg.x, shmem);

    if (tidx == 0) {
        device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
        dst_f32[r0] = sum;
    }
}

// Macro to instantiate a GCN k-quant kernel for each type.
// Adding a new k-quant type = one line here + dequantize_* in dequant.metal.
#define GCN_KQUANT_KERNEL(name, block_type, deq_fn) \
kernel void kernel_mul_mv_gcn_##name##_f32( \
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]], \
        device const char * src0 [[buffer(1)]], \
        device const char * src1 [[buffer(2)]], \
        device       char * dst  [[buffer(3)]], \
        threadgroup float  * shmem [[threadgroup(0)]], \
        uint3  tgpig [[threadgroup_position_in_grid]], \
        uint3  ntpg  [[threads_per_threadgroup]], \
        uint   tidx  [[thread_index_in_threadgroup]]) { \
    kernel_mul_mv_gcn_kquant_impl<block_type, deq_fn>(args, src0, src1, dst, shmem, tgpig, ntpg, tidx); \
}

GCN_KQUANT_KERNEL(q4_K, block_q4_K, dequantize_q4_K)
GCN_KQUANT_KERNEL(q3_K, block_q3_K, dequantize_q3_K)
GCN_KQUANT_KERNEL(q5_K, block_q5_K, dequantize_q5_K)
GCN_KQUANT_KERNEL(q6_K, block_q6_K, dequantize_q6_K)
