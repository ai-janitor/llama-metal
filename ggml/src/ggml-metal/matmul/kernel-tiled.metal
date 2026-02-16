#include "dequant.metal"

// Tiled matmul kernel for non-simdgroup_mm GPUs (AMD Radeon, Intel UHD)
//
// Template design (REFACTOR-007):
//   The kernel body is shared across quant types. Only the A-matrix loading
//   differs per type (pointer type, stride semantics, K-tracking, dequant).
//   Three options were considered:
//
//   Option A: Functor — encapsulate all 5 divergence points behind one
//     load_a interface. Clean but requires designing a new uniform signature
//     across types with fundamentally different memory layouts (typed vs byte
//     pointers, element vs block K-tracking).
//
//   Option B: Reuse dequantize_*_t4 — adapt the tiled kernel to call the
//     existing per-element dequant functions. Doesn't work because the tiled
//     divergence is cooperative loading (threads coordinate to fill shared
//     memory), not per-element decompression.
//
//   Option C (chosen): Keep existing load_a functions unchanged, wrap each
//     in a loader struct with init/load/advance methods, template the kernel
//     body on the struct. Simplest — the load functions already exist and are
//     tested. The compiler inlines everything, producing identical machine
//     code to the hand-duplicated kernels.
//
// Threadgroup size: 64 threads (1D)
// Output tile per threadgroup: BM=64 rows × BN=64 cols
// K-dimension tile: BK=32
// Per-warp tile: WM=32 × WN=32 (2 warps per M, 2 per N = 4 warps)
// Per-thread accumulator: TM=4 × TN=2, WMITER=2
//
// Uses flat tid (thread_index_in_threadgroup) for thread identity - no simd intrinsics.
// Works with variable SIMD widths (Intel Gen9 may use simd_size=8 or 16, AMD uses 64).

constant constexpr int BM = 64;
constant constexpr int BN = 64;
constant constexpr int BK = 32;

constant constexpr int WM = 32;
constant constexpr int WN = 32;
constant constexpr int WMITER = 2;
constant constexpr int TM = 4;
constant constexpr int TN = 2;

constant constexpr int WARP = 32;
constant constexpr int BLOCK_SIZE = 128;

constant constexpr int SHMEM_STRIDE = BK/2 + 1;  // +1 padding to avoid bank conflicts

constant constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
constant constexpr int WSUBM = WM / WMITER;
constant constexpr int WSUBN = WN / WNITER;

constant constexpr int LOAD_VEC_A = 2;
constant constexpr int LOAD_VEC_B = 2;

// Shared memory buffers: float2[BM × SHMEM_STRIDE] + float2[BN × SHMEM_STRIDE]
// = 64×17×8 + 64×17×8 = 17408 bytes
typedef struct {
    float2 buf_a[BM * SHMEM_STRIDE];  // 64 × 17 float2
    float2 buf_b[BN * SHMEM_STRIDE];  // 64 × 17 float2
} TiledSharedMemory;

#include "kernel-tiled-loaders-qk32.metal"
#include "kernel-tiled-loaders-kquant.metal"
#include "kernel-tiled-loaders-iq.metal"

// Cooperative load: A tile (BM rows × BK cols) into shared memory
// Matches Vulkan mul_mm_funcs.glsl:1-27 (LOAD_VEC_BATCH_A=2 path)
// pos_a = K-dimension offset, data_a already points to tile's first row
// idx_m = global row index, checked against ne01 (M dimension)
// block + loadr*2 checked against end_k (K dimension)
inline void load_a_to_shmem(
    device const half * data_a,
    threadgroup float2 * buf_a,
    uint pos_a,
    uint loadr,
    uint loadc,
    uint idx_m,
    uint ne01,
    uint block,
    uint end_k,
    uint stride_a
) {
    const uint idx = pos_a + loadc * stride_a + loadr * LOAD_VEC_A;
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    // Vulkan ref: idx_m < p.M && block + row * 2 + 1 < end_k
    if (idx_m < ne01 && block + loadr * LOAD_VEC_A + 1 < end_k) {
        buf_a[buf_idx] = float2(float(data_a[idx]), float(data_a[idx + 1]));
    } else if (idx_m < ne01 && block + loadr * LOAD_VEC_A < end_k) {
        buf_a[buf_idx] = float2(float(data_a[idx]), 0.0f);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

// Cooperative load: A tile (BM rows × BK cols) from mxfp4 blocks into shared memory
// Each K-tile is exactly 1 mxfp4 block per row (BK=32=QK_MXFP4)
// loadr ranges 0..15 (BK/LOAD_VEC_A = 32/2), identifies which byte in qs[] to read
// Each thread reads 1 byte from qs[], extracts 2 nibbles, dequantizes via LUT, scales by e8m0
inline void load_a_mxfp4_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr,
    uint loadc,
    uint idx_m,
    uint ne01,
    uint block_k,
    uint num_blocks,
    uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_k < num_blocks) {
        device const block_mxfp4 * block_ptr = (device const block_mxfp4 *)(src0_row + loadc * nb01) + block_k;

        const float scale = e8m0_to_fp32(block_ptr->e);

        // mxfp4 nibble packing: low nibbles = elements 0-15, high nibbles = elements 16-31
        // float2 at shmem[loadr] = elements (loadr*2, loadr*2+1)
        // loadr 0..7:  elements 0-15 from low nibbles of 2 consecutive bytes
        // loadr 8..15: elements 16-31 from high nibbles of 2 consecutive bytes
        if (loadr < 8) {
            const uint8_t byte0 = block_ptr->qs[2*loadr];
            const uint8_t byte1 = block_ptr->qs[2*loadr + 1];
            buf_a[buf_idx] = float2(scale * kvalues_mxfp4_f[byte0 & 0x0F], scale * kvalues_mxfp4_f[byte1 & 0x0F]);
        } else {
            const uint8_t byte0 = block_ptr->qs[2*(loadr - 8)];
            const uint8_t byte1 = block_ptr->qs[2*(loadr - 8) + 1];
            buf_a[buf_idx] = float2(scale * kvalues_mxfp4_f[byte0 >> 4], scale * kvalues_mxfp4_f[byte1 >> 4]);
        }
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

// Cooperative load: B tile (BN rows × BK cols) into shared memory
// B is f32 (src1 type). idx_n checked against ne11 (N dimension)
inline void load_b_to_shmem(
    device const float * data_b,
    threadgroup float2 * buf_b,
    uint pos_b,
    uint loadr,
    uint loadc,
    uint idx_n,
    uint ne_b_rows,
    uint block,
    uint end_k,
    uint stride_b
) {
    const uint idx = pos_b + loadc * stride_b + loadr * LOAD_VEC_B;
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_n < ne_b_rows && block + loadr * LOAD_VEC_B + 1 < end_k) {
        buf_b[buf_idx] = float2(data_b[idx], data_b[idx + 1]);
    } else if (idx_n < ne_b_rows && block + loadr * LOAD_VEC_B < end_k) {
        buf_b[buf_idx] = float2(data_b[idx], 0.0f);
    } else {
        buf_b[buf_idx] = float2(0.0f);
    }
}

// ---------------------------------------------------------------------------
// Loader structs — each wraps one of the load_a functions above.
// The template parameter to kernel_mul_mat_tiled_impl selects which loader
// to use. The compiler inlines everything — identical machine code to the
// hand-duplicated kernels.
// ---------------------------------------------------------------------------

// f16: typed pointer, element-based stride, pos_a tracks element offset
struct tiled_loader_f16 {
    device const half * data_a;
    uint stride_a;
    uint pos_a;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        data_a = (device const half *)(src0 + offset0 + args.nb01 * (ir * BM));
        stride_a = args.nb01 / sizeof(half);
        pos_a = 0;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_to_shmem(data_a, buf_a, pos_a, loadr, loadc, idx_m, ne01, block, end_k, stride_a);
    }

    void advance() { pos_a += BK; }
};

// mxfp4: byte pointer, block-based K tracking, LUT dequant
struct tiled_loader_mxfp4 {
    device const char * src0_row;
    uint num_blocks;
    uint block_k;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK_MXFP4;
        block_k = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_mxfp4_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_k, num_blocks, nb01);
    }

    void advance() { block_k++; }
};

// ---------------------------------------------------------------------------
// Shared kernel body — templated on LoadA strategy.
// Adding a new quant type = add one loader struct + one instantiation line.
// ---------------------------------------------------------------------------

template <typename LoadA>
kernel void kernel_mul_mat_tiled_impl(
    constant ggml_metal_kargs_mul_mm & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],
    device const char * src1 [[buffer(2)]],
    device       char * dst  [[buffer(3)]],
    threadgroup  char * shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]]
) {
    threadgroup TiledSharedMemory * shared = (threadgroup TiledSharedMemory *)shmem;
    threadgroup float2 * buf_a = shared->buf_a;
    threadgroup float2 * buf_b = shared->buf_b;

    const uint ir = tgpig.x + args.tg_x_offset;  // M dimension threadgroup index (offset for chunked dispatch)
    const uint ic = tgpig.y;  // N dimension threadgroup index
    const uint im = tgpig.z;  // batch index

    // Batch indexing with broadcast (from mul-mm.metal lines 48-51)
    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;
    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    // Thread identity (flat tid, not simdgroup-based)
    const uint warp_i = tid / WARP;
    const uint tiw = tid % WARP;

    const uint warp_r = warp_i % (BM / WM);  // warp row within threadgroup tile
    const uint warp_c = warp_i / (BM / WM);  // warp col within threadgroup tile

    const uint tiwr = tiw % (WSUBM / TM);
    const uint tiwc = tiw / (WSUBM / TM);

    // Cooperative load indices (from Vulkan mul_mm.comp lines 194-200)
    const uint loadr_a = tid % (BK / LOAD_VEC_A);
    const uint loadc_a = tid / (BK / LOAD_VEC_A);
    const uint loadr_b = tid % (BK / LOAD_VEC_B);
    const uint loadc_b = tid / (BK / LOAD_VEC_B);

    const uint loadstride_a = BLOCK_SIZE * LOAD_VEC_A / BK;
    const uint loadstride_b = BLOCK_SIZE * LOAD_VEC_B / BK;

    // Type-specific A-matrix loader
    LoadA loader;
    loader.init(src0, offset0, ir, args);

    // B matrix setup (shared across all types — B is always f32)
    device const float * data_b = (device const float *)(src1 + args.nb13*i13 + args.nb12*i12 + args.nb11*(ic*BN));
    const uint stride_b = args.nb11 / sizeof(float);  // B row stride in elements

    const uint end_k = args.ne00;  // K dimension

    const uint ne01 = args.ne0;  // M (output rows)
    const uint ne11 = args.ne1;  // N (output cols, also B rows for non-broadcast)

    uint pos_b = 0;

    float2 sums[WMITER * TM * WNITER * TN / 2];
    for (uint i = 0; i < WMITER * TM * WNITER * TN / 2; i++) {
        sums[i] = float2(0.0f);
    }

    for (uint block = 0; block < end_k; block += BK) {
        // Load A tile — dispatched to type-specific loader
        for (uint l = 0; l < BM; l += loadstride_a) {
            loader.load(buf_a, loadr_a, loadc_a + l, ir * BM + loadc_a + l, ne01, block, end_k);
        }

        // Load B tile (shared — B is always f32)
        for (uint l = 0; l < BN; l += loadstride_b) {
            load_b_to_shmem(
                data_b,
                buf_b,
                pos_b,
                loadr_b,
                loadc_b + l,
                ic * BN + loadc_b + l,
                ne11,
                block,
                end_k,
                stride_b
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        loader.advance();
        pos_b += BK;

        // Compute: nested FMA loops (from Vulkan mul_mm.comp lines 304-343)
        // BK_STEP=4 (processes 4 K elements = 2 float2 per iteration)
        float4 cache_a[WMITER * TM];
        float4 cache_b;

        for (uint i = 0; i < BK / 4; i++) {
            // Load cache_a[WMITER*TM] from buf_a
            for (uint wsir = 0; wsir < WMITER; wsir++) {
                for (uint j = 0; j < TM; j++) {
                    const uint row = warp_r * WM + wsir * WSUBM + tiwr * TM + j;
                    cache_a[wsir * TM + j].xy = buf_a[row * SHMEM_STRIDE + 2*i    ];
                    cache_a[wsir * TM + j].zw = buf_a[row * SHMEM_STRIDE + 2*i + 1];
                }
            }

            // Inner loop: iterate over columns (WNITER=2, TN=2)
            for (uint wsic = 0; wsic < WNITER; wsic++) {
                for (uint cc = 0; cc < TN; cc++) {
                    const uint col = warp_c * WN + wsic * WSUBN + tiwc * TN + cc;
                    cache_b.xy = buf_b[col * SHMEM_STRIDE + 2*i    ];
                    cache_b.zw = buf_b[col * SHMEM_STRIDE + 2*i + 1];

                    // FMA accumulation (from Vulkan lines 326-334)
                    for (uint wsir = 0; wsir < WMITER; wsir++) {
                        for (uint cr = 0; cr < TM / 2; cr++) {
                            const uint sums_idx = (wsic * TN + cc) * WMITER * (TM / 2) + wsir * (TM / 2) + cr;

                            sums[sums_idx].x = fma(cache_a[wsir * TM + 2*cr    ].x, cache_b.x,
                                               fma(cache_a[wsir * TM + 2*cr    ].y, cache_b.y,
                                               fma(cache_a[wsir * TM + 2*cr    ].z, cache_b.z,
                                               fma(cache_a[wsir * TM + 2*cr    ].w, cache_b.w, sums[sums_idx].x))));

                            sums[sums_idx].y = fma(cache_a[wsir * TM + 2*cr + 1].x, cache_b.x,
                                               fma(cache_a[wsir * TM + 2*cr + 1].y, cache_b.y,
                                               fma(cache_a[wsir * TM + 2*cr + 1].z, cache_b.z,
                                               fma(cache_a[wsir * TM + 2*cr + 1].w, cache_b.w, sums[sums_idx].y))));
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output tile to dst
    device float * C = (device float *)dst + im * args.ne1 * args.ne0;

    // Store: each thread writes TM×TN = 4×2 = 8 elements
    for (uint wsic = 0; wsic < WNITER; wsic++) {
        for (uint wsir = 0; wsir < WMITER; wsir++) {
            for (uint cc = 0; cc < TN; cc++) {
                for (uint cr = 0; cr < TM / 2; cr++) {
                    const uint sums_idx = (wsic * TN + cc) * WMITER * (TM / 2) + wsir * (TM / 2) + cr;

                    const uint dst_row0 = ir * BM + warp_r * WM + wsir * WSUBM + tiwr * TM + 2*cr;
                    const uint dst_row1 = dst_row0 + 1;
                    const uint dst_col  = ic * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc;

                    if (dst_row0 < args.ne0 && dst_col < args.ne1) {
                        C[dst_col * args.ne0 + dst_row0] = sums[sums_idx].x;
                    }
                    if (dst_row1 < args.ne0 && dst_col < args.ne1) {
                        C[dst_col * args.ne0 + dst_row1] = sums[sums_idx].y;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Template instantiations — adding a new quant type = one line here.
// host_name preserves the kernel name visible to the pipeline/dispatch code.
// ---------------------------------------------------------------------------

template [[host_name("kernel_mul_mat_tiled_f16")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_f16>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_mxfp4")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_mxfp4>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_f32")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_f32>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mat_tiled_bf16")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_bf16>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);
#endif

template [[host_name("kernel_mul_mat_tiled_q8_0")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q8_0>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q4_0")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q4_0>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q4_1")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q4_1>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q5_0")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q5_0>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q5_1")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q5_1>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq4_nl")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq4_nl>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

// k-quant types (QK_K=256, sub-block tracking)

template [[host_name("kernel_mul_mat_tiled_q4_K")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q4_K>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q5_K")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q5_K>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q6_K")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q6_K>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q2_K")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q2_K>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q3_K")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q3_K>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

// IQ types (QK_K=256, grid LUT + signs)

template [[host_name("kernel_mul_mat_tiled_iq4_xs")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq4_xs>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq2_xxs")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq2_xxs>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq2_xs")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq2_xs>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq2_s")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq2_s>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq3_xxs")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq3_xxs>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq3_s")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq3_s>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq1_s")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq1_s>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq1_m")]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq1_m>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);
