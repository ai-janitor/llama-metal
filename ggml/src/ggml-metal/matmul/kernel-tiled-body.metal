// The tiled matmul kernel for ONE work plan. Included by kernel-tiled.metal once per compiled plan, inside that plan's
// namespace, after the plan's eleven values (constant constexpr uint tiled_<field>, the fields of struct
// ggml_metal_tiled_plan in matmul/tiled-plan.h) and TILED_PLAN_NAME (the suffix every entry name of the plan ends with).
// Everything here is a compile-time constant of the plan: a value that reaches a kernel as a function constant
// made this kernel 2.2x slower on a Vega 64 (measured 2026-11-20, 4B prompt of 512 tokens: 137 against 281 t/s).

constant constexpr uint WARP       = tiled_warp;
constant constexpr uint BLOCK_SIZE = tiled_threads;
constant constexpr uint BM         = tiled_bm;
constant constexpr uint BN         = tiled_bn;
constant constexpr uint BK         = tiled_bk;
constant constexpr uint WM         = tiled_wm;
constant constexpr uint WN         = tiled_wn;
constant constexpr uint WMITER     = tiled_wmiter;
constant constexpr uint TM         = tiled_tm;
constant constexpr uint TN         = tiled_tn;

constant constexpr uint SHMEM_STRIDE = BK/2 + tiled_pad;  // float2 slots per buffer row

constant constexpr uint WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
constant constexpr uint WSUBM = WM / WMITER;
constant constexpr uint WSUBN = WN / WNITER;

constant constexpr int LOAD_VEC_A = 2;
constant constexpr int LOAD_VEC_B = 2;

// Threadgroup memory: float2 buf_a[BM * SHMEM_STRIDE], then float2 buf_b[BN * SHMEM_STRIDE].

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
// B is src1: f32, or the copy at the model's precision (ActT). idx_n checked against ne11 (N dimension)
// Contiguous path: K elements are adjacent (stride_b is row stride only).
template <typename ActT>
inline void load_b_to_shmem(
    device const ActT * data_b,
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
        buf_b[buf_idx] = float2(float(data_b[idx]), float(data_b[idx + 1]));
    } else if (idx_n < ne_b_rows && block + loadr * LOAD_VEC_B < end_k) {
        buf_b[buf_idx] = float2(float(data_b[idx]), 0.0f);
    } else {
        buf_b[buf_idx] = float2(0.0f);
    }
}

// Strided B-loading for transposed src1: K elements are ks1 apart in device memory.
// Separate function to avoid any overhead on the contiguous path — only called when
// src1 is transposed (delta-net SSM), eliminating ggml_cont(ggml_transpose(...)) dispatches.
template <typename ActT>
inline void load_b_to_shmem_strided(
    device const ActT * data_b,
    threadgroup float2 * buf_b,
    uint pos_b,
    uint loadr,
    uint loadc,
    uint idx_n,
    uint ne_b_rows,
    uint block,
    uint end_k,
    uint stride_b,
    uint ks1
) {
    // pos_b is in K-element units; scale by ks1 for correct strided indexing
    const uint idx_base = pos_b * ks1 + loadc * stride_b + loadr * LOAD_VEC_B * ks1;
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_n < ne_b_rows && block + loadr * LOAD_VEC_B + 1 < end_k) {
        buf_b[buf_idx] = float2(float(data_b[idx_base]), float(data_b[idx_base + ks1]));
    } else if (idx_n < ne_b_rows && block + loadr * LOAD_VEC_B < end_k) {
        buf_b[buf_idx] = float2(float(data_b[idx_base]), 0.0f);
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

// ActT: element type the activations are read in (float, or the copy's type).
template <typename LoadA, typename ActT = float>
kernel void kernel_mul_mat_tiled_impl(
    constant ggml_metal_kargs_mul_mm & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],
    device const char * src1 [[buffer(2)]],
    device       char * dst  [[buffer(3)]],
    threadgroup  char * shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]]
) {
    threadgroup float2 * buf_a = (threadgroup float2 *)shmem;
    threadgroup float2 * buf_b = buf_a + BM * SHMEM_STRIDE;

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

    // B matrix setup (shared across all weight types)
    device const ActT * data_b = (device const ActT *)(src1 + args.nb13*i13 + args.nb12*i12 + args.nb11*(ic*BN));
    const uint stride_b = args.nb11 / sizeof(ActT);  // B row stride in elements
    // ks1: K-element stride (1=contiguous, >1=transposed src1)
    const uint ks1 = args.nb10 / sizeof(ActT);

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

        // Load B tile — branch once on ks1 to avoid per-element overhead on contiguous path
        if (ks1 > 1) {
            for (uint l = 0; l < BN; l += loadstride_b) {
                load_b_to_shmem_strided(data_b, buf_b, pos_b, loadr_b, loadc_b + l,
                    ic * BN + loadc_b + l, ne11, block, end_k, stride_b, ks1);
            }
        } else {
            for (uint l = 0; l < BN; l += loadstride_b) {
                load_b_to_shmem(data_b, buf_b, pos_b, loadr_b, loadc_b + l,
                    ic * BN + loadc_b + l, ne11, block, end_k, stride_b);
            }
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

template [[host_name("kernel_mul_mat_tiled_f16" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_f16>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_mxfp4" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_mxfp4>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_f32" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_f32>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

#if defined(GGML_METAL_BF16_ANY)
template [[host_name("kernel_mul_mat_tiled_bf16" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_bf16>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);
#endif

template [[host_name("kernel_mul_mat_tiled_q8_0" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q8_0>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q4_0" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q4_0>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q4_1" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q4_1>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q5_0" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q5_0>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q5_1" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q5_1>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq4_nl" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq4_nl>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

// k-quant types (QK_K=256, sub-block tracking)

template [[host_name("kernel_mul_mat_tiled_q4_K" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q4_K>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q5_K" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q5_K>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q6_K" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q6_K>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q2_K" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q2_K>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q3_K" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q3_K>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

// IQ types (QK_K=256, grid LUT + signs)

template [[host_name("kernel_mul_mat_tiled_iq4_xs" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq4_xs>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq2_xxs" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq2_xxs>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq2_xs" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq2_xs>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq2_s" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq2_s>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq3_xxs" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq3_xxs>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq3_s" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq3_s>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq1_s" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq1_s>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_iq1_m" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_iq1_m>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

// Activations read from the copy at the model's precision (act-copy/): one entry per weight type that has a row.
#if defined(GGML_METAL_BF16_ANY)
template [[host_name("kernel_mul_mat_tiled_q8_0_abf16" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q8_0, GGML_BF16_T>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q4_K_abf16" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q4_K, GGML_BF16_T>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q5_K_abf16" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q5_K, GGML_BF16_T>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);

template [[host_name("kernel_mul_mat_tiled_q6_K_abf16" TILED_PLAN_NAME)]]
kernel void kernel_mul_mat_tiled_impl<tiled_loader_q6_K, GGML_BF16_T>(
    constant ggml_metal_kargs_mul_mm &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort);
#endif
