// =============================================================================
// Scalar matmul kernels for basic quant types (q4_0, q4_1, q5_0, q5_1, q8_0, f16, bf16, f32).
// =============================================================================
//
// These are hand-tuned per-type implementations — not templated — each type has
// custom load/multiply logic optimized for that specific dequant pattern.
//
// The key optimization trick in these kernels is "pre-scaled activation loads":
// instead of dequantizing weights then multiplying by activations, the activations
// (yl[]) are pre-divided by powers of 256 (1, 1/16, 1/256, 1/4096) so that the
// masked quant values can be multiplied directly without shifting. This avoids
// integer shift operations in the inner loop at the cost of pre-dividing yl.
//
// The block_q_n_dot_y() helpers compute the dot product between a HALF-block
// (il=0 for lower half, il=QK/4 for upper half) and 16 pre-scaled floats.

#include "block-dot.metal"

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];
// When true, use shared memory reduction instead of simd_sum (for Intel iGPU)
constant bool  FC_mul_mv_shmem_reduce [[function_constant(FC_MUL_MV + 2)]];
// When true, src1 is transposed: K elements are nb10/sizeof(T1) apart, not contiguous.
// Compile-time dispatch avoids runtime branching in the hot loop — zero overhead when false.
constant bool  FC_mul_mv_src1_trans [[function_constant(FC_MUL_MV + 3)]];
// Dynamic NR0: rows per simdgroup, computed from model/hardware profile.
// Set by host pipeline compilation. When NR0=8 (small model, low L1 pressure),
// each simdgroup reuses activation data across 8 weight rows instead of 4.
constant short FC_mul_mv_nr0 [[function_constant(FC_MUL_MV + 4)]];
// Row-sum tail of the shared-memory reduce: 0 serial (default), 1 pairwise tree.
constant short FC_mul_mv_tail_raw [[function_constant(FC_MUL_MV + 7)]];
constant short FC_mul_mv_tail = is_function_constant_defined(FC_mul_mv_tail_raw) ? FC_mul_mv_tail_raw : 0;
// Q8_0 weight load width in bytes: 1 (default) or 4 (aligned words, same bytes out).
constant short FC_mul_mv_q8_load_raw [[function_constant(FC_MUL_MV + 10)]];
constant short FC_mul_mv_q8_load = is_function_constant_defined(FC_mul_mv_q8_load_raw) ? FC_mul_mv_q8_load_raw : 1;
// Q8_0 activation load width in bytes: 4 (default) or 16 (two aligned loads per lane, same bytes out).
constant short FC_mul_mv_q8_act_raw [[function_constant(FC_MUL_MV + 11)]];
constant short FC_mul_mv_q8_act = is_function_constant_defined(FC_mul_mv_q8_act_raw) ? FC_mul_mv_q8_act_raw : 4;

// Shared memory reduction for GPUs where simd_sum() produces wrong results or where
// actual SIMD width differs from N_SIMDWIDTH (Intel UHD 630: th_width=16, not 32).
// Uses thread_index_in_threadgroup for SIMD-width-independent indexing.
// local_tid: thread index within the reduction group (tidx for cross-group, tidx%32 for per-group)
// group_size: number of threads participating in the reduction
// buf: pointer to group_size * NR0 floats of threadgroup memory
template<short NR0>
static inline void helper_mv_reduce_and_write_shmem(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort local_tid,
        short  group_size,
        threadgroup float * buf) {
    if (FC_mul_mv_tail == 1) {
        // Pairwise tree: log2(group_size) dependent steps instead of group_size.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short row = 0; row < NR0; ++row) {
            buf[group_size * row + local_tid] = sumf[row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short step = group_size / 2; step > 0; step /= 2) {
            if (local_tid < step) {
                for (short row = 0; row < NR0; ++row) {
                    buf[group_size * row + local_tid] += buf[group_size * row + local_tid + step];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (local_tid == 0) {
            for (short row = 0; row < NR0; ++row) {
                if (r0 + row < ne01) {
                    dst_f32[r0 + row] = buf[group_size * row];
                }
            }
        }
        // buf is rewritten by the next slice of the persistent loop: wait for lane 0's reads.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return;
    }
    // All rows written, one barrier, then lane r sums row r in the original order.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) {
        buf[group_size * row + local_tid] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = local_tid; row < NR0; row += group_size) {
        if (r0 + row < ne01) {
            float tot = 0.0f;
            for (short i = 0; i < group_size; i++) {
                tot += buf[group_size * row + i];
            }
            dst_f32[r0 + row] = tot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        ushort tidx,
        threadgroup char * shmem) {
    const short NW = N_SIMDWIDTH;
    const short nsg = FC_mul_mv_nsg;

    if (FC_mul_mv_shmem_reduce) {
        // Intel iGPU: use tidx for SIMD-width-independent reduction
        helper_mv_reduce_and_write_shmem<NR0>(dst_f32, sumf, r0, ne01, tidx, nsg * NW, (threadgroup float *) shmem);
        return;
    }

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
    for (short row = 0; row < NR0; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0 && r0 + row < ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template<typename block_q_type, short NR0, typename args_t, short ARITHMETIC_LANE = block_dot_lane_fp32>
void mul_vec_q_n_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;

    const short NW = N_SIMDWIDTH;

    // Logical SIMD grouping: on Intel (th_width=16), hardware sgitg/tiisg don't match
    // the assumed NW=32 layout. Use tidx to derive correct logical indices.
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    // Native NW-wide work distribution: 2 threads collaborate on each quant block
    // (one per half-block). All NW threads stay active at any SIMD width.
    constexpr short THREADS_PER_BLOCK = basic_block_dot::schedule::threads_per_block;
    const short CONCURRENT_BLOCKS = NW / THREADS_PER_BLOCK;
    const short block_id      = eff_tiisg / THREADS_PER_BLOCK;
    const short lane_in_block = eff_tiisg % THREADS_PER_BLOCK;

    const int nb = args.ne00/QK4_0;

    // Grid position: slice-invariant coordinates (constant across persistent loop iterations)
    const int r1 =  tgpig.y + args.tg_y_offset;
    const int im =  tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const float * y = (device const float *) (src1 + offset1);

    const short ix = block_id;
    const short il = lane_in_block * 8;
    const basic_block_dot::owned_fragment owned = basic_block_dot::make_owned_fragment(lane_in_block);

    // Persistent threadgroup loop setup.
    // When n_tg_x == 0 (normal mode), grid_stride == total_slices → loop runs once.
    const int rows_per_slice = NSG * NR0;
    const int total_slices = (args.ne01 + rows_per_slice - 1) / rows_per_slice;
    const int grid_stride = args.n_tg_x > 0 ? args.n_tg_x : total_slices;

    // -------------------------------------------------------------------------
    // Persistent threadgroup loop: each TG processes multiple row slices.
    // -------------------------------------------------------------------------
    for (int slice = (int)tgpig.x + args.tg_x_offset; slice < total_slices; slice += grid_stride) {

    const int r0 = (slice * NSG + eff_sgitg) * NR0;

    // pointers to src0 rows (change per slice — different weight rows each iteration)
    device const block_q_type * ax[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const int rb = mv_tail_row(r0 + row, args.ne01);
        const uint64_t offset0 = (uint64_t) rb*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = {0.f};  // reset each slice

    device const float * yb = y + ix*QK4_0 + il;

    // Main loop: iterate over quant blocks, striding by CONCURRENT_BLOCKS.
    for (int ib = ix; ib < nb; ib += CONCURRENT_BLOCKS) {
        basic_block_dot::prepared_activation prepared = basic_block_dot::prepare_activation<ARITHMETIC_LANE>(yb);

        // Activation reuse: same yl serves NR0 weight rows
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            basic_block_dot::accumulate<ARITHMETIC_LANE>(ax[row] + ib, owned, prepared, sumf[row]);
        }

        yb += QK4_0 * CONCURRENT_BLOCKS;
    }

    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;

    // Q4/Q5 path: each logical simdgroup computes the full sum independently (NSG loop commented out).
    // Reduce within each logical 32-thread group, not across groups.
    if (FC_mul_mv_shmem_reduce) {
        const ushort ltid = tidx % NW;
        // Each logical simdgroup gets its own shmem region to avoid races
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0;
        helper_mv_reduce_and_write_shmem<NR0>(dst_f32, sumf, r0, args.ne01, ltid, NW, buf);
    } else {
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
        for (int row = 0; row < NR0; ++row) {
            const float tot = simd_sum(sumf[row]);

            if (tiisg == 0 && r0 + row < args.ne01) {
                dst_f32[r0 + row] = tot;
            }
        }
    }

    } // end persistent threadgroup loop (slice loop)
}

// Dynamic NR0 dispatch: FC_mul_mv_nr0 is a Metal function constant set by the host
// at pipeline compile time. Since function constants are compile-time values, the
// Metal compiler eliminates the dead branch — zero runtime overhead.
// Supported values: 4 (default, safe), 8 (small models with low L1 pressure).
#define BASIC_MV_ENTRY(host_name_str, block_type, arithmetic_lane, nr0_first)       \
kernel void host_name_str(                                                          \
        constant ggml_metal_kargs_mul_mv & args,                                    \
        device const char * src0,                                                   \
        device const char * src1,                                                   \
        device       char * dst,                                                    \
        threadgroup  char * shmem [[threadgroup(0)]],                               \
        uint3  tgpig[[threadgroup_position_in_grid]],                               \
        ushort tiisg[[thread_index_in_simdgroup]],                                  \
        ushort sgitg[[simdgroup_index_in_threadgroup]],                             \
        ushort tidx [[thread_index_in_threadgroup]]) {                              \
    if (FC_mul_mv_nr0 == nr0_first) {                                              \
        mul_vec_q_n_f32_impl<block_type, nr0_first, constant ggml_metal_kargs_mul_mv &, arithmetic_lane>( \
            args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);              \
    } else if (FC_mul_mv_nr0 == 8) {                                              \
        mul_vec_q_n_f32_impl<block_type, 8, constant ggml_metal_kargs_mul_mv &, arithmetic_lane>( \
            args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);              \
    } else {                                                                        \
        mul_vec_q_n_f32_impl<block_type, 4, constant ggml_metal_kargs_mul_mv &, arithmetic_lane>( \
            args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);              \
    }                                                                               \
}

BASIC_MV_ENTRY(kernel_mul_mv_q4_0_f32, block_q4_0, block_dot_lane_fp32, N_R0_BASIC)
BASIC_MV_ENTRY(kernel_mul_mv_q4_1_f32, block_q4_1, block_dot_lane_fp32, N_R0_BASIC)
BASIC_MV_ENTRY(kernel_mul_mv_q5_0_f32, block_q5_0, block_dot_lane_fp32, N_R0_Q5_0)
BASIC_MV_ENTRY(kernel_mul_mv_q5_1_f32, block_q5_1, block_dot_lane_fp32, N_R0_BASIC)

#undef BASIC_MV_ENTRY

template<short NR0, typename args_t, short ARITHMETIC_LANE = block_dot_lane_fp32>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;

    const short NW = N_SIMDWIDTH;

    // Logical SIMD grouping for Intel (th_width=16)
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    // Native NW-wide work distribution: 4 threads per quant block,
    // each handling 8 consecutive elements (QK8_0/4 = 8).
    const short CONCURRENT_BLOCKS = NW/block_dot_q8_0_scalar::threads_per_block;
    const short block_id = eff_tiisg/block_dot_q8_0_scalar::threads_per_block;
    const short lane_in_block = eff_tiisg%block_dot_q8_0_scalar::threads_per_block;
    const block_dot_q8_0_fragment owned = block_dot_q8_0_scalar::owned_fragment(lane_in_block);

    const int nb = args.ne00/QK8_0;

    const int r0 = (tgpig.x + args.tg_x_offset)*NR0;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q8_0 * x = (device const block_q8_0 *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q8_0 * ax[NR0];
    uint64_t aoff[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const int rb = mv_tail_row(r0 + row, args.ne01);
        const uint64_t offset0 = (uint64_t) rb*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
        aoff[row] = offset0;
    }

    float sumf[NR0] = { 0.f };

    const int ib0 = eff_sgitg*CONCURRENT_BLOCKS + block_id;

    device const float * yb = y + ib0*QK8_0;

    for (int ib = ib0; ib < nb; ib += NSG*CONCURRENT_BLOCKS) {
        const block_dot_q8_0_activation prepared = block_dot_q8_0_scalar::prepare_activation<ARITHMETIC_LANE>(owned, yb, FC_mul_mv_q8_act);

        for (short row = 0; row < NR0; row++) {
            if (FC_mul_mv_q8_load == 4 && ARITHMETIC_LANE == block_dot_lane_fp32) {
                block_dot_q8_0_scalar::accumulate_words(src0, aoff[row] + (uint64_t) ib*sizeof(block_q8_0), owned, prepared, sumf[row]);
            } else {
                block_dot_q8_0_scalar::accumulate<ARITHMETIC_LANE>(ax[row] + ib, owned, prepared, sumf[row]);
            }
        }

        yb += NSG*CONCURRENT_BLOCKS*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, eff_tiisg, eff_sgitg, tidx, shmem);
}

// q8_0 kernel entry point — fixed NR0 via N_R0_Q8_0 (currently 4).
// The q8_0 impl uses a different structure than basic quants (4 threads per quant block,
// explicit row pointer array). Tuning history on D700 (NW=64, NSG=2):
//
//   NR0=2: original default, correct, ~15 t/s on Qwen3.5-0.8B
//   NR0=4: correct after shmem fix, ~34.6 t/s (2x win from activation reuse)
//   NR0=8: correct but -27% (25.2 t/s) — register pressure / occupancy collapse
//   char4 int8 loads: no effect — compiler already coalesces byte loads
//   THREADS_PER_BLOCK=2 (16 elem/thread): correct but -40% across ALL shapes.
//     With NW=64, CONCURRENT_BLOCKS=32 → stride=64 > nb for ne00=1024,
//     leaving sgitg=1 entirely idle (50% TG waste). Even larger shapes regressed
//     (+60-90%), suggesting the wider per-thread work hurts occupancy on GCN.
//
// The kernel body at THREADS_PER_BLOCK=4, NR0=4, NSG=2 is at a local optimum.
// Next gains require reducing dispatch count (persistent TGs), not kernel-body changes.
// All tested 2026-03-29.
[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

constant short FC_mul_mvg_nmat [[function_constant(FC_MUL_MV_GROUP + 0)]];

// Rows r0 .. r0 + NR0 - 1 of one matrix of a group: the body of kernel_mul_mv_q8_0_f32_impl.
template<short NR0>
static inline void mul_mvg_q8_0_rows(
        const int ne00, const int rows, const uint64_t nb01, const int r0,
        device const char * w, device const float * x, device float * y,
        threadgroup char * shmem, ushort eff_tiisg, ushort eff_sgitg, ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const short CONCURRENT_BLOCKS = NW/block_dot_q8_0_scalar::threads_per_block;
    const short block_id = eff_tiisg/block_dot_q8_0_scalar::threads_per_block;
    const short lane_in_block = eff_tiisg%block_dot_q8_0_scalar::threads_per_block;
    const block_dot_q8_0_fragment owned = block_dot_q8_0_scalar::owned_fragment(lane_in_block);
    const int nb = ne00/QK8_0;

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const int rb = mv_tail_row(r0 + row, rows);
        ax[row] = (device const block_q8_0 *) (w + (uint64_t) rb*nb01);
    }

    float sumf[NR0] = { 0.f };

    const int ib0 = eff_sgitg*CONCURRENT_BLOCKS + block_id;

    device const float * yb = x + ib0*QK8_0;

    for (int ib = ib0; ib < nb; ib += NSG*CONCURRENT_BLOCKS) {
        const block_dot_q8_0_activation prepared = block_dot_q8_0_scalar::prepare_activation<block_dot_lane_fp32>(owned, yb, FC_mul_mv_q8_act);

        for (short row = 0; row < NR0; row++) {
            if (FC_mul_mv_q8_load == 4) {
                const int rb = mv_tail_row(r0 + row, rows);
                block_dot_q8_0_scalar::accumulate_words(w, (uint64_t) rb*nb01 + (uint64_t) ib*sizeof(block_q8_0), owned, prepared, sumf[row]);
            } else {
                block_dot_q8_0_scalar::accumulate<block_dot_lane_fp32>(ax[row] + ib, owned, prepared, sumf[row]);
            }
        }

        yb += NSG*CONCURRENT_BLOCKS*QK8_0;
    }

    helper_mv_reduce_and_write<NR0>(y, sumf, r0, rows, eff_tiisg, eff_sgitg, tidx, shmem);
}

// K Q8_0 matvecs that read one activation, one launch. K is compile-time; each matrix has its own weight and
// output argument.
template<short K>
void kernel_mul_mvg_q8_0_f32_impl(
        constant ggml_metal_kargs_norm_mul_mv & args,
        device const float * x,
        device const char * w0, device const char * w1, device const char * w2, device const char * w3,
        device float * y0, device float * y1, device float * y2, device float * y3,
        threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg, ushort tidx) {
    constexpr short NR0 = N_R0_Q8_0;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;
    const int gid = (int) tgpig.x;

    // every threadgroup walks the matrices in turn; the row test is the same for all its threads
    const int r0 = gid*NR0;
    if (r0 < args.rows0) {
        mul_mvg_q8_0_rows<NR0>(args.ne00, args.rows0, args.nb01_0, r0, w0, x, y0, shmem, eff_tiisg, eff_sgitg, tidx);
    }
    if (K >= 2 && r0 < args.rows1) {
        mul_mvg_q8_0_rows<NR0>(args.ne00, args.rows1, args.nb01_1, r0, w1, x, y1, shmem, eff_tiisg, eff_sgitg, tidx);
    }
    if (K >= 3 && r0 < args.rows2) {
        mul_mvg_q8_0_rows<NR0>(args.ne00, args.rows2, args.nb01_2, r0, w2, x, y2, shmem, eff_tiisg, eff_sgitg, tidx);
    }
    if (K >= 4 && r0 < args.rows3) {
        mul_mvg_q8_0_rows<NR0>(args.ne00, args.rows3, args.nb01_3, r0, w3, x, y3, shmem, eff_tiisg, eff_sgitg, tidx);
    }
}

#define MVG_ARGS args, x, w0, w1, w2, w3, y0, y1, y2, y3, shmem, tgpig, tiisg, sgitg, tidx
[[host_name("kernel_mul_mvg_q8_0_f32")]]
kernel void kernel_mul_mvg_q8_0_f32(
        constant ggml_metal_kargs_norm_mul_mv & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device const char * w0 [[buffer(4)]], device const char * w1 [[buffer(5)]],
        device const char * w2 [[buffer(6)]], device const char * w3 [[buffer(7)]],
        device float * y0 [[buffer(8)]], device float * y1 [[buffer(9)]],
        device float * y2 [[buffer(10)]], device float * y3 [[buffer(11)]],
        threadgroup char * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    if      (FC_mul_mvg_nmat == 2) { kernel_mul_mvg_q8_0_f32_impl<2>(MVG_ARGS); }
    else if (FC_mul_mvg_nmat == 3) { kernel_mul_mvg_q8_0_f32_impl<3>(MVG_ARGS); }
    else                           { kernel_mul_mvg_q8_0_f32_impl<4>(MVG_ARGS); }
}
#undef MVG_ARGS

// Q8_0 matvec where each simdgroup owns NR0 whole rows (the default kernel splits a row's blocks over its simdgroups);
// weights as aligned words. For launches with many row groups.
template<short NR0>
void kernel_mul_mv_q8_0_f32_rows_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tidx) {
    const short NW = N_SIMDWIDTH;
    const ushort lane = tidx % NW;
    const ushort sg   = tidx / NW;
    const short CONCURRENT_BLOCKS = NW/block_dot_q8_0_scalar::threads_per_block;
    const short block_id = lane/block_dot_q8_0_scalar::threads_per_block;
    const block_dot_q8_0_fragment owned = block_dot_q8_0_scalar::owned_fragment(lane%block_dot_q8_0_scalar::threads_per_block);
    const int nb = args.ne00/QK8_0;
    const int r0 = ((int) tgpig.x*FC_mul_mv_nsg + sg)*NR0;

    uint64_t aoff[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        aoff[row] = (uint64_t) mv_tail_row(r0 + row, args.ne01)*args.nb01;
    }

    float sumf[NR0] = { 0.f };

    device const float * yb = (device const float *) src1 + block_id*QK8_0;

    for (int ib = block_id; ib < nb; ib += CONCURRENT_BLOCKS) {
        const block_dot_q8_0_activation prepared = block_dot_q8_0_scalar::prepare_activation<block_dot_lane_fp32>(owned, yb);

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            block_dot_q8_0_scalar::accumulate_words(src0, aoff[row] + (uint64_t) ib*sizeof(block_q8_0), owned, prepared, sumf[row]);
        }

        yb += CONCURRENT_BLOCKS*QK8_0;
    }

    threadgroup float * buf = (threadgroup float *) shmem + sg*NW*NR0;
    helper_mv_reduce_and_write_shmem<NR0>((device float *) dst, sumf, r0, args.ne01, lane, NW, buf);
}

[[host_name("kernel_mul_mv_q8_0_f32_rows")]]
kernel void kernel_mul_mv_q8_0_f32_rows(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    if (FC_mul_mv_nr0 == 8) {
        kernel_mul_mv_q8_0_f32_rows_impl<8>(args, src0, src1, dst, shmem, tgpig, tidx);
    } else {
        kernel_mul_mv_q8_0_f32_rows_impl<4>(args, src0, src1, dst, shmem, tgpig, tidx);
    }
}
