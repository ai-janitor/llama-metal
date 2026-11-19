// INTENT: distribute IQ blocks across the actual SIMD width.
// PDL: advance each lane by NW blocks, matching its NW-wide reduction.
// =============================================================================
// Scalar matmul kernels for k-quant types (q2_K through q6_K, iq2_xxs through iq4_xs).
// =============================================================================
//
// These kernels compute matrix-vector products for GGML's "k-quant" formats.
// k-quants use multi-level scale packing: each superblock (QK_K=256 elements)
// has a superblock-level scale (d/dmin) plus per-group sub-scales packed in
// a compact format that requires bitmask extraction.
//
// All kernels follow the same pattern:
//   1. Thread-to-data mapping: NW threads are split into groups (THREADS_PER_BLOCK),
//      each group collaborates on one superblock.
//   2. Activation reuse: activation elements are loaded once per superblock and
//      reused across nr0 weight rows (controlled by N_R0_* constants).
//   3. Cross-lane reduction: partial sums from all threads are combined via
//      simd_sum (NW=32) or threadgroup-memory reduction (shmem_reduce for NW!=32).
//
// Key variables used across all kernels:
//   NSG          — simdgroups per threadgroup (function constant FC_mul_mv_nsg)
//   NW           — SIMD width (function constant N_SIMDWIDTH, 32 or 64)
//   eff_tiisg    — effective lane index (remapped when using shmem_reduce)
//   eff_sgitg    — effective simdgroup index
//   first_row    — first output row assigned to this simdgroup
//   nr0          — how many consecutive rows each simdgroup processes
//   nb           — number of superblocks per row (ne00 / QK_K)
//
#include "dequant.metal"
#include "block-dot.metal"
#define MUL_MV_ENTRY_HEAD \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst
#define MUL_MV_ENTRY_TAIL \
        threadgroup  char * shmem [[threadgroup(0)]], \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]], \
        ushort tidx [[thread_index_in_threadgroup]]

// K-quant entry points: the loop is mul-mv.metal, the format is a struct of mv-formats.metal.
[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_q2_K, N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_q3_K, N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// Residual at buffer 4: with FC_mul_mv_residual the row result plus residual[row] is stored (one column, one plane).
[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        MUL_MV_ENTRY_HEAD,
        device const float * residual [[buffer(4)]],
        MUL_MV_ENTRY_TAIL) {
    mul_mv_rows<mv_fmt_q4_K<false>, N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, residual, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// Block headers unpacked once at load at buffer 5 (hw mv rows with prep); one plane, one column, no residual.
[[host_name("kernel_mul_mv_q4_K_prep_f32")]]
kernel void kernel_mul_mv_q4_K_prep_f32(
        MUL_MV_ENTRY_HEAD,
        device const char * prep [[buffer(5)]],
        MUL_MV_ENTRY_TAIL) {
    mul_mv_rows<mv_fmt_q4_K<true>, N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, (device const float *) dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_rows<mv_fmt_q5_K, N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_rows<mv_fmt_q6_K, N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_q4_K_cols_f32")]]
kernel void kernel_mul_mv_q4_K_cols_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_cols<mv_fmt_q4_K<false>, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_q4_K_prep_cols_f32")]]
kernel void kernel_mul_mv_q4_K_prep_cols_f32(
        MUL_MV_ENTRY_HEAD,
        device const char * prep [[buffer(5)]],
        MUL_MV_ENTRY_TAIL) {
    mul_mv_cols<mv_fmt_q4_K<true>, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_q6_K_cols_f32")]]
kernel void kernel_mul_mv_q6_K_cols_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_cols<mv_fmt_q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// i-quant and mxfp4 entry points.
[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_iq2_xxs, N_R0_IQ2_XXS, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_iq2_xs, N_R0_IQ2_XS, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_iq3_xxs, N_R0_IQ3_XXS, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_iq3_s, N_R0_IQ3_S, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_iq2_s, N_R0_IQ2_S, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// iq1_s keeps its own loop: through the shared loop its pipelines lose threads per threadgroup on 64-wide cards.
template<int nr0, typename args_t>
void kernel_mul_mv_iq1_s_f32_impl(
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
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_s * x = (device const block_iq1_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    const short ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += NW) {
        float sumy = 0;
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint16_t * qh = xr->qh + ib;
        device const half     * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));

            float sum = 0;
            for (short j = 0; j < 4; ++j) {
                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01/2;
        }

        y4 += 32 * NW;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    kernel_mul_mv_iq1_s_f32_impl<N_R0_IQ1_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_iq1_m, N_R0_IQ1_M, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_iq4_nl, N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_iq4_xs, N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_mxfp4_f32")]]
kernel void kernel_mul_mv_mxfp4_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_mxfp4, N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// 1-bit, 2-bit, ternary and nvfp4 entry points.
[[host_name("kernel_mul_mv_q1_0_f32")]]
kernel void kernel_mul_mv_q1_0_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_q1_0, N_R0_Q1_0, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_q2_0_f32")]]
kernel void kernel_mul_mv_q2_0_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_q2_0, N_R0_Q2_0, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_tq1_0_f32")]]
kernel void kernel_mul_mv_tq1_0_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_tq1_0, N_R0_TQ1_0, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_tq2_0_f32")]]
kernel void kernel_mul_mv_tq2_0_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_tq2_0, N_R0_TQ2_0, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_nvfp4_f32")]]
kernel void kernel_mul_mv_nvfp4_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_nvfp4, N_R0_NVFP4, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_f8_e4m3_f32")]]
kernel void kernel_mul_mv_f8_e4m3_f32(
        MUL_MV_ENTRY_HEAD,
        MUL_MV_ENTRY_TAIL) {
    mul_mv_impl<mv_fmt_f8_e4m3, N_R0_F8_E4M3, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}
