// Scalar matmul kernels for basic quant types (q4_0, q4_1, q5_0, q5_1, q8_0, f16, bf16, f32).
// Hand-tuned per-type implementations — not templated, each type has custom load/multiply logic.
#include "dequant.metal"

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];
// When true, use shared memory reduction instead of simd_sum (for Intel iGPU)
constant bool  FC_mul_mv_shmem_reduce [[function_constant(FC_MUL_MV + 2)]];

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
    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        buf[group_size * row + local_tid] = sumf[row];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local_tid == 0) {
            float tot = 0.0f;
            for (short i = 0; i < group_size; i++) {
                tot += buf[group_size * row + i];
            }
            dst_f32[r0 + row] = tot;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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
    constexpr short NW = N_SIMDWIDTH;
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

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template<typename block_q_type, short NR0, typename args_t>
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

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 16;

    // Logical SIMD grouping: on Intel (th_width=16), hardware sgitg/tiisg don't match
    // the assumed NW=32 layout. Use tidx to derive correct logical indices.
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK4_0;

    const int r0 = ((tgpig.x + args.tg_x_offset)*NSG + eff_sgitg)*NR0;
  //const int r0 =  tgpig.x*NR0;
    const int r1 =  tgpig.y + args.tg_y_offset;
    const int im =  tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q_type * x = (device const block_q_type *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q_type * ax[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = {0.f};

    const short ix = (eff_tiisg/(NW/NQ));
    const short il = (eff_tiisg%(NW/NQ))*8;

    //const int ib0 = sgitg*NQ + ix;
    const int ib0 = ix;

    float yl[16]; // src1 vector cache

    //device const float * yb = y + ix*QK4_0 + il;
    device const float * yb = y + ib0*QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    //for (int ib = ib0; ib < nb; ib += NSG*NQ) {
    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
        //yb += NSG*NQ*QK4_0;
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
        for (int row = 0; row < NR0; ++row) {
            const float tot = simd_sum(sumf[row]);

            if (tiisg == 0 && r0 + row < args.ne01) {
                dst_f32[r0 + row] = tot;
            }
        }
    }
}

kernel void kernel_mul_mv_q4_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

kernel void kernel_mul_mv_q4_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
     mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

kernel void kernel_mul_mv_q5_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

kernel void kernel_mul_mv_q5_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<short NR0, typename args_t>
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

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    // Logical SIMD grouping for Intel (th_width=16)
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

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
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = eff_tiisg/(NW/NQ);
    const short il = eff_tiisg%(NW/NQ);

    const int ib0 = eff_sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    // each thread in a SIMD group deals with NQ quants at a time
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, eff_tiisg, eff_sgitg, tidx, shmem);
}

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
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}
