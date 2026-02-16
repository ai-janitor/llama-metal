// Scalar matmul kernels for k-quant types (q2_K through q6_K, iq2_xxs through iq4_xs).
// Multi-level scale unpacking — more complex dequant than basic types.
#include "dequant.metal"

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
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

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const short ix = eff_tiisg/8;  // 0...3
    const short it = eff_tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3
    const short is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q3_K_f32_impl(
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

    device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float yl[32];

    //const uint16_t kmask1 = 0x3030;
    //const uint16_t kmask2 = 0x0f0f;

    const short tid = eff_tiisg/4;
    const short ix  = eff_tiisg%4;
    const short ip  = tid/4;          // 0 or 1
    const short il  = 2*((tid%4)/2);  // 0 or 2
    const short ir  = tid%2;
    const short l0  = 8*ir;

    // One would think that the Metal compiler would figure out that ip and il can only have
    // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it
    // with these two tales.
    //
    // Possible masks for the high bit
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

    // Possible masks for the low 2 bits
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const short shift = 2*il;

    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const short q_offset = 32*ip + l0;
    const short y_offset = 128*ip + 32*il + l0;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[nr0] = {0.f};
    float sumf2[nr0] = {0.f};

    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += args.nb01/2;
            h  += args.nb01/2;
            a  += args.nb01/2;
            dh += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    for (int row = 0; row < nr0; ++row) {
        sumf1[row] = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf1[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0; ++row) {
            sumf1[row] = simd_sum(sumf1[row]);
        }
        if (tiisg == 0) {
            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                dst_f32[first_row + row] = sumf1[row];
            }
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
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
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    // 16-thread-per-superblock mapping (matches Vulkan mul_mat_vec_q4_k)
    const short ix   = eff_tiisg/16;         // 0 or 1 (2 block groups)
    const short itid = eff_tiisg%16;         // 0..15
    const short il   = itid/4;           // 0..3
    const short ir   = itid%4;           // 0..3
    const short v_im = il/2;             // 0 or 1 (which 128-element half)
    const short v_in = il%2;             // 0 or 1 (interleaved within half)
    const short l0   = 4*(2*ir + v_in);  // 0,4,8,...,28

    const uint q_offset = 32*v_im + l0;
    const uint y_offset = 64*v_im + l0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float sumf[nr0]={0.f};

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 2) {
        device const float * yb = y + ib*QK_K + y_offset;

        // 4 vec4 loads (16 floats per thread, half of old 32)
        float4 by10  = *(device const float4 *)(yb +   0);
        float4 by132 = *(device const float4 *)(yb +  32);
        float4 by20  = *(device const float4 *)(yb + 128);
        float4 by232 = *(device const float4 *)(yb + 160);

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + v_im;
        device const uint32_t * qs_ptr = (device const uint32_t *)(x[ib].qs + q_offset);
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            // Extract nibbles from 4 bytes of quants + 4 bytes at +64
            uint32_t qs0  = qs_ptr[0];
            uint32_t qs64 = qs_ptr[16];  // +64 bytes

            uchar4 qs0_lo  = as_type<uchar4>(qs0  & 0x0F0F0F0Fu);
            uchar4 qs0_hi  = as_type<uchar4>((qs0  >> 4) & 0x0F0F0F0Fu);
            uchar4 qs64_lo = as_type<uchar4>(qs64 & 0x0F0F0F0Fu);
            uchar4 qs64_hi = as_type<uchar4>((qs64 >> 4) & 0x0F0F0F0Fu);

            // Dot products matching Vulkan's sx/sy/sz/sw
            float sx = dot(by10,  float4(qs0_lo));
            float sy = dot(by132, float4(qs0_hi));
            float sz = dot(by20,  float4(qs64_lo));
            float sw = dot(by232, float4(qs64_hi));

            // smin computed inline (no separate sumy registers)
            float smin = (by10.x  + by10.y  + by10.z  + by10.w)  * float(sc8[2]) +
                         (by132.x + by132.y + by132.z + by132.w) * float(sc8[3]) +
                         (by20.x  + by20.y  + by20.z  + by20.w)  * float(sc8[6]) +
                         (by232.x + by232.y + by232.z + by232.w) * float(sc8[7]);

            float2 dm = float2(dh[0], dh[1]);
            sumf[row] = fma(dm.x, fma(sx, float(sc8[0]), fma(sy, float(sc8[1]),
                            fma(sz, float(sc8[4]), sw * float(sc8[5])))),
                        fma(-dm.y, smin, sumf[row]));

            qs_ptr += args.nb01/4;
            sc     += args.nb01/2;
            dh     += args.nb01/2;
        }
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q5_K_f32_impl(
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

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0]={0.f};

    float yl[16], yh[16];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = eff_tiisg/4;
    const short ix  = eff_tiisg%4;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }

            sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                  sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                  sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                  sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01;
            qh += args.nb01;
            dh += args.nb01/2;
            a  += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            const float tot = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = tot;
            }
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q6_K_f32_impl(
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
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = { 0.f };

    float yl[16];

    const short tid = eff_tiisg/2;
    const short ix  = eff_tiisg%2;
    const short ip  = tid/8;         // 0 or 1
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01/2;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// ======================= "True" 2-bit

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
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

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*eff_sgitg + eff_tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*eff_sgitg + eff_tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot * 0.25f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all * 0.25f;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xs_f32_impl(
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

    device const block_iq2_xs * x = (device const block_iq2_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 512);
    {
        int nval = 8;
        int pos  = (32*eff_sgitg + eff_tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xs_grid[pos + i];
        nval = 2;
        pos  = (32*eff_sgitg + eff_tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (short l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (short l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += args.nb01/2;
            q2 += args.nb01/2;
            sc += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot * 0.25f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all * 0.25f;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<N_R0_IQ2_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_xxs_f32_impl(
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

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*eff_sgitg + eff_tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos  = (32*eff_sgitg + eff_tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += args.nb01/2;
            q3  += args.nb01;
            gas += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot * 0.5f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all * 0.5f;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_s_f32_impl(
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

    device const block_iq3_s * x = (device const block_iq3_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        int nval = 8;
        int pos  = (32*eff_sgitg + eff_tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<N_R0_IQ3_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_s_f32_impl(
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

    device const block_iq2_s * x = (device const block_iq2_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    //threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
    //{
    //    int nval = 32;
    //    int pos  = (32*sgitg + tiisg)*nval;
    //    for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2s_grid[pos + i];
    //    threadgroup_barrier(mem_flags::mem_threadgroup);
    //}

    const short ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >>  4));

            float2 sum = {0};
            for (short l = 0; l < 2; ++l) {
                //const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                //const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                for (short j = 0; j < 8; ++j) {
                    sum[0] += yl[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot * 0.25f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all * 0.25f;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<N_R0_IQ2_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

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
    constexpr short NW = N_SIMDWIDTH;
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

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
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

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<N_R0_IQ1_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_m_f32_impl(
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

    device const block_iq1_m * x = (device const block_iq1_m *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    const short ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        float4 sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint8_t  * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (short row = 0; row < nr0; row++) {
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f};
            for (short j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

            sc += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int NR0, typename args_t>
void kernel_mul_mv_iq4_nl_f32_impl(
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
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * NR0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK4_NL;
    const int ns01 = args.nb01/args.nb00;

    const short ix = eff_tiisg/2;  // 0...15
    const short it = eff_tiisg%2;  // 0 or 1

    shmem_f32[eff_tiisg] = kvalues_iq4nl_f[eff_tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK4_NL + it*8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    // [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            device const block_iq4_nl & xb = x[row*ns01 + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 16 * QK4_NL;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0;
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int NR0, typename args_t>
void kernel_mul_mv_iq4_xs_f32_impl(
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
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + eff_sgitg) * NR0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK_K;
    const int ns01 = args.nb01/args.nb00;

    const short ix = eff_tiisg/16;  // 0 or 1
    const short it = eff_tiisg%16;  // 0...15
    const short ib = it/2;
    const short il = it%2;

    shmem_f32[eff_tiisg] = kvalues_iq4nl_f[eff_tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    // [TAG_MUL_MV_WEIRD]
    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += 2) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; ++row) {
            device const block_iq4_xs & xb = x[row*ns01 + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0;
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int NR0, typename args_t>
void kernel_mul_mv_mxfp4_f32_impl(
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
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * NR0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_mxfp4 * x = (device const block_mxfp4 *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    const int nb   = args.ne00/QK_MXFP4;
    const int ns01 = args.nb01/args.nb00; // this can be larger than nb for permuted src0 tensors

    const short ix = eff_tiisg/2;  // 0...15
    const short it = eff_tiisg%2;  // 0 or 1

    shmem_f32[eff_tiisg] = kvalues_mxfp4_f[eff_tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK_MXFP4 + it*8;

    // note: just the check `ib < nb` is enough, but adding the redundant `&& ib < ns01` check makes the kernel a bit faster
    //       no idea why that is - needs some deeper investigation [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *) yb;

        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_mxfp4 & xb = x[row*ns01 + ib];
            device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 8*it);

            float4 acc1 = yl[0]*float4(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
            float4 acc2 = yl[1]*float4(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
            float4 acc3 = yl[2]*float4(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
            float4 acc4 = yl[3]*float4(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            sumf[row] += e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }

        yb += 16 * QK_MXFP4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0;
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_mxfp4_f32")]]
kernel void kernel_mul_mv_mxfp4_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}
