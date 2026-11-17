// Q8_0 x f32 matvec for batch 1: activation to int8 per 32-block once per simdgroup, int8 dot, row sum by lane exchange.
// A lane owns a pair of 34-byte blocks (68 bytes, 4-byte aligned when ne00 % 64 == 0); the host checks that before routing here.

#define Q8LAB_QK 32

// INTENT: one row's Q8_0 dot over blocks [begin, end) for this lane; shared by the plain and the fused (NORM) matvec.
// NORM: the activation is formed as (x*scale)*nw per value (rms_norm then its weight), never written to memory.
template<short NW, short NL, bool NORM>
inline float q8lab_row_acc(
        int begin,
        int end,
        device const char  * wrow,
        device const float * y,
        device const float * nw,
        float scale,
        ushort tiisg) {
    constexpr short RB  = 2*NL;                    // blocks per round of one row
    constexpr short AV  = RB*Q8LAB_QK/NW;          // activation values per lane per round
    constexpr short AW  = AV/4;                    // packed int8 words per lane
    constexpr short LB  = Q8LAB_QK/AV;             // lanes holding one block's activation
    static_assert(NW % NL == 0 && AV % 4 == 0 && Q8LAB_QK % AV == 0 && AV <= 32, "unsupported lane geometry");

    const short lane = tiisg % NL;
    float acc = 0.0f;

    for (int base = begin; base < end; base += RB) {
        const int count = min((int)RB, end - base);

        // reads first: this lane's activation values (and norm weight), then its pair of weight blocks
        const bool has_act = tiisg*AV < count*Q8LAB_QK;
        const int  aoff    = base*Q8LAB_QK + (has_act ? tiisg*AV : 0);
        float4 xv[AW], nv[AW];
        FOR_UNROLL (short i = 0; i < AW; ++i) {
            xv[i] = has_act ? float4(((device const packed_float4 *)(y + aoff))[i]) : float4(0.0f);
            if (NORM) {
                nv[i] = has_act ? float4(((device const packed_float4 *)(nw + aoff))[i]) : float4(0.0f);
            }
        }
        const bool owner = 2*lane < count;
        const int  b0    = base + (owner ? 2*lane : 0);
        device const char * wp = wrow + (uint64_t) b0*sizeof(block_q8_0);
        uint w[17];
        FOR_UNROLL (short i = 0; i < 4; ++i) {
            const uint4 v = uint4(((device const packed_uint4 *) wp)[i]);
            w[4*i + 0] = v.x; w[4*i + 1] = v.y; w[4*i + 2] = v.z; w[4*i + 3] = v.w;
        }
        w[16] = ((device const uint *) wp)[16];

        float act[AV];
        FOR_UNROLL (short i = 0; i < AW; ++i) {
            const float4 v = NORM ? (xv[i]*scale)*nv[i] : xv[i];
            act[4*i + 0] = v.x; act[4*i + 1] = v.y; act[4*i + 2] = v.z; act[4*i + 3] = v.w;
        }

        // activation to int8 per block: d = amax/127 stored as half, q = round(x/d)
        float amax = 0.0f;
        FOR_UNROLL (short i = 0; i < AV; ++i) {
            amax = max(amax, fabs(act[i]));
        }
        FOR_UNROLL (short off = LB/2; off > 0; off /= 2) {
            amax = max(amax, simd_shuffle_xor(amax, (ushort) off));
        }
        const float d  = amax/127.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;
        const float dh = (float)(half) d;
        int packed[AW];
        FOR_UNROLL (short k = 0; k < AW; ++k) {
            const int4 q = int4(round(float4(act[4*k + 0], act[4*k + 1], act[4*k + 2], act[4*k + 3])*id));
            packed[k] = (q.x & 0xff) | ((q.y & 0xff) << 8) | ((q.z & 0xff) << 16) | ((q.w & 0xff) << 24);
        }

        // weight data is used last: block 0 starts 2 bytes into w[0], block 1 is word-aligned at w[9]
        FOR_UNROLL (short b = 0; b < 2; ++b) {
            const short j  = 2*lane + b;
            const float da = simd_shuffle(dh, (ushort)(j*LB));
            int sum = 0;
            FOR_UNROLL (short p = 0; p < LB; ++p) {
                FOR_UNROLL (short k = 0; k < AW; ++k) {
                    const int  a  = simd_shuffle(packed[k], (ushort)(j*LB + p));
                    const short wi = p*AW + k;
                    const uint qw = b == 0 ? ((w[wi] >> 16) | (w[wi + 1] << 16)) : w[9 + wi];
                    // bytes sign-extended by shifts; reading the words as char4 gave wrong products on the 5500M
                    sum += ((a << 24) >> 24)*((int)(qw << 24) >> 24) + ((a << 16) >> 24)*((int)(qw << 16) >> 24)
                         + ((a <<  8) >> 24)*((int)(qw <<  8) >> 24) + (a >> 24)*((int) qw >> 24);
                }
            }
            const ushort dbits = b == 0 ? (ushort)(w[0] & 0xffffu) : (ushort)(w[8] >> 16);
            if (j < count) {
                acc = fma((float) sum, (float) as_type<half>(dbits)*da, acc);
            }
        }
    }
    return acc;
}

template<short NW, short NL>
void kernel_mul_mv_q8_0_f32_lab_impl(
        int ne00,
        int ne01,
        uint64_t nb01,
        short NS,
        device const char  * src0,
        device const float * y,
        device       float * y_out,
        threadgroup  float * partial,
        int group,
        ushort tiisg,
        ushort sgitg) {
    constexpr short RPW = NW/NL;                   // rows per simdgroup

    const short lane = tiisg % NL;
    const int   row  = group*RPW + tiisg/NL;
    const bool  valid = row < ne01;

    const int nb    = ne00/Q8LAB_QK;
    const int slice = ((nb + NS - 1)/NS + 1) & ~1;
    const int begin = min((int)sgitg*slice, nb);
    const int end   = min(begin + slice, nb);

    device const char  * wrow = src0 + (uint64_t)(valid ? row : 0)*nb01;

    float acc = q8lab_row_acc<NW, NL, false>(begin, end, wrow, y, nullptr, 1.0f, tiisg);

    FOR_UNROLL (short off = NL/2; off > 0; off /= 2) {
        acc += simd_shuffle_xor(acc, (ushort) off);
    }

    if (NS == 1) {
        if (lane == 0 && valid) {
            y_out[row] = acc;
        }
        return;
    }
    if (lane == 0) {
        partial[sgitg*RPW + tiisg/NL] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && lane == 0 && valid) {
        float tot = 0.0f;
        for (short s = 0; s < NS; ++s) {
            tot += partial[s*RPW + tiisg/NL];
        }
        y_out[row] = tot;
    }
}

[[host_name("kernel_mul_mv_q8_0_f32_lab")]]
kernel void kernel_mul_mv_q8_0_f32_lab(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup float * partial = (threadgroup float *) shmem;
    if (N_SIMDWIDTH == 64 && FC_mul_mv_nxpsg == 8) {
        kernel_mul_mv_q8_0_f32_lab_impl<64, 8>(args.ne00, args.ne01, args.nb01, FC_mul_mv_nsg, src0, (device const float *) src1, (device float *) dst, partial, (int) tgpig.x, tiisg, sgitg);
    } else if (N_SIMDWIDTH == 64 && FC_mul_mv_nxpsg == 4) {
        kernel_mul_mv_q8_0_f32_lab_impl<64, 4>(args.ne00, args.ne01, args.nb01, FC_mul_mv_nsg, src0, (device const float *) src1, (device float *) dst, partial, (int) tgpig.x, tiisg, sgitg);
    } else if (N_SIMDWIDTH == 32 && FC_mul_mv_nxpsg == 8) {
        kernel_mul_mv_q8_0_f32_lab_impl<32, 8>(args.ne00, args.ne01, args.nb01, FC_mul_mv_nsg, src0, (device const float *) src1, (device float *) dst, partial, (int) tgpig.x, tiisg, sgitg);
    } else if (N_SIMDWIDTH == 32 && FC_mul_mv_nxpsg == 4) {
        kernel_mul_mv_q8_0_f32_lab_impl<32, 4>(args.ne00, args.ne01, args.nb01, FC_mul_mv_nsg, src0, (device const float *) src1, (device float *) dst, partial, (int) tgpig.x, tiisg, sgitg);
    }
}

constant short FC_norm_mv_nmat  [[function_constant(FC_NORM_MUL_MV + 0)]];
constant bool  FC_norm_mv_mean  [[function_constant(FC_NORM_MUL_MV + 1)]];
constant short FC_norm_mv_lanes [[function_constant(FC_NORM_MUL_MV + 2)]];

// INTENT: rms_norm (scale from the mean of squares the residual add stored, or computed here) times its weight,
// then K Q8_0 matvecs that read it; groups walk matrix 0's rows, then matrix 1's, and so on.
// K and the mean source are compile-time: each pipeline holds one K and one mean path. The matrix of a group is
// picked by comparing with the first group of each later matrix, each matrix's pointer, rows and stride a separate
// argument (2026-11-16: the earlier form indexed args.rows[k], groups[k], nb01[k] and the buffers by a run-time k
// and passed the plain matvec's split parameters; it ran 14-32 % slower than the same step in its own entry and
// lowered the 5500M thread limit to 896).
template<short NW, short NL, short K, bool MEAN>
void kernel_norm_mul_mv_q8_0_f32_lab_impl(
        constant ggml_metal_kargs_norm_mul_mv & args,
        device const float * x,
        device const float * nw,
        device const float * mean,
        device const char * w0, device const char * w1, device const char * w2, device const char * w3,
        device       char * y0, device       char * y1, device       char * y2, device       char * y3,
        uint3  tgpig,
        ushort tiisg) {
    constexpr short RPW = NW/NL;
    float ms;
    if (MEAN) {
        ms = mean[0];
    } else {
        float s = 0.0f;
        for (int i = tiisg; i < args.ne00/4; i += NW) {
            const float4 v = float4(((device const packed_float4 *) x)[i]);
            s += dot(v, v);
        }
        ms = simd_sum(s)/args.ne00;
    }
    const float scale = 1.0f/sqrt(ms + args.eps);

    const int gid = (int) tgpig.x;
    device const char * w = w0;
    device       char * y = y0;
    int      rows  = args.rows0;
    uint64_t nb01  = args.nb01_0;
    int      first = 0;
    if (K >= 2 && gid >= args.start1) { w = w1; y = y1; rows = args.rows1; nb01 = args.nb01_1; first = args.start1; }
    if (K >= 3 && gid >= args.start2) { w = w2; y = y2; rows = args.rows2; nb01 = args.nb01_2; first = args.start2; }
    if (K >= 4 && gid >= args.start3) { w = w3; y = y3; rows = args.rows3; nb01 = args.nb01_3; first = args.start3; }

    const short lane  = tiisg % NL;
    const int   row   = (gid - first)*RPW + tiisg/NL;
    const bool  valid = row < rows;
    device const char * wrow = w + (uint64_t)(valid ? row : 0)*nb01;

    float acc = q8lab_row_acc<NW, NL, true>(0, args.ne00/Q8LAB_QK, wrow, x, nw, scale, tiisg);
    FOR_UNROLL (short off = NL/2; off > 0; off /= 2) {
        acc += simd_shuffle_xor(acc, (ushort) off);
    }
    if (lane == 0 && valid) {
        ((device float *) y)[row] = acc;
    }
}

#define NORM_MV_ARGS args, (device const float *) x, (device const float *) nw, (device const float *) mean, w0, w1, w2, w3, y0, y1, y2, y3, tgpig, tiisg
#define NORM_MV_K(NW, NL, K) \
    if (FC_norm_mv_mean) { kernel_norm_mul_mv_q8_0_f32_lab_impl<NW, NL, K, true >(NORM_MV_ARGS); } \
    else                 { kernel_norm_mul_mv_q8_0_f32_lab_impl<NW, NL, K, false>(NORM_MV_ARGS); }
#define NORM_MV_W(NW, NL) \
    if      (FC_norm_mv_nmat == 1) { NORM_MV_K(NW, NL, 1) } \
    else if (FC_norm_mv_nmat == 2) { NORM_MV_K(NW, NL, 2) } \
    else if (FC_norm_mv_nmat == 3) { NORM_MV_K(NW, NL, 3) } \
    else if (FC_norm_mv_nmat == 4) { NORM_MV_K(NW, NL, 4) }

[[host_name("kernel_norm_mul_mv_q8_0_f32_lab")]]
kernel void kernel_norm_mul_mv_q8_0_f32_lab(
        constant ggml_metal_kargs_norm_mul_mv & args,
        device const char * x,
        device const char * nw,
        device const char * mean,
        device const char * w0, device const char * w1, device const char * w2, device const char * w3,
        device       char * y0, device       char * y1, device       char * y2, device       char * y3,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    if (N_SIMDWIDTH == 64 && FC_norm_mv_lanes == 8) {
        NORM_MV_W(64, 8)
    } else if (N_SIMDWIDTH == 32 && FC_norm_mv_lanes == 8) {
        NORM_MV_W(32, 8)
    }
}

#undef NORM_MV_W
#undef NORM_MV_K
#undef NORM_MV_ARGS

// Residual add that also stores the mean of squares of its sum in a slot the fused norm reads.
[[host_name("kernel_add_scale_f32")]]
kernel void kernel_add_scale_f32(
        constant ggml_metal_kargs_add_scale & args,
        device const char * a,
        device const char * b,
        device       char * dst,
        device       float * mean,
        threadgroup  float * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort nsg  [[simdgroups_per_threadgroup]]) {
    device const packed_float4 * a4 = (device const packed_float4 *) a;
    device const packed_float4 * b4 = (device const packed_float4 *) b;
    device       packed_float4 * d4 = (device       packed_float4 *) dst;
    float s = 0.0f;
    for (int i = tpitg; i < args.ne0/4; i += ntg) {
        const float4 v = float4(a4[i]) + float4(b4[i]);
        d4[i] = packed_float4(v);
        s += dot(v, v);
    }
    s = simd_sum(s);
    if (tiisg == 0) {
        shmem[sgitg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg == 0) {
        float t = 0.0f;
        for (short i = 0; i < nsg; ++i) {
            t += shmem[i];
        }
        mean[0] = t/args.ne0;
    }
}
