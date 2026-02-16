#include "00-common.metal"

constant short FC_solve_tri_nsg [[function_constant(FC_SOLVE_TRI + 0)]];
constant short FC_solve_tri_n   [[function_constant(FC_SOLVE_TRI + 1)]];
constant short FC_solve_tri_k   [[function_constant(FC_SOLVE_TRI + 2)]];

kernel void kernel_solve_tri_f32(
        constant ggml_metal_kargs_solve_tri & args,
        device   const char * src0,
        device   const char * src1,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        ushort3 tgpig[[threadgroup_position_in_grid]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const short NSG = FC_solve_tri_nsg;
    const short N   = FC_solve_tri_n;
    const short K   = FC_solve_tri_k;
    const short NP  = PAD2(N, NW);

    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x*NSG + sgitg;

    threadgroup float * sh0 = (threadgroup float *) shmem;

    device const float * src0_ptr = (device const float *)(src0 + i02 * args.nb02 + i03 * args.nb03) + sgitg*N;
    device const float * src1_ptr = (device const float *)(src1 + i02 * args.nb12 + i03 * args.nb13) + i01;
    device       float * dst_ptr  = (device       float *)(dst  + i02 * args.nb2  + i03 * args.nb3)  + i01;

    for (short rr = 0; rr < N; rr += NSG) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            threadgroup float * sh0_cur = sh0 + sgitg*NP;

            for (short t = 0; t*NW < N; ++t) {
                const short idx = t*NW + tiisg;
                sh0_cur[idx] = src0_ptr[idx];
            }

            src0_ptr += NSG*N;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (i01 >= args.ne10) {
            continue;
        }

        for (short ir = 0; ir < NSG && rr + ir < N; ++ir) {
            const short r = rr + ir;

            threadgroup float * sh0_cur = sh0 + ir*NP;

            float sum = 0.0f;

            for (short t = 0; t*NW < r; ++t) {
                const short idx = t*NW + tiisg;
                sum += sh0_cur[idx] * dst_ptr[idx*K] * (idx < r);
            }

            sum = simd_sum(sum);

            if (tiisg == 0) {
                const float diag = sh0_cur[r];

                dst_ptr[r*K] = (src1_ptr[r*K] - sum) / diag;
            }
        }
    }
}

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        ushort  tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort simd_width[[threads_per_simdgroup]],
        ushort    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) (shmem + 128*sizeof(float));

    if (ntg > simd_width) {
        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            const ushort nsg = (ntg + simd_width - 1) / simd_width;

            float best = -INFINITY;
            int32_t arg = -1;

            for (ushort i = tiisg; i < nsg; i += simd_width) {
                if (shared_maxval[i] > best) {
                    best = shared_maxval[i];
                    arg = shared_argmax[i];
                }
            }

            float max_val_reduced   = simd_max(best);
            int32_t arg_val_reduced = simd_max(select(-1, arg, best == max_val_reduced));

            dst_i32[tgpig] = arg_val_reduced;
        }

        return;
    }

    dst_i32[tgpig] = arg_val;
}
