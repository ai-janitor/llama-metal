#ifndef GGML_METAL_MATMUL_MV_FORMATS_METAL
#define GGML_METAL_MATMUL_MV_FORMATS_METAL

#include "block-dot.metal"

// One struct per weight format for the mul_mv templates (mul-mv.metal): the block, the lanes that share it, a lane's
// piece, the decode and the dot. The three flags record how the format's row sums leave the simdgroup.

struct mv_acc2 { float first; float second; };

// Hooks of a format with whole blocks per lane group, no table, row slices walked by the threadgroup.
#define MV_FMT_PLAIN_HOOKS \
    enum : short { units_per_block = 1, has_table = 0, rows_by_blocks = 0, persistent = 1, bound_in_loop = 0, unroll_rows = 0 }; \
    static inline void fill_table(threadgroup char * shmem, ushort lane, ushort sg, short NW, short NSG) {} \
    static inline float scale(float tot) { return tot; }

struct mv_fmt_q2_K {
    typedef block_q2_K block_t;
    typedef block_dot_q2_k_fragment fragment_t;
    typedef block_dot_q2_k_activation activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 8, block_elems = QK_K, shmem_stored = 0, simd_sum_first = 0, has_residual = 0 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW / 8; }
    static inline fragment_t owned_fragment(short lane) { return block_dot_q2_k_legacy::owned_fragment(lane); }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        return block_dot_q2_k_legacy::prepare_activation<LANE>(f, y);
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        block_dot_q2_k_legacy::accumulate<LANE>(block, f, a, acc);
    }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
};

struct mv_fmt_q3_K {
    typedef block_q3_K block_t;
    typedef block_dot_q3_k_fragment fragment_t;
    typedef block_dot_q3_k_activation activation_t;
    typedef mv_acc2 acc_t;
    enum : short { threads_per_block = 8, block_elems = QK_K, shmem_stored = 0, simd_sum_first = 1, has_residual = 0 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW / 8; }
    static inline fragment_t owned_fragment(short lane) { return block_dot_q3_k_legacy::owned_fragment(lane); }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        return block_dot_q3_k_legacy::prepare_activation<LANE>(f, y);
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        block_dot_q3_k_legacy::accumulate<LANE>(block, f, a, acc.first, acc.second);
    }
    // Normalized once per row after every block; per-block normalization changes rounding.
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) {
        float sum = 0.f;
        sum += (acc.first + 0.25f * acc.second) / (1 << f.quant_shift);
        return sum;
    }
};

// PREP: block headers come from entries unpacked once at load (32 bytes per block, entry = row * blocks per row + block).
template<bool PREP>
struct mv_fmt_q4_K {
    typedef block_q4_K block_t;
    typedef block_dot_q4_k_fragment fragment_t;
    typedef block_dot_q4_k_activation activation_t;
    typedef float acc_t;
    typedef block_dot_q4_k_swar::weights weights_t;
    typedef block_dot_q4_k_swar::raw raw_t;
    enum : short { threads_per_block = 16, block_elems = QK_K, shmem_stored = 1, simd_sum_first = 1, has_residual = 1 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW / 16; }
    static inline fragment_t owned_fragment(short lane) { return block_dot_q4_k_swar::owned_fragment(lane); }
    static inline device const uint * header(device const char * prep, int entry) {
        return (device const uint *) (prep + (uint64_t) entry * 32);
    }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        return block_dot_q4_k_swar::prepare_activation<block_dot_lane_fp32>(f, y);
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        if (PREP) {
            block_dot_q4_k_swar::accumulate_prepared<block_dot_lane_fp32>(block, f, a, acc, header(prep, entry));
        } else {
            block_dot_q4_k_swar::accumulate<block_dot_lane_fp32>(block, f, a, acc);
        }
    }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    static inline weights_t load_weights(device const block_t * block, thread const fragment_t & f, device const char * prep, int entry) {
        if (PREP) {
            return block_dot_q4_k_swar::load_weights_prepared(block, f, header(prep, entry));
        }
        return block_dot_q4_k_swar::load_weights(block, f);
    }
    static inline void accumulate_weights(thread const weights_t & w, thread const activation_t & a, thread float & acc) {
        block_dot_q4_k_swar::accumulate_weights(w, a, acc);
    }
    static inline raw_t load_raw(device const block_t * block, thread const fragment_t & f, device const char * prep, int entry) {
        if (PREP) {
            return block_dot_q4_k_swar::load_raw_prepared(block, f, header(prep, entry));
        }
        return block_dot_q4_k_swar::load_raw(block, f);
    }
    static inline void accumulate_raw(thread const raw_t & w, thread const activation_t & a, thread float & acc) {
        block_dot_q4_k_swar::accumulate_raw(w, a, acc);
    }
};

struct mv_fmt_q5_K {
    typedef block_q5_K block_t;
    typedef block_dot_q5_k_fragment fragment_t;
    typedef block_dot_q5_k_activation activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 8, block_elems = QK_K, shmem_stored = 1, simd_sum_first = 0, has_residual = 0 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW / 8; }
    static inline fragment_t owned_fragment(short lane) { return block_dot_q5_k_scalar_swar::owned_fragment(lane); }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        return block_dot_q5_k_scalar_swar::prepare_activation<LANE>(f, y);
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        block_dot_q5_k_scalar_swar::accumulate<LANE>(block, f, a, acc);
    }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
};

struct mv_fmt_q6_K {
    typedef block_q6_K block_t;
    typedef block_dot_q6_k_fragment fragment_t;
    typedef block_dot_q6_k_activation activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 16, block_elems = QK_K, shmem_stored = 1, simd_sum_first = 1, has_residual = 0 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW / 16; }
    static inline fragment_t owned_fragment(short lane) { return block_dot_q6_k_scalar_swar::owned_fragment(lane); }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        return block_dot_q6_k_scalar_swar::prepare_activation<LANE>(f, y);
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        block_dot_q6_k_scalar_swar::accumulate<LANE>(block, f, a, acc);
    }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    typedef block_dot_q6_k_scalar_swar::weights weights_t;
    typedef block_dot_q6_k_scalar_swar::raw raw_t;
    static inline weights_t load_weights(device const block_t * block, thread const fragment_t & f, device const char * prep, int entry) {
        return block_dot_q6_k_scalar_swar::load_weights(block, f);
    }
    static inline void accumulate_weights(thread const weights_t & w, thread const activation_t & a, thread float & acc) {
        block_dot_q6_k_scalar_swar::accumulate_weights(w, a, acc);
    }
    static inline raw_t load_raw(device const block_t * block, thread const fragment_t & f, device const char * prep, int entry) {
        return block_dot_q6_k_scalar_swar::load_raw(block, f);
    }
    static inline void accumulate_raw(thread const raw_t & w, thread const activation_t & a, thread float & acc) {
        block_dot_q6_k_scalar_swar::accumulate_raw(w, a, acc);
    }
};


// Activation holders of the formats below.
struct mv_act_32 { float v[32]; };
struct mv_act_32_sum4 { float v[32]; float4 sumy; };
struct mv_act_4x4 { float4 v[4]; };
struct mv_frag_half { short it; };
struct mv_frag_iq4_xs { short ib; short il; };

// mxfp4, iq4_nl, iq4_xs: values come from a 16-entry table in threadgroup memory; a lane takes 16 elements.
struct mv_fmt_mxfp4 {
    typedef block_mxfp4 block_t;
    typedef mv_frag_half fragment_t;
    typedef mv_act_4x4 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 2, block_elems = QK_MXFP4, shmem_stored = 0, simd_sum_first = 0, has_residual = 0 };
    enum : short { units_per_block = 1, has_table = 1, rows_by_blocks = 1, persistent = 0, bound_in_loop = 0, unroll_rows = 1 };
    static inline short stride(short NW) { return NW / 2; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    static inline float scale(float tot) { return tot; }
    // One table slot per lane; 16 values, so every wave width of 16 or more fills it.
    static inline void fill_table(threadgroup char * shmem, ushort lane, ushort sg, short NW, short NSG) {
        ((threadgroup float *) shmem)[lane] = kvalues_mxfp4_f[lane%16];
    }
    static inline fragment_t owned_fragment(short lane) { fragment_t f; f.it = lane; return f; }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        device const float4 * y4 = (device const float4 *) (y + f.it*8);
        activation_t a;
        a.v[0] = y4[0];
        a.v[1] = y4[4];
        a.v[2] = y4[1];
        a.v[3] = y4[5];
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        threadgroup float * shmem_f32 = (threadgroup float *) shmem;
        device const uint8_t * q2 = (device const uint8_t *)(block->qs + 8*f.it);

        float4 acc1 = a.v[0]*float4(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
        float4 acc2 = a.v[1]*float4(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
        float4 acc3 = a.v[2]*float4(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
        float4 acc4 = a.v[3]*float4(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);

        acc1 = (acc1 + acc3) + (acc2 + acc4);

        acc += e8m0_to_fp32(block->e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
    }
};

struct mv_fmt_iq4_nl {
    typedef block_iq4_nl block_t;
    typedef mv_frag_half fragment_t;
    typedef mv_act_4x4 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 2, block_elems = QK4_NL, shmem_stored = 0, simd_sum_first = 0, has_residual = 0 };
    enum : short { units_per_block = 1, has_table = 1, rows_by_blocks = 1, persistent = 0, bound_in_loop = 1, unroll_rows = 0 };
    static inline short stride(short NW) { return NW / 2; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    static inline float scale(float tot) { return tot; }
    // One table slot per lane; 16 values, so every wave width of 16 or more fills it.
    static inline void fill_table(threadgroup char * shmem, ushort lane, ushort sg, short NW, short NSG) {
        ((threadgroup float *) shmem)[lane] = kvalues_iq4nl_f[lane%16];
    }
    static inline fragment_t owned_fragment(short lane) { fragment_t f; f.it = lane; return f; }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        device const float4 * y4 = (device const float4 *) (y + f.it*8);
        activation_t a;
        a.v[0] = y4[0];
        a.v[1] = y4[4];
        a.v[2] = y4[1];
        a.v[3] = y4[5];
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        threadgroup float * shmem_f32 = (threadgroup float *) shmem;
        device const uint16_t * q4 = (device const uint16_t *)(block->qs + 8*f.it);
        uint32_t aux32[2];
        thread const uint8_t * q8 = (thread const uint8_t *)aux32;
        float4 qf1, qf2;

        float4 acc1 = {0.f}, acc2 = {0.f};

        aux32[0] = q4[0] | (q4[1] << 16);
        aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
        aux32[0] &= 0x0f0f0f0f;
        qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
        qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
        acc1 += a.v[0] * qf1;
        acc2 += a.v[1] * qf2;

        aux32[0] = q4[2] | (q4[3] << 16);
        aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
        aux32[0] &= 0x0f0f0f0f;
        qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
        qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
        acc1 += a.v[2] * qf1;
        acc2 += a.v[3] * qf2;

        acc1 += acc2;

        acc += (float)block->d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
    }
};

struct mv_fmt_iq4_xs {
    typedef block_iq4_xs block_t;
    typedef mv_frag_iq4_xs fragment_t;
    typedef mv_act_4x4 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 16, block_elems = QK_K, shmem_stored = 0, simd_sum_first = 0, has_residual = 0 };
    enum : short { units_per_block = 1, has_table = 1, rows_by_blocks = 1, persistent = 0, bound_in_loop = 1, unroll_rows = 0 };
    static inline short stride(short NW) { return NW / 16; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    static inline float scale(float tot) { return tot; }
    // One table slot per lane; 16 values, so every wave width of 16 or more fills it.
    static inline void fill_table(threadgroup char * shmem, ushort lane, ushort sg, short NW, short NSG) {
        ((threadgroup float *) shmem)[lane] = kvalues_iq4nl_f[lane%16];
    }
    static inline fragment_t owned_fragment(short lane) { fragment_t f; f.ib = lane/2; f.il = lane%2; return f; }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        device const float4 * y4 = (device const float4 *) (y + f.ib * 32 + f.il * 8);
        activation_t a;
        a.v[0] = y4[0];
        a.v[1] = y4[4];
        a.v[2] = y4[1];
        a.v[3] = y4[5];
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        threadgroup float * shmem_f32 = (threadgroup float *) shmem;
        const short ib = f.ib;
        device const uint32_t * q4 = (device const uint32_t *)(block->qs + 16*ib + 8*f.il);
        uint32_t aux32[2];
        thread const uint8_t * q8 = (thread const uint8_t *)aux32;
        float4 qf1, qf2;

        float4 acc1 = {0.f}, acc2 = {0.f};

        aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
        aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
        qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
        qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
        acc1 += a.v[0] * qf1;
        acc2 += a.v[1] * qf2;

        aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
        aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
        qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
        qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
        acc1 += a.v[2] * qf1;
        acc2 += a.v[3] * qf2;

        acc1 += acc2;

        const int ls = (((block->scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((block->scales_h >> 2*ib) & 3) << 4)) - 32;
        acc += (float)block->d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
    }
};

// The formats below work in units of 32 elements, one lane per unit; unit is the unit's index inside its block.
struct mv_fmt_iq1_m {
    typedef block_iq1_m block_t;
    typedef short fragment_t;
    typedef mv_act_32_sum4 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 1, block_elems = 32, shmem_stored = 0, simd_sum_first = 0, has_residual = 0 };
    enum : short { units_per_block = QK_K / 32, has_table = 0, rows_by_blocks = 0, persistent = 0, bound_in_loop = 0, unroll_rows = 0 };
    static inline short stride(short NW) { return NW; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    static inline float scale(float tot) { return tot; }
    static inline void fill_table(threadgroup char * shmem, ushort lane, ushort sg, short NW, short NSG) {}
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        a.sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            a.v[i+ 0] = y[i+ 0]; a.sumy[0] += a.v[i+ 0];
            a.v[i+ 8] = y[i+ 8]; a.sumy[1] += a.v[i+ 8];
            a.v[i+16] = y[i+16]; a.sumy[2] += a.v[i+16];
            a.v[i+24] = y[i+24]; a.sumy[3] += a.v[i+24];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        const short ib = unit;
        device const uint8_t  * qs = block->qs + 4 * ib;
        device const uint8_t  * qh = block->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)block->scales;

        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
        constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
        constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
        constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

        float2 sum = {0.f};
        for (short j = 0; j < 4; ++j) {
            sum[0] += a.v[j+ 0] * (grid1[j] & 0xf) + a.v[j+ 4] * (grid1[j] >> 4)
                    + a.v[j+ 8] * (grid2[j] & 0xf) + a.v[j+12] * (grid2[j] >> 4);
            sum[1] += a.v[j+16] * (grid3[j] & 0xf) + a.v[j+20] * (grid3[j] >> 4)
                    + a.v[j+24] * (grid4[j] & 0xf) + a.v[j+28] * (grid4[j] >> 4);
        }
        const float delta1 = a.sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + a.sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
        const float delta2 = a.sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + a.sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

        acc += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                   (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));
    }
};

struct mv_fmt_iq2_s {
    typedef block_iq2_s block_t;
    typedef short fragment_t;
    typedef mv_act_32 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 1, block_elems = 32, shmem_stored = 0, simd_sum_first = 0, has_residual = 0 };
    enum : short { units_per_block = QK_K / 32, has_table = 0, rows_by_blocks = 0, persistent = 0, bound_in_loop = 0, unroll_rows = 0 };
    static inline short stride(short NW) { return NW; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    static inline float scale(float tot) { return tot * 0.25f; }
    static inline void fill_table(threadgroup char * shmem, ushort lane, ushort sg, short NW, short NSG) {}
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        for (short i = 0; i < 32; ++i) {
            a.v[i] = y[i];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        const short ib = unit;
        device const uint8_t * qs = block->qs + 4 * ib;
        device const uint8_t * qh = block->qh + ib;
        device const uint8_t * sc = block->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;

        const float db = block->d;
        const float d1 = db * (0.5f + (sc[0] & 0xf));
        const float d2 = db * (0.5f + (sc[0] >>  4));

        float2 sum = {0};
        for (short l = 0; l < 2; ++l) {
            constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
            for (short j = 0; j < 8; ++j) {
                sum[0] += a.v[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                sum[1] += a.v[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
            }
        }
        acc += d1 * sum[0] + d2 * sum[1];
    }
};

struct mv_fmt_iq2_xxs {
    typedef block_iq2_xxs block_t;
    typedef short fragment_t;
    typedef mv_act_32 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 1, block_elems = 32, shmem_stored = 0, simd_sum_first = 0, has_residual = 0 };
    enum : short { units_per_block = QK_K / 32, has_table = 1, rows_by_blocks = 0, persistent = 0, bound_in_loop = 0, unroll_rows = 0 };
    static inline short stride(short NW) { return NW; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    static inline float scale(float tot) { return tot * 0.25f; }
    // Every thread of the group copies its share of the grid and the sign table.
    static inline void fill_table(threadgroup char * shmem, ushort lane, ushort sg, short NW, short NSG) {
        threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
        const int total_threads = NW * NSG;
        const int nval = 4;
        for (int pos = (NW*sg + lane)*nval; pos < 256; pos += total_threads*nval) {
            for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        }
        threadgroup uint8_t * ssigns = (threadgroup uint8_t *) (svalues + 256);
        const int nval2 = 2;
        for (int pos = (NW*sg + lane)*nval2; pos < 128; pos += total_threads*nval2) {
            for (int i = 0; i < nval2; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        }
    }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        for (short i = 0; i < 32; ++i) {
            a.v[i] = y[i];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
        threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *) (svalues + 256);
        device const uint16_t * q2 = block->qs + 4 * unit;

        const float db = block->d;
        device const uint8_t * aux8 = (device const uint8_t *)q2;
        const uint32_t aux32 = q2[2] | (q2[3] << 16);
        const float d = db * (0.5f + (aux32 >> 28));

        float sum = 0;
        for (short l = 0; l < 4; ++l) {
            const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
            const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
            for (short j = 0; j < 8; ++j) {
                sum += a.v[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            }
        }
        acc += d * sum;
    }
};

struct mv_fmt_iq2_xs {
    typedef block_iq2_xs block_t;
    typedef short fragment_t;
    typedef mv_act_32 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 1, block_elems = 32, shmem_stored = 0, simd_sum_first = 0, has_residual = 0 };
    enum : short { units_per_block = QK_K / 32, has_table = 1, rows_by_blocks = 0, persistent = 0, bound_in_loop = 0, unroll_rows = 0 };
    static inline short stride(short NW) { return NW; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    static inline float scale(float tot) { return tot * 0.25f; }
    // Every thread of the group copies its share of the grid and the sign table.
    static inline void fill_table(threadgroup char * shmem, ushort lane, ushort sg, short NW, short NSG) {
        threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
        const int total_threads = NW * NSG;
        const int nval = 8;
        for (int pos = (NW*sg + lane)*nval; pos < 512; pos += total_threads*nval) {
            for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xs_grid[pos + i];
        }
        threadgroup uint8_t * ssigns = (threadgroup uint8_t *) (svalues + 512);
        const int nval2 = 2;
        for (int pos = (NW*sg + lane)*nval2; pos < 128; pos += total_threads*nval2) {
            for (int i = 0; i < nval2; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        }
    }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        for (short i = 0; i < 32; ++i) {
            a.v[i] = y[i];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
        threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *) (svalues + 512);
        device const uint16_t * q2 = block->qs + 4 * unit;
        device const uint8_t  * sc = block->scales + unit;

        const float db = block->d;
        const uint8_t ls1 = sc[0] & 0xf;
        const uint8_t ls2 = sc[0] >>  4;
        const float d1 = db * (0.5f + ls1);
        const float d2 = db * (0.5f + ls2);

        float sum1 = 0, sum2 = 0;
        for (short l = 0; l < 2; ++l) {
            const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
            const uint8_t signs = ssigns[(q2[l] >> 9)];
            for (short j = 0; j < 8; ++j) {
                sum1 += a.v[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            }
        }
        for (short l = 2; l < 4; ++l) {
            const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
            const uint8_t signs = ssigns[(q2[l] >> 9)];
            for (short j = 0; j < 8; ++j) {
                sum2 += a.v[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            }
        }
        acc += d1 * sum1 + d2 * sum2;
    }
};

struct mv_fmt_iq3_xxs {
    typedef block_iq3_xxs block_t;
    typedef short fragment_t;
    typedef mv_act_32 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 1, block_elems = 32, shmem_stored = 0, simd_sum_first = 0, has_residual = 0 };
    enum : short { units_per_block = QK_K / 32, has_table = 1, rows_by_blocks = 0, persistent = 0, bound_in_loop = 0, unroll_rows = 0 };
    static inline short stride(short NW) { return NW; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    static inline float scale(float tot) { return tot * 0.5f; }
    // Every thread of the group copies its share of the grid and the sign table.
    static inline void fill_table(threadgroup char * shmem, ushort lane, ushort sg, short NW, short NSG) {
        threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
        const int total_threads = NW * NSG;
        const int nval = 4;
        for (int pos = (NW*sg + lane)*nval; pos < 256; pos += total_threads*nval) {
            for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        }
        threadgroup uint8_t * ssigns = (threadgroup uint8_t *) (svalues + 256);
        const int nval2 = 2;
        for (int pos = (NW*sg + lane)*nval2; pos < 128; pos += total_threads*nval2) {
            for (int i = 0; i < nval2; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        }
    }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        for (short i = 0; i < 32; ++i) {
            a.v[i] = y[i];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
        threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *) (svalues + 256);
        device const uint8_t  * q3 = block->qs + 8 * unit;
        device const uint16_t * gas = (device const uint16_t *)(block->qs + QK_K/4) + 2 * unit;

        const float db = block->d;
        const uint32_t aux32 = gas[0] | (gas[1] << 16);
        const float d = db * (0.5f + (aux32 >> 28));

        float2 sum = {0};
        for (short l = 0; l < 4; ++l) {
            const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
            const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
            const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
            for (short j = 0; j < 4; ++j) {
                sum[0] += a.v[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                sum[1] += a.v[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
            }
        }
        acc += d * (sum[0] + sum[1]);
    }
};

struct mv_fmt_iq3_s {
    typedef block_iq3_s block_t;
    typedef short fragment_t;
    typedef mv_act_32 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 1, block_elems = 32, shmem_stored = 0, simd_sum_first = 0, has_residual = 0 };
    enum : short { units_per_block = QK_K / 32, has_table = 1, rows_by_blocks = 0, persistent = 0, bound_in_loop = 0, unroll_rows = 0 };
    static inline short stride(short NW) { return NW; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    static inline float scale(float tot) { return tot; }
    // Every thread of the group copies its share of the grid.
    static inline void fill_table(threadgroup char * shmem, ushort lane, ushort sg, short NW, short NSG) {
        threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
        const int total_threads = NW * NSG;
        const int nval = 8;
        for (int pos = (NW*sg + lane)*nval; pos < 512; pos += total_threads*nval) {
            for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        }
    }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        for (short i = 0; i < 32; ++i) {
            a.v[i] = y[i];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
        const short ib = unit;
        device const uint8_t * qs = block->qs + 8 * ib;
        device const uint8_t * qh = block->qh + ib;
        device const uint8_t * sc = block->scales + (ib/2);
        device const uint8_t * signs = block->signs + 4 * ib;

        const float db = block->d;
        const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

        float2 sum = {0};
        for (short l = 0; l < 4; ++l) {
            const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
            const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
            const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
            const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
            for (short j = 0; j < 4; ++j) {
                sum[0] += a.v[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                sum[1] += a.v[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
            }
        }
        acc += d * (sum[0] + sum[1]);
    }
};


struct mv_act_16 { float v[16]; };

// 1-bit, 2-bit, ternary and nvfp4 weights: a lane takes 32 elements of its block in order (16 for nvfp4).
struct mv_fmt_q1_0 {
    typedef block_q1_0 block_t;
    typedef short fragment_t;
    typedef mv_act_32 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 4, block_elems = QK1_0, shmem_stored = 1, simd_sum_first = 1, has_residual = 0 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW / 4; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        for (short i = 0; i < 32; ++i) {
            a.v[i] = y[32*f + i];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        float sum = 0;
        for (short b = 0; b < 4; ++b) {
            const uint8_t bits = block->qs[4*f + b];
            for (short j = 0; j < 8; ++j) {
                sum += ((bits >> j) & 1) ? a.v[8*b + j] : -a.v[8*b + j];
            }
        }
        acc += (float) block->d * sum;
    }
};

struct mv_fmt_q2_0 {
    typedef block_q2_0 block_t;
    typedef short fragment_t;
    typedef mv_act_32 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 2, block_elems = QK2_0, shmem_stored = 1, simd_sum_first = 1, has_residual = 0 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW / 2; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        for (short i = 0; i < 32; ++i) {
            a.v[i] = y[32*f + i];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        float sum = 0;
        for (short b = 0; b < 8; ++b) {
            const uint8_t bits = block->qs[8*f + b];
            for (short j = 0; j < 4; ++j) {
                sum += (float) ((int) ((bits >> (2*j)) & 3) - 1) * a.v[4*b + j];
            }
        }
        acc += (float) block->d * sum;
    }
};

struct mv_fmt_tq1_0 {
    typedef block_tq1_0 block_t;
    typedef short fragment_t;
    typedef mv_act_32 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 8, block_elems = QK_K, shmem_stored = 1, simd_sum_first = 1, has_residual = 0 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW / 8; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        for (short i = 0; i < 32; ++i) {
            a.v[i] = y[32*f + i];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        float sum = 0;
        for (short i = 0; i < 32; ++i) {
            sum += (float) tq1_0_trit(block, 32*f + i) * a.v[i];
        }
        acc += (float) block->d * sum;
    }
};

struct mv_fmt_tq2_0 {
    typedef block_tq2_0 block_t;
    typedef short fragment_t;
    typedef mv_act_32 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 8, block_elems = QK_K, shmem_stored = 1, simd_sum_first = 1, has_residual = 0 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW / 8; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        for (short i = 0; i < 32; ++i) {
            a.v[i] = y[32*f + i];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        // Lane f reads bit pair f % 4 of 32 bytes in half f / 4.
        device const uint8_t * qs = block->qs + (f / 4) * 32;
        const short shift = (f % 4) * 2;
        float sum = 0;
        for (short m = 0; m < 32; ++m) {
            sum += (float) ((int) ((qs[m] >> shift) & 3) - 1) * a.v[m];
        }
        acc += (float) block->d * sum;
    }
};

// fp8 weights: a block is one element, a lane takes every NW-th element of the row (any row width).
struct mv_fmt_f8_e4m3 {
    typedef block_f8_e4m3 block_t;
    typedef short fragment_t;
    typedef float activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 1, block_elems = 1, shmem_stored = 1, simd_sum_first = 1, has_residual = 0 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        return y[0];
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        acc += f8_e4m3_to_fp32(block->bits) * a;
    }
};

struct mv_fmt_nvfp4 {
    typedef block_nvfp4 block_t;
    typedef short fragment_t;
    typedef mv_act_16 activation_t;
    typedef float acc_t;
    enum : short { threads_per_block = 4, block_elems = QK_NVFP4, shmem_stored = 1, simd_sum_first = 1, has_residual = 0 };
    MV_FMT_PLAIN_HOOKS
    static inline short stride(short NW) { return NW / 4; }
    static inline fragment_t owned_fragment(short lane) { return lane; }
    static inline float finish(thread const fragment_t & f, thread const acc_t & acc) { return acc; }
    template<short LANE>
    static inline activation_t prepare_activation(thread const fragment_t & f, device const float * y) {
        activation_t a;
        for (short i = 0; i < 16; ++i) {
            a.v[i] = y[16*f + i];
        }
        return a;
    }
    template<short LANE>
    static inline void accumulate(device const block_t * block, thread const fragment_t & f, thread const activation_t & a, thread acc_t & acc, device const char * prep, int entry, threadgroup char * shmem, short unit) {
        device const uint8_t * qs = block->qs + 8*f;
        float sum = 0;
        for (short j = 0; j < 8; ++j) {
            sum += kvalues_mxfp4_f[qs[j] & 0x0F] * a.v[j] + kvalues_mxfp4_f[qs[j] >> 4] * a.v[8 + j];
        }
        acc += ue4m3_to_fp32(block->d[f]) * sum;
    }
};

#endif
