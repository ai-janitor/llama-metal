#ifndef GGML_METAL_MATMUL_BLOCK_DOT_Q6_GROUPED_METAL
#define GGML_METAL_MATMUL_BLOCK_DOT_Q6_GROUPED_METAL

#include "block-dot-lane.metal"

// Grouped Q6_K preserves the legacy per-element arithmetic. It shares only
// the facade's ownership map; the scalar SWAR unpack and bias fold stay separate.
struct block_dot_q6_k_grouped_activation {
    // Interleaved by element position: [l+0, l+32, l+64, l+96] for each l in 0..3.
    float values[16];
};

struct block_dot_q6_k_grouped_legacy {
    enum : short { threads_per_block = 16 };

    // Thread-to-data mapping for q6_K (16 threads per superblock).
    //   ip (0..1): which 128-element half of the superblock
    //   il (0..7): position within half
    //   l0:        starting element offset (0,4,8,...,28)
    //   is:        scale index for this thread's data
    // This delegates to the facade so grouped and scalar callers share the map only.
    static inline block_dot_q6_k_fragment owned_fragment(short lane_in_block) {
        return block_dot_q6_k_scalar_swar::owned_fragment(lane_in_block);
    }

    template <short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline block_dot_q6_k_grouped_activation prepare_activation(
            thread const block_dot_q6_k_fragment & owned,
            device const float * block_y) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "grouped Q6_K only preserves the fp32 lane");

        // Local activation cache — loaded once per superblock, reused across NR0 rows.
        block_dot_q6_k_grouped_activation cache;
        device const float * yy = block_y + owned.y_offset;

        // Load 16 activation elements in an interleaved pattern.
        // Elements are gathered from 4 groups of 4, spaced 32 apart.
        for (short l = 0; l < 4; ++l) {
            cache.values[4*l + 0] = yy[l +  0];
            cache.values[4*l + 1] = yy[l + 32];
            cache.values[4*l + 2] = yy[l + 64];
            cache.values[4*l + 3] = yy[l + 96];
        }

        return cache;
    }

    template <short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline void accumulate(
            device const block_q6_K * block,
            thread const block_dot_q6_k_fragment & owned,
            thread const block_dot_q6_k_grouped_activation & prepared,
            thread float & accumulator) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "grouped Q6_K only preserves the fp32 lane");

        device const uint8_t * lower_ptr = block->ql + owned.q_offset_l;
        device const uint8_t * second_ptr = lower_ptr + 32;
        device const uint8_t * high_ptr = block->qh + owned.q_offset_h;
        device const int8_t  * signed_scale_ptr = block->scales + owned.scale_index;
        device const half    * half_d_ptr = &block->d;

        // Bitmasks for extracting the 2 high bits from qh.
        // q6_K stores 6-bit weights split as: ql (lower 4 bits) + qh (upper 2 bits).
        // The 4 masks select different 2-bit fields within each qh byte.
        constexpr uint8_t kmask1 = 0x03;   // bits 0-1
        constexpr uint8_t kmask2 = 0x0C;   // bits 2-3
        constexpr uint8_t kmask3 = 0x30;   // bits 4-5
        constexpr uint8_t kmask4 = 0xC0;   // bits 6-7

        float4 sums = {0.f, 0.f, 0.f, 0.f};

        // Reconstruct 6-bit weights and compute dot products.
        // Each weight = (ql_nibble | (qh_2bits << 4)) - 32
        // The -32 centers the 6-bit range [0,63] to [-32,31].
        FOR_UNROLL (short l = 0; l < 4; ++l) {
            sums[0] += prepared.values[4*l + 0] * ((int8_t)((lower_ptr[l] & 0xF) | ((high_ptr[l] & kmask1) << 4)) - 32);
            sums[1] += prepared.values[4*l + 1] * ((int8_t)((second_ptr[l] & 0xF) | ((high_ptr[l] & kmask2) << 2)) - 32);
            sums[2] += prepared.values[4*l + 2] * ((int8_t)((lower_ptr[l]  >> 4) | ((high_ptr[l] & kmask3) << 0)) - 32);
            sums[3] += prepared.values[4*l + 3] * ((int8_t)((second_ptr[l]  >> 4) | ((high_ptr[l] & kmask4) >> 2)) - 32);
        }

        // Weighted sum: each of the 4 sub-groups has its own scale.
        // sc[0], sc[2], sc[4], sc[6] are the scales (stride 2 because
        // adjacent scales belong to different thread positions).
        accumulator += half_d_ptr[0] * (sums[0] * signed_scale_ptr[0] + sums[1] * signed_scale_ptr[2] + sums[2] * signed_scale_ptr[4] + sums[3] * signed_scale_ptr[6]);
    }
};

#endif
