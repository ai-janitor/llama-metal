#ifndef GGML_METAL_MATMUL_BLOCK_DOT_Q2_K_METAL
#define GGML_METAL_MATMUL_BLOCK_DOT_Q2_K_METAL

#include "lane-fma-f32.metal"

// Scalar Q2_K block-dot adapter. It owns the scalar lane map and exact native
// legacy arithmetic; traversal, reduction, and output storage remain kernel-owned.
struct block_dot_q2_k_fragment {
    short activation_offset;     // first activation in this lane's four eight-value groups
    short scale_byte_offset;     // packed scale/min byte for this lane
    short quant_halfword_offset; // first packed eight-quant uint16 for this lane
};

struct block_dot_q2_k_activation {
    float values[32]; // four sequential eight-value groups in native load order
    float4 group_sums; // sums for the corresponding min-value correction groups
};

struct block_dot_q2_k_legacy {
    enum : short { threads_per_block = 8 };

    static inline block_dot_q2_k_fragment owned_fragment(short lane_in_block) {
        //   it (0..7)       — thread position within 8-thread block
        //   iq (0 or 1)     — which 128-element half of the superblock
        //   ir (0..3)       — position within the half (selects 8-element chunk)
        //   is (0 or 1)     — which scale pair to use (every 16 elements = 1 scale)
        const short half_index = lane_in_block/4; // 0 = lower 128 elements, 1 = upper
        const short chunk = lane_in_block%4; // selects one eight-element chunk in the half
        const short scale_pair = (8*chunk)/16; // selects its packed scale/min pair
        return { short(128*half_index + 8*chunk), short(8*half_index + scale_pair), short(16*half_index + 4*chunk) };
    }

    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline block_dot_q2_k_activation prepare_activation(
            thread const block_dot_q2_k_fragment & owned,
            device const float * block_y) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "only the fp32 lane exists");
        // local activation cache — 32 elements loaded per superblock iteration
        // Starting activation pointer for this thread's data within the superblock
        device const float * base = block_y + owned.activation_offset;
        block_dot_q2_k_activation prepared = {};
        // Load 32 activation elements (4 groups of 8) and compute their sums.
        // sumy[0..3] are needed for the min-value correction term in q2_K dequant.
        // The 4 groups are spaced 32 elements apart within the 128-element half.
        for (short element = 0; element < 8; ++element) {
            prepared.values[element+ 0] = base[element+ 0]; prepared.group_sums[0] += prepared.values[element+ 0];
            prepared.values[element+ 8] = base[element+32]; prepared.group_sums[1] += prepared.values[element+ 8];
            prepared.values[element+16] = base[element+64]; prepared.group_sums[2] += prepared.values[element+16];
            prepared.values[element+24] = base[element+96]; prepared.group_sums[3] += prepared.values[element+24];
        }
        return prepared;
    }

    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline void accumulate(
            device const block_q2_K * block,
            thread const block_dot_q2_k_fragment & owned,
            thread const block_dot_q2_k_activation & prepared,
            thread float & accumulator) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "only the fp32 lane exists");
        // Pointers to this superblock's scale, quant, and d/dmin data.
        // sc: 16 scale bytes, indexed by this lane's packed scale/min offset
        // sc: 16 scale bytes, indexed by (8*iq + is) to reach this thread's scale
        // Historical scalar index: (8*iq + is); owned.scale_byte_offset carries that same offset.
        // qs: 64 bytes of packed 2-bit quants, viewed as uint16 for 8-at-a-time extraction
        device const uint8_t  * sc = (device const uint8_t  *)block->scales + owned.scale_byte_offset;
        device const uint16_t * qs = (device const uint16_t *)block->qs + owned.quant_halfword_offset;
        device const half     * dh = &block->d;

        // q2_K packs 2-bit quants into uint16 values. Each uint16 holds 8 quants.
        // The masks extract pairs of 2-bit quants at different bit positions:
        //   acc1 extracts from the low byte,  acc2 from the high byte.
        //   The pairs are at bit offsets: 0,2,4,6 (low byte) and 8,10,12,14 (high byte).
        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; i += 2) {
            acc1[0] += prepared.values[i+ 0] * (qs[i/2] & 0x0003);  // bits [1:0] of low byte
            acc2[0] += prepared.values[i+ 1] * (qs[i/2] & 0x0300);  // bits [1:0] of high byte
            acc1[1] += prepared.values[i+ 8] * (qs[i/2] & 0x000c);  // bits [3:2] of low byte
            acc2[1] += prepared.values[i+ 9] * (qs[i/2] & 0x0c00);  // bits [3:2] of high byte
            acc1[2] += prepared.values[i+16] * (qs[i/2] & 0x0030);  // bits [5:4] of low byte
            acc2[2] += prepared.values[i+17] * (qs[i/2] & 0x3000);  // bits [5:4] of high byte
            acc1[3] += prepared.values[i+24] * (qs[i/2] & 0x00c0);  // bits [7:6] of low byte
            acc2[3] += prepared.values[i+25] * (qs[i/2] & 0xc000);  // bits [7:6] of high byte
        }
        // q2_K dequantization: value = dall * quant * scale - dmin * min_scale
        //   dall = superblock scale (d), dmin = superblock min scale (dm)
        //   sc[n] & 0xF  = lower 4 bits = quantization scale for group n
        //   sc[n] & 0xF0 = upper 4 bits = min-value scale for group n
        //
        // The 1/256 factor on acc2 compensates for the masks extracting from
        // the high byte (bit-shifted left by 8). Similarly, 1/4, 1/16, 1/64
        // compensate for the 2-bit shifts within each byte pair.
        float dall = dh[0];
        float dmin = dh[1] * 1.f/16.f;  // pre-scale dmin by 1/16 for the 0xF0 mask
        accumulator += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                               (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                               (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                               (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                       dmin * (prepared.group_sums[0] * (sc[0] & 0xF0) + prepared.group_sums[1] * (sc[2] & 0xF0) + prepared.group_sums[2] * (sc[4] & 0xF0) + prepared.group_sums[3] * (sc[6] & 0xF0));
    }
};

#endif
