#ifndef GGML_METAL_MATMUL_BLOCK_DOT_METAL
#define GGML_METAL_MATMUL_BLOCK_DOT_METAL

#include "block-dot-lane.metal"

#include "block-dot-basic.metal"

#include "block-dot-q2-k.metal"

#include "block-dot-q3-k.metal"

#include "block-dot-q8-0.metal"

#include "block-dot-q4-k.metal"

// Scalar Q6_K block-dot adapter. It owns the scalar lane map and exact SWAR
// unpack; traversal, reduction, and output storage remain kernel-owned.
struct block_dot_q6_k_fragment { short q_offset_l; short q_offset_h; short scale_index; short y_offset; };
struct block_dot_q6_k_activation { float4 act0; float4 act1; float4 act2; float4 act3; float4 act_sums; half4 half_act0; half4 half_act1; half4 half_act2; half4 half_act3; };

struct block_dot_q6_k_scalar_swar {
    // An enum is a compile-time value without MSL program-scope variable storage.
    enum : short { threads_per_block = 16 };
    static inline block_dot_q6_k_fragment owned_fragment(short lane_in_block) {
        const short superblock_half = lane_in_block/8; // selects one of two 128-weight halves
        const short lane_in_half = lane_in_block%8; // selects one of eight lanes
        const short byte_offset_in_half = 4*lane_in_half; // selects four adjacent byte positions
        const short scale_base_index = 8*superblock_half + byte_offset_in_half/16; // addresses this lane's first signed scale
        return { short(64*superblock_half + byte_offset_in_half), short(32*superblock_half + byte_offset_in_half), scale_base_index, short(128*superblock_half + byte_offset_in_half) };
    }
    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline block_dot_q6_k_activation prepare_activation(thread const block_dot_q6_k_fragment & f, device const float * block_y) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32 || ARITHMETIC_LANE == block_dot_lane_packed_fp16, "Q6_K scalar SWAR supports fp32 or packed fp16 lanes");
        device const float * y = block_y + f.y_offset;
        const float4 act0 = *((device const float4 *)(y +  0));
        const float4 act1 = *((device const float4 *)(y + 32));
        const float4 act2 = *((device const float4 *)(y + 64));
        const float4 act3 = *((device const float4 *)(y + 96));
        const float4 act_sums = float4(act0.x + act0.y + act0.z + act0.w, act1.x + act1.y + act1.z + act1.w, act2.x + act2.y + act2.z + act2.w, act3.x + act3.y + act3.z + act3.w);
        if constexpr (ARITHMETIC_LANE == block_dot_lane_packed_fp16) {
            return { act0, act1, act2, act3, act_sums, half4(act0), half4(act1), half4(act2), half4(act3) };
        }
        return { act0, act1, act2, act3, act_sums };
    }
    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline void accumulate(device const block_q6_K * block, thread const block_dot_q6_k_fragment & f, thread const block_dot_q6_k_activation & a, thread float & acc) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32 || ARITHMETIC_LANE == block_dot_lane_packed_fp16, "Q6_K scalar SWAR supports fp32 or packed fp16 lanes");
        // A 210-byte block can start at address 2 mod 4: retain ushort loads, never uint casts.
        device const uint8_t * low_ptr = block->ql + f.q_offset_l;
        device const uint8_t * low_hi_ptr = low_ptr + 32;
        device const uint8_t * high_ptr = block->qh + f.q_offset_h;
        device const int8_t * scale_ptr = block->scales + f.scale_index;
        device const half * d_ptr = &block->d;
        device const ushort * low_halves_a = (device const ushort *) low_ptr;
        device const ushort * low_halves_b = (device const ushort *) low_hi_ptr;
        device const ushort * high_halves = (device const ushort *) high_ptr;
        constexpr uint HALF_WORD_BITS = 16; // high ushort occupies the upper half of the packed word
        const uint low_word_a = uint(low_halves_a[0]) | (uint(low_halves_a[1]) << HALF_WORD_BITS);
        const uint low_word_b = uint(low_halves_b[0]) | (uint(low_halves_b[1]) << HALF_WORD_BITS);
        const uint high_word = uint(high_halves[0]) | (uint(high_halves[1]) << HALF_WORD_BITS);
        // Reconstruct 6-bit weights: ql nibble | qh pair shifted into bits 4..5.
        constexpr uint LOW_NIBBLES = 0x0F0F0F0F; // lower four bits in each byte lane
        constexpr uint HIGH_PAIR_0 = 0x03030303; // qh bits 0..1
        constexpr uint HIGH_PAIR_1 = 0x0C0C0C0C; // qh bits 2..3
        constexpr uint HIGH_PAIR_2 = 0x30303030; // qh bits 4..5
        const uint q_g0 = (low_word_a & LOW_NIBBLES) | ((high_word & HIGH_PAIR_0) << 4);
        const uint q_g1 = (low_word_b & LOW_NIBBLES) | ((high_word & HIGH_PAIR_1) << 2);
        const uint q_g2 = ((low_word_a >> 4) & LOW_NIBBLES) | (high_word & HIGH_PAIR_2);
        const uint q_g3 = ((low_word_b >> 4) & LOW_NIBBLES) | ((high_word >> 2) & HIGH_PAIR_2);
        if constexpr (ARITHMETIC_LANE == block_dot_lane_packed_fp16) {
            const float4 unsigned_dots = float4(block_dot_packed_half_dot4(a.half_act0, as_type<uchar4>(q_g0)), block_dot_packed_half_dot4(a.half_act1, as_type<uchar4>(q_g1)), block_dot_packed_half_dot4(a.half_act2, as_type<uchar4>(q_g2)), block_dot_packed_half_dot4(a.half_act3, as_type<uchar4>(q_g3)));
            constexpr float QUANT_CENTER = 32.f; // fold -32 after dot; preserves fp32 operation order
            const float4 sums = unsigned_dots - QUANT_CENTER * a.act_sums;
            const float superblock_scale = float(d_ptr[0]);
            acc += superblock_scale * (sums[0] * scale_ptr[0] + sums[1] * scale_ptr[2] + sums[2] * scale_ptr[4] + sums[3] * scale_ptr[6]);
            return;
        }
        const float4 unsigned_dots = float4(dot(float4(as_type<uchar4>(q_g0)), a.act0), dot(float4(as_type<uchar4>(q_g1)), a.act1), dot(float4(as_type<uchar4>(q_g2)), a.act2), dot(float4(as_type<uchar4>(q_g3)), a.act3));
        constexpr float QUANT_CENTER = 32.f; // fold -32 after dot; preserves fp32 operation order
        const float4 sums = unsigned_dots - QUANT_CENTER * a.act_sums;
        const float superblock_scale = float(d_ptr[0]);
        acc += superblock_scale * (sums[0] * scale_ptr[0] + sums[1] * scale_ptr[2] + sums[2] * scale_ptr[4] + sums[3] * scale_ptr[6]);
    }
};

// Scalar Q5_K block-dot adapter. It owns the scalar lane map and exact SWAR
// unpack; traversal, reduction, and output storage remain kernel-owned.
struct block_dot_q5_k_fragment {
    short quant_byte_offset;      // qs[] byte: 32*scale_pair_index + 8*lane_in_pair
    short high_plane_byte_offset; // qh[] byte: 8*lane_in_pair
    short scale_pair_index;       // scales[] uint16 pair: lane_in_block/4
    short activation_offset;      // activation float: 64*scale_pair_index + 8*lane_in_pair
    short high_bit_shift;         // selected qh bit base: 2*scale_pair_index
};
struct block_dot_q5_k_activation {
    float4 act_a[2]; float4 act_b[2]; float4 act_c[2]; float4 act_d[2]; float4 activation_sums;
    half4 packed_act_a[2]; half4 packed_act_b[2]; half4 packed_act_c[2]; half4 packed_act_d[2];
};

struct block_dot_q5_k_scalar_swar {
    // An enum is a compile-time value without MSL program-scope variable storage.
    enum : short { threads_per_block = 8 };
    static inline block_dot_q5_k_fragment owned_fragment(short lane_in_block) {
        // Historical thread-map comments preserved verbatim:
        // -------------------------------------------------------------------------
        // Thread-to-data mapping for q5_K (8 threads per superblock).
        // -------------------------------------------------------------------------
        //   iq (0 or 1) — which 128-element half of the superblock
        //   ir (0..3)   — which 8-element chunk within the half
        //   l0 (0,8,16,24) — starting element offset
        // Scale-pair clarification: one pair spans 64 activation floats; lane_in_pair selects its eight-float chunk.
        const short scale_pair_index = lane_in_block/4;
        const short lane_in_pair = lane_in_block%4;
        const short byte_offset_in_pair = 8*lane_in_pair;
        return { short(32*scale_pair_index + byte_offset_in_pair), byte_offset_in_pair, scale_pair_index, short(64*scale_pair_index + byte_offset_in_pair), short(2*scale_pair_index) };
    }
    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline block_dot_q5_k_activation prepare_activation(thread const block_dot_q5_k_fragment & owned, device const float * block_y) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32 || ARITHMETIC_LANE == block_dot_lane_packed_fp16, "Q5_K adapter supports fp32 or packed fp16 lanes");
        // Historical cache comment preserved verbatim:
        // float yl[16], yh[16];  // local activation cache: yl for lower 128, yh for upper 128
        // The cache now names the selected lower and upper 64-weight bands directly.
        device const float * lower_activation = block_y + owned.activation_offset;
        device const float * upper_activation = lower_activation + 128;
        // Four 8-element activation groups, each split into two aligned float4 loads.
        // Original terminology: y_offset is a multiple of eight floats; all vector starts are 16-byte aligned.
        // activation_offset is a multiple of eight floats; all vector starts are 16-byte aligned.
        // chunk iq, low nibble
        const float4 act_a[2] = {*((device const float4 *)(lower_activation + 0)), *((device const float4 *)(lower_activation + 4))};   // scale-pair band, low nibble
        // chunk iq, high nibble
        const float4 act_b[2] = {*((device const float4 *)(lower_activation + 32)), *((device const float4 *)(lower_activation + 36))}; // scale-pair band, high nibble
        // chunk iq+2, low nibble
        const float4 act_c[2] = {*((device const float4 *)(upper_activation + 0)), *((device const float4 *)(upper_activation + 4))};   // second 64-weight band, low nibble
        // chunk iq+2, high nibble
        const float4 act_d[2] = {*((device const float4 *)(upper_activation + 32)), *((device const float4 *)(upper_activation + 36))}; // second 64-weight band, high nibble
        float4 activation_sums = {0.f, 0.f, 0.f, 0.f};
        // Preserve the original sequential eight-element sums used by the dmin term.
        FOR_UNROLL (short activation_element = 0; activation_element < 8; ++activation_element) {
            activation_sums[0] += act_a[activation_element / 4][activation_element % 4]; // register element from the same lower/low group
            activation_sums[1] += act_b[activation_element / 4][activation_element % 4]; // lower/high group
            activation_sums[2] += act_c[activation_element / 4][activation_element % 4]; // upper/low group
            activation_sums[3] += act_d[activation_element / 4][activation_element % 4]; // upper/high group
        }
        if constexpr (ARITHMETIC_LANE == block_dot_lane_fp32) {
            return { { act_a[0], act_a[1] }, { act_b[0], act_b[1] }, { act_c[0], act_c[1] }, { act_d[0], act_d[1] }, activation_sums };
        } else {
            block_dot_q5_k_activation prepared = { { act_a[0], act_a[1] }, { act_b[0], act_b[1] }, { act_c[0], act_c[1] }, { act_d[0], act_d[1] }, activation_sums };
            FOR_UNROLL (short word = 0; word < 2; ++word) {
                prepared.packed_act_a[word] = half4(prepared.act_a[word]);
                prepared.packed_act_b[word] = half4(prepared.act_b[word]);
                prepared.packed_act_c[word] = half4(prepared.act_c[word]);
                prepared.packed_act_d[word] = half4(prepared.act_d[word]);
            }
            return prepared;
        }
    }
    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline void accumulate(device const block_q5_K * block, thread const block_dot_q5_k_fragment & owned, thread const block_dot_q5_k_activation & prepared, thread float & acc) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32 || ARITHMETIC_LANE == block_dot_lane_packed_fp16, "Q5_K adapter supports fp32 or packed fp16 lanes");
        device const uint8_t * lower_quant_bytes = block->qs + owned.quant_byte_offset;
        device const uint8_t * upper_quant_bytes = lower_quant_bytes + 64;  // upper 64 bytes of qs[]
        device const uint8_t * high_plane_bytes = block->qh + owned.high_plane_byte_offset;
        device const half * block_scales = &block->d;
        device const uint16_t * packed_scale_words = (device const uint16_t *)block->scales + owned.scale_pair_index;
        // Bitmasks for unpacking 6-bit scales (same as q4_K — shared format)
        constexpr uint16_t SCALE_LOW_SIX_BITS = 0x3f3f;  // lower 6 bits of each byte
        constexpr uint16_t SCALE_LOW_FOUR_BITS = 0x0f0f;  // lower 4 bits of each byte
        constexpr uint16_t SCALE_HIGH_TWO_BITS = 0xc0c0;  // upper 2 bits of each byte
        uint16_t unpacked_scale_pairs[4];  // unpacked scale buffer (same layout as q4_K)
        thread const uint8_t * unpacked_scales = (thread const uint8_t *)unpacked_scale_pairs;
        // Unpack 6-bit scales (same format as q4_K)
        unpacked_scale_pairs[0] = packed_scale_words[0] & SCALE_LOW_SIX_BITS;
        unpacked_scale_pairs[1] = packed_scale_words[2] & SCALE_LOW_SIX_BITS;
        unpacked_scale_pairs[2] = ((packed_scale_words[4] >> 0) & SCALE_LOW_FOUR_BITS) | ((packed_scale_words[0] & SCALE_HIGH_TWO_BITS) >> 2);
        unpacked_scale_pairs[3] = ((packed_scale_words[4] >> 4) & SCALE_LOW_FOUR_BITS) | ((packed_scale_words[2] & SCALE_HIGH_TWO_BITS) >> 2);
        // 176 = 11*16: block/row strides, qs+48, qh+16 and lane offsets permit aligned uint2 loads.
        // eight bytes in chunk iq
        const uint2 low_words = *((device const uint2 *) lower_quant_bytes);   // eight bytes in scale-pair band
        // eight bytes in chunk iq+2
        const uint2 upper_words = *((device const uint2 *) upper_quant_bytes); // eight bytes in second 64-weight band
        // shared eight high-plane bytes
        const uint2 high_words = *((device const uint2 *) high_plane_bytes);  // shared eight high-plane bytes
        // Former acc1: dot product with 4-bit low quants (from qs[]).
        // Former acc2: dot product contribution from the 5th high bit (from qh[]).
        // Merge the fifth bit before conversion, replacing separate select-accumulates.
        float4 quant_dots = {0.f};
        constexpr short WORDS_PER_LANE = 2; // eight bytes owned by this lane, four bytes per uint
        constexpr uint LOW_NIBBLES = 0x0F0F0F0F; // lower four bits of each independent byte
        constexpr uint ONE_BIT_PER_BYTE = 0x01010101; // retain the selected qh bit without cross-byte spill
        // q5_K stores 5-bit weights: 4 low bits in qs[] + 1 high bit in qh[].
        // Each qh byte holds 8 high bits. The masks select the correct bit
        // for this thread's (iq, position) within the qh byte.
        //   hm1: high bit for 1st element group (yl), low half
        //   hm2: high bit for 2nd element group (yl+8), low half
        //   hm3: high bit for 1st element group (yh), upper half
        //   hm4: high bit for 2nd element group (yh+8), upper half
        const short high_shift = owned.high_bit_shift; // original hm1..hm4 select bits shift+0,+1,+4,+5 respectively
        FOR_UNROLL (short word = 0; word < WORDS_PER_LANE; ++word) {
            const uint high_word = high_words[word]; // eight high bits packed in each byte
            // Low 4 bits from qs: shift high nibbles down before merging the fifth bit.
            const uint quant_a = (low_words[word] & LOW_NIBBLES) | (((high_word >> high_shift) & ONE_BIT_PER_BYTE) << 4); // chunk scale pair low
            const uint quant_b = ((low_words[word] >> 4) & LOW_NIBBLES) | (((high_word >> (high_shift + 1)) & ONE_BIT_PER_BYTE) << 4); // chunk scale pair high
            const uint quant_c = (upper_words[word] & LOW_NIBBLES) | (((high_word >> (high_shift + 4)) & ONE_BIT_PER_BYTE) << 4); // chunk scale pair+2 low
            const uint quant_d = ((upper_words[word] >> 4) & LOW_NIBBLES) | (((high_word >> (high_shift + 5)) & ONE_BIT_PER_BYTE) << 4); // chunk scale pair+2 high
            // 5th bit from qh adds 16 to each 4-bit value when set, now inside the packed byte.
            // Reinterpret to byte lanes before numerical conversion: every quant is exactly 0..31.
            if constexpr (ARITHMETIC_LANE == block_dot_lane_fp32) {
                quant_dots[0] += dot(float4(as_type<uchar4>(quant_a)), prepared.act_a[word]);
                quant_dots[1] += dot(float4(as_type<uchar4>(quant_b)), prepared.act_b[word]);
                quant_dots[2] += dot(float4(as_type<uchar4>(quant_c)), prepared.act_c[word]);
                quant_dots[3] += dot(float4(as_type<uchar4>(quant_d)), prepared.act_d[word]);
            } else {
                quant_dots[0] += block_dot_packed_half_dot4(prepared.packed_act_a[word], as_type<uchar4>(quant_a));
                quant_dots[1] += block_dot_packed_half_dot4(prepared.packed_act_b[word], as_type<uchar4>(quant_b));
                quant_dots[2] += block_dot_packed_half_dot4(prepared.packed_act_c[word], as_type<uchar4>(quant_c));
                quant_dots[3] += block_dot_packed_half_dot4(prepared.packed_act_d[word], as_type<uchar4>(quant_d));
            }
        }
        // Combine low 4-bit and high 1-bit contributions:
        //   5-bit value = low_4bits + 16*high_bit
        // Former /16.f on acc1[1,3] compensated for the unshifted 0xF0 mask.
        // Packed quants already contain the whole 5-bit value, so both /16 and separate *16 disappear.
        // Scale/min indices and the dmin term remain unchanged; dot regrouping changes fp32 rounding.
        acc += block_scales[0] * (unpacked_scales[0] * quant_dots[0] +
                                  unpacked_scales[1] * quant_dots[1] +
                                  unpacked_scales[4] * quant_dots[2] +
                                  unpacked_scales[5] * quant_dots[3]) -
               block_scales[1] * (prepared.activation_sums[0] * unpacked_scales[2] + prepared.activation_sums[1] * unpacked_scales[3] + prepared.activation_sums[2] * unpacked_scales[6] + prepared.activation_sums[3] * unpacked_scales[7]);
    }
};

#include "block-dot-q6-grouped.metal"

#endif
