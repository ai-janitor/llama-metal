#ifndef GGML_METAL_MATMUL_BLOCK_DOT_Q3_K_METAL
#define GGML_METAL_MATMUL_BLOCK_DOT_Q3_K_METAL

// Native Q3_K block-dot adapter. The caller owns persistent accumulation and
// reduction; this module owns the fixed eight-lane map and one row-body.
// -------------------------------------------------------------------------
// q3_K format: 3-bit weights stored as 2 low bits (qs) + 1 high bit (hmask).
// -------------------------------------------------------------------------
// block_q3_K layout (110 bytes per superblock of 256 elements):
//   - hmask:  uint8[32]  — 1-bit high bits packed, one bit per element
//   - qs:     uint8[64]  — 2-bit low quants packed, 4 values per byte
//   - scales: uint8[12]  — 6-bit quantized per-group scales (packed)
//   - d:      half       — super-block scale
//
struct block_dot_q3_k_fragment {
    short low_quant_byte_offset;
    short high_quant_byte_offset;
    short activation_offset;
    short quarter_index;
    short quant_shift;
    uint16_t scale_low_shift;
    uint16_t scale_high_shift;
    ushort4 high_masks;
    int4 low_masks;
    float correction_first;
    float correction_second;
};

struct block_dot_q3_k_activation {
    float values[32];
    // The fast lane keeps two packed four-value activation groups for each
    // original eight-value Q3 group. The faithful lane continues to use values.
    half4 fast_groups[8];
};

struct block_dot_q3_k_legacy {
    enum : short { threads_per_block = 8 };

    static inline block_dot_q3_k_fragment owned_fragment(short lane_in_block) {
        // One would think that the Metal compiler would figure out that ip and il can only have
        // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it
        // with these two tables.
        //
        // The high-bit mask tables (mm) and low-2-bit mask tables (qm) are
        // pre-computed to avoid the compiler failing to optimize the ip/il branches.
        // Each entry selects the correct bits for a given (ip, il) combination.
        //
        // mm[i]: 4 masks to extract the high bit from hmask (viewed as uint16).
        //   mm[i][0] selects high bit for 1st element pair in low byte
        //   mm[i][1] selects high bit for 1st element pair in high byte
        //   mm[i][2] selects high bit for 2nd element pair in low byte
        //   mm[i][3] selects high bit for 2nd element pair in high byte
        // The compiler needs literal mask tables for the four (half, quarter) states.
        const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip=0, il=0: bits 0,8,1,9
                               {0x0004, 0x0400, 0x0008, 0x0800},  // ip=0, il=2: bits 2,10,3,11
                               {0x0010, 0x1000, 0x0020, 0x2000},  // ip=1, il=0: bits 4,12,5,13
                               {0x0040, 0x4000, 0x0080, 0x8000}}; // ip=1, il=2: bits 6,14,7,15
        // qm[i]: 4 masks to extract the low 2 bits from qs (viewed as uint16).
        //   il=0: bits at positions 0-1, 8-9, 2-3, 10-11
        //   il=2: bits at positions 4-5, 12-13, 6-7, 14-15
        const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00},    // il/2 = 0
                            {0x0030, 0x3000, 0x00c0, 0xc000}};   // il/2 = 1

        // -------------------------------------------------------------------------
        // Thread-to-data mapping within 8-thread block.
        // -------------------------------------------------------------------------
        //   ip (0 or 1) — which 128-element half of the 256-element superblock
        //   il (0 or 2) — which 64-element quarter within the half
        //   ir (0 or 1) — selects 8-element chunk within the quarter
        //   l0 (0 or 8) — byte offset within the 16-byte qs chunk
        const short half_index = lane_in_block / 4;
        const short quarter_selector = 2*((lane_in_block%4)/2);
        const short chunk_selector = lane_in_block % 2;
        const short chunk_byte_offset = 8 * chunk_selector;
        // shift/v1/v2: compensate for the bit position of the 2-bit quants.
        // q3_K extracts quants from different bit positions within each byte;
        // the normalizing factors (v1, v2) undo the implicit left-shift.
        const short quant_shift = 2*quarter_selector;
        const float correction_first = quarter_selector == 0 ? 4.f : 64.f;
        const float correction_second = 4.f * correction_first;
        // Scale reconstruction shifts: select the correct 4-bit nibble from
        // the packed 12-byte scale array (a[0..5]).
        const uint16_t scale_low_shift = 4*half_index;
        const uint16_t scale_high_shift = scale_low_shift + quarter_selector;
        // quarter_index is 0 or 1; scale entries remain the original a[il+0/+1].
        const short quarter_index = quarter_selector/2;
        return { short(32*half_index + chunk_byte_offset), chunk_byte_offset,
                 short(128*half_index + 32*quarter_selector + chunk_byte_offset), quarter_index, quant_shift,
                 scale_low_shift, scale_high_shift, mm[2*half_index + quarter_index], qm[quarter_index],
                 correction_first, correction_second };
    }

    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline block_dot_q3_k_activation prepare_activation(
            thread const block_dot_q3_k_fragment & owned, device const float * block_y) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32 || ARITHMETIC_LANE == block_dot_lane_packed_fp16,
                "Q3_K arithmetic lane must be fp32 or packed fp16");
        block_dot_q3_k_activation prepared_values;
        device const float * y1 = block_y + owned.activation_offset;
        for (short l = 0; l < 8; ++l) {
            prepared_values.values[l+ 0] = y1[l+ 0];
            prepared_values.values[l+ 8] = y1[l+16];
            prepared_values.values[l+16] = y1[l+32];
            prepared_values.values[l+24] = y1[l+48];
        }
        if constexpr (ARITHMETIC_LANE == block_dot_lane_packed_fp16) {
            for (short group = 0; group < 4; ++group) {
                const short offset = 8*group;
                prepared_values.fast_groups[2*group + 0] = half4(
                        prepared_values.values[offset + 0], prepared_values.values[offset + 1],
                        prepared_values.values[offset + 2], prepared_values.values[offset + 3]);
                prepared_values.fast_groups[2*group + 1] = half4(
                        prepared_values.values[offset + 4], prepared_values.values[offset + 5],
                        prepared_values.values[offset + 6], prepared_values.values[offset + 7]);
            }
        }
        return prepared_values;
    }

    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline void accumulate(
            device const block_q3_K * block,
            thread const block_dot_q3_k_fragment & owned,
            thread const block_dot_q3_k_activation & prepared_values,
            thread float & first_accumulator,
            thread float & second_accumulator) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32 || ARITHMETIC_LANE == block_dot_lane_packed_fp16,
                "Q3_K arithmetic lane must be fp32 or packed fp16");
        if constexpr (ARITHMETIC_LANE == block_dot_lane_packed_fp16) {
            // The packed lane decodes each original Q3 group as two dot4 calls.
            // Quant shifts restore the original masked low-quant magnitude before
            // the fp32 high-mask correction and scale application.
            device const uint16_t * q = (device const uint16_t *)(block->qs + owned.low_quant_byte_offset);
            device const uint16_t * h = (device const uint16_t *)(block->hmask + owned.high_quant_byte_offset);
            device const uint16_t * a = (device const uint16_t *)(block->scales);
            device const half * dh = &block->d;
            const float d_all = (float)dh[0];
            uint32_t scales32, aux32;
            thread uint16_t * scales16 = (thread uint16_t *)&scales32;
            thread const int8_t * scales = (thread const int8_t *)&scales32;
            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> owned.scale_high_shift) << 4) & 0x30303030;
            scales16[0] = a[owned.quarter_index*2+0];
            scales16[1] = a[owned.quarter_index*2+1];
            scales32 = ((scales32 >> owned.scale_low_shift) & 0x0f0f0f0f) | aux32;

            for (short group = 0; group < 4; ++group) {
                const short value_offset = 8*group;
                const short q_offset = (group & 1) ? 8 : 0;
                const short quant_shift = owned.quant_shift + (group >= 2 ? 2 : 0);
                const ushort low_bit_shift = quant_shift;
                const ushort high_byte_shift = low_bit_shift + 8;
                uchar4 low_quant_first;
                uchar4 low_quant_second;
                float correction_sum = 0.0f;
                const ushort4 high_masks = owned.high_masks;
                const ushort high_mask_first = group < 2 ? high_masks[0] : high_masks[2];
                const ushort high_mask_second = group < 2 ? high_masks[1] : high_masks[3];
                const float correction_scale = group < 2 ? owned.correction_first : owned.correction_second;

                for (short pair = 0; pair < 4; ++pair) {
                    const ushort packed_q = q[q_offset + pair];
                    const ushort packed_h = h[q_offset + pair];
                    const uchar q0 = uchar((packed_q >> low_bit_shift) & 0x3);
                    const uchar q1 = uchar((packed_q >> high_byte_shift) & 0x3);
                    if (pair < 2) {
                        low_quant_first[2*pair + 0] = q0;
                        low_quant_first[2*pair + 1] = q1;
                    } else {
                        low_quant_second[2*(pair - 2) + 0] = q0;
                        low_quant_second[2*(pair - 2) + 1] = q1;
                    }
                    const short correction_offset = 2*pair;
                    correction_sum += ((packed_h & high_mask_first) ? 0.0f : prepared_values.values[value_offset + correction_offset + 0])
                                   + ((packed_h & high_mask_second) ? 0.0f : prepared_values.values[value_offset + correction_offset + 1]);
                }
                const float restored_low_dot = (block_dot_packed_half_dot4(prepared_values.fast_groups[2*group + 0], low_quant_first)
                        + block_dot_packed_half_dot4(prepared_values.fast_groups[2*group + 1], low_quant_second)) * float(1 << quant_shift);
                const float group_dot = d_all * (restored_low_dot - correction_sum*correction_scale);
                if (group < 2) {
                    first_accumulator += group_dot * (scales[group] - 32);
                } else {
                    second_accumulator += group_dot * (scales[group] - 32);
                }
            }
            return;
        }
        // scales32/scales16/scales are overlapping views of packed scale bytes.
        // scales32/scales16/scales: overlapping views of 4-byte scale buffer.
        // Used to reconstruct 6-bit per-group scales from the packed 12-byte format.
        uint32_t scales32, aux32;
        thread uint16_t * scales16 = (thread uint16_t *)&scales32;
        thread const int8_t * scales = (thread const int8_t *)&scales32;

        // A 110-byte Q3_K block can start at address 2 mod 4: retain uint16 loads.
        device const uint16_t * q = (device const uint16_t *)(block->qs + owned.low_quant_byte_offset);
        device const uint16_t * h = (device const uint16_t *)(block->hmask + owned.high_quant_byte_offset);
        device const uint16_t * a = (device const uint16_t *)(block->scales);
        device const half * dh = &block->d;
        const float d_all = (float)dh[0];

        // Reconstruct 6-bit per-group scales from the packed 12-byte array.
        // q3_K packs 16 × 6-bit scales into a[0..5] (12 bytes = 96 bits).
        // Step 1: extract upper 2 bits from a[4..5] into aux32
        scales16[0] = a[4];
        scales16[1] = a[5];
        aux32 = ((scales32 >> owned.scale_high_shift) << 4) & 0x30303030;
        // Step 2: extract lower 4 bits from a[il..il+1]
        scales16[0] = a[owned.quarter_index*2+0];
        scales16[1] = a[owned.quarter_index*2+1];
        scales32 = ((scales32 >> owned.scale_low_shift) & 0x0f0f0f0f) | aux32;

        // Dot product with first qs chunk (first 8 uint16 values).
        // s1/s2: weighted 2-bit quant sums (low byte / high byte)
        // s3: high-bit correction — subtracted when hmask bit is 0
        //     (q3_K value = low_2bits - 4*high_bit, centered at 0)
        //     Here high_bit is the inverted hmask representation: raw hmask bit 0 subtracts 4.
        // s4/s5/s6: same pattern for the interleaved sub-group
        float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
        for (short l = 0; l < 8; l += 2) {
            const int32_t qs = q[l/2];
            s1 += prepared_values.values[l+0] * (qs & owned.low_masks[0]);
            s2 += prepared_values.values[l+1] * (qs & owned.low_masks[1]);
            s3 += ((h[l/2] & owned.high_masks[0]) ? 0.f : prepared_values.values[l+0]) + ((h[l/2] & owned.high_masks[1]) ? 0.f : prepared_values.values[l+1]);
            s4 += prepared_values.values[l+16] * (qs & owned.low_masks[2]);
            s5 += prepared_values.values[l+17] * (qs & owned.low_masks[3]);
            s6 += ((h[l/2] & owned.high_masks[2]) ? 0.f : prepared_values.values[l+16]) + ((h[l/2] & owned.high_masks[3]) ? 0.f : prepared_values.values[l+17]);
        }
        // Combine: quant_sum * scale, where scale is (scales[i] - 32) to center at 0.
        // The 1/256 factor on s2 compensates for high-byte extraction.
        // v1/v2 compensate for the bit-position shift of the high bit.
        float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*owned.correction_first);
        float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*owned.correction_second);
        first_accumulator += d1 * (scales[0] - 32);
        second_accumulator += d2 * (scales[2] - 32);

        // Same pattern for second qs chunk (q[8..15], h[8..15])
        s1 = s2 = s3 = s4 = s5 = s6 = 0;
        for (short l = 0; l < 8; l += 2) {
            const int32_t qs = q[l/2+8];
            s1 += prepared_values.values[l+8] * (qs & owned.low_masks[0]);
            s2 += prepared_values.values[l+9] * (qs & owned.low_masks[1]);
            s3 += ((h[l/2+8] & owned.high_masks[0]) ? 0.f : prepared_values.values[l+8]) + ((h[l/2+8] & owned.high_masks[1]) ? 0.f : prepared_values.values[l+9]);
            s4 += prepared_values.values[l+24] * (qs & owned.low_masks[2]);
            s5 += prepared_values.values[l+25] * (qs & owned.low_masks[3]);
            s6 += ((h[l/2+8] & owned.high_masks[2]) ? 0.f : prepared_values.values[l+24]) + ((h[l/2+8] & owned.high_masks[3]) ? 0.f : prepared_values.values[l+25]);
        }
        d1 = d_all * (s1 + 1.f/256.f * s2 - s3*owned.correction_first);
        d2 = d_all * (s4 + 1.f/256.f * s5 - s6*owned.correction_second);
        first_accumulator += d1 * (scales[1] - 32);
        second_accumulator += d2 * (scales[3] - 32);
    }
};

#endif
