#ifndef GGML_METAL_MATMUL_BLOCK_DOT_Q4_K_METAL
#define GGML_METAL_MATMUL_BLOCK_DOT_Q4_K_METAL

#include "block-dot-lane.metal"

// Q4_K SWAR adapter. Callers retain traversal, reduction, and row scheduling.
struct block_dot_q4_k_fragment {
    // Selects the pair of packed scale words for this lane.
    short scale_pair_index;
    // Byte offset into packed Q4 quant bytes for this lane's four quants.
    uint quant_byte_offset;
    // Float offset into the activation superblock for this lane's first load.
    uint activation_offset;
};

struct block_dot_q4_k_activation {
    float4 lower_low;
    float4 lower_high;
    float4 upper_low;
    float4 upper_high;
};

struct block_dot_q4_k_swar {
    // Sixteen lanes cover one 256-element Q4_K superblock.
    enum : short { threads_per_block = 16 };

    static inline block_dot_q4_k_fragment owned_fragment(short lane_in_block) {
        // Thread-to-data mapping within 16-thread block (matches Vulkan mul_mat_vec_q4_k).
        // Historical names: il element group, ir position, v_im half, v_in interleave, l0 offset.
        const short element_group = lane_in_block / 4;
        const short position_in_group = lane_in_block % 4;
        const short scale_pair_index = element_group / 2;
        const short interleaved_band = element_group % 2;
        const short chunk_offset = 4 * (2 * position_in_group + interleaved_band);

        // Quant bytes use 32-byte bands; activation positions use 64-float bands,
        // rather than the inaccurate old description of a 128-element half offset.
        return {
            scale_pair_index,
            uint(32 * scale_pair_index + chunk_offset),
            uint(64 * scale_pair_index + chunk_offset),
        };
    }

    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline block_dot_q4_k_activation prepare_activation(
            thread const block_dot_q4_k_fragment & owned,
            device const float * block_y) {
        // This faithful adapter accepts only the original fp32 representation.
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32,
                "Q4_K faithful adapter requires the fp32 arithmetic lane");
        // Preserve the original four float4 loads and their order.
        device const float * activation = block_y + owned.activation_offset;
        float4 lower_low = *(device const float4 *)(activation + 0);
        float4 lower_high = *(device const float4 *)(activation + 32);
        float4 upper_low = *(device const float4 *)(activation + 128);
        float4 upper_high = *(device const float4 *)(activation + 160);
        return { lower_low, lower_high, upper_low, upper_high };
    }

    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline void accumulate(
            device const block_q4_K * block,
            thread const block_dot_q4_k_fragment & owned,
            thread const block_dot_q4_k_activation & prepared,
            thread float & accumulator) {
        // This faithful adapter accepts only the original fp32 representation.
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32,
                "Q4_K faithful adapter requires the fp32 arithmetic lane");
        // Bitmasks for unpacking 6-bit scales from the 12-byte scales array.
        constexpr uint16_t kmask1 = 0x3f3f;
        constexpr uint16_t kmask2 = 0x0f0f;
        constexpr uint16_t kmask3 = 0xc0c0;

        // Select this lane's packed scales, quant bytes, and d/dmin pair.
        device const uint16_t * packed_scale_words =
                (device const uint16_t *) block->scales + owned.scale_pair_index;
        device const uint32_t * quant_words =
                (device const uint32_t *) (block->qs + owned.quant_byte_offset);
        device const half * block_scales = &block->d;
        uint16_t unpacked_scale_pairs[4];

        // sc16[0..1] are quantization scales; sc16[2..3] are min scales.
        unpacked_scale_pairs[0] = packed_scale_words[0] & kmask1;
        unpacked_scale_pairs[1] = packed_scale_words[2] & kmask1;
        unpacked_scale_pairs[2] = ((packed_scale_words[4] >> 0) & kmask2) |
                                  ((packed_scale_words[0] & kmask3) >> 2);
        unpacked_scale_pairs[3] = ((packed_scale_words[4] >> 4) & kmask2) |
                                  ((packed_scale_words[2] & kmask3) >> 2);
        // Declare after the writes so Intel Metal reads freshly-unpacked bytes.
        thread const uint8_t * unpacked_scales =
                (thread const uint8_t *) unpacked_scale_pairs;

        // qs0 is this lane's lower word; qs64 is 64 bytes later in qs.
        uint32_t lower_quant_word = quant_words[0];
        uint32_t upper_quant_word = quant_words[16];
        uchar4 lower_low_quants = as_type<uchar4>(lower_quant_word & 0x0F0F0F0Fu);
        uchar4 lower_high_quants = as_type<uchar4>((lower_quant_word >> 4) & 0x0F0F0F0Fu);
        uchar4 upper_low_quants = as_type<uchar4>(upper_quant_word & 0x0F0F0F0Fu);
        uchar4 upper_high_quants = as_type<uchar4>((upper_quant_word >> 4) & 0x0F0F0F0Fu);

        // Dot products: activation × dequantized weights for four sub-groups.
        float dot_lower_low = dot(prepared.lower_low, float4(lower_low_quants));
        float dot_lower_high = dot(prepared.lower_high, float4(lower_high_quants));
        float dot_upper_low = dot(prepared.upper_low, float4(upper_low_quants));
        float dot_upper_high = dot(prepared.upper_high, float4(upper_high_quants));

        // q4_K dequant: d * (quant * scale) - dmin * min_scale * sum(activation).
        float min_correction =
                (prepared.lower_low.x + prepared.lower_low.y + prepared.lower_low.z + prepared.lower_low.w) * float(unpacked_scales[2]) +
                (prepared.lower_high.x + prepared.lower_high.y + prepared.lower_high.z + prepared.lower_high.w) * float(unpacked_scales[3]) +
                (prepared.upper_low.x + prepared.upper_low.y + prepared.upper_low.z + prepared.upper_low.w) * float(unpacked_scales[6]) +
                (prepared.upper_high.x + prepared.upper_high.y + prepared.upper_high.z + prepared.upper_high.w) * float(unpacked_scales[7]);

        // Preserve nested FMA and the incoming accumulator's inner position.
        float2 scale_and_min = float2(block_scales[0], block_scales[1]);
        accumulator = fma(scale_and_min.x, fma(dot_lower_low, float(unpacked_scales[0]), fma(dot_lower_high, float(unpacked_scales[1]),
                        fma(dot_upper_low, float(unpacked_scales[4]), dot_upper_high * float(unpacked_scales[5])))),
                    fma(-scale_and_min.y, min_correction, accumulator));
    }
};

#endif
