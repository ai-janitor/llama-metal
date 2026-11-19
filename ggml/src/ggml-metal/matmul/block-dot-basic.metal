#ifndef GGML_METAL_MATMUL_BLOCK_DOT_BASIC_METAL
#define GGML_METAL_MATMUL_BLOCK_DOT_BASIC_METAL

#include "dequant.metal"
#include "lane-fma-f32.metal"

// ---- q4_0 half-block dot product ----
// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
//
// q4_0 dequant: value = d * (q - 8), where q is a 4-bit unsigned integer.
// The -8 centering is handled by the sumy * -8 term (sum of all yl values × -8 × d).
//
// Parameters:
//   qb_curr — pointer to the q4_0 block
//   sumy    — sum of all 16 yl values (for the centering correction)
//   yl      — 16 pre-scaled activation values (see file header note)
//   il      — starting offset: 0 for lower half, QK4_0/4 for upper half
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;  // per-block scale

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // View quant bytes as uint16 for 2-at-a-time extraction.
    // Skip 1 uint16 (2 bytes) for the d field, then offset by il/2.
    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    // Extract 4-bit quants from packed uint16 values using bitmasks.
    // The yl values are pre-scaled so that the mask positions align:
    //   0x000F → bits [3:0]  of low byte  → yl[i+0] (scale 1)
    //   0x0F00 → bits [3:0]  of high byte → yl[i+1] (scale 1/256)
    //   0x00F0 → bits [7:4]  of low byte  → yl[i+8] (scale 1/16)
    //   0xF000 → bits [7:4]  of high byte → yl[i+9] (scale 1/4096)
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    // Final: d * (quant_sum - 8*activation_sum)
    // The centering (-8 per element) is applied via sumy * -8.
    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// ---- q4_1 half-block dot product ----
// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
//
// Same as q4_0 but with additive min: dequant = d * q + m.
// No centering term needed — m handles the offset via sumy * m.
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;  // per-block scale
    float m = qb_curr->m;  // per-block min value

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // Skip 2 uint16 (4 bytes) for d and m fields
    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);

    // Same mask pattern as q4_0
    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    // d * quant_sum + m * activation_sum
    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// ---- q5_0 half-block dot product ----
// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
//
// q5_0: 5-bit quants = 4 low bits (qs) + 1 high bit (qh). Centered at 16.
// The high bit is extracted from a packed uint32 (qh) — one bit per element.
// Same pre-scaled yl trick as q4_0, with extra logic for the 5th bit.
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

// ---- q5_1 half-block dot product ----
// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
//
// q5_1: same as q5_0 but with additive min (m) instead of centering at 16.
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

namespace basic_block_dot {

struct schedule {
    enum : short { threads_per_block = 2 };
};

struct owned_fragment {
    short half_block_byte_offset;  // 0 for lane 0; 8 for lane 1.
};

struct prepared_activation {
    float values[16];          // Pre-scaled activation cache consumed by legacy helpers.
    float unscaled_sum[2];     // Original unscaled halves for centering/min corrections.
};

static inline owned_fragment make_owned_fragment(short lane_in_block) {
    return { short(lane_in_block * 8) };
}

template<short ARITHMETIC_LANE = block_dot_lane_fp32>
static inline prepared_activation prepare_activation(device const float * activation_half) {
    // This adapter preserves the established fp32 arithmetic until a separate fast lane specifies its math.
    static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "only the fp32 lane exists");
    prepared_activation prepared;
    prepared.unscaled_sum[0] = 0.f;
    prepared.unscaled_sum[1] = 0.f;

    // Load 16 activation elements with pre-scaling.
    // The prepared.values are divided by powers of 256 to compensate for the
    // bit positions of the quant masks in block_q_n_dot_y():
    //   prepared.values[i+0] = activation_half[i+0]           — mask 0x000F extracts bits [3:0]  (×1)
    //   prepared.values[i+1] = activation_half[i+1] / 256     — mask 0x0F00 extracts bits [11:8] (×256)
    //   prepared.values[i+8] = activation_half[i+16] / 16     — mask 0x00F0 extracts bits [7:4]  (×16)
    //   prepared.values[i+9] = activation_half[i+17] / 4096   — mask 0xF000 extracts bits [15:12] (×4096)
    // This avoids integer shifts in the inner loop — the division is done once here.
    // prepared.unscaled_sum accumulates the un-scaled sums for the centering/min correction term.
    FOR_UNROLL (short i = 0; i < 8; i += 2) {
        prepared.unscaled_sum[0]  += activation_half[i +  0] + activation_half[i +  1];
        prepared.values[i + 0] = activation_half[i +  0];
        prepared.values[i + 1] = activation_half[i +  1]/256.f;

        prepared.unscaled_sum[1]  += activation_half[i + 16] + activation_half[i + 17];
        prepared.values[i + 8] = activation_half[i + 16]/16.f;
        prepared.values[i + 9] = activation_half[i + 17]/4096.f;
    }

    return prepared;
}

template<short ARITHMETIC_LANE = block_dot_lane_fp32>
static inline void accumulate(
        device const block_q4_0 * block,
        thread const owned_fragment & owned,
        thread prepared_activation & prepared,
        thread float & accumulator) {
    static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "only the fp32 lane exists");
    accumulator += block_q_n_dot_y(block, prepared.unscaled_sum[0] + prepared.unscaled_sum[1], prepared.values, owned.half_block_byte_offset);
}

template<short ARITHMETIC_LANE = block_dot_lane_fp32>
static inline void accumulate(
        device const block_q4_1 * block,
        thread const owned_fragment & owned,
        thread prepared_activation & prepared,
        thread float & accumulator) {
    static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "only the fp32 lane exists");
    // Keep the established helper invocation and accumulation order in lane 0.
    accumulator += block_q_n_dot_y(block, prepared.unscaled_sum[0] + prepared.unscaled_sum[1], prepared.values, owned.half_block_byte_offset);
}

template<short ARITHMETIC_LANE = block_dot_lane_fp32>
static inline void accumulate(
        device const block_q5_0 * block,
        thread const owned_fragment & owned,
        thread prepared_activation & prepared,
        thread float & accumulator) {
    static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "only the fp32 lane exists");
    accumulator += block_q_n_dot_y(block, prepared.unscaled_sum[0] + prepared.unscaled_sum[1], prepared.values, owned.half_block_byte_offset);
}

template<short ARITHMETIC_LANE = block_dot_lane_fp32>
static inline void accumulate(
        device const block_q5_1 * block,
        thread const owned_fragment & owned,
        thread prepared_activation & prepared,
        thread float & accumulator) {
    static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "only the fp32 lane exists");
    accumulator += block_q_n_dot_y(block, prepared.unscaled_sum[0] + prepared.unscaled_sum[1], prepared.values, owned.half_block_byte_offset);
}

template<short ARITHMETIC_LANE = block_dot_lane_fp32, typename block_q_type>
static inline void accumulate(
        device const block_q_type * block,
        thread const owned_fragment & owned,
        thread prepared_activation & prepared,
        thread float & accumulator) {
    // Reject alternate lane policies before they can silently change the established accumulator arithmetic.
    // Other basic formats remain restricted to their faithful fp32 arithmetic lane.
    static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "basic block-dot supports only the faithful fp32 lane");
    accumulator += block_q_n_dot_y(block, prepared.unscaled_sum[0] + prepared.unscaled_sum[1], prepared.values, owned.half_block_byte_offset);
}

} // namespace basic_block_dot

#endif
