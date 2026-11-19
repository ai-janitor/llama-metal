#ifndef GGML_METAL_MATMUL_BLOCK_DOT_Q8_0_METAL
#define GGML_METAL_MATMUL_BLOCK_DOT_Q8_0_METAL

#include "dequant.metal"
#include "lane-fma-f32.metal"

// One native lane owns eight consecutive Q8_0 elements within a 32-weight block.
struct block_dot_q8_0_fragment {
    short element_offset; // first Q8_0 element owned by this lane
};

// Cached activation values match the scalar lane's eight sequential loads.
struct block_dot_q8_0_activation {
    float values[8];
};

// This adapter owns Q8_0 element ownership and scalar arithmetic; caller owns traversal and reduction.
struct block_dot_q8_0_scalar {
    enum : short {
        threads_per_block = 4,
        elements_per_lane = QK8_0/threads_per_block,
    };

    static inline block_dot_q8_0_fragment owned_fragment(short lane_in_block) {
        return { short(lane_in_block*elements_per_lane) };
    }

    // This adapter is faithful-only until a separately specified fast lane is added.
    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline block_dot_q8_0_activation prepare_activation(
            thread const block_dot_q8_0_fragment & owned,
            device const float * block_y,
            const short act_load = 4) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "only the fp32 lane exists");
        block_dot_q8_0_activation prepared;
        device const float * lane_y = block_y + owned.element_offset;

        if (act_load == 16) {
            // The same eight values as two 16-byte loads; lane_y is 16-byte aligned when the host passes 16.
            device const float4 * lane_y4 = (device const float4 *) lane_y;
            const float4 y0 = lane_y4[0];
            const float4 y1 = lane_y4[1];
            prepared.values[0] = y0.x; prepared.values[1] = y0.y; prepared.values[2] = y0.z; prepared.values[3] = y0.w;
            prepared.values[4] = y1.x; prepared.values[5] = y1.y; prepared.values[6] = y1.z; prepared.values[7] = y1.w;
        } else {
            for (short i = 0; i < elements_per_lane; ++i) {
                prepared.values[i] = lane_y[i];
            }
        }

        return prepared;
    }

    // Keep lane policy compile-time so the faithful body has no hot-loop route choice.
    template<short ARITHMETIC_LANE = block_dot_lane_fp32>
    static inline void accumulate(
            device const block_q8_0 * block,
            thread const block_dot_q8_0_fragment & owned,
            thread const block_dot_q8_0_activation & prepared,
            thread float & accumulator) {
        static_assert(ARITHMETIC_LANE == block_dot_lane_fp32, "only the fp32 lane exists");
        device const int8_t * quant = block->qs + owned.element_offset;

        float quant_dot = 0.f;

        FOR_UNROLL (short i = 0; i < elements_per_lane; ++i) {
            quant_dot += quant[i]*prepared.values[i];
        }

        accumulator += quant_dot*block->d;
    }

    // The fp32 lane of accumulate, same products in the same order: the lane's eight weights are read as three
    // 4-byte words at aligned addresses (base is 4-byte aligned; block_offset is the block's byte offset from it).
    static inline void accumulate_words(
            device const char * base,
            const uint64_t block_offset,
            thread const block_dot_q8_0_fragment & owned,
            thread const block_dot_q8_0_activation & prepared,
            thread float & accumulator) {
        const uint64_t qoff = block_offset + 2 + owned.element_offset;
        const uint off = (uint) (qoff & 3);
        device const uint * wp = (device const uint *) (base + (qoff - off));
        const uint w0 = wp[0];
        const uint w1 = wp[1];
        const uint w2 = wp[2];
        const char4 q0 = as_type<char4>(off != 0 ? ((w0 >> 16) | (w1 << 16)) : w0);
        const char4 q1 = as_type<char4>(off != 0 ? ((w1 >> 16) | (w2 << 16)) : w1);
        float quant_dot = 0.f;
        quant_dot += q0.x*prepared.values[0];
        quant_dot += q0.y*prepared.values[1];
        quant_dot += q0.z*prepared.values[2];
        quant_dot += q0.w*prepared.values[3];
        quant_dot += q1.x*prepared.values[4];
        quant_dot += q1.y*prepared.values[5];
        quant_dot += q1.z*prepared.values[6];
        quant_dot += q1.w*prepared.values[7];
        accumulator += quant_dot*((device const block_q8_0 *) (base + block_offset))->d;
    }
};

#endif
