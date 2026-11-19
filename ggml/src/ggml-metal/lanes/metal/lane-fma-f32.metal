#ifndef GGML_METAL_LANES_LANE_FMA_F32_METAL
#define GGML_METAL_LANES_LANE_FMA_F32_METAL

// Lane id a block-dot adapter takes as its template argument.
enum : short {
    block_dot_lane_fp32 = 0,
};

// Base lane: fp32 products and fp32 sums. Every route has it; a format keeps its own fp32 expression.
struct lane_fma_f32 {
    typedef float4 act4_t;
    static inline act4_t load(float4 x) { return x; }
    static inline float dot4(act4_t a, uchar4 q) { return dot(a, float4(q)); }
    static inline float dot4(act4_t a, char4 q) { return dot(a, float4(q)); }
    static inline float4 mul4(float4 a, float4 b) { return a*b; }
};

#endif
