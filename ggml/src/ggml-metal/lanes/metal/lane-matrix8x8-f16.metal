#ifndef GGML_METAL_LANES_LANE_MATRIX8X8_F16_METAL
#define GGML_METAL_LANES_LANE_MATRIX8X8_F16_METAL

// Matrix unit lane staged in half, as upstream ships it: the types a kernel holds its operands in while it forms products.
// Matrix matmul: weight element, 4x4 load, 8x8 tile, then the same for the activation (2x4 load).
#define MM_STAGE_F16 half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8

// Matrix flash attention: query, key, value, attention scores, scratch, output.
#define FA_STAGE_F16 \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8

// Vec flash attention (no matrix unit): query, key, value as half4, sums in float.
#define FA_VEC_STAGE_F16 \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#endif
