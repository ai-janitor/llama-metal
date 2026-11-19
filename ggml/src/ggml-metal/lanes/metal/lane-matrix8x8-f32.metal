#ifndef GGML_METAL_LANES_LANE_MATRIX8X8_F32_METAL
#define GGML_METAL_LANES_LANE_MATRIX8X8_F32_METAL

// Flash attention with f32 keys and values, as upstream ships it: the query is staged in half, key and value in float.
#define FA_STAGE_F32 \
    half,   half4,     simdgroup_half8x8,  \
    float,  float4x4,  simdgroup_float8x8, \
    float,  float4x4,  simdgroup_float8x8, \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8

// Vec flash attention (no matrix unit): query as half4, key and value as float4.
#define FA_VEC_STAGE_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

#endif
