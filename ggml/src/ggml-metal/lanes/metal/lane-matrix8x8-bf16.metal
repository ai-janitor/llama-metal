#ifndef GGML_METAL_LANES_LANE_MATRIX8X8_BF16_METAL
#define GGML_METAL_LANES_LANE_MATRIX8X8_BF16_METAL

// Matrix unit lane staged in bfloat, as upstream ships it for bf16 data (shaders with native bfloat only).
#define MM_STAGE_BF16 bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8

// Matrix flash attention: query, key, value in bfloat, attention scores and scratch in float, output in half.
#define FA_STAGE_BF16 \
    bfloat, bfloat4,   simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    float,             simdgroup_float8x8,  \
    float,  float2,    simdgroup_float8x8,  \
    half,   half4,     simdgroup_half8x8

#endif
