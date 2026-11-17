#ifndef GGML_METAL_MATMUL_BLOCK_DOT_LANE_METAL
#define GGML_METAL_MATMUL_BLOCK_DOT_LANE_METAL

// Named enum values are compile-time arguments without MSL program-scope storage.
enum : short {
    block_dot_lane_fp32 = 0,
    block_dot_lane_packed_fp16 = 1,
};

// Shared host policy binding; current adapters retain explicit fp32 literals.
constant short FC_matvec_lane [[function_constant(FC_MATVEC_LANE)]];

// Experimental true-half dot for one four-quant group (V3 form). Each quant is
// rebuilt as an exact half by placing it in the mantissa of binary16 1024.0
// (0x6400 | q = 1024 + q) and subtracting 1024 in half BEFORE the multiply; this
// is exact for 0 <= q < 1024, and the Q6_K caller supplies 0..63. Every
// activation*quant product is then formed and rounded at half precision, widened
// to float, and only then added: there is no half FMA, half add or half partial
// sum, and the operands are not widened before multiplying. The fp32 caller owns
// scales, centering and row accumulation. Dispatch is dormant: the host lane policy
// still selects block_dot_lane_fp32 and every caller instantiates it explicitly.
static inline float block_dot_packed_half_dot4(
        thread const half4 & activation,
        uchar4 quant) {
    constexpr ushort half_1024_bits = 0x6400; // binary16 1024.0; mantissa bits 0..9 hold q
    const half quant_bias = half(1024.0f); // exact subtrahend that removes the placed bias
    const half quant_x = as_type<half>(ushort(half_1024_bits | ushort(quant.x))) - quant_bias; // exact q
    const half quant_y = as_type<half>(ushort(half_1024_bits | ushort(quant.y))) - quant_bias;
    const half quant_z = as_type<half>(ushort(half_1024_bits | ushort(quant.z))) - quant_bias;
    const half quant_w = as_type<half>(ushort(half_1024_bits | ushort(quant.w))) - quant_bias;
    const half product_x = activation.x * quant_x; // half product, rounded to half
    const half product_y = activation.y * quant_y;
    const half product_z = activation.z * quant_z;
    const half product_w = activation.w * quant_w;
    float sum = float(product_x); // widen each product, then sum in float only
    sum += float(product_y);
    sum += float(product_z);
    sum += float(product_w);
    return sum;
}

// Q8_0 preserves signed char4 quants. Bias each signed value in integer space,
// materialize 1024 + (q + 128), then subtract half 1152 before multiplication.
// This leaves the frozen unsigned helper unchanged for its existing callers.
static inline float block_dot_packed_half_dot4(
        thread const half4 & activation,
        char4 quant) {
    constexpr ushort half_1024_bits = 0x6400;
    const half signed_bias = half(1152.0f);
    const ushort quant_x_bits = ushort(short(quant.x) + 128);
    const ushort quant_y_bits = ushort(short(quant.y) + 128);
    const ushort quant_z_bits = ushort(short(quant.z) + 128);
    const ushort quant_w_bits = ushort(short(quant.w) + 128);
    const half quant_x = as_type<half>(ushort(half_1024_bits | quant_x_bits)) - signed_bias;
    const half quant_y = as_type<half>(ushort(half_1024_bits | quant_y_bits)) - signed_bias;
    const half quant_z = as_type<half>(ushort(half_1024_bits | quant_z_bits)) - signed_bias;
    const half quant_w = as_type<half>(ushort(half_1024_bits | quant_w_bits)) - signed_bias;
    const half product_x = activation.x * quant_x;
    const half product_y = activation.y * quant_y;
    const half product_z = activation.z * quant_z;
    const half product_w = activation.w * quant_w;
    float sum = float(product_x);
    sum += float(product_y);
    sum += float(product_z);
    sum += float(product_w);
    return sum;
}

#endif
