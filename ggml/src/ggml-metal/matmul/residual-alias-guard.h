#pragma once

#include <cstddef>
#include <limits>

// Exact output reuse is safe for the scalar residual epilogue; partial overlap
// remains unsafe because a write can replace a source before its final read.
static inline bool ggml_metal_ranges_exact(
        const void * a_buffer, size_t a_offset, size_t a_size,
        const void * b_buffer, size_t b_offset, size_t b_size) {
    return a_buffer && b_buffer && a_size != 0 && b_size != 0 &&
        a_size <= std::numeric_limits<size_t>::max() - a_offset &&
        b_size <= std::numeric_limits<size_t>::max() - b_offset &&
        a_buffer == b_buffer && a_offset == b_offset && a_size == b_size;
}

// The only permitted overlap is the ADD destination exactly reusing the
// MUL_MAT output. All kernel inputs remain disjoint from that destination.
static inline bool ggml_metal_q4_k_residual_ranges_safe(
        const void * op_buffer, size_t op_offset, size_t op_size,
        const void * add_buffer, size_t add_offset, size_t add_size,
        const void * residual_buffer, size_t residual_offset, size_t residual_size,
        const void * weights_buffer, size_t weights_offset, size_t weights_size,
        const void * activation_buffer, size_t activation_offset, size_t activation_size) {
    const auto disjoint = [](const void * a_buffer, size_t a_offset, size_t a_size,
                             const void * b_buffer, size_t b_offset, size_t b_size) {
        if (!a_buffer || !b_buffer || a_size == 0 || b_size == 0 ||
            a_size > std::numeric_limits<size_t>::max() - a_offset ||
            b_size > std::numeric_limits<size_t>::max() - b_offset) {
            return false;
        }
        return a_buffer != b_buffer ||
            a_offset + a_size <= b_offset || b_offset + b_size <= a_offset;
    };
    return disjoint(op_buffer, op_offset, op_size, residual_buffer, residual_offset, residual_size) &&
        (disjoint(op_buffer, op_offset, op_size, add_buffer, add_offset, add_size)
#if !defined(GGML_METAL_TEST_DISJOINT_ONLY)
         || ggml_metal_ranges_exact(op_buffer, op_offset, op_size, add_buffer, add_offset, add_size)
#endif
        ) &&
        disjoint(residual_buffer, residual_offset, residual_size, add_buffer, add_offset, add_size) &&
        disjoint(add_buffer, add_offset, add_size, weights_buffer, weights_offset, weights_size) &&
        disjoint(add_buffer, add_offset, add_size, activation_buffer, activation_offset, activation_size);
}
