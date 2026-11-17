#include "../ggml/src/ggml-metal/matmul/residual-alias-guard.h"

#include <cstdio>

static int check(const char * name, bool actual, bool expected) {
    if (actual == expected) {
        std::printf("PASS %s\n", name);
        return 0;
    }
    std::fprintf(stderr, "FAIL %s: got %d expected %d\n", name, actual, expected);
    return 1;
}

int main() {
    int failures = 0;
    const int output = 0, destination = 0, residual = 0, weights = 0, activation = 0;
    const auto safe = [&](const void * add_buffer, size_t add_offset, size_t add_size,
                          const void * residual_buffer, size_t residual_offset, size_t residual_size) {
        return ggml_metal_q4_k_residual_ranges_safe(
            &output, 32, 16, add_buffer, add_offset, add_size,
            residual_buffer, residual_offset, residual_size, &weights, 0, 16, &activation, 0, 16);
    };
    failures += check("disjoint_accept_control", safe(&destination, 64, 16, &residual, 96, 16), true);
#if defined(GGML_METAL_TEST_DISJOINT_ONLY)
    failures += check("exact_output_alias_old_red", safe(&output, 32, 16, &residual, 96, 16), true);
#else
    failures += check("exact_output_alias_accept", safe(&output, 32, 16, &residual, 96, 16), true);
#endif
    failures += check("partial_overlap_reject", safe(&output, 40, 16, &residual, 96, 16), false);
    failures += check("output_residual_overlap_reject", safe(&destination, 64, 16, &output, 32, 16), false);
    return failures == 0 ? 0 : 1;
}
