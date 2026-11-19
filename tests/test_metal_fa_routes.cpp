// Flash attention route table check, host only: no GPU, no model.
// For every combination of the six facts and three head sizes the table must give the route the if-chain it replaced gave.
#include "flash-attn/routes.h"

#include <cstdio>
#include <cstring>

// The chain as it was in the encoder: Intel tiled when it has a kernel for the head size (64 only), then scalar, matrix, vec.
static int reference(bool has_matrix, bool intel, bool width32, bool small, bool kv_same, bool fp8, int head) {
    if (!has_matrix && intel && head % 16 == 0 && kv_same && !fp8 && head == 64) {
        return FA_ROUTE_INTEL_TILED;
    }
    if (!has_matrix && (intel || !width32 || !small || !kv_same || fp8)) {
        return FA_ROUTE_SCALAR;
    }
    return small ? FA_ROUTE_VEC : FA_ROUTE_MATRIX;
}

// The walk of the router: the first matching row, or -1.
static int table(bool has_matrix, bool intel, bool width32, bool small, bool kv_same, bool fp8, int head) {
    for (const struct fa_route_row & row : fa_route_rows) {
        if (fa_route_row_matches(&row, has_matrix, intel, width32, small, kv_same, fp8, head)) {
            return row.route;
        }
    }
    return -1;
}

// No argument: the main run, which must have 0 failures. "control <name>": a table changed on purpose, which must fail.
int main(int argc, char ** argv) {
    const bool control = argc > 2 && std::strcmp(argv[1], "control") == 0;
    int cases = 0, failures = 0;
    const int heads[] = { 64, 128, 40 };
    for (int head : heads) {
        for (int bits = 0; bits < 64; ++bits) {
            const bool has_matrix = bits & 1, intel = bits & 2, width32 = bits & 4, small = bits & 8, kv_same = bits & 16, fp8 = bits & 32;
            cases++;
            if (reference(has_matrix, intel, width32, small, kv_same, fp8, head) != table(has_matrix, intel, width32, small, kv_same, fp8, head)) {
                if (!control) {
                    std::printf("FAIL matrix=%d intel=%d width32=%d small=%d kv_same=%d fp8=%d head=%d\n", has_matrix, intel, width32, small, kv_same, fp8, head);
                }
                failures++;
            }
        }
    }
    if (control) {
        std::printf("CONTROL %s (expected to fail): %d failures%s\n", argv[2], failures, failures == 0 ? " -- NOT CAUGHT" : "");
        return failures > 0 ? 0 : 1;
    }
    std::printf("MAIN fa routes: %d rows, %d cases, %d failures\n", (int) (sizeof(fa_route_rows) / sizeof(fa_route_rows[0])), cases, failures);
    return failures == 0 ? 0 : 1;
}
