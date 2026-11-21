// Work plan of the tiled matmul kernel (matmul/kernel-tiled.metal): how much work one group takes (tile sizes, threads,
// sums per thread, shared memory), picked by one rule from the card object. First slice of the work planner.
// How a plan reaches the kernel: as compile-time constants. The kernel body is compiled once per row of
// data/matmul.tsv [tiles] (the build generates a namespace block per row and the host's list from that table); an
// entry of a plan is named <entry>_lay_<the values>. (Function constants were measured 2.2x slower on a Vega 64,
// 2026-11-20.) dispatch-matmul.cpp uses the same struct for the name, the grid, the thread count and the group memory.
// A card runs the default plan (row 0) unless a row of data/matmul.tsv [tiles] names another compiled plan for it.
// To add a plan: one row in data/matmul.tsv [tiles] (its header says how).
#pragma once

#include "card-object.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_METAL_TILED_PLAN_N GGML_METAL_CARD_TILED_N

struct ggml_metal_tiled_plan {
    // 1 = the kernel can run with this layout on this card; 0 = it cannot, and nothing else in the struct is used.
    int32_t ok;
    // plan-fields-begin (one int32_t per line; this order is the order of the values everywhere)
    int32_t warp;    // threads of one warp tile; divides threads
    int32_t threads; // threads per threadgroup; a whole number of waves of the card
    int32_t bm;      // output rows per threadgroup tile
    int32_t bn;      // output columns per threadgroup tile
    int32_t bk;      // K step; 32, the block size the weight loaders are written for
    int32_t wm;      // output rows per warp tile
    int32_t wn;      // output columns per warp tile
    int32_t wmiter;  // row passes per warp tile
    int32_t tm;      // rows of sums per thread per pass; even
    int32_t tn;      // columns of sums per thread per pass
    int32_t pad;     // float2 slots added to each buffer row
    // plan-fields-end
    // Bytes of threadgroup memory the two buffers take: (bm + bn) * (bk / 2 + pad) * 8.
    int32_t group_mem_bytes;
    // The row of data/matmul.tsv [tiles] this plan is (0 = default), or -1 for values checked outside the table.
    int32_t row;
    // 1 = the plan selected for this card and weight type cannot run on it, and this is the default plan instead.
    int32_t from_default;
};

// One row of data/matmul.tsv [tiles]: the card with this Metal device name runs compiled plan `plan` (a row of
// data/matmul.tsv [tiles]) for weight type `weight` (a ggml_type value, -1 = every type), for multiplies of `cols`
// activation columns (0 = every column count).
struct ggml_metal_tiled_tune_row {
    const char * card;
    int32_t      weight;
    int32_t      plan;
    int32_t      cols;
    uint32_t     wave;        // the wave width and group memory the card reported when the row was measured
    uint32_t     group_bytes;
};

// Loads the rows of data/matmul.tsv [tiles] whose card is exactly this Metal device name into the object (none first;
// NULL or "" loads none).
void ggml_metal_tiled_tune_load(struct ggml_metal_card * card, const char * device_name);

// The checks: can the kernel run these eleven values (field order) on this card? Bounds, the kernel's own arithmetic
// (the same conditions matmul/gen-tiles.cmake applies to every table row), whole waves, group memory, and
// thread_limit (a built pipeline's maxTotalThreadsPerThreadgroup, or 0 = not known yet).
struct ggml_metal_tiled_plan ggml_metal_tiled_plan_check(const struct ggml_metal_card * card, const int32_t values[GGML_METAL_TILED_PLAN_N], uint32_t thread_limit);

// The rule: the compiled plan of the first entry of the card that exists, in this order: for this weight type at this
// column count; for the type at every count; for every type at this count; for every type at every count; else the
// default plan. Then the checks. weight is the ggml_type value of the op's weights, cols its activation column count
// (an entry for one count is that count only: 63 and 65 columns do not read the entry of 64).
struct ggml_metal_tiled_plan ggml_metal_tiled_plan_for(const struct ggml_metal_card * card, int32_t weight, int64_t cols, uint32_t thread_limit);

// The plan the tiled route runs: the rule's plan (or, with forced_row >= 0, that row of the compiled table instead of
// the card's selection, whatever the column count) when the card's facts admit it, else the default plan with from_default = 1, so a selection
// never drops the op off the tiled route. The thread limit is checked on the plan chosen. ok = 0 when neither can run.
struct ggml_metal_tiled_plan ggml_metal_tiled_plan_to_run(const struct ggml_metal_card * card, int32_t weight, int64_t cols, uint32_t thread_limit, int32_t forced_row);

// The compiled plans (rows of data/matmul.tsv [tiles]): how many; row i as its values in field order and its id
// (NULL outside); the row of an id, or -1.
int ggml_metal_tiled_compiled_count(void);
const int32_t * ggml_metal_tiled_compiled_row(int i);
const char * ggml_metal_tiled_compiled_id(int i);
int ggml_metal_tiled_compiled_find(const char * id);

// Test hook GGML_METAL_TEST_TILED_PLAN (defined in matmul/dispatch-matmul.cpp, where the hook is described): reads the
// hook once (a bad value aborts) and, when it is set, makes the process print its receipt and check it when it ends.
void ggml_metal_tiled_test_arm(void);

// The fields between the markers, in order.
void ggml_metal_tiled_plan_values(const struct ggml_metal_tiled_plan * lay, int32_t out[GGML_METAL_TILED_PLAN_N]);

#ifdef __cplusplus
}
#endif
