// The launch numbers of the matmul launches that are rows of data/matmul.tsv [launches]; the section's header lines
// say what each launch reads.
#pragma once

#include "device/launch-plan.h"
#include "mv-launch-ids.h" // generated: enum ggml_metal_mv_launch_id

#ifdef __cplusplus
extern "C" {
#endif

// The numbers of launch id on a card: see ggml_metal_launch_pick_rows.
struct ggml_metal_launch ggml_metal_mv_launch_pick(int id, const char * card, int32_t wave);

static inline struct ggml_metal_launch ggml_metal_mv_launch_of(const struct ggml_device_profile * profile, int id) {
    return ggml_metal_mv_launch_pick(id, profile->name, (int32_t) profile->simd_width);
}

// Rows of the weight one threadgroup of the small-batch matvec works: nsg simdgroups of width threads, k_threads of
// them along K. 0 = the kernel compiled at this width cannot run that K width (it does not divide the width).
static inline int ggml_metal_mv_ext_rows(int width, int k_threads, int nsg) {
    if (width <= 0 || k_threads <= 0 || nsg <= 0 || width % k_threads != 0) {
        return 0;
    }
    return (width / k_threads)*nsg;
}

#ifdef __cplusplus
}
#endif
