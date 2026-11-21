// The launch numbers of one kernel launch on one card: data/device.tsv [launches], whose header lines say what each
// number is. A number is 0 when the launch has none (its cell is - in every row that applies).
#pragma once

#include <stdint.h>

#include "ggml-device-profile.h"
#include "launch-ids.h" // generated: enum ggml_metal_launch_id

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_launch {
    int32_t start;     // threads of a threadgroup before it grows
    int32_t work_min;  // the amount of work up to which it grows
    int32_t cap;       // the most threads the launch asks for
    int32_t nsg;       // simdgroups per threadgroup
    int32_t units;     // work units per threadgroup
    int32_t mem_units; // threadgroup memory in 4-byte units
};

// One row of a section [launches], as the generator writes it. card "" = any; wave -1 = any. In the number cells
// -1 = the cell is -, -2 = the cell is wave. A table of rows ends with a row of id -1.
struct ggml_metal_launch_row {
    int32_t      id;
    const char * card;
    int32_t      wave;
    int32_t      start;
    int32_t      work_min;
    int32_t      cap;
    int32_t      nsg;
    int32_t      units;
    int32_t      mem_units;
};

// The numbers of launch id in rows on the card named card with a wave of wave threads: the default row, then the row
// of the wave, then the card's own row at that wave, a written cell replacing the earlier value. A cell that says
// wave gives wave. card may be null; an id with no row gives zeros.
struct ggml_metal_launch ggml_metal_launch_pick_rows(const struct ggml_metal_launch_row * rows, int id, const char * card, int32_t wave);

// The same over data/device.tsv [launches].
struct ggml_metal_launch ggml_metal_launch_pick(int id, const char * card, int32_t wave);

// The same for the card of a device profile.
static inline struct ggml_metal_launch ggml_metal_launch_of(const struct ggml_device_profile * profile, int id) {
    return ggml_metal_launch_pick(id, profile->name, (int32_t) profile->simd_width);
}

#ifdef __cplusplus
}
#endif
