// The card object: what a kernel's shape is chosen from. This comment is its definition; nothing else goes in.
//
//   Card object (one per GPU, built at startup)
//   |
//   +-- Facts (read from the hardware through Metal)
//   |     +-- wave width
//   |     +-- shared memory per group
//   |
//   +-- Tuning (measured on this card, beats the default)
//         +-- tile shape of the tiled matmul, per weight type (when measured)
//                   |
//                   v
//           Rule (one, for all cards): reads the object, picks the kernel's shape;
//           a tuning entry wins where present
//                   |
//                   v
//           Kernel shape for this card: group width, tile sizes, sums per thread, shared memory used
//                   |
//                   v
//           Kernel compiled once with that shape
//                   |
//                   v
//           Check: the shape's threads <= the compiled kernel's thread limit
//           (the limit belongs to one compiled kernel, not to the card; when it fails
//           the route does not take the op)
//
//   Layer  | What goes in                      | Who fills it                 | How it changes
//   Facts  | what the hardware is              | Metal, when the device is set up | a new card brings its own
//   Tuning | what was measured to work better  | a table row citing the measurement (date, what was measured) | a new row per measurement
//   Rule   | how facts become a shape          | written once                 | edited only when its logic is wrong
//   Shape  | the numbers the kernel uses       | the rule                     | never by hand
//
// A fact or a tuning entry is added only in the commit of the rule that reads it, marked with its source beside it.
// Built in device/init.m; handed out by ggml_metal_device_get_card (ggml-metal-device.h).
// Rules that read it: ggml_metal_tiled_plan_for (matmul/tiled-plan.cpp).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_METAL_CARD_TILED_N 11   // fields of the tiled matmul work plan
#define GGML_METAL_CARD_TILED_MAX 32 // tuning entries one card may carry for it

// One measured tuning entry: for weight type `weight` (a ggml_type value, or -1 = every type) this card runs the
// compiled plan of row `plan` of the compiled plans (a whole plan, never one field of it).
struct ggml_metal_card_tiled_entry {
    int32_t weight;
    int32_t plan;
};

struct ggml_metal_card {
    // Threads of one wave. Source: threadExecutionWidth of a probe pipeline, read once after the library is built.
    // wave_probed = 1 once that read succeeded; until then wave holds a placeholder no tuned row may be matched against.
    uint32_t wave;
    uint32_t wave_probed;
    // Bytes of threadgroup memory one group may declare. Source: MTLDevice maxThreadgroupMemoryLength.
    uint32_t group_mem;
    // Measured tuning of this card for the tiled matmul work plan: an entry names the compiled plan to run for one
    // weight type or for every type; an entry for the type wins over one for every type.
    // Source of every entry: its tuned row of data/matmul.tsv [tiles], which cites the measurement.
    // Loaded by ggml_metal_tiled_tune_load (matmul/tiled-plan.cpp).
    struct ggml_metal_card_tiled_entry tiled_tune[GGML_METAL_CARD_TILED_MAX];
    int32_t n_tiled_tune;
};

#ifdef __cplusplus
}
#endif
