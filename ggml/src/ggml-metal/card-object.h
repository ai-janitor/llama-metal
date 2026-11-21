// The card object: what a kernel's shape is chosen from. This comment is its definition; nothing else goes in.
//
//   Card object (one per GPU, built at startup)
//   |
//   +-- Facts (read from the hardware through Metal)
//   |     +-- wave width
//   |     +-- shared memory per group
//   |     +-- level-1 cache per compute unit (the card's row of data/device.tsv; not known without one)
//   |
//   +-- Tuning (measured on this card, beats the default)
//         +-- tile shape of the tiled matmul, per weight type and column count (when measured)
//         +-- matvec settings (data/matmul.tsv [matvec_defaults]: the default rows, then the card's own row)
//         +-- matvec rows per weight type and shape (data/matmul.tsv [matvec_shapes], the card's rows)
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
// Rules that read it: ggml_metal_tiled_plan_for (matmul/tiled-plan.cpp), ggml_metal_matvec_plan_for (matmul/matvec-plan.cpp).
#pragma once

#include <stdint.h>

#include "ggml-device-profile.h" // struct ggml_device_mv_row: one row of [matvec_shapes]

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_METAL_CARD_TILED_N 11   // fields of the tiled matmul work plan
#define GGML_METAL_CARD_TILED_MAX 32 // tuning entries one card may carry for it

// One measured tuning entry: for weight type `weight` (a ggml_type value, or -1 = every type) this card runs the
// compiled plan of row `plan` of the compiled plans (a whole plan, never one field of it), for multiplies of `cols`
// activation columns (0 = every column count).
struct ggml_metal_card_tiled_entry {
    int32_t weight;
    int32_t plan;
    int32_t cols;
};

struct ggml_metal_card {
    // Threads of one wave. Source: threadExecutionWidth of a probe pipeline, read once after the library is built.
    // wave_probed = 1 once that read succeeded; until then wave holds a placeholder no tuned row may be matched against.
    uint32_t wave;
    uint32_t wave_probed;
    // Bytes of threadgroup memory one group may declare. Source: MTLDevice maxThreadgroupMemoryLength.
    uint32_t group_mem;
    // Measured tuning of this card for the tiled matmul work plan: an entry names the compiled plan to run for one
    // weight type or for every type, at one column count or at every count; an entry for the type wins over one for
    // every type, and inside each an entry for the op's column count wins over one for every count.
    // Source of every entry: its tuned row of data/matmul.tsv [tiles], which cites the measurement.
    // Loaded by ggml_metal_tiled_tune_load (matmul/tiled-plan.cpp).
    struct ggml_metal_card_tiled_entry tiled_tune[GGML_METAL_CARD_TILED_MAX];
    int32_t n_tiled_tune;

    // Level-1 data cache per compute unit, bytes; 0 = not known. Source: the card's row of data/device.tsv [devices].
    // Read by the matvec cache rule.
    uint32_t l1_per_cu;

    // Matvec settings of this card. Source of every cell: data/matmul.tsv [matvec_defaults], in this order, a written
    // cell replacing the earlier value: the default row, the default row of the card's wave (once probed), the card's
    // own row (while the card reports the wave and threadgroup memory written in it). Loaded by
    // ggml_metal_matvec_tune_load (matmul/matvec-plan.cpp); the header lines of the section say what each cell does.
    uint32_t mv_sum;            // a single matvec sums a row with the wave's own reduction (0) or through shared memory (1)
    uint32_t mv_reduce;         // shared-memory row sum: 0 serial, 1 tree
    uint32_t f_scalar_w64;      // f16 and bf16 scalar matvec at wave width 64: 1 on
    uint32_t mv_group;          // Q8_0 matvecs of one activation in one launch: 1 on
    uint32_t mv_group_red;      // grouped K-quant matvec stores every row before the row sums: 1 on
    uint32_t q8_load_bytes;     // Q8_0 weight load width the card prefers: 1 or 4
    uint32_t q8_act_load_bytes; // Q8_0 activation load width the card prefers: 4 or 16
    uint32_t q8_rows8_min;      // smallest row count of a single Q8_0 matvec run at 8 whole rows per simdgroup, 0 = never
    float    l1_share_cut;      // the cache rule: share of l1_per_cu one activation column may take for mv_rows_below
    uint32_t mv_rows_below;     // rows per simdgroup under the cut
    uint32_t mv_rows_from;      // rows per simdgroup from the cut
    uint32_t mv_rows_unknown;   // rows per simdgroup without a model profile or an L1 fact
    uint32_t f_nsg_max;         // dense float matvec: most simdgroups per threadgroup
    uint32_t f_k_per_sg;        // dense float matvec: input columns per simdgroup

    // The card's rows of data/matmul.tsv [matvec_shapes] (rows per simdgroup, columns, loop form, prepared headers
    // per weight type and shape), each used only while the card reports the wave and threadgroup memory written in
    // it; none = the settings above. kq_prep 1 = a usable row asks for prepared Q4_K block headers; kq_cols 1 = a
    // usable row is for 2 to 4 columns. Loaded by ggml_metal_matvec_tune_load.
    const struct ggml_device_mv_row * mv_rows;
    uint32_t n_mv_rows;
    uint32_t kq_prep;
    uint32_t kq_cols;

    // The family of data/device.tsv [families] the card is a member of, or "". Its rows are candidates for a sweep
    // and are never applied.
    char family[24];
};

#ifdef __cplusplus
}
#endif
