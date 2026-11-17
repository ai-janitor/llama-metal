// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_cgraph;

enum ggml_mem_range_type {
    MEM_RANGE_TYPE_SRC = 0,
    MEM_RANGE_TYPE_DST = 1,
};

// a helper object that can be used for reordering operations to improve concurrency
//
// the fundamental idea is that a set of tasks (either ggml ops, or something else) can run concurrently if they
//   don't write to a memory that is being read by another task or written to by another task in the set
//
// with this structure, we can add tasks to the set, setting memory constraints. we can also check if a new task
//   can be added to the set without violating the constraints (i.e. if it can be executed concurrently with the
//   tasks already in the set)
//
typedef struct ggml_mem_ranges * ggml_mem_ranges_t;

ggml_mem_ranges_t ggml_mem_ranges_init(int debug);
void ggml_mem_ranges_free(ggml_mem_ranges_t mrs);

// remove all ranges from the set
void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs);

// add src or dst ranges to track
bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// return false if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// reorder the nodes in the graph to improve concurrency, while respecting fusion
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
//       pack_matvecs keeps the Q8_0 matvecs that read one normed activation adjacent, for the grouped launch
//       attn_gate keeps the attention gate copy, sigmoid, output copy and multiply adjacent, for their one launch
void ggml_graph_optimize(struct ggml_cgraph * gf, bool pack_matvecs, bool attn_gate);

// Fusion GGML_METAL_NORM_MATVEC (off unless set): rms_norm + its weight + 1-4 Q8_0 matvecs in one launch, and the
// residual add that leaves the mean of squares of its output in a device-owned slot for that norm.
// debug values: pack = optimizer packing only, nm-self = fused launch with every norm computing its own mean,
// 1 or full = fused launch and residual adds storing the means; 0 or unset = off
int  ggml_metal_norm_matvec_mode(void);
bool ggml_metal_norm_matvec_enabled(void);
bool ggml_metal_norm_matvec_add_scale(void);
// GGML_METAL_STATE_IN_PLACE (on unless 0): decode conv reads and shifts its state in the cache cell (one launch)
bool ggml_metal_state_in_place(void);
bool ggml_metal_add_scale_applies(const struct ggml_tensor * add);
// Gate copy, its sigmoid, attention output copy and their multiply: the match fills the three inner nodes.
bool ggml_metal_attn_gate_match(const struct ggml_tensor * mul, const struct ggml_tensor ** cont_gate,
                                const struct ggml_tensor ** sigmoid, const struct ggml_tensor ** cont_out);
bool ggml_metal_norm_matvec_mm_ok(const struct ggml_tensor * mm, const struct ggml_tensor * act);

#ifdef __cplusplus
}
#endif
