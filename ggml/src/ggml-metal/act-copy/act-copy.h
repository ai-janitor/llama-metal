#pragma once

// Convert-once: an op that reads its activations more than once asks here for a copy at the model's precision.
// One scratch buffer per backend context holds the copy of one source at a time; no family keeps its own.
#include "decide.h"
#include "ggml-metal-ops.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_act_copy;

struct ggml_metal_act_copy * ggml_metal_act_copy_init(void);
void ggml_metal_act_copy_free(struct ggml_metal_act_copy * st);

// The model's precision; GGML_TYPE_COUNT (the default) = not known = no copy anywhere.
void ggml_metal_act_copy_set_precision(struct ggml_metal_act_copy * st, enum ggml_type type);

// Sizes the scratch buffer for a graph. Called once per graph, before any op is encoded.
void ggml_metal_act_copy_reserve(struct ggml_metal_act_copy * st, ggml_metal_device_t dev, struct ggml_cgraph * gf);

#ifdef __cplusplus
}

// The source whose copy the scratch buffer holds, per encoding context.
struct ggml_metal_act_slot {
    bool     valid;
    void *   metal;
    size_t   offs;
    size_t   span;
    int64_t  ne[4];
    size_t   nb[4];
    int      type;
};

// What was decided for one op. applies = 0: read the source as it is.
struct ggml_metal_act_plan {
    int                                    applies;
    int                                    why;       // enum ggml_metal_act_copy_why
    enum ggml_type                         type;      // element type of the copy
    struct ggml_metal_buffer_id            bid;       // the copy: contiguous, same shape as the source
    uint64_t                               nb[4];     // strides of the copy
    const struct ggml_metal_act_copy_row * row;       // the row that decided, NULL when none matched
    const struct ggml_metal_lane_row     * lane;      // arithmetic lane for the copy's type
    int                                    lane_step; // enum ggml_metal_lane_holds
};

// Decides for the op at idx; encodes nothing. rereads = how often the op's kernel reads each activation.
struct ggml_metal_act_plan ggml_metal_act_copy_plan(ggml_metal_op_t ctx, int idx, int family, int route, const struct ggml_tensor * src, int64_t rereads);

// Encodes the conversion of src into the scratch buffer unless it already holds it. Call only when plan.applies,
// before the op's own kernel is encoded.
void ggml_metal_act_copy_encode(ggml_metal_op_t ctx, int idx, const struct ggml_metal_act_plan * plan, const struct ggml_tensor * src);

// A node was encoded: the held copy is dropped when the node writes over its source.
void ggml_metal_act_copy_note_write(ggml_metal_op_t ctx, const struct ggml_tensor * node);
#endif
