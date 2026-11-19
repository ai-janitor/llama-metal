#pragma once

// Whether an op converts its activations once to the model's precision. The rows are act-copy/act-copy.tsv.
#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One row of act-copy/act-copy.tsv. family / route are the lane enums of lanes/select.h.
struct ggml_metal_act_copy_row {
    int          family;
    int          route;
    uint32_t     pci_vendor;
    uint32_t     pci_device;
    int          copy;        // ggml_type of the copy
    int          applies;
    int          min_rereads;
    uint64_t     max_bytes;
    const char * reason;
    const char * test;
};

// Why an op got a copy or did not, in the order the checks run.
enum ggml_metal_act_copy_why {
    GGML_METAL_ACT_COPY_APPLIES,
    GGML_METAL_ACT_COPY_NO_PRECISION, // the model's precision is not known, or is f32
    GGML_METAL_ACT_COPY_NOT_F32,      // the activation tensor is not f32
    GGML_METAL_ACT_COPY_NO_ROW,       // no row for this family, route, card and precision
    GGML_METAL_ACT_COPY_ROW_OFF,
    GGML_METAL_ACT_COPY_FEW_REREADS,
    GGML_METAL_ACT_COPY_TOO_BIG,
};

const struct ggml_metal_act_copy_row * ggml_metal_act_copy_rows_get(int * n);

// The row that decides for one op and, in why, the step that decided. NULL when no row matches.
// precision GGML_TYPE_COUNT = not known. The copy is never narrower than the model's precision: a row is
// matched on copy == precision only.
const struct ggml_metal_act_copy_row * ggml_metal_act_copy_decide_in(
        const struct ggml_metal_act_copy_row * rows, int n, uint32_t pci_vendor, uint32_t pci_device,
        int family, int route, enum ggml_type precision, enum ggml_type src, int64_t rereads, uint64_t bytes, int * why);

const char * ggml_metal_act_copy_why_name(int why);

#ifdef __cplusplus
}
#endif
