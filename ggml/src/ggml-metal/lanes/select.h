#pragma once

#include "ggml.h"
#include "ggml-device-profile.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_metal_lane_source { GGML_METAL_LANE_SOURCE_METAL, GGML_METAL_LANE_SOURCE_LLVM_ARCHIVE, GGML_METAL_LANE_SOURCE_LLVM_NATIVE };
enum ggml_metal_lane_family { GGML_METAL_LANE_FAMILY_MATMUL, GGML_METAL_LANE_FAMILY_FLASH_ATTN };
enum ggml_metal_lane_route {
    GGML_METAL_LANE_ROUTE_MV_SCALAR,
    GGML_METAL_LANE_ROUTE_MV_GROUPED,
    GGML_METAL_LANE_ROUTE_MM_TILED,
    GGML_METAL_LANE_ROUTE_MM_MATRIX,
    GGML_METAL_LANE_ROUTE_FA_SCALAR,
    GGML_METAL_LANE_ROUTE_FA_VEC,
    GGML_METAL_LANE_ROUTE_FA_MATRIX,
    GGML_METAL_LANE_ROUTE_FA_INTEL_TILED,
};
// What a lane does to the values: nothing, rounds the product, or rounds an operand.
enum ggml_metal_lane_class { GGML_METAL_LANE_CLASS_EXACT, GGML_METAL_LANE_CLASS_PRODUCT_ROUNDED, GGML_METAL_LANE_CLASS_NARROWS };
enum ggml_metal_lane_origin { GGML_METAL_LANE_ORIGIN_OURS, GGML_METAL_LANE_ORIGIN_UPSTREAM };
// How a format holds a tensor type, in the order the negotiation prefers them; NARROWER is never picked.
enum ggml_metal_lane_holds {
    GGML_METAL_LANE_HOLDS_NATIVE = 1,
    GGML_METAL_LANE_HOLDS_EXACT_OTHER,
    GGML_METAL_LANE_HOLDS_WIDER,
    GGML_METAL_LANE_HOLDS_NARROWER,
};

// One row of lanes/lanes.tsv. weight / activation GGML_TYPE_COUNT = any type.
struct ggml_metal_lane_row {
    const char * lane;
    int          source;
    const char * file;
    int          family;
    int          route;
    int          weight;
    int          activation;
    int          operand;    // format the lane holds an activation in while it forms products
    int          holds;      // holds column as written; for activation any the request decides
    const char * entry;      // suffix of the kernel entry point, "" for the base lane
    int          capability; // enum ggml_metal_lane_cap, -1 = needs none
    int          cls;
    int          origin;
    int          on;         // default column: 1 = may be picked
    int          seam_us;    // cost of crossing to another submit route, 0 = none
    int          id;         // value of the pipeline lane constant
    const char * reason;
    const char * test;
};

// One row of lanes/capability.tsv: 1 yes, 0 no, -1 unknown.
struct ggml_metal_lane_cap_row {
    uint32_t pci_vendor;
    uint32_t pci_device;
    int      capability;
    int      in_hardware;
    int      reached;
    int      correct;
};

const struct ggml_metal_lane_row     * ggml_metal_lane_rows_get(int * n);
const struct ggml_metal_lane_cap_row * ggml_metal_lane_cap_rows_get(int * n);

// Bit i set = capability i is reached and correct on this card. A card with no rows gets 0.
uint32_t ggml_metal_lane_ok_mask(uint32_t pci_vendor, uint32_t pci_device);

// How operand holds tensor: equal = native (exact-other when the row declares it), f32 over f16 / bf16 = wider, else narrower.
int ggml_metal_lane_holds(int operand, enum ggml_type tensor, int declared);

// Negotiates the activation type of the request against the card: of the rows of (family, route) that are on, have no
// seam, cover the operand types and whose capability the card has, the one that holds the activation best (native,
// exact-other, wider; table order within a step). A row narrower than the tensor is dropped unless upstream ships the
// route that way. NULL = no path: a coverage gap. holds (may be NULL) gets the step of the picked row.
const struct ggml_metal_lane_row * ggml_metal_negotiate_lane_in(
        const struct ggml_metal_lane_row * rows, int n, uint32_t lane_ok,
        int family, int route, enum ggml_type weight, enum ggml_type activation, int * holds);

// The same without the step.
const struct ggml_metal_lane_row * ggml_metal_select_lane_in(
        const struct ggml_metal_lane_row * rows, int n, uint32_t lane_ok,
        int family, int route, enum ggml_type weight, enum ggml_type activation);

// The same over the built-in table; lane constant of the picked row, 0 when none.
int16_t ggml_metal_select_lane_id(
        const struct ggml_device_profile * hw,
        int family, int route, enum ggml_type weight, enum ggml_type activation);

#ifdef __cplusplus
}
#endif
