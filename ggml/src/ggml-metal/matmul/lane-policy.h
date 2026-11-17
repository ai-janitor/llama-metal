#pragma once

#include "../ggml-metal-device.h"

// These names describe the numerical multiplication lane, not a kernel variant.
enum ggml_metal_matvec_lane : int16_t {
    GGML_METAL_MATVEC_LANE_FP32 = 0,
    GGML_METAL_MATVEC_LANE_PACKED_FP16 = 1,
};

enum ggml_metal_matvec_route : int16_t {
    GGML_METAL_MATVEC_ROUTE_SCALAR = 0,
    GGML_METAL_MATVEC_ROUTE_GROUPED = 1,
};

// Selects a compile-time lane. This faithful unit has no enabled fast rows.
int16_t ggml_metal_select_matvec_lane(
        ggml_metal_library_t lib,
        enum ggml_type format,
        enum ggml_metal_matvec_route route);
