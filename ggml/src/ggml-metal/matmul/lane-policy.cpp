#include "lane-policy.h"

#include "../ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>

namespace {

struct lane_format_support {
    enum ggml_type format;
    bool scalar_packed_fp16;
    bool grouped_packed_fp16;
};

// Capability belongs in data: adding a format changes a row, not selection logic.
constexpr lane_format_support k_lane_format_support[] = {
    { GGML_TYPE_Q2_K, false, false }, { GGML_TYPE_Q3_K, false, false },
    { GGML_TYPE_Q4_K, false, false }, { GGML_TYPE_Q5_K, false, false },
    { GGML_TYPE_Q6_K, false, false }, { GGML_TYPE_Q4_0, false, false },
    { GGML_TYPE_Q4_1, false, false }, { GGML_TYPE_Q5_0, true, false },
    { GGML_TYPE_Q5_1, false, false }, { GGML_TYPE_Q8_0, false, false },
};

enum requested_policy { requested_faithful, requested_fast };

requested_policy get_requested_policy() {
    static const requested_policy policy = [] {
        const char * value = std::getenv("GGML_METAL_MATVEC_LANE");
        if (!value || value[0] == '\0' || strcmp(value, "0") == 0) return requested_faithful;
        if (strcmp(value, "1") == 0) return requested_fast;
        GGML_LOG_WARN("ggml-metal: GGML_METAL_MATVEC_LANE='%s' is invalid; using faithful lane 0\n", value);
        return requested_faithful;
    }();
    return policy;
}

void log_fast_fallback_once(ggml_metal_library_t lib, enum ggml_type format,
                            enum ggml_metal_matvec_route route, const char * reason) {
    static std::mutex mutex;
    static std::set<std::string> emitted;
    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(lib)) + ":" +
        std::to_string((int) format) + ":" + std::to_string((int) route) + ":1:" + reason;
    std::lock_guard<std::mutex> lock(mutex);
    if (emitted.insert(key).second) {
        GGML_LOG_WARN("ggml-metal: requested packed-fp16 matvec lane falls back to faithful lane 0 (%s, format %d, route %d)\n",
                      reason, (int) format, (int) route);
    }
}

} // namespace

int16_t ggml_metal_select_matvec_lane(ggml_metal_library_t lib, enum ggml_type format,
                                      enum ggml_metal_matvec_route route) {
    if (get_requested_policy() == requested_faithful) return GGML_METAL_MATVEC_LANE_FP32;

    const struct ggml_device_profile * profile = ggml_metal_library_get_device_profile(lib);
    const bool device_eligible = profile && profile->vendor == GGML_GPU_VENDOR_AMD &&
        profile->simd_width == 32 && !profile->has_matrix_hw;
    const lane_format_support * support = nullptr;
    for (const auto & row : k_lane_format_support) {
        if (row.format == format) { support = &row; break; }
    }
    const bool route_supported = support && (route == GGML_METAL_MATVEC_ROUTE_SCALAR
        ? support->scalar_packed_fp16 : support->grouped_packed_fp16);
    if (device_eligible && route_supported) return GGML_METAL_MATVEC_LANE_PACKED_FP16;

    log_fast_fallback_once(lib, format, route,
        !support ? "format has no lane-policy row" : !device_eligible ? "device is not eligible" : "route has no enabled fast implementation");
    return GGML_METAL_MATVEC_LANE_FP32;
}
