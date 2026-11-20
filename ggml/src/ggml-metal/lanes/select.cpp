#include "select.h"

#include "lanes-table.h"

#include <cstring>

const struct ggml_metal_lane_row * ggml_metal_lane_rows_get(int * n) {
    *n = (int) (sizeof(ggml_metal_lane_rows) / sizeof(ggml_metal_lane_rows[0]));
    return ggml_metal_lane_rows;
}

const struct ggml_metal_lane_cap_row * ggml_metal_lane_cap_rows_get(int * n) {
    *n = (int) (sizeof(ggml_metal_lane_cap_rows) / sizeof(ggml_metal_lane_cap_rows[0]));
    return ggml_metal_lane_cap_rows;
}

uint32_t ggml_metal_lane_ok_mask(const char * card) {
    uint32_t mask = 0;
    if (card == nullptr) {
        return mask;
    }
    for (const auto & row : ggml_metal_lane_cap_rows) {
        if (std::strcmp(row.card, card) == 0 && row.reached == 1 && row.correct == 1) {
            mask |= 1u << row.capability;
        }
    }
    return mask;
}

int ggml_metal_lane_holds(int operand, enum ggml_type tensor, int declared) {
    if (operand == (int) tensor) {
        return declared == GGML_METAL_LANE_HOLDS_EXACT_OTHER ? GGML_METAL_LANE_HOLDS_EXACT_OTHER : GGML_METAL_LANE_HOLDS_NATIVE;
    }
    if (operand == GGML_TYPE_F32 && (tensor == GGML_TYPE_F16 || tensor == GGML_TYPE_BF16)) {
        return GGML_METAL_LANE_HOLDS_WIDER;
    }
    return GGML_METAL_LANE_HOLDS_NARROWER;
}

const struct ggml_metal_lane_row * ggml_metal_negotiate_lane_in(
        const struct ggml_metal_lane_row * rows, int n, uint32_t lane_ok,
        int family, int route, enum ggml_type weight, enum ggml_type activation, int * holds) {
    const struct ggml_metal_lane_row * best = nullptr;
    int best_step = GGML_METAL_LANE_HOLDS_NARROWER + 1;
    for (int i = 0; i < n; ++i) {
        const struct ggml_metal_lane_row & row = rows[i];
        if (row.family != family || row.route != route) continue;
        if (row.weight != GGML_TYPE_COUNT && row.weight != (int) weight) continue;
        if (row.activation != GGML_TYPE_COUNT && row.activation != (int) activation) continue;
        if (!row.on || row.seam_us != 0) continue;
        if (row.capability >= 0 && !(lane_ok & (1u << row.capability))) continue;
        const int step = ggml_metal_lane_holds(row.operand, activation, row.holds);
        if (step == GGML_METAL_LANE_HOLDS_NARROWER && row.origin != GGML_METAL_LANE_ORIGIN_UPSTREAM) continue;
        if (step < best_step) {
            best = &row;
            best_step = step;
        }
    }
    if (holds && best) *holds = best_step;
    return best;
}

const struct ggml_metal_lane_row * ggml_metal_select_lane_in(
        const struct ggml_metal_lane_row * rows, int n, uint32_t lane_ok,
        int family, int route, enum ggml_type weight, enum ggml_type activation) {
    return ggml_metal_negotiate_lane_in(rows, n, lane_ok, family, route, weight, activation, nullptr);
}

int16_t ggml_metal_select_lane_id(
        const struct ggml_device_profile * hw,
        int family, int route, enum ggml_type weight, enum ggml_type activation) {
    int n = 0;
    const struct ggml_metal_lane_row * rows = ggml_metal_lane_rows_get(&n);
    const struct ggml_metal_lane_row * row = ggml_metal_select_lane_in(rows, n, hw ? hw->lane_ok : 0, family, route, weight, activation);
    return row ? (int16_t) row->id : 0;
}
