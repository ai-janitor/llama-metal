// Internal dispatch strategy declarations for matmul dispatch routing.
// Each function attempts one dispatch strategy and returns:
//   0  — not handled (caller should try next strategy)
//   1+ — dispatched N ops (1 for scalar/ext/tiled/gcn, N for grouped fusion)
#pragma once

#include "../ops/ggml-metal-ops-internal.h"

#include <cstdlib>
#include <cstring>

// The first per-shape matvec row of this card that applies to a K-quant weight of this kind (0 single, 1 group
// member), type, shape and activation column count; a prep row applies only when prep_ok (prepared headers exist).
static inline const struct ggml_device_mv_row * ggml_metal_mv_hw_row(
        const struct ggml_device_profile * profile, int kind, ggml_type type, int64_t ne00, int64_t ne01, int cols, bool prep_ok) {
    if (cols > 1 && profile->kq_cols != 1) {
        return nullptr;
    }
    for (uint32_t i = 0; i < profile->n_mv_rows; i++) {
        const struct ggml_device_mv_row * row = &profile->mv_rows[i];
        if (row->kind == kind && row->type == (int32_t) type && row->ne00 == ne00 && row->ne01 == ne01 && row->cols == cols && (row->prep != 1 || prep_ok)) {
            return row;
        }
    }
    return nullptr;
}

// GGML_METAL_KQ_PREP_TEST=1: every Q4_K weight gets prepared headers and its single matvec reads them (test only).
static inline bool ggml_metal_kq_prep_test_requested() {
    const char * value = std::getenv("GGML_METAL_KQ_PREP_TEST");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// Column count from which this K-quant type takes mul_mv_ext on this card (hw rows); no row = 4.
static inline int ggml_metal_kq_ext_min(const struct ggml_device_profile * profile, ggml_type type) {
    for (uint32_t i = 0; i < profile->n_mv_ext_min; i++) {
        if (profile->mv_ext_min[i].type == (int32_t) type) {
            return profile->mv_ext_min[i].ext_min;
        }
    }
    return 4;
}

// GGML_METAL_KQ_COLS_TEST=<rows><form> (two digits, e.g. 20; form 4 or 6 = paired halves, rows 2, 4 or 8): every 2-4 column
// K-quant batch takes the N-column kernel (test only).
static inline bool ggml_metal_kq_cols_test(int * nr0, int * form) {
    const char * value = std::getenv("GGML_METAL_KQ_COLS_TEST");
    if (value == nullptr || std::strlen(value) != 2) {
        return false;
    }
    const int r = value[0] - '0';
    const int f = value[1] - '0';
    if ((r != 1 && r != 2 && r != 4 && r != 8) || f < 0 || (f > 3 && f != 4 && f != 6) || (f > 3 && r == 1)) {
        return false;
    }
    if (nr0 != nullptr) {
        *nr0 = r;
        *form = f;
    }
    return true;
}

// Break-even batch size where matmul kernels become more efficient than matvec.
// Used by ext (ne11 <= threshold) and matmul (ne11 > threshold) strategies.
static constexpr int NE11_MM_MIN = 8;

// P17.14: the ONE admission predicate for the opt-in Q4_K packed-fp16 tiled prefill route
// (GGML_METAL_TILED_FP16_MUL exact "1", read per call): Q4_K weights, AMD, no matrix hw, f32 src1,
// both operands non-transposed, ne01 >= 64, ne11 > NE11_MM_MIN. Shared by the ext strategy (which
// must step aside, since its K-quant predicate would otherwise take long AMD Q4_K prompts first)
// and the tiled strategy (V3 kernel selection) so the two can never disagree. Defined in
// dispatch-matmul.cpp. False when the flag is off: nothing else changes.
bool ggml_metal_tiled_fp16_mul_admitted(ggml_metal_op_t ctx, int idx);

// Grouped q4_K/q6_K fusion — fuses 2-8 consecutive matvecs into one dispatch.
int dispatch_mul_mat_grouped(ggml_metal_op_t ctx, int idx);

// Small-batch ext kernels — efficient for batch sizes 2..8.
int dispatch_mul_mat_ext(ggml_metal_op_t ctx, int idx);

// Matrix-matrix multiplication — simdgroup_mm (Apple Silicon) and tiled (AMD/Intel).
int dispatch_mul_mat_matmul(ggml_metal_op_t ctx, int idx);

// Scalar mul_mv — workhorse for single-token generation (batch=1).
int dispatch_mul_mat_scalar(ggml_metal_op_t ctx, int idx);

// Load width in bytes for this Q8_0 weight tensor: the hw row's value when 4-byte reads stay aligned and inside
// the tensor's Metal buffer, else 1.
int ggml_metal_q8_load_bytes(const struct ggml_device_profile * hw, const ggml_tensor * src0);
int ggml_metal_q8_act_load_bytes(const struct ggml_device_profile * hw, const ggml_tensor * src1);

// GCN fallback — f32/f16/bf16 on AMD GCN (64-wide wavefronts).
// Used when scalar mul_mv can't handle these types at NW=64.
int dispatch_mul_mat_gcn(ggml_metal_op_t ctx, int idx);
