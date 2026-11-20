// Internal dispatch strategy declarations for matmul dispatch routing.
// Each function attempts one dispatch strategy and returns:
//   0  — not handled (caller should try next strategy)
//   1+ — dispatched N ops (1 for scalar/ext/tiled/gcn, N for grouped fusion)
#pragma once

#include "../ops/ggml-metal-ops-internal.h"
#include "formats.h"
#include "batch-route.h"
#include "matvec-plan.h"

#include <cstdlib>
#include <cstring>

// GGML_METAL_KQ_PREP_TEST=1: every Q4_K weight gets prepared headers and its single matvec reads them (test only).
static inline bool ggml_metal_kq_prep_test_requested() {
    const char * value = std::getenv("GGML_METAL_KQ_PREP_TEST");
    return value != nullptr && std::strcmp(value, "1") == 0;
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
static constexpr int NE11_MM_MIN = GGML_METAL_NE11_MM_MIN;

// Vendors the tiled kernel of a weight type is verified on.
static inline uint8_t ggml_metal_tiled_verified_vendors(ggml_type type) {
    uint8_t verified = GGML_METAL_VERIFIED_AMD;
    if (type != GGML_TYPE_MXFP4) {
        verified |= GGML_METAL_VERIFIED_INTEL; // f16 tiled is needed on Intel; mxfp4 tiled times out there
    }
    return verified;
}

// GGML_METAL_KQ_HW_ROWS=0 or GGML_METAL_KQ_EXT_ROWS=0: the card's batch route rows are not read.
static inline bool ggml_metal_batch_rows_off() {
    static const bool off = [] {
        const char * rows = std::getenv("GGML_METAL_KQ_HW_ROWS");
        const char * ext  = std::getenv("GGML_METAL_KQ_EXT_ROWS");
        return (rows != nullptr && std::strcmp(rows, "0") == 0) || (ext != nullptr && std::strcmp(ext, "0") == 0);
    }();
    return off;
}

// The batch route of a MUL_MAT op on this card. The small-batch and the tiled route both ask here.
static inline enum mv_batch_route ggml_metal_batch_route(ggml_metal_op_t ctx, int idx) {
    const ggml_tensor * op = ctx->node(idx);
    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
    const ggml_type type = op->src[0]->type;

    int n_rows = 0;
    const struct mv_batch_row * rows = ggml_metal_batch_rows_get(&n_rows);
    const struct mv_batch_row * row = ggml_metal_batch_rows_off() ? nullptr
        : ggml_metal_batch_row_in(rows, n_rows, profile->name, type);

    struct mv_batch_in in = {};
    in.fmt             = ggml_metal_mv_format(type);
    in.ext_from        = row ? row->ext_from : 0;
    in.tiled_from      = row ? row->tiled_from : 0;
    in.has_matrix_hw   = profile->has_matrix_hw;
    in.tiled_vendor_ok = (ggml_metal_tiled_verified_vendors(type) & ggml_metal_vendor_to_verified_bit(profile->vendor)) != 0;
    in.act_f32         = op->src[1]->type == GGML_TYPE_F32;
    in.trans0          = ggml_is_transposed(op->src[0]);
    in.trans1          = ggml_is_transposed(op->src[1]);
    in.k               = op->src[0]->ne[0];
    in.m               = op->src[0]->ne[1];
    in.n               = op->src[1]->ne[1];
    in.cols_takes      = in.n <= 4 && (ggml_metal_matvec_shape_row(ggml_metal_device_get_card(ctx->dev), 0, (int32_t) type, in.k, in.m, (int32_t) in.n, 1) >= 0 || ggml_metal_kq_cols_test(nullptr, nullptr) ||
                                           ggml_metal_matvec_forced_is_for(ggml_metal_matvec_forced_env(), 0, (int32_t) type, in.k, in.m, (int32_t) in.n));
    return ggml_metal_batch_route_decide(&in);
}

// Grouped q4_K/q6_K fusion — fuses 2-8 consecutive matvecs into one dispatch.
int dispatch_mul_mat_grouped(ggml_metal_op_t ctx, int idx);

// Small-batch ext kernels — efficient for batch sizes 2..8.
int dispatch_mul_mat_ext(ggml_metal_op_t ctx, int idx);

// Matrix-matrix multiplication — simdgroup_mm (Apple Silicon) and tiled (AMD/Intel).
int dispatch_mul_mat_matmul(ggml_metal_op_t ctx, int idx);
// Test hook GGML_METAL_TEST_TILED_PLAN (matmul/dispatch-matmul.cpp): a route other than the matmul route took nodes
// idx .. idx + n - 1; the mul_mat nodes among them that the hook applies to are counted under that route's name.
void ggml_metal_tiled_test_route(ggml_metal_op_t ctx, int idx, int n, const char * route);

// Scalar mul_mv — workhorse for single-token generation (batch=1).
int dispatch_mul_mat_scalar(ggml_metal_op_t ctx, int idx);

// Load width in bytes for this Q8_0 weight tensor: the card's preference (the plan's q8_load_pref) when 4-byte reads
// stay aligned and inside the tensor's Metal buffer, else 1. The same for the activation (q8_act_pref: 16 or 4).
int ggml_metal_q8_load_bytes(int pref, const ggml_tensor * src0);
int ggml_metal_q8_act_load_bytes(int pref, const ggml_tensor * src1);

// GGML_METAL_TEST_MV_RECEIPT=1, inert unless set: every matvec launch of the one-column and grouped routes prints one
// line to stderr: MV_RECEIPT attempt=<the column plan selected, cols:nr0:form:prep, or none> fallback=<none | tail: ... |
// width: ...> final=<kind:cols:nr0:form:prep:pipeline name> (test only).
static inline bool ggml_metal_mv_receipt_requested() {
    static const bool on = [] {
        const char * value = std::getenv("GGML_METAL_TEST_MV_RECEIPT");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return on;
}

// GCN fallback — f32/f16/bf16 on AMD GCN (64-wide wavefronts).
// Used when scalar mul_mv can't handle these types at NW=64.
int dispatch_mul_mat_gcn(ggml_metal_op_t ctx, int idx);
