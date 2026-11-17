// Internal dispatch strategy declarations for matmul dispatch routing.
// Each function attempts one dispatch strategy and returns:
//   0  — not handled (caller should try next strategy)
//   1+ — dispatched N ops (1 for scalar/ext/tiled/gcn, N for grouped fusion)
#pragma once

#include "../ops/ggml-metal-ops-internal.h"

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

// GCN fallback — f32/f16/bf16 on AMD GCN (64-wide wavefronts).
// Used when scalar mul_mv can't handle these types at NW=64.
int dispatch_mul_mat_gcn(ggml_metal_op_t ctx, int idx);
