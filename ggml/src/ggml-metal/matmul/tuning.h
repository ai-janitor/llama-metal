#pragma once

// =============================================================================
// Kernel tuning functions: combine hardware + model profiles into parameters.
// =============================================================================
//
// These pure functions replace the hardcoded N_R0_*/N_SG_* constants with
// dynamic values computed from the device's cache hierarchy and the model's
// dimensions. Each function documents the rationale for its decision tree.
//
// NOTE: NR0 is currently a template parameter compiled into the scalar kernels
// (e.g., kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>) and a constexpr in the grouped
// kernel. To use dynamic NR0, we need to JIT-compile kernel variants with different
// template args — this is Phase 2 work. For now, these functions compute the
// optimal NR0 but it can only be applied when the pipeline system supports
// multiple NR0 variants per quant type.
//
// The model profile infrastructure (struct, API, population) is in place so that
// when Phase 2 lands, the tuning decisions are ready to use.
//
// When the model profile is not populated (e.g., during model loading),
// these functions return the existing hardcoded defaults for backward
// compatibility. Zero regression risk.
//
// =============================================================================
// Architecture note: why runtime if/else, not C++ templates or inheritance
// =============================================================================
//
// Alternative considered: template specialization per SIMD width, e.g.
//   template<uint32_t SIMD_W> struct KernelTuning;
//   template<> struct KernelTuning<64> { ... };  // GCN (D700)
//   template<> struct KernelTuning<32> { ... };  // RDNA (5500M)
//
// This would give compile-time polymorphism and let the compiler inline/fold
// per-target. Clean architecturally.
//
// Also considered: C++ inheritance with virtual dispatch:
//   struct DeviceProfile { virtual int optimal_nr0(...) = 0; };
//   struct GCN_Profile : DeviceProfile { ... };
//
// Both rejected for these reasons:
//
// 1. The tuning logic is ~10 lines of if/else on 1-2 float thresholds.
//    Template specialization adds structural overhead (3 struct definitions,
//    a dispatch mechanism) for a problem that isn't complex enough to need it.
//    Two GPU families with one threshold each doesn't warrant an abstraction.
//
// 2. These headers are shared with the ObjC layer (ggml-metal-device.m) via
//    extern "C" structs. C++ templates and vtables don't cross that boundary.
//    The flat C struct + inline function approach works everywhere.
//
// 3. The real bottleneck is NOT host-side tuning organization — it's getting
//    the computed NR0 into the Metal kernel. The scalar kernels use
//    template<int nr0> where nr0 is a compile-time constant (N_R0_KQUANT=4).
//    The grouped kernel solved this with Metal function constants (FC_grouped_nr0).
//    The 15 scalar kernels still need the same treatment (Phase 2 / Task D).
//    Refactoring the host-side if/else into templates doesn't move this needle.
//
// 4. If we grow beyond 2-3 GPU families, revisit. But for GCN + RDNA + Apple,
//    the if/else is more readable and debuggable than template specializations.
//
// Bottom line: the right abstraction investment is Metal function constants
// for all kernel types, not C++ template gymnastics on the host side.
// =============================================================================

#include "ggml-device-profile.h"
#include "ggml-model-profile.h"
#include "ggml.h"
#include "../ggml-metal-impl.h"

// Compute optimal NR0 (rows per simdgroup) for a given matmul operation.
//
// NR0 controls activation data reuse in the L1 cache:
//   - Each simdgroup loads the input vector (activation) once and multiplies
//     it against NR0 different weight rows.
//   - Higher NR0 = better L1 reuse (amortize activation load over more rows).
//   - But higher NR0 = more accumulator registers (sumf[NR0]), which can
//     reduce wavefront occupancy on register-limited GPUs.
//   - And higher NR0 = the activation must stay in L1 for NR0 iterations,
//     competing with weight data for cache lines.
//
// The key relationship: as activation_bytes approaches L1_size, NR0 should
// decrease because there's less room for weight data alongside the activation.
//
// Parameters:
//   hw        — hardware profile (L1 cache size, compute units, etc.)
//   model     — model profile (n_embd, l1_pressure_ratio, etc.) — may be NULL
//   ne00      — K dimension of THIS specific matmul (may be n_embd or n_ff)
//
// Returns: optimal NR0 value (2, 4, or 8 for k-quant types)
static inline int ggml_metal_compute_nr0(
        const struct ggml_device_profile * hw,
        const struct ggml_model_profile  * model,
        int ne00) {

    // ---- Fallback: no model profile or no L1 data → hardcoded default ----
    // This preserves exact current behavior for:
    //   - Apple Silicon (L1 unknown in hw_table)
    //   - Any code path that doesn't set a model profile
    //   - Intel GPUs (L1 unknown)
    if (!model || !model->populated || hw->l1_cache_per_cu == 0) {
        return N_R0_KQUANT;  // current default: 4
    }

    // ---- Dynamic NR0 based on L1 pressure ----
    // Use ne00 (the actual K dimension of this matmul) rather than model->n_embd,
    // because FFN down-projections have ne00=n_ff which can be much larger.
    const uint32_t act_bytes = (uint32_t)ne00 * (uint32_t)sizeof(float);
    const float pressure = (float)act_bytes / (float)hw->l1_cache_per_cu;

    // Decision thresholds (tuned on D700 with 16KB L1):
    //
    //   pressure < 0.3 (act < 5KB on D700):
    //     Small model (0.6B, n_embd=1024, 4KB). Plenty of L1 room.
    //     NR0=8: each activation load serves 8 dot products.
    //     Register pressure: 8 float accumulators = 32 bytes, negligible.
    //
    //   pressure >= 0.3 (act >= 5KB on D700):
    //     Medium to large model. NR0=4 proven optimal on D700 across 3B-9B range.
    //     Even at pressure=1.0 (9B, activation=L1), NR0=4 beats NR0=2 because
    //     the activation reuse benefit (4 dot products per load) outweighs the
    //     cache pressure cost. NR0=2 was tested and regressed on 9B (6.87 vs 7.18).
    if (pressure < 0.3f) {
        return 8;
    }
    return 4;
}

// Compute optimal NR0 for basic quant types (q4_0, q4_1, q5_0, q5_1).
// These use a different baseline NR0 than k-quants (currently 4 for both,
// but basic types have different register pressure characteristics).
// For now, delegates to the same logic as k-quants.
static inline int ggml_metal_compute_nr0_basic(
        const struct ggml_device_profile * hw,
        const struct ggml_model_profile  * model,
        int ne00) {
    return ggml_metal_compute_nr0(hw, model, ne00);
}

// =============================================================================
// Device-gated mat-vec tuning: nsg and nr0 per quant type
// =============================================================================
//
// Returns {nsg, nr0} for a given quant type based on hardware traits.
// This replaces the global N_R0_*/N_SG_* macros with device-aware selection.
//
// Historical classification (not used for the shared template geometry below):
//   GCN legacy: vendor=AMD, simd_width=64, no matrix_hw (D700, W9100, etc.)
//   RDNA:       vendor=AMD, simd_width=32, no matrix_hw (5500M, 5700XT, etc.)
//   Apple:      vendor=Apple, has matrix_hw (M1+)
//   Other:      safe defaults
//
// Validated on D700 (GCN) — 2026-03-29.
// Other devices require re-benchmarking before adopting non-default values.

struct mv_tuning {
    int nsg;  // simdgroups per threadgroup
    int nr0;  // rows per simdgroup
};

// Static template geometry must be the same number on the host and in Metal.
// UNTIED: these conservative values are named once in ggml-metal-impl.h.
static inline struct mv_tuning ggml_metal_get_mv_tuning(
        const struct ggml_device_profile * hw, enum ggml_type type) {
    (void) hw;
    switch (type) {
        case GGML_TYPE_Q4_0: return mv_tuning{N_SG_Q4_0, N_R0_Q4_0};
        case GGML_TYPE_Q4_1: return mv_tuning{N_SG_Q4_1, N_R0_Q4_1};
        case GGML_TYPE_Q5_0: return mv_tuning{N_SG_Q5_0, N_R0_Q5_0};
        case GGML_TYPE_Q5_1: return mv_tuning{N_SG_Q5_1, N_R0_Q5_1};
        case GGML_TYPE_Q8_0: return mv_tuning{N_SG_Q8_0, N_R0_Q8_0};
        case GGML_TYPE_Q2_K: return mv_tuning{N_SG_Q2_K, N_R0_Q2_K};
        case GGML_TYPE_Q3_K: return mv_tuning{N_SG_Q3_K, N_R0_Q3_K};
        case GGML_TYPE_Q4_K: return mv_tuning{N_SG_Q4_K, N_R0_Q4_K};
        case GGML_TYPE_Q5_K: return mv_tuning{N_SG_Q5_K, N_R0_Q5_K};
        case GGML_TYPE_Q6_K: return mv_tuning{N_SG_Q6_K, N_R0_Q6_K};
        case GGML_TYPE_IQ1_S: return mv_tuning{N_SG_IQ1_S, N_R0_IQ1_S};
        case GGML_TYPE_IQ1_M: return mv_tuning{N_SG_IQ1_M, N_R0_IQ1_M};
        case GGML_TYPE_IQ2_XXS: return mv_tuning{N_SG_IQ2_XXS, N_R0_IQ2_XXS};
        case GGML_TYPE_IQ2_XS: return mv_tuning{N_SG_IQ2_XS, N_R0_IQ2_XS};
        case GGML_TYPE_IQ2_S: return mv_tuning{N_SG_IQ2_S, N_R0_IQ2_S};
        case GGML_TYPE_IQ3_XXS: return mv_tuning{N_SG_IQ3_XXS, N_R0_IQ3_XXS};
        case GGML_TYPE_IQ3_S: return mv_tuning{N_SG_IQ3_S, N_R0_IQ3_S};
        case GGML_TYPE_IQ4_NL: return mv_tuning{N_SG_IQ4_NL, N_R0_IQ4_NL};
        case GGML_TYPE_IQ4_XS: return mv_tuning{N_SG_IQ4_XS, N_R0_IQ4_XS};
        case GGML_TYPE_MXFP4: return mv_tuning{N_SG_MXFP4, N_R0_MXFP4};
        default: return mv_tuning{N_SG_Q8_0, N_R0_Q8_0};
    }
}
