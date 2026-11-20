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
#include "formats.h"

// Rows per simdgroup from the card's L1 and the activation size (the cache rule) are part of the matvec work plan:
// ggml_metal_matvec_plan_for (matmul/matvec-plan.cpp).

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

// Float and q8_0 matvec kernels take nr0 rows per threadgroup; every other format takes nr0 rows per simdgroup.
static inline bool ggml_metal_mv_rows_per_threadgroup(enum ggml_type type) {
    const struct mv_format_row * row = ggml_metal_mv_format(type);
    return row && row->rows_per_threadgroup;
}

struct mv_tuning {
    int nsg;  // simdgroups per threadgroup
    int nr0;  // rows per simdgroup
};

// Static template geometry must be the same number on the host and in Metal.
// UNTIED: these conservative values are named once in ggml-metal-impl.h.
static inline struct mv_tuning ggml_metal_get_mv_tuning(
        const struct ggml_device_profile * hw, enum ggml_type type) {
    (void) hw;
    const struct mv_format_row * row = ggml_metal_mv_format(type);
    if (row && row->nsg > 0) {
        return mv_tuning{row->nsg, row->nr0};
    }
    return mv_tuning{N_SG_Q8_0, N_R0_Q8_0};
}
