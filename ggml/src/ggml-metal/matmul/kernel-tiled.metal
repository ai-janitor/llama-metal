#include "dequant.metal"

// Tiled matmul kernel for non-simdgroup_mm GPUs (AMD Radeon, Intel UHD)
//
// Template design (REFACTOR-007):
//   The kernel body is shared across quant types. Only the A-matrix loading
//   differs per type (pointer type, stride semantics, K-tracking, dequant).
//   Three options were considered:
//
//   Option A: Functor — encapsulate all 5 divergence points behind one
//     load_a interface. Clean but requires designing a new uniform signature
//     across types with fundamentally different memory layouts (typed vs byte
//     pointers, element vs block K-tracking).
//
//   Option B: Reuse dequantize_*_t4 — adapt the tiled kernel to call the
//     existing per-element dequant functions. Doesn't work because the tiled
//     divergence is cooperative loading (threads coordinate to fill shared
//     memory), not per-element decompression.
//
//   Option C (chosen): Keep existing load_a functions unchanged, wrap each
//     in a loader struct with init/load/advance methods, template the kernel
//     body on the struct. Simplest — the load functions already exist and are
//     tested. The compiler inlines everything, producing identical machine
//     code to the hand-duplicated kernels.
//
// Work plans: the kernel body (kernel-tiled-body.metal) is compiled once per row of data/matmul.tsv [tiles], each
// in its own namespace with the plan's eleven values as compile-time constants; the blocks are generated
// (tiled-plans.metal, by matmul/gen-tiles.cmake) and never written here. An entry of a plan is named
// kernel_mul_mat_tiled_<type>_lay_<the eleven values>. The host rule (matmul/tiled-plan.cpp) picks the plan for a card.
// The field of the same name in struct ggml_metal_tiled_plan (matmul/tiled-plan.h) is the authority on a value's meaning.
//
// Uses flat tid (thread_index_in_threadgroup) for thread identity - no simd intrinsics.

#include "tiled-plans.metal"
