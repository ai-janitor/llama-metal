#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Model profile: captures model dimensions that affect kernel tuning decisions.
// =============================================================================
//
// Parallel to ggml_device_profile (hardware characteristics), this struct
// captures model characteristics. The dispatch layer combines both profiles
// to select optimal kernel parameters (NR0, NSG, grouped dispatch thresholds).
//
// Why a separate module instead of reading tensor dimensions per-op?
//   - Tensor ne00 tells you the K dimension of ONE matmul, but not whether
//     this is an attention projection (ne00 = n_embd) or FFN down-projection
//     (ne00 = n_ff). The model profile provides the full picture.
//   - Some tuning decisions depend on model-wide properties (n_layer for
//     grouped dispatch sizing) that no single tensor carries.
//   - Surfacing these as a named struct with documented fields prevents
//     tuning logic from being buried in one-line dispatch hacks.
//
// Lifecycle:
//   1. Initialized to defaults (populated=false) at device init time.
//   2. Populated via ggml_backend_metal_set_model_profile() after GGUF load.
//   3. Read by dispatch code on every graph compute.
//   When populated=false, all tuning functions return hardcoded defaults —
//   zero regression risk for code paths that don't set a profile.

struct ggml_model_profile {
    // false until explicitly set by the backend API.
    // When false, all tuning functions fall back to hardcoded compile-time defaults.
    bool populated;

    // -------------------------------------------------------------------------
    // Core dimensions (from GGUF metadata)
    // -------------------------------------------------------------------------

    // Hidden dimension (GGUF: {arch}.embedding_length, hparams: n_embd).
    // Directly determines the activation vector size loaded into L1 per matvec:
    //   activation_bytes = n_embd * sizeof(float)
    // This is THE critical value for cache-aware tuning:
    //   - n_embd=1024 (0.6B) → 4KB activation → 25% of D700's 16KB L1 → high NR0 OK
    //   - n_embd=3584 (9B)   → 14KB activation → 87% of D700's 16KB L1 → lower NR0
    //   - n_embd=4096 (7B)   → 16KB activation → exactly fills D700's L1 → NR0=2
    uint32_t n_embd;

    // Feed-forward hidden dimension (GGUF: {arch}.feed_forward_length, hparams: n_ff).
    // FFN matmuls have different shapes than attention matmuls:
    //   - gate/up projections: [n_ff, n_embd] → ne00=n_embd, ne01=n_ff
    //   - down projection:     [n_embd, n_ff] → ne00=n_ff, ne01=n_embd
    // n_ff is typically 2.5-4x n_embd. The down projection has ne00=n_ff,
    // which may exceed L1 even when n_embd fits. Different NR0 may be optimal.
    uint32_t n_ff;

    // Number of transformer layers (GGUF: {arch}.block_count, hparams: n_layer).
    // Determines how many consecutive fusible matmul ops exist in the compute graph.
    // More layers = more opportunities for grouped dispatch fusion.
    // Also determines total matmuls per token: ~6*n_layer (Q/K/V/O + gate/up/down).
    uint32_t n_layer;

    // -------------------------------------------------------------------------
    // Attention geometry
    // -------------------------------------------------------------------------

    // Number of query attention heads (GGUF: {arch}.attention.head_count).
    // head_dim = n_embd / n_head. Affects attention kernel tile sizes and
    // whether attention matmuls (ne00=head_dim) qualify as "small K" for mul_mv_ext.
    uint32_t n_head;

    // Number of key-value attention heads (GGUF: {arch}.attention.head_count_kv).
    // GQA ratio = n_head / n_head_kv. Higher ratio means more src1 heads
    // share each src0 head (broadcast in the r2 dimension of mul_mv).
    uint32_t n_head_kv;

    // -------------------------------------------------------------------------
    // Derived quantities (computed at population time for convenience).
    // These save the dispatch layer from recomputing on every op.
    // -------------------------------------------------------------------------

    // activation_bytes_f32 = n_embd * sizeof(float).
    // The size of one activation vector in the common case (f32 src1 in mul_mv).
    // Compared directly against device_profile.l1_cache_per_cu to determine
    // cache pressure. This is the number that drives NR0 selection.
    uint32_t activation_bytes_f32;

    // l1_pressure_ratio = activation_bytes_f32 / l1_cache_per_cu.
    // 0.0 if L1 size is unknown (no hardware profile data).
    // Interpretation:
    //   < 0.3: activation comfortably fits L1 with room for weight data.
    //          Use high NR0 (8) for maximum activation reuse.
    //   0.3-0.6: moderate pressure. Baseline NR0 (4) is optimal.
    //   > 0.6: activation nearly fills L1, leaving little room for weights.
    //          Reduce NR0 (2) to minimize cache thrashing.
    //   > 1.0: activation exceeds L1. NR0=2 mandatory; consider tiling.
    float l1_pressure_ratio;
};

// Initialize with safe defaults. All fields zero, populated=false.
// Tuning functions will return hardcoded compile-time defaults.
static inline struct ggml_model_profile ggml_model_profile_default(void) {
    struct ggml_model_profile p;
    p.populated = false;
    p.n_embd = 0;
    p.n_ff = 0;
    p.n_layer = 0;
    p.n_head = 0;
    p.n_head_kv = 0;
    p.activation_bytes_f32 = 0;
    p.l1_pressure_ratio = 0.0f;
    return p;
}

// Populate from model hyperparameters + hardware L1 cache size.
// l1_cache_per_cu comes from the device profile; pass 0 if unknown.
static inline void ggml_model_profile_set(
        struct ggml_model_profile * p,
        uint32_t n_embd,
        uint32_t n_ff,
        uint32_t n_layer,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t l1_cache_per_cu) {
    p->n_embd    = n_embd;
    p->n_ff      = n_ff;
    p->n_layer   = n_layer;
    p->n_head    = n_head;
    p->n_head_kv = n_head_kv;

    // Derived: activation size for the common f32 case
    p->activation_bytes_f32 = n_embd * (uint32_t)sizeof(float);

    // Derived: how much of L1 does one activation vector fill?
    p->l1_pressure_ratio = (l1_cache_per_cu > 0)
        ? (float)p->activation_bytes_f32 / (float)l1_cache_per_cu
        : 0.0f;

    p->populated = true;
}

#ifdef __cplusplus
}
#endif
