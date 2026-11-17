// Match Metal fusion patterns and retain their allocator inputs.
#pragma once
#include "ggml-impl.h"
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
#define GGML_METAL_FUSION_MAX 16
typedef enum ggml_metal_fusion_mode {
    GGML_METAL_FUSION_STRUCTURAL = 0,
    GGML_METAL_FUSION_FULL,
} ggml_metal_fusion_mode;
typedef enum ggml_metal_fusion_id {
    GGML_METAL_FUSION_NONE = 0,
    GGML_METAL_FUSION_NORM_SCALE,
    GGML_METAL_FUSION_GDN_CACHE,
    GGML_METAL_FUSION_SSM_CONV_SILU,
    GGML_METAL_FUSION_CONV_STATE,
    GGML_METAL_FUSION_GATE_G,
    GGML_METAL_FUSION_GATED_RMS_NORM,
    GGML_METAL_FUSION_SPLIT_L2,
} ggml_metal_fusion_id;
bool ggml_metal_fuse_small_enabled(void);
struct ggml_metal_fusion;
typedef struct ggml_metal_fusion ggml_metal_fusion;
ggml_metal_fusion_id ggml_metal_fusion_get_id(const ggml_metal_fusion * fusion);
const ggml_metal_fusion * ggml_metal_fusion_next(const struct ggml_cgraph * gf, const int * node_idxs,
        int n_idxs, int idx, ggml_metal_fusion_mode mode, int * n_out);
int ggml_metal_fusion_max(const struct ggml_cgraph * gf, int idx);
void ggml_metal_fusion_add_alloc_deps(void * user_data,
        void (*add_alloc_dep)(void *, struct ggml_tensor *, struct ggml_tensor *), struct ggml_cgraph * gf);
#ifdef __cplusplus
}
#endif
