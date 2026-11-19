#include "fusion/ggml-metal-fusion-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(fusion) static bool ggml_metal_fusion_small_rows(const ggml_tensor * tensor) {
    if (!tensor || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(tensor)) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] <= 0 || tensor->ne[d] > INT32_MAX) {
            return false;
        }
    }
    return ggml_nrows(tensor) <= INT32_MAX;
}

static bool ggml_metal_fusion_small_vector(const ggml_tensor * tensor, int64_t width) {
    return ggml_metal_fusion_small_rows(tensor) && ggml_is_contiguous(tensor) &&
           tensor->ne[0] == width && ggml_nrows(tensor) == 1;
}

bool ggml_metal_fusion_small_subgraph(const ggml_metal_fusion * fusion,
        const ggml_cgraph * gf, const int * node_idxs, int idx) {
    int outputs[GGML_METAL_FUSION_MAX];
    outputs[0] = node_idxs[idx + fusion->ops.size() - 1];
    for (size_t j = 0; j < fusion->outs.size(); ++j) {
        outputs[j + 1] = node_idxs[idx + fusion->outs[j]];
    }
    return ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, fusion->ops.size(),
            fusion->ops.data(), outputs, 1 + fusion->outs.size());
}

bool ggml_metal_fusion_check_gate_g(const ggml_metal_fusion * fusion,
        const ggml_tensor * const * nodes, const ggml_cgraph * gf,
        const int * node_idxs, int idx, ggml_metal_fusion_mode mode) {
    GGML_UNUSED(fusion); GGML_UNUSED(gf); GGML_UNUSED(node_idxs); GGML_UNUSED(idx); GGML_UNUSED(mode);
    const ggml_tensor * add = nodes[0];
    const ggml_tensor * un = nodes[1];
    const ggml_tensor * mul = nodes[2];
    return un->src[0] == add && ggml_get_unary_op(un) == GGML_UNARY_OP_SOFTPLUS &&
           mul->src[0] == un && ggml_metal_fusion_small_rows(add->src[0]) &&
           ggml_metal_fusion_small_rows(add) && ggml_metal_fusion_small_rows(un) &&
           ggml_metal_fusion_small_rows(mul) &&
           ggml_metal_fusion_small_vector(add->src[1], add->ne[0]) &&
           ggml_metal_fusion_small_vector(mul->src[1], add->ne[0]);
}

bool ggml_metal_fusion_check_gated_rms_norm(const ggml_metal_fusion * fusion,
        const ggml_tensor * const * nodes, const ggml_cgraph * gf,
        const int * node_idxs, int idx, ggml_metal_fusion_mode mode) {
    GGML_UNUSED(mode);
    const ggml_tensor * norm = nodes[0];
    const ggml_tensor * weighted = nodes[1];
    const ggml_tensor * silu = nodes[2];
    const ggml_tensor * out = nodes[3];
    if (weighted->src[0] != norm || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
        out->src[0] != weighted || out->src[1] != silu ||
        !ggml_metal_fusion_small_rows(norm->src[0]) ||
        !ggml_metal_fusion_small_rows(silu->src[0]) ||
        !ggml_metal_fusion_small_vector(weighted->src[1], norm->ne[0])) {
        return false;
    }
    for (int j = 0; j < 4; ++j) {
        if (!ggml_metal_fusion_small_rows(nodes[j]) || !ggml_are_same_shape(norm, nodes[j])) {
            return false;
        }
        for (const ggml_tensor * gate = silu->src[0]; gate; gate = gate->view_src) {
            if (gate == nodes[j]) {
                return false;
            }
        }
    }
    return ggml_are_same_shape(norm, silu->src[0]) &&
           ggml_metal_fusion_small_subgraph(fusion, gf, node_idxs, idx);
}

bool ggml_metal_fusion_check_split_l2(const ggml_metal_fusion * fusion,
        const ggml_tensor * const * nodes, const ggml_cgraph * gf,
        const int * node_idxs, int idx, ggml_metal_fusion_mode mode) {
    GGML_UNUSED(mode);
    for (int j : { 0, 2 }) {
        const ggml_tensor * norm = nodes[j];
        const ggml_tensor * scale = nodes[j + 1];
        if (!ggml_metal_fusion_small_rows(norm->src[0]) ||
            !ggml_metal_fusion_small_rows(norm) || !ggml_metal_fusion_small_rows(scale) ||
            scale->src[0] != norm || scale->src[1] ||
            !ggml_are_same_shape(norm, scale) || ggml_get_op_params_f32(scale, 1) != 0.0f) {
            return false;
        }
    }
    return nodes[0]->ne[0] == nodes[2]->ne[0] &&
           ggml_nrows(nodes[0]) + ggml_nrows(nodes[2]) <= INT32_MAX &&
           ggml_metal_fusion_small_subgraph(fusion, gf, node_idxs, idx);
}

