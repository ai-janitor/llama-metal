// Match Metal fusion patterns and retain their external inputs.
#include "ggml-metal-fusion.h"
#include "ggml-metal-common.h"
#include "ggml-backend-impl.h"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

bool ggml_metal_fuse_small_enabled(void) {
    static const bool enabled = []() {
        const char * value = std::getenv("GGML_METAL_FUSE_SMALL");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// derive the non-empty op sequence from the raw `ops_all` sequence
static std::vector<ggml_op> ggml_metal_fusion_filter_ops(const std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    for (ggml_op op : ops_all) {
        if (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    return ops;
}

struct ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<int>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    bool (*check)(const struct ggml_metal_fusion   * fusion,
                  const struct ggml_tensor * const * nodes,
                  const struct ggml_cgraph         * gf,
                  const int                        * node_idxs,
                        int                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            const std::vector<ggml_op> & ops_all,
            const std::vector<int> & outs,
            bool unsafe,
            bool (*check)(const struct ggml_metal_fusion   * fusion,
                          const struct ggml_tensor * const * nodes,
                          const struct ggml_cgraph         * gf,
                          const int                        * node_idxs,
                                int                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(const ggml_metal_fusion * fusion) {
    return fusion->id;
}

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    if (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        const ggml_tensor * scale = nodes[1];
        if (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    return false;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
static bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * gdn = nodes[0];
    const ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    if (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const int64_t S_v      = gdn->src[2]->ne[0];
    const int64_t H        = gdn->src[2]->ne[1];
    const int64_t n_tokens = gdn->src[2]->ne[2];
    const int64_t n_seqs   = gdn->src[2]->ne[3];
    const int64_t K        = ggml_get_op_params_i32(gdn, 0);
    const size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    const ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    const ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        return false;
    }

    const int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    if (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst->data == nullptr) {
            return false;
        }
    }

    return true;
}

// SSM_CONV + UNARY (silu)
static bool ggml_metal_fusion_check_ssm_conv_silu(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * conv = nodes[0];
    const ggml_tensor * un   = nodes[1];

    if (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        return false;
    }

    if (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        return false;
    }

    if (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        return false;
    }

    return true;
}

// CONCAT(state, input) + CPY(shifted state -> cache cell) + SSM_CONV + silu at one decode token:
// one launch reads the state, computes the conv and writes the shifted state (GGML_METAL_STATE_IN_PLACE).
static bool ggml_metal_fusion_check_conv_state(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    if (!ggml_metal_state_in_place()) {
        return false;
    }

    const ggml_tensor * cat  = nodes[0];
    const ggml_tensor * cpy  = nodes[1];
    const ggml_tensor * conv = nodes[2];
    const ggml_tensor * un   = nodes[3];
    const ggml_tensor * st   = cat->src[0];
    const ggml_tensor * in   = cat->src[1];
    const ggml_tensor * w    = conv->src[1];
    const ggml_tensor * tail = cpy->src[0];
    const ggml_tensor * cell = cpy->src[1];

    if (ggml_get_op_params_i32(cat, 0) != 0 || cat->type != GGML_TYPE_F32 || st->type != GGML_TYPE_F32 ||
        in->type != GGML_TYPE_F32 || !ggml_is_contiguous(cat) || (cat->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const int64_t nc = cat->ne[0];
    const int64_t ch = cat->ne[1];
    if (nc < 2 || cat->ne[2] != 1 || cat->ne[3] != 1 || st->ne[0] != nc - 1 || st->ne[1] != ch || st->nb[0] != sizeof(float) ||
        in->ne[0] != 1 || in->ne[1] != ch) {
        return false;
    }
    if (conv->src[0] != cat || !w || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != nc || w->ne[1] != ch ||
        conv->type != GGML_TYPE_F32 || (conv->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (un->op != GGML_OP_UNARY || ggml_get_unary_op(un) != GGML_UNARY_OP_SILU || un->src[0] != conv || un->src[1] ||
        un->type != GGML_TYPE_F32 || !ggml_is_contiguous(un)) {
        return false;
    }
    // the cpy stores the last nc - 1 columns of the concat into a contiguous cache cell
    if (tail->op != GGML_OP_VIEW || tail->view_src != cat || tail->view_offs != sizeof(float) || tail->ne[0] != nc - 1 ||
        tail->ne[1] != ch || cell->type != GGML_TYPE_F32 || !ggml_is_contiguous(cell) || ggml_nelements(cell) != (nc - 1)*ch) {
        return false;
    }
    if (mode == GGML_METAL_FUSION_FULL && cell->data == nullptr) {
        return false;
    }
    // nothing else reads the concat, its tail view or the raw conv
    int uses_cat = 0, uses_tail = 0, uses_conv = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = gf->nodes[i]->src[s];
            uses_cat  += src == cat;
            uses_tail += src == tail;
            uses_conv += src == conv;
        }
    }
    return uses_cat == 2 && uses_tail == 1 && uses_conv == 1;
}

static const std::vector<ggml_op> ops_rms_norm_scale = { GGML_OP_RMS_NORM, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };
static const std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };
static const std::vector<ggml_op> ops_conv_state = { GGML_OP_CONCAT, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_UNARY };

static bool ggml_metal_fusion_small_rows(const ggml_tensor * tensor) {
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

static bool ggml_metal_fusion_small_subgraph(const ggml_metal_fusion * fusion,
        const ggml_cgraph * gf, const int * node_idxs, int idx) {
    int outputs[GGML_METAL_FUSION_MAX];
    outputs[0] = node_idxs[idx + fusion->ops.size() - 1];
    for (size_t j = 0; j < fusion->outs.size(); ++j) {
        outputs[j + 1] = node_idxs[idx + fusion->outs[j]];
    }
    return ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, fusion->ops.size(),
            fusion->ops.data(), outputs, 1 + fusion->outs.size());
}

static bool ggml_metal_fusion_check_gate_g(const ggml_metal_fusion * fusion,
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

static bool ggml_metal_fusion_check_gated_rms_norm(const ggml_metal_fusion * fusion,
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

static bool ggml_metal_fusion_check_split_l2(const ggml_metal_fusion * fusion,
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

static bool ggml_metal_fusion_is_small(const ggml_metal_fusion * fusion) {
    return fusion->id == GGML_METAL_FUSION_GATE_G ||
           fusion->id == GGML_METAL_FUSION_GATED_RMS_NORM ||
           fusion->id == GGML_METAL_FUSION_SPLIT_L2;
}

static const std::vector<ggml_metal_fusion> ggml_metal_fusions = []() {
    std::vector<ggml_metal_fusion> result = {
        { GGML_METAL_FUSION_NORM_SCALE, ops_rms_norm_scale, {}, false, ggml_metal_fusion_check_norm },
        { GGML_METAL_FUSION_GDN_CACHE, ops_gdn_cache, {}, true, ggml_metal_fusion_check_gdn_cache },
        { GGML_METAL_FUSION_SSM_CONV_SILU, ops_ssm_conv_silu, {}, false, ggml_metal_fusion_check_ssm_conv_silu },
        { GGML_METAL_FUSION_CONV_STATE, ops_conv_state, {}, true, ggml_metal_fusion_check_conv_state },
    };
    if (ggml_metal_fuse_small_enabled()) {
        result.emplace_back(GGML_METAL_FUSION_GATE_G,
                std::vector<ggml_op>{GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL},
                std::vector<int>{}, false, ggml_metal_fusion_check_gate_g);
        result.emplace_back(GGML_METAL_FUSION_GATED_RMS_NORM,
                std::vector<ggml_op>{GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_MUL},
                std::vector<int>{}, true, ggml_metal_fusion_check_gated_rms_norm);
        result.emplace_back(GGML_METAL_FUSION_SPLIT_L2,
                std::vector<ggml_op>{GGML_OP_RMS_NORM, GGML_OP_SCALE, GGML_OP_RMS_NORM, GGML_OP_SCALE},
                std::vector<int>{1}, true, ggml_metal_fusion_check_split_l2);
    }
    return result;
}();

// New patterns retain inputs across intervening views, as the encoder does.
static int ggml_metal_fusion_small_span(const ggml_cgraph * gf, int first,
        const ggml_metal_fusion * fusion) {
    if (ggml_op_is_empty(gf->nodes[first]->op)) {
        return 0;
    }
    int indices[GGML_METAL_FUSION_MAX];
    const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];
    int n = 0;
    for (int i = first; i < gf->n_nodes && n < (int) fusion->ops.size(); ++i) {
        if (ggml_op_is_empty(gf->nodes[i]->op)) {
            continue;
        }
        if (gf->nodes[i]->op != fusion->ops[n]) {
            return 0;
        }
        indices[n] = i;
        nodes[n++] = gf->nodes[i];
    }
    if (n != (int) fusion->ops.size() ||
        !fusion->check(fusion, nodes, gf, indices, 0, GGML_METAL_FUSION_STRUCTURAL)) {
        return 0;
    }
    if (!fusion->unsafe && !ggml_metal_fusion_small_subgraph(fusion, gf, indices, 0)) {
        return 0;
    }
    return indices[n - 1] - first + 1;
}

static void ggml_metal_fusion_add_span_alloc_deps(void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf, int first, int count) {
    std::set<ggml_tensor *> seen;
    for (int i = first; i < first + count; ++i) {
        ggml_tensor * node = gf->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = node->src[s];
            if (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[first + count - 1]);
            }
        }
        seen.insert(node);
    }
}

static bool ggml_metal_fusion_match_raw_pattern(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops) {
    if (node_idx < 0 || node_idx + (int) ops.size() > gf->n_nodes) {
        return false;
    }

    for (int i = 0; i < (int) ops.size(); ++i) {
        if (gf->nodes[node_idx + i]->op != ops[i]) {
            return false;
        }
    }

    return true;
}

static void ggml_metal_fusion_add_pattern_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        const ggml_metal_fusion * fusion,
        int node_idx) {
    const int last_node = node_idx + (int) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    for (int j = 0; j < (int) fusion->ops_all.size(); ++j) {
        ggml_tensor * node = gf->nodes[node_idx + j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = node->src[s];
            if (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_node]);
            }
        }
        seen.insert(node);
    }
}

void ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_metal_fusion * best = nullptr;
        int best_raw = 0;

        for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
            if (ggml_metal_fusion_is_small(&fusion)) {
                const int span = ggml_metal_fusion_small_span(gf, i, &fusion);
                if (span > best_raw) {
                    best = &fusion;
                    best_raw = span;
                }
                continue;
            }
            if ((int) fusion.ops_all.size() <= best_raw) {
                continue;
            }
            if (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (int) fusion.ops_all.size();
            }
        }

        if (best) {
            if (ggml_metal_fusion_is_small(best)) {
                ggml_metal_fusion_add_span_alloc_deps(user_data, add_alloc_dep, gf, i, best_raw);
            } else {
                ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            }
            i += best_raw - 1;
        }
    }
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
static bool ggml_metal_fusion_check_memory_ranges(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        int                            node_count) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a->data && b->data && a->data == b->data) {
            return true;
        }
        for (const ggml_tensor * v = a; v; v = v->view_src) {
            if (v == b) {
                return true;
            }
        }
        for (const ggml_tensor * v = b; v; v = v->view_src) {
            if (v == a) {
                return true;
            }
        }
        return false;
    };

    auto nodes_overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            return false;
        }

        if (a->buffer != b->buffer) {
            return false;
        }

        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        return (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](const ggml_tensor * src, const ggml_tensor * const * nodes, int j) {
        for (int k = 0; k < j; ++k) {
            if (src == nodes[k]) {
                return true;
            }
            for (const ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                if (view_src == nodes[k]) {
                    return true;
                }
            }
        }
        return false;
    };

    auto check_dst = [&](const ggml_tensor * dst) {
        for (int j = 0; j < node_count; ++j) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = nodes[j]->src[s];
                if (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    continue;
                }

                if (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    return false;
                }
            }
        }
        return true;
    };

    if (!check_dst(nodes[node_count - 1])) {
        return false;
    }

    for (int offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < node_count);
        if (!check_dst(nodes[offset])) {
            return false;
        }
    }

    return true;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const ggml_metal_fusion * ggml_metal_fusion_next(
        const ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        ggml_metal_fusion_mode mode,
        int * n_out) {
    const ggml_metal_fusion * res = nullptr;
    int best = 1;

    for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        const int n_ops = (int) fusion.ops.size();

        // only look for a longer match than the current best
        if (n_ops <= best) {
            continue;
        }
        if (idx + n_ops > n_idxs) {
            continue;
        }

        const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        for (int j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion.ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            for (int j = 1; j < n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            int outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = node_idxs[idx + n_ops - 1];
            for (size_t i = 0; i < fusion.outs.size(); ++i) {
                const int out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = node_idxs[idx + out_offset];
            }

            const int n_outputs = 1 + (int) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion.check && !fusion.check(&fusion, nodes, gf, node_idxs, idx, mode)) {
            continue;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        if (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            continue;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    return res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
int ggml_metal_fusion_max(const ggml_cgraph * gf, int idx) {
    // a view node cannot start a pattern - pack it alone
    if (ggml_op_is_empty(gf->nodes[idx]->op)) {
        return 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    int idxs[GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    for (int i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        if (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    int total = 0;
    int i_f = 0;

    while (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        int len = 1;
        const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        if (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    return std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
