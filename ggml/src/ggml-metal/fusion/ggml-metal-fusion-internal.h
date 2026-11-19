#pragma once

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


// shared across the split files
std::vector<ggml_op> ggml_metal_fusion_filter_ops(const std::vector<ggml_op> & ops_all);
bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode);
bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode);
bool ggml_metal_fusion_check_ssm_conv_silu(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode);
bool ggml_metal_fusion_check_conv_state(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode);
bool ggml_metal_fusion_small_subgraph(const ggml_metal_fusion * fusion,
        const ggml_cgraph * gf, const int * node_idxs, int idx);
bool ggml_metal_fusion_check_gate_g(const ggml_metal_fusion * fusion,
        const ggml_tensor * const * nodes, const ggml_cgraph * gf,
        const int * node_idxs, int idx, ggml_metal_fusion_mode mode);
bool ggml_metal_fusion_check_gated_rms_norm(const ggml_metal_fusion * fusion,
        const ggml_tensor * const * nodes, const ggml_cgraph * gf,
        const int * node_idxs, int idx, ggml_metal_fusion_mode mode);
bool ggml_metal_fusion_check_split_l2(const ggml_metal_fusion * fusion,
        const ggml_tensor * const * nodes, const ggml_cgraph * gf,
        const int * node_idxs, int idx, ggml_metal_fusion_mode mode);

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

