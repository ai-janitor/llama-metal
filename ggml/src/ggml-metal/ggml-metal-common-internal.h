#pragma once

#include "ggml-metal-common.h"
#include "ggml-metal-fusion.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
struct ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

struct ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    int debug = 0;
};

struct node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;

    // a packed norm + matvecs group writes every member, not only the last one
    bool multi_dst = false;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }
};


// shared across the split files
bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor);
bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor);
bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor);
bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor);
std::vector<int> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes);
