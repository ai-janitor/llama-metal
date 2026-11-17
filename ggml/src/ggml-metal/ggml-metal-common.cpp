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

ggml_mem_ranges_t ggml_mem_ranges_init(int debug) {
    auto * res = new ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

static bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    return true;
}

static ggml_mem_range ggml_mem_range_from_tensor(const ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    if (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mr;
}

static ggml_mem_range ggml_mem_range_from_tensor_src(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static ggml_mem_range ggml_mem_range_from_tensor_dst(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

static bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    return ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        if (mr.pb != cmp.pb) {
            continue;
        }

        // intersecting source ranges are allowed
        if (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            if (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            return false;
        }
    }

    return true;
}

static bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

static bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            if (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                return false;
            }
        }
    }

    return ggml_mem_ranges_check_dst(mrs, tensor);
}

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

static std::vector<int> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add node src and dst ranges
    const auto & h_add = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!ggml_mem_ranges_add_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        // keep track of the sources of the fused nodes as well
        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!ggml_mem_ranges_add_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
        }

        if (node.multi_dst) {
            if (!ggml_mem_ranges_add_dst(mrs, node.node)) {
                return false;
            }
            for (const auto * fused : node.fused) {
                if (!ggml_mem_ranges_add_dst(mrs, fused)) {
                    return false;
                }
            }
            return true;
        }

        return ggml_mem_ranges_add_dst(mrs, node.dst());
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    const auto & h_check = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!ggml_mem_ranges_check_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!ggml_mem_ranges_check_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
        }

        if (node.multi_dst) {
            if (!ggml_mem_ranges_check_dst(mrs, node.node)) {
                return false;
            }
            for (const auto * fused : node.fused) {
                if (!ggml_mem_ranges_check_dst(mrs, fused)) {
                    return false;
                }
            }
            return true;
        }

        return ggml_mem_ranges_check_dst(mrs, node.dst());
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    const auto & h_safe = [](ggml_op op) {
        switch (op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_ROPE:
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_SSM_CONV:
            case GGML_OP_SSM_SCAN:
            case GGML_OP_CLAMP:
            case GGML_OP_TRI:
            case GGML_OP_DIAG:
            case GGML_OP_MUL:
            case GGML_OP_ADD:
            case GGML_OP_DIV:
            case GGML_OP_GLU:
            case GGML_OP_SCALE:
            case GGML_OP_UNARY:
            case GGML_OP_GET_ROWS:
            case GGML_OP_SET_ROWS:
            case GGML_OP_SET:
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_REPEAT:
                return true;
            default:
                return ggml_op_is_empty(op);
        }
    };

    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        if (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr int N_FORWARD = 64;

            for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                if (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                if (!h_safe(node1.op())) {
                    break;
                }

                const bool is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                if ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } else {
                    // expand the set of nodes that haven't been processed yet
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_add(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    return res;
}

int ggml_metal_norm_matvec_mode(void) {
    static const int mode = [] {
        const char * value = std::getenv("GGML_METAL_NORM_MATVEC");
        if (value == nullptr) {
            return 0;
        }
        if (std::strcmp(value, "pack") == 0) {
            return 1;
        }
        if (std::strcmp(value, "nm-self") == 0) {
            return 2;
        }
        return std::strcmp(value, "1") == 0 || std::strcmp(value, "full") == 0 ? 3 : 0;
    }();
    return mode;
}

bool ggml_metal_norm_matvec_enabled(void) {
    return ggml_metal_norm_matvec_mode() >= 2;
}

bool ggml_metal_norm_matvec_add_scale(void) {
    return ggml_metal_norm_matvec_mode() == 3;
}

bool ggml_metal_state_in_place(void) {
    static const bool on = [] {
        const char * value = std::getenv("GGML_METAL_STATE_IN_PLACE");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return on;
}

bool ggml_metal_add_scale_applies(const ggml_tensor * add) {
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    return add->op == GGML_OP_ADD && add->type == GGML_TYPE_F32 && add->view_src == nullptr &&
           a && b && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(add, a) && ggml_are_same_shape(add, b) &&
           ggml_is_contiguous(add) && ggml_is_contiguous(a) && ggml_is_contiguous(b) &&
           ggml_nrows(add) == 1 && add->ne[0] % 4 == 0;
}

bool ggml_metal_norm_matvec_mm_ok(const ggml_tensor * mm, const ggml_tensor * act) {
    const ggml_tensor * w = mm->src[0];
    return mm->op == GGML_OP_MUL_MAT && mm->src[1] == act && mm->type == GGML_TYPE_F32 && ggml_is_contiguous(mm) &&
           w && w->type == GGML_TYPE_Q8_0 && w->op == GGML_OP_NONE && w->view_src == nullptr && ggml_is_contiguous(w) &&
           w->ne[2] == 1 && w->ne[3] == 1 && w->ne[0] % 64 == 0 && w->ne[0] == act->ne[0] &&
           act->type == GGML_TYPE_F32 && ggml_is_contiguous(act) && ggml_nrows(act) == 1;
}

// true when the view's elements lie in memory in index order with no gaps
static bool ggml_metal_view_is_dense(const ggml_tensor * t) {
    size_t expect = ggml_element_size(t);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (t->ne[d] > 1 && t->nb[d] != expect) {
            return false;
        }
        expect *= t->ne[d];
    }
    return true;
}

bool ggml_metal_attn_gate_match(const ggml_tensor * mul, const ggml_tensor ** cont_gate,
                                const ggml_tensor ** sigmoid, const ggml_tensor ** cont_out) {
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 || !mul->src[0] || !mul->src[1]) {
        return false;
    }
    const ggml_tensor * out = mul->src[0];
    const ggml_tensor * sig = mul->src[1];
    if (out->op != GGML_OP_CONT || sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID ||
        !sig->src[0] || sig->src[0]->op != GGML_OP_CONT) {
        return false;
    }
    const ggml_tensor * cg = sig->src[0];
    const ggml_tensor * gv = cg->src[0];
    const ggml_tensor * ov = out->src[0];
    const int64_t n = ggml_nelements(mul);
    const bool flat = ggml_is_contiguous(mul) && ggml_is_contiguous(out) && ggml_is_contiguous(sig) && ggml_is_contiguous(cg) &&
                      ggml_nrows(mul) == 1 && ggml_are_same_shape(mul, out) && ggml_are_same_shape(mul, sig) && ggml_are_same_shape(mul, cg);
    if (!flat || !gv || !ov || gv->type != GGML_TYPE_F32 || ov->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
        sig->type != GGML_TYPE_F32 || cg->type != GGML_TYPE_F32 ||
        ggml_nelements(gv) != n || ggml_nelements(ov) != n || !ggml_metal_view_is_dense(ov) ||
        gv->nb[0] != sizeof(float) || gv->ne[2] != 1 || gv->ne[3] != 1 || gv->ne[0] % 4 != 0 ||
        ((out->flags | sig->flags | cg->flags) & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    *cont_gate = cg;
    *sigmoid   = sig;
    *cont_out  = out;
    return true;
}

// Delay the gate copy and its sigmoid to sit right before the attention output copy and the multiply.
static void ggml_metal_graph_optimize_attn_gate(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, int> uses;
    for (int i = 0; i < gf->n_nodes; ++i) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (gf->nodes[i]->src[s]) {
                uses[gf->nodes[i]->src[s]]++;
            }
        }
    }
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * cg = nullptr, * sig = nullptr, * out = nullptr;
        if (!ggml_metal_attn_gate_match(gf->nodes[i], &cg, &sig, &out) ||
            uses[cg] != 1 || uses[sig] != 1 || uses[out] != 1) {
            continue;
        }
        int pg = -1, ps = -1, po = -1;
        for (int j = 0; j < i; ++j) {
            pg = gf->nodes[j] == cg  ? j : pg;
            ps = gf->nodes[j] == sig ? j : ps;
            po = gf->nodes[j] == out ? j : po;
        }
        if (pg < 0 || ps < 0 || po < 0 || !(pg < ps && ps < po) || po != i - 1) {
            continue;
        }
        // [.. cg .. sig .. out mul] -> [.. .. cg sig out mul]
        std::rotate(gf->nodes + ps, gf->nodes + ps + 1, gf->nodes + po);
        std::rotate(gf->nodes + pg, gf->nodes + pg + 1, gf->nodes + po - 1);
    }
}

// Delay a single-consumer SiLU to form the validated gated norm group.
static void ggml_metal_graph_optimize_small(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, int> positions;
    for (int i = 0; i < gf->n_nodes; ++i) {
        positions.emplace(gf->nodes[i], i);
    }
    const auto previous = [gf](int i) {
        do { --i; } while (i >= 0 && ggml_op_is_empty(gf->nodes[i]->op));
        return i;
    };
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * out = gf->nodes[i];
        if (out->op != GGML_OP_MUL || !out->src[0] || !out->src[1]) {
            continue;
        }
        const ggml_tensor * weighted = out->src[0];
        const ggml_tensor * silu = out->src[1];
        if (weighted->op != GGML_OP_MUL || !weighted->src[0] ||
            weighted->src[0]->op != GGML_OP_RMS_NORM || silu->op != GGML_OP_UNARY ||
            ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || !silu->src[0]) {
            continue;
        }
        const auto norm_pos = positions.find(weighted->src[0]);
        const auto weighted_pos = positions.find(weighted);
        const auto silu_pos = positions.find(silu);
        if (norm_pos == positions.end() || weighted_pos == positions.end() || silu_pos == positions.end()) {
            continue;
        }
        const int ni = norm_pos->second;
        const int wi = weighted_pos->second;
        const int si = silu_pos->second;
        if (si >= ni || previous(wi) != ni || previous(i) != wi) {
            continue;
        }
        const auto z_pos = positions.find(silu->src[0]);
        if (z_pos != positions.end() && z_pos->second >= si) {
            continue;
        }
        const int indices[] = {ni, wi, si, i};
        int count = 1;
        const auto * fusion = ggml_metal_fusion_next(gf, indices, 4, 0,
                GGML_METAL_FUSION_STRUCTURAL, &count);
        if (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_GATED_RMS_NORM) {
            continue;
        }
        std::rotate(gf->nodes + si, gf->nodes + si + 1, gf->nodes + wi + 1);
        for (int j = si; j <= wi; ++j) {
            positions[gf->nodes[j]] = j;
        }
    }
}

void ggml_graph_optimize(ggml_cgraph * gf, bool pack_matvecs, bool attn_gate) {
    constexpr int MAX_FUSE = 16;

    const int n = gf->n_nodes;

    // with GGML_METAL_NORM_MATVEC, the Q8_0 matvecs that read a norm's weighted output are packed right after it
    const bool norm_mv = ggml_metal_norm_matvec_mode() >= 1 || pack_matvecs;
    std::unordered_map<const ggml_tensor *, int> uses;
    std::vector<bool> taken(n, false);
    if (norm_mv) {
        for (int i = 0; i < n; i++) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (gf->nodes[i]->src[s]) {
                    uses[gf->nodes[i]->src[s]]++;
                }
            }
        }
    }

    enum ggml_op ops[MAX_FUSE];

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    for (int i = 0; i < n; i++) {
        if (taken[i]) {
            continue;
        }
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        int legacy_span = 1;

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_NORM ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            legacy_span = f;
        }

        // INTENT: keep both legacy groups and the ported table's groups adjacent during reorder.
        // PDL: derive each raw span first; pack the larger span, including transparent VIEWs.
        const int f = std::max(legacy_span, ggml_metal_fusion_max(gf, i));
        for (int k = 1; k < f; k++) {
            ++i;
            node.add_fused(gf->nodes[i]);
        }
        // the conv-state span also writes the cache cell (its CPY), not only its last node
        if (f > 1 && node.op() == GGML_OP_CONCAT && ggml_metal_state_in_place()) {
            node.multi_dst = true;
        }

        if (norm_mv && f == 2 && node.op() == GGML_OP_RMS_NORM && gf->nodes[i]->op == GGML_OP_MUL &&
            gf->nodes[i]->src[0] == node.node && uses[node.node] == 1 && !(gf->nodes[i]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            const ggml_tensor * act = gf->nodes[i];
            std::vector<int> mms;
            for (int j = i + 1; j < n && (int) mms.size() < 4; j++) {
                if (!taken[j] && ggml_metal_norm_matvec_mm_ok(gf->nodes[j], act)) {
                    mms.push_back(j);
                }
            }
            if (!mms.empty() && uses[act] == (int) mms.size()) {
                for (const int j : mms) {
                    taken[j] = true;
                    node.add_fused(gf->nodes[j]);
                }
                node.multi_dst = true;
            }
        }

        node.multi_dst = node.multi_dst || (ggml_metal_fuse_small_enabled() && !node.fused.empty());
        nodes.push_back(std::move(node));
    }

#if 1
    // reorder to improve concurrency
    const auto order = ggml_metal_graph_optimize_reorder(nodes);
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    if (ggml_metal_fuse_small_enabled()) {
        ggml_metal_graph_optimize_small(gf);
    }
    if (attn_gate) {
        ggml_metal_graph_optimize_attn_gate(gf);
    }
}
