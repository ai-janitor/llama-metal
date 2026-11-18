#pragma once

#include "ggml-metal-ops.h"
#include "ggml-metal-fusion.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-metal-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"
#include "ggml-metal-test-observer-helpers.h"
#include "ggml-device-profile.h"

#include <cassert>
#include <algorithm>
#include <limits>
#include <cmath>
#include <vector>
#include <string>

// struct ggml_metal_op — encoding context shared by all op dispatch functions
struct ggml_metal_op {
    ggml_metal_op(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        int  idx_start,
        int  idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph,
        int  debug_fusion,
        ggml_metal_profiler_t profiler = NULL,
        int  profiler_cb_idx = -1
#ifdef GGML_METAL_TEST_OBSERVER
        , ggml_backend_metal_test_observer_callback observer = NULL,
        void * observer_user_data = NULL
#endif
        ) {
        this->dev             = dev;
        this->lib             = ggml_metal_device_get_library(dev);
        // when profiler is active, create encoder with counter sample buffer attached
        this->enc             = profiler
            ? ggml_metal_encoder_init_profiled(cmd_buf, use_concurrency, profiler, profiler_cb_idx)
            : ggml_metal_encoder_init(cmd_buf, use_concurrency);
        this->mem_ranges      = ggml_mem_ranges_init(debug_graph);
        this->idx_start       = idx_start;
        this->idx_end         = idx_end;
        this->use_fusion      = use_fusion;
        this->use_concurrency = use_concurrency;
#ifdef GGML_METAL_TEST_OBSERVER
        this->encoder_concurrent = ggml_metal_encoder_is_concurrent(this->enc);
#endif
        this->use_capture     = use_capture;
        this->debug_graph     = debug_graph;
        this->debug_fusion    = debug_fusion;
#ifdef GGML_METAL_TEST_OBSERVER
        this->observer        = observer;
        this->observer_user_data = observer_user_data;
#endif
        this->gf              = gf;

        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        for (int i = idx_start; i < idx_end; i++) {
            if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }
    }

    ~ggml_metal_op() {
        ggml_metal_encoder_end_encoding(this->enc);
        ggml_metal_encoder_free(this->enc);
        ggml_mem_ranges_free(this->mem_ranges);
    }

    int n_nodes() const {
        return idxs.size();
    }

    ggml_tensor * node(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return ggml_graph_node(gf, idxs[i]);
    }

    // the whole graph, for fusions that must know every consumer of a tensor
    int graph_n_nodes() const {
        return gf->n_nodes;
    }

    ggml_tensor * graph_node(int i) const {
        return gf->nodes[i];
    }

    int graph_index(int i) const {
        return idxs[i];
    }

    bool can_fuse(int i0, const ggml_op * ops, int n_ops) const {
        assert(use_fusion);
        assert(i0 >= 0 && i0 < n_nodes());

        if (i0 + n_ops > n_nodes()) {
            return false;
        }

        return ggml_can_fuse_ext(gf, idxs.data() + i0, ops, n_ops);
    }

    // INTENT: report groups only after the encoder has consumed them, not pattern candidates.
    // PDL: debug2 builds the op chain; raw graph indices tie the event to its graph-compute marker.
    void trace_fusion(int idx, int count) const {
        if (debug_fusion < 2 || count <= 1) {
            return;
        }
        std::string pattern;
        for (int i = 0; i < count; ++i) {
            if (i) {
                pattern += ">";
            }
            pattern += ggml_op_name(node(idx + i)->op);
        }
        GGML_LOG_DEBUG("FUSION_FIRE · %s · graph=%p · raw=%d..%d\n",
                pattern.c_str(), (void *) gf, idxs[idx], idxs[idx + count - 1]);
    }

    const ggml_metal_fusion * can_fuse(int idx, ggml_metal_fusion_mode mode, int * n) const {
        assert(use_fusion);
        return ggml_metal_fusion_next(gf, idxs.data(), (int) idxs.size(), idx, mode, n);
    }

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    bool use_fusion;

    // split_l2 heads (first RMS_NORM) whose outputs a conv_state launch already wrote (hw row conv_l2_fold)
    std::vector<const ggml_tensor *> split_l2_folded;
    bool use_concurrency;
    bool use_capture;

#ifdef GGML_METAL_TEST_OBSERVER
    ggml_backend_metal_test_observer_callback observer;
    void * observer_user_data;

    void observe(int idx, enum ggml_backend_metal_test_observer_stage stage,
            bool result, bool barrier_issued) const {
        if (observer) {
            struct ggml_metal_test_observer_record record = {};
            const bool assembled = ggml_metal_test_observer_assemble_record(
                (uintptr_t) gf, (uintptr_t) node(idx), idxs.data(), idxs.size(), idx,
                (int) stage, result, barrier_issued, encoder_concurrent, &record);
            assert(assembled);
            if (!assembled) {
                return;
            }
            const struct ggml_backend_metal_test_observer_event event = {
                record.graph, record.node, record.graph_index, stage, record.result,
                record.barrier_issued, record.concurrency_enabled,
            };
            ggml_metal_test_observer_forward(observer, &event, observer_user_data);
        }
    }
#endif

    int debug_graph;
    int debug_fusion;

#ifdef GGML_METAL_TEST_OBSERVER
    // Keep requested policy separate from the dispatch mode of the created encoder.
    bool encoder_concurrent;
#endif
private:
    ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    // non-empty node indices
    std::vector<int> idxs;
};

// Buffer helper used by all op files
static inline ggml_metal_buffer_id ggml_metal_get_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

// Prepared Q4_K block headers of a weight tensor (nil buffer when it has none)
static inline ggml_metal_buffer_id ggml_metal_get_kq_prep_id(const ggml_tensor * t) {
    if (!t || t->view_src || !t->buffer) {
        return { nullptr, 0 };
    }

    return ggml_metal_buffer_kq_prep_id((ggml_metal_buffer_t) t->buffer->context, t);
}

// Concurrency helpers used by multiple op files
static inline bool ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx, int idx) {
    if (!ctx->mem_ranges) {
#ifdef GGML_METAL_TEST_OBSERVER
        ctx->observe(idx, GGML_BACKEND_METAL_TEST_OBSERVER_RESET, true, false);
#endif
        return true;
    }

    ggml_metal_encoder_memory_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

#ifdef GGML_METAL_TEST_OBSERVER
    ctx->observe(idx, GGML_BACKEND_METAL_TEST_OBSERVER_RESET, true, true);
#endif

    return true;
}

static inline bool ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, const ggml_tensor * node, int idx) {
    if (!ctx->mem_ranges) {
#ifdef GGML_METAL_TEST_OBSERVER
        ctx->observe(idx, GGML_BACKEND_METAL_TEST_OBSERVER_CHECK, false, false);
#endif
        return false;
    }
    const bool result = ggml_mem_ranges_check(ctx->mem_ranges, node);
#ifdef GGML_METAL_TEST_OBSERVER
    ctx->observe(idx, GGML_BACKEND_METAL_TEST_OBSERVER_CHECK, result, false);
#endif
    return result;
}

static inline bool ggml_metal_op_concurrency_add(ggml_metal_op_t ctx, const ggml_tensor * node, int idx) {
    if (!ctx->mem_ranges) {
#ifdef GGML_METAL_TEST_OBSERVER
        ctx->observe(idx, GGML_BACKEND_METAL_TEST_OBSERVER_ADD, true, false);
#endif
        return true;
    }
    const bool result = ggml_mem_ranges_add(ctx->mem_ranges, node);
#ifdef GGML_METAL_TEST_OBSERVER
    ctx->observe(idx, GGML_BACKEND_METAL_TEST_OBSERVER_ADD, result, false);
#endif
    return result;
}
