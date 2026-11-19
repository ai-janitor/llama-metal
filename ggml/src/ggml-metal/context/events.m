#import "context/ggml-metal-context-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(context) void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    // skip graph optimization when profiling per-op (profile>=2) — fusion would merge
    // nodes and prevent individual op timing measurement
    if (ctx->use_graph_optimize && ctx->profile < 2) {
        const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
        ggml_graph_optimize(gf, profile->mv_group == 1, profile->attn_gate_fuse == 1);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = ggml_metal_new_command_buffer(queue, false);

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = ggml_metal_new_command_buffer(queue, false);

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    // a hw row with cb_seq sets the number of command buffers after the first
    const int cb_seq = (int) ggml_metal_device_get_profile(ctx->dev)->cb_seq;
    if (cb_seq > 0) {
        n_cb = cb_seq;
    }
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2 && cb_seq == 0) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        // pass profiler and cb_idx to op encoding context for profile=3 dispatch sampling
        // explicit window: only the graph that began profiling under the window gets the profiler
        ggml_metal_profiler_t prof = (ctx->profile == 3 && (!ctx->window_enabled || ctx->window_graph_active)) ? ctx->profiler : NULL; GGML_METAL_PROBE_ONLY(if (ctx->probe_graph_active) prof = ctx->probe_profiler;)

        const bool l0_trace = ggml_l0_enabled(); GGML_METAL_ZONE_BEGIN(encode);
        const double l0_encode_start = l0_trace ? ggml_l0_now() : 0;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            idx_start,
            idx_end,
            ctx->use_fusion,
            ctx->use_concurrency,
            ctx->capture_next_compute,
            ctx->debug_graph,
            ctx->debug_fusion,
            prof,
            cb_idx
#ifdef GGML_METAL_TEST_OBSERVER
            , ctx->observer, ctx->observer_user_data
#endif
            );
        ggml_metal_op_set_act_copy(ctx_op, ctx->act_copy);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op); GGML_METAL_ZONE_END(encode);

        if (l0_trace) {
            ggml_l0_event("encode", l0_encode_start, ggml_l0_now(), cb_idx);
        }

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            const double l0_commit_start = l0_trace ? ggml_l0_now() : 0;
            GGML_METAL_ZONE_BEGIN(commit); [cmd_buf commit]; GGML_METAL_ZONE_END(commit);
            if (l0_trace) {
                ggml_l0_event("commit", l0_commit_start, ggml_l0_now(), cb_idx);
            }
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

ggml_metal_device_t ggml_metal_get_dev(ggml_metal_t ctx) {
    return ctx->dev;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_next_compute = true;
}

#ifdef GGML_METAL_TEST_OBSERVER
void ggml_metal_test_set_observer(ggml_metal_t ctx,
        ggml_backend_metal_test_observer_callback observer, void * user_data) {
    ctx->observer = observer;
    ctx->observer_user_data = user_data;
}
#endif

// Arm every eligible registered context for an explicit n_graphs profile window; see ggml-metal.h for the contract.
// Eligible: the context requested profile 3 at init and actually owns a profiler (unsupported-counter fallback
// contexts and profile 0/1/2 contexts are left untouched), and has no graph in flight. No GPU work is done here.
int ggml_metal_profile_window(int n_graphs) {
    if (n_graphs <= 0) {
        return 0;
    }

    int n_armed = 0;

    pthread_mutex_lock(&g_registry_mutex);
    for (struct ggml_metal * ctx = g_registry_head; ctx; ctx = ctx->registry_next) {
        if (ctx->profile_ctor != 3 || ctx->profiler == NULL || ctx->window_graph_active) {
            continue;
        }
        ctx->window_enabled      = true;
        ctx->window_left         = n_graphs;
        ctx->window_graph_active = false;
        ctx->profile             = 3;
        n_armed++;
    }
    pthread_mutex_unlock(&g_registry_mutex);

    return n_armed;
}
