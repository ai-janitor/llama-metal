#import "context/ggml-metal-context-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(context) enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) { GGML_METAL_ZONE_BEGIN(submit);
    GGML_METAL_ZONE_BEGIN(setup); ggml_metal_get_pending_complete(ctx);
    ggml_metal_act_copy_reserve(ctx->act_copy, ctx->dev, gf);
    // INTENT/PDL: delimit actual graph encodes for debug-only fusion firing counts.
    if (ctx->debug_fusion > 1) {
        GGML_LOG_DEBUG("FUSION_GRAPH · graph=%p · nodes=%d · fusion=%d\n",
                (void *) gf, gf->n_nodes, (int) ctx->use_fusion);
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        // stamp CPU start time for profile=1 CPU vs GPU comparison
        if (ctx->profile >= 1) {
            ctx->profile_cpu_start = mach_absolute_time();
        }

        // GGML_METAL_PROFILE=3: per-dispatch timestamp sampling in async path
        // runs for profile_n graphs, then drops to profile=1
        if (ctx->profile == 3 && ctx->profiler && ctx->window_enabled) {
            // explicit window: profile (and mark active) only while budget remains; outside the budget the context
            // stays at profile 3 (no drop to 1) and attaches no profiler. window_left is decremented in synchronize.
            if (ctx->window_left > 0) {
                ggml_metal_profiler_begin_graph(ctx->profiler, gf->n_nodes);
                ctx->window_graph_active = true;
            }
        } else if (ctx->profile == 3 && ctx->profiler) {
            if (ctx->profile_n > 0) {
                ggml_metal_profiler_begin_graph(ctx->profiler, gf->n_nodes);
                // profile_n is decremented in synchronize after the report prints
            } else {
                // done profiling — drop to profile=1 for ongoing graph timing
                ctx->profile = 1;
            }
        }

        // GGML_METAL_PROFILE=2: per-op GPU timing breakdown
        // Encodes and completes one command buffer per op, measuring GPU time for each.
        // This serializes all GPU work — runs for profile_n graph computes, then drops to level 1.
        // Note: profile=3 uses a different path (async counter sampling), so only enter here for exactly level 2.
        if (ctx->profile == 2 && ctx->profile_n > 0) {
            return ggml_metal_graph_compute_per_op(ctx, gf);
        }

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes); GGML_METAL_PROBE_ONLY(ctx->probe_graph_active = ggml_metal_probe_profiler_begin(ctx->probe_profiler, gf->n_nodes);)
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        const bool use_capture = ctx->capture_next_compute;
        if (use_capture) {
            ctx->capture_next_compute = false;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithFormat:@"/tmp/perf-metal.gputrace"]];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev); GGML_METAL_ZONE_END(setup);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = ggml_metal_new_command_buffer(queue, true);
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ggml_metal_new_command_buffer(queue, true);
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        if (ggml_metal_device_get_profile(ctx->dev)->cb_seq > 0) {
            // the hw row asks for the buffers in order on this thread: each is committed as soon as it is encoded
            for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
                ctx->encode_async(cb_idx);
            }
        } else {
            dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);
        }

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (!use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }
                    return ggml_metal_graph_failed(ctx, cmd_buf, n_cb);
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }
                    return ggml_metal_graph_failed(ctx, cmd_buf, i);
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
        }
    }

    // Registration opts into a checked synchronous graph boundary so that a failed command
    // buffer reaches the backend status path; no registration retains the legacy async path.
    if (ggml_metal_gpu_error_callback_registered()) {
        return ggml_metal_synchronize_checked(ctx);
    }
    return GGML_STATUS_SUCCESS;
}

