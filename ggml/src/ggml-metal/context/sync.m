#import "context/ggml-metal-context-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(context) // Wait for the pending readbacks of this context and copy each to its destination.
void ggml_metal_get_pending_complete(ggml_metal_t ctx) { GGML_METAL_ZONE_BEGIN(readback);
    if ([ctx->get_pending count] == 0) {
        return;
    }
    @autoreleasepool {
        for (NSArray * p in ctx->get_pending) {
            id<MTLCommandBuffer> cmd = [p objectAtIndex:0];
            id<MTLBuffer>        stg = [p objectAtIndex:1];
            GGML_METAL_ZONE_BEGIN(wait); [cmd waitUntilCompleted]; GGML_METAL_ZONE_END(wait);
            memcpy([[p objectAtIndex:2] pointerValue], [stg contents], [[p objectAtIndex:3] unsignedLongValue]);
        }
        [ctx->get_pending removeAllObjects];
    }
}

enum ggml_status ggml_metal_synchronize_checked(ggml_metal_t ctx) {
    ggml_metal_get_pending_complete(ctx);
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        const bool l0_trace = ggml_l0_enabled();
        const double l0_wait_start = l0_trace ? ggml_l0_now() : 0;
        GGML_METAL_ZONE_BEGIN(wait); [ctx->cmd_buf_last waitUntilCompleted]; GGML_METAL_ZONE_END(wait);
        if (l0_trace) {
            ggml_l0_event("host_wait", l0_wait_start, ggml_l0_now(), 0);
            for (int cb_idx = 0; cb_idx <= ctx->n_cb; ++cb_idx) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
                if (cmd_buf) {
                    const double start = [cmd_buf GPUStartTime];
                    const double end = [cmd_buf GPUEndTime];
                    ggml_l0_event(start == 0 || end == 0 || end < start ? "gpu_graph_cb_invalid" : "gpu_graph_cb", start, end, cb_idx);
                }
            }
        }

        // GGML_METAL_PROFILE=3: resolve counter sample buffers and print per-dispatch report.
        // Must happen after waitUntilCompleted but before cmd_buf_last is cleared.
        if (ctx->profile == 3 && ctx->profiler) {
            if (!ctx->window_enabled) {
                ggml_metal_profiler_end_graph(ctx->profiler);
            } else if (ctx->window_graph_active) {
                // explicit window: report only a graph that began under the window, and consume exactly one
                // unit of the budget for it; a repeated synchronize or an unrelated command finds active == false
                ggml_metal_profiler_end_graph(ctx->profiler);
                ctx->window_graph_active = false;
                ctx->window_left--;
            }
        }

        // GGML_METAL_PROFILE=1: read GPU timestamps immediately after completion,
        // before cmd_buf_last is cleared. GPUStartTime/GPUEndTime are only
        // valid after the command buffer has completed.
        // Note: profile>=2 does per-op timing inline in graph_compute, so skip here.
        if (ctx->profile == 1) {
            const int n_cb = ctx->n_cb;
            double gpu_min_start = 1e30, gpu_max_end = 0;
            double gpu_total = 0;
            int n_active = 0;

            for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
                if (!cmd_buf) continue;

                double start = [cmd_buf GPUStartTime];
                double end   = [cmd_buf GPUEndTime];
                if (start == 0 && end == 0) continue; // not yet scheduled
                double dt = (end - start) * 1000.0;

                if (start < gpu_min_start) gpu_min_start = start;
                if (end   > gpu_max_end)   gpu_max_end   = end;
                gpu_total += dt;
                n_active++;
            }

            if (n_active > 0) {
                double wall_ms = (gpu_max_end - gpu_min_start) * 1000.0;

                // CPU wall time: from graph_compute start to synchronize completion.
                // The gap between CPU wall and GPU wall reveals dispatch overhead —
                // time the CPU spends encoding ops before the GPU starts, plus
                // time waiting for the GPU to finish after the last encode.
                mach_timebase_info_data_t tb;
                mach_timebase_info(&tb);
                double cpu_ms = (double)(mach_absolute_time() - ctx->profile_cpu_start) * tb.numer / tb.denom / 1e6;

                // Print absolute GPU start/end times for cross-device overlap analysis.
                // GPUStartTime/GPUEndTime are CFAbsoluteTime (seconds since 2001-01-01),
                // so differences between devices are directly comparable.
                // We print modulo 1000s to keep the numbers readable while preserving
                // sub-millisecond precision for overlap calculation.
                fprintf(stderr, "[METAL_PROFILE] %s: %d cmd_bufs, GPU wall=%.2f ms, CPU wall=%.2f ms, gap=%.2f ms "
                    "(abs start=%.6f end=%.6f, %d nodes)\n",
                    ctx->name, n_active, wall_ms, cpu_ms, cpu_ms - wall_ms,
                    fmod(gpu_min_start, 1000.0), fmod(gpu_max_end, 1000.0),
                    ctx->gf->n_nodes);
            }
        }

        ctx->cmd_buf_last = nil; GGML_METAL_PROBE_ONLY(ggml_metal_probe_profiler_finish(ctx->probe_profiler);)
    }

    if (ggml_l0_enabled()) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];
            const double start = [cmd_buf GPUStartTime];
            const double end = [cmd_buf GPUEndTime];
            ggml_l0_event(start == 0 || end == 0 || end < start ? "gpu_ext_cb_invalid" : "gpu_ext_cb", start, end, i);
        }
    }

    // Final status/drain is the shared policy in the error module: with a registered handler every buffer is
    // waited/checked/released before FAILED returns; without one the first failure aborts as before.
    return ggml_metal_complete_command_buffers(ctx->cmd_bufs, (size_t) ctx->n_cb + 1, ctx->cmd_bufs_ext);
}

// Failure of a synchronous profile/capture command buffer: shared nonfatal policy (reports to the hook if any,
// FAILED either way). With a hook, pending main/extra buffers are drained of references before returning.
enum ggml_status ggml_metal_graph_failed(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf, int index) {
    const enum ggml_status status = ggml_metal_check_command_buffer(cmd_buf, index, false, false);
    if (ggml_metal_gpu_error_callback_registered()) {
        ctx->cmd_buf_last = nil;
        ggml_metal_discard_command_buffers(ctx->cmd_bufs, (size_t) ctx->n_cb + 1, ctx->cmd_bufs_ext);
    }
    return status;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    (void) ggml_metal_synchronize_checked(ctx);
}

