#import "context/ggml-metal-context-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(memory) static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) { GGML_METAL_ZONE_BEGIN(upload);
    const int64_t t0 = ggml_time_us();

    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = ggml_metal_new_command_buffer(queue, false);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        GGML_METAL_PROBE_ONLY(ggml_metal_probe_transfer(cmd_buf, "upload");) [cmd_buf commit];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }

    ggml_metal_xfer_stats.h2d_bytes   += (int64_t)size;
    ggml_metal_xfer_stats.h2d_time_us += ggml_time_us() - t0;
    ggml_metal_xfer_stats.h2d_count   += 1;
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) { GGML_METAL_ZONE_BEGIN(readback);
    const int64_t t0 = ggml_time_us();

    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = ggml_metal_new_command_buffer(queue, false);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        if (props_dev->use_shared_buffers) {
            // UMA: wrap the destination pointer directly — zero-copy readback
            id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                              length:size
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
            GGML_ASSERT(buf_dst);

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
            GGML_METAL_PROBE_ONLY(ggml_metal_probe_transfer(cmd_buf, "readback");) [cmd_buf commit];
            [buf_dst release];

            // do not wait here for completion
            //[cmd_buf waitUntilCompleted];

            // instead, remember a reference to the command buffer and wait for it later if needed
            [ctx->cmd_bufs_ext addObject:cmd_buf];
            ctx->cmd_buf_last = cmd_buf;

            [cmd_buf retain];
        } else {
            // Discrete GPU: a shared staging buffer (kept and reused per context when the hw row says staging_keep), blit, wait, then memcpy
            const bool keep = ggml_metal_device_get_profile(ctx->dev)->staging_keep == 1;
            id<MTLBuffer> buf_staging = nil;
            // a pending readback owns the kept buffer until it completes; the next one gets its own
            if (keep && [ctx->get_pending count] == 0) {
                if (ctx->staging_kept == nil || [ctx->staging_kept length] < size) {
                    [ctx->staging_kept release];
                    ctx->staging_kept = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
                }
                buf_staging = [ctx->staging_kept retain];
            } else {
                buf_staging = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
            }
            GGML_ASSERT(buf_staging);

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_staging
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
            GGML_METAL_PROBE_ONLY(ggml_metal_probe_transfer(cmd_buf, "readback");) [cmd_buf commit];
            if (ctx->get_defer) {
                // as on shared memory: the data is valid after synchronize
                [ctx->get_pending addObject:@[cmd_buf, buf_staging, [NSValue valueWithPointer:data], [NSNumber numberWithUnsignedLong:size]]];
                [buf_staging release];
                ggml_metal_xfer_stats.d2h_bytes   += (int64_t)size;
                ggml_metal_xfer_stats.d2h_time_us += ggml_time_us() - t0;
                ggml_metal_xfer_stats.d2h_count   += 1;
                return;
            }
            const bool l0_trace = ggml_l0_enabled();
            const double l0_wait_start = l0_trace ? ggml_l0_now() : 0;
            [cmd_buf waitUntilCompleted];
            if (l0_trace) {
                const double start = [cmd_buf GPUStartTime];
                const double end = [cmd_buf GPUEndTime];
                ggml_l0_event(start == 0 || end == 0 || end < start ? "gpu_d2h_cb_invalid" : "gpu_d2h_cb", start, end, size);
                ggml_l0_event("d2h_wait", l0_wait_start, ggml_l0_now(), size);
            }

            const double l0_memcpy_start = l0_trace ? ggml_l0_now() : 0;
            memcpy(data, [buf_staging contents], size);
            if (l0_trace) {
                ggml_l0_event("d2h_memcpy", l0_memcpy_start, ggml_l0_now(), size);
            }
            [buf_staging release];
        }
    }

    ggml_metal_xfer_stats.d2h_bytes   += (int64_t)size;
    ggml_metal_xfer_stats.d2h_time_us += ggml_time_us() - t0;
    ggml_metal_xfer_stats.d2h_count   += 1;
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = ggml_metal_new_command_buffer(queue, false);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        GGML_METAL_PROBE_ONLY(ggml_metal_probe_transfer(cmd_buf, "copy");) [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

