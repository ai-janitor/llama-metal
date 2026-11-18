#pragma once

#include <stdbool.h>
#include <stddef.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ggml.h"

// One retained main-graph command buffer slot owned by the Metal context.
struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

bool ggml_metal_gpu_error_callback_registered(void);
id<MTLCommandBuffer> ggml_metal_new_command_buffer(id<MTLCommandQueue> queue, bool unretained);
void ggml_metal_upload_flush(id<MTLCommandQueue> queue);
bool ggml_metal_upload_add(id<MTLCommandQueue> queue, id<MTLDevice> device, const void * data, size_t size, id<MTLBuffer> dst, size_t dst_offs);
void ggml_metal_report_command_buffer_error(void * command_buffer, int index, bool extra);

// Shared failed-buffer policy (R2). Completed -> SUCCESS, no callback.
// Failed + hook -> report, FAILED. Failed + no hook -> GGML_ABORT when fatal_without_hook, else FAILED.
enum ggml_status ggml_metal_check_command_buffer(id<MTLCommandBuffer> command_buffer, int index, bool extra, bool fatal_without_hook);

// Final status/drain of a graph's main buffers and extras (R3). With a hook: waits committed/scheduled buffers
// (never NotEnqueued), checks every buffer, and on any failure releases/nils main slots; extras are always
// released and cleared once. Without a hook: legacy behavior (abort on first failure, extras released as checked).
enum ggml_status ggml_metal_complete_command_buffers(struct ggml_metal_command_buffer * main, size_t count, NSMutableArray * extras);

// After a failure already reported through the check above: wait committed/scheduled buffers (no re-report),
// then release/nil main slots and release/clear extras.
void ggml_metal_discard_command_buffers(struct ggml_metal_command_buffer * main, size_t count, NSMutableArray * extras);
