#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <pthread.h>

#import "ggml-metal.h"
#import "ggml-metal-command-buffer-error.h"
#import "ggml-impl.h"

static pthread_mutex_t g_gpu_error_mutex = PTHREAD_MUTEX_INITIALIZER;
static ggml_backend_metal_gpu_error_callback g_gpu_error_callback;
static void * g_gpu_error_user_data;

void ggml_backend_metal_set_gpu_error_callback(ggml_backend_metal_gpu_error_callback callback, void * user_data) {
    pthread_mutex_lock(&g_gpu_error_mutex);
    g_gpu_error_callback = callback;
    g_gpu_error_user_data = user_data;
    pthread_mutex_unlock(&g_gpu_error_mutex);
}

bool ggml_metal_gpu_error_callback_registered(void) {
    pthread_mutex_lock(&g_gpu_error_mutex);
    const bool registered = g_gpu_error_callback != NULL;
    pthread_mutex_unlock(&g_gpu_error_mutex);
    return registered;
}

id<MTLCommandBuffer> ggml_metal_new_command_buffer(id<MTLCommandQueue> queue, bool unretained) {
    if (!ggml_metal_gpu_error_callback_registered()) {
        return unretained ? [queue commandBufferWithUnretainedReferences] : [queue commandBuffer];
    }

    MTLCommandBufferDescriptor * descriptor = [MTLCommandBufferDescriptor new];
    descriptor.retainedReferences = !unretained;
    descriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> command_buffer = [queue commandBufferWithDescriptor:descriptor];
    [descriptor release];
    return command_buffer;
}

void ggml_metal_report_command_buffer_error(void * raw_command_buffer, int index, bool extra) {
    pthread_mutex_lock(&g_gpu_error_mutex);
    ggml_backend_metal_gpu_error_callback callback = g_gpu_error_callback;
    void * user_data = g_gpu_error_user_data;
    pthread_mutex_unlock(&g_gpu_error_mutex);
    if (!callback) return;

    id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>) raw_command_buffer;
    NSError * error = [command_buffer error];
    NSArray<id<MTLCommandBufferEncoderInfo>> * infos = error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
    const NSUInteger n_infos = infos ? infos.count : 0;
    struct ggml_backend_metal_gpu_error_encoder encoders[n_infos ? n_infos : 1];
    for (NSUInteger i = 0; i < n_infos; ++i) {
        id<MTLCommandBufferEncoderInfo> info = infos[i];
        encoders[i] = (struct ggml_backend_metal_gpu_error_encoder) {
            info.label.UTF8String ? info.label.UTF8String : "", [[NSString stringWithFormat:@"%ld", (long) info.errorState] UTF8String],
            [info.debugSignposts componentsJoinedByString:@" | "].UTF8String ? [info.debugSignposts componentsJoinedByString:@" | "].UTF8String : ""
        };
    }
    NSString * description = error ? error.localizedDescription : @"missing NSError";
    struct ggml_backend_metal_gpu_error report = { error ? error.code : 0, description.UTF8String ? description.UTF8String : "", encoders, n_infos };
    (void) index;
    (void) extra;
    callback(&report, user_data);
}

// Retrospective legacy-policy variant (test build only): abort before the callback, as the original code did.
#ifdef GGML_METAL_TEST_LEGACY_ABORT_BEFORE_HOOK
#define GGML_METAL_LEGACY_ABORT 1
#else
#define GGML_METAL_LEGACY_ABORT 0
#endif

enum ggml_status ggml_metal_check_command_buffer(id<MTLCommandBuffer> command_buffer, int index, bool extra, bool fatal_without_hook) {
    const MTLCommandBufferStatus status = [command_buffer status];
    if (status == MTLCommandBufferStatusCompleted) {
        return GGML_STATUS_SUCCESS;
    }
    GGML_LOG_ERROR("%s: error: %scommand buffer %d failed with status %d\n", __func__, extra ? "ext " : "", index, (int) status);
    if (status == MTLCommandBufferStatusError) {
        GGML_LOG_ERROR("error: %s\n", [[command_buffer error].localizedDescription UTF8String]);
    }
    if (GGML_METAL_LEGACY_ABORT) {
        GGML_ABORT("fatal error");
    }
    if (ggml_metal_gpu_error_callback_registered()) {
        ggml_metal_report_command_buffer_error((__bridge void *) command_buffer, index, extra);
        return GGML_STATUS_FAILED;
    }
    if (fatal_without_hook) {
        GGML_ABORT("fatal error");
    }
    return GGML_STATUS_FAILED;
}

static void ggml_metal_wait_if_pending(id<MTLCommandBuffer> command_buffer) {
    const MTLCommandBufferStatus status = [command_buffer status];
    if (status == MTLCommandBufferStatusCommitted || status == MTLCommandBufferStatusScheduled) {
        [command_buffer waitUntilCompleted];
    }
}

void ggml_metal_discard_command_buffers(struct ggml_metal_command_buffer * main, size_t count, NSMutableArray * extras) {
    // drain first: a buffer still committed/scheduled is waited before its reference is dropped
    for (size_t i = 0; i < count; ++i) {
        if (main[i].obj) ggml_metal_wait_if_pending(main[i].obj);
    }
    for (NSUInteger i = 0; i < extras.count; ++i) {
        ggml_metal_wait_if_pending(extras[i]);
    }
    for (size_t i = 0; i < count; ++i) {
        if (main[i].obj) {
            [main[i].obj release];
            main[i].obj = nil;
        }
    }
    // the array owns one manual retain per element (taken where the buffers are appended)
    for (NSUInteger i = 0; i < extras.count; ++i) {
        [extras[i] release];
    }
    [extras removeAllObjects];
}

enum ggml_status ggml_metal_complete_command_buffers(struct ggml_metal_command_buffer * main, size_t count, NSMutableArray * extras) {
    if (!ggml_metal_gpu_error_callback_registered()) {
        for (size_t i = 0; i < count; ++i) {
            if (main[i].obj) {
                (void) ggml_metal_check_command_buffer(main[i].obj, (int) i, false, true);
            }
        }
        for (NSUInteger i = 0; i < extras.count; ++i) {
            id<MTLCommandBuffer> command_buffer = extras[i];
            (void) ggml_metal_check_command_buffer(command_buffer, (int) i, true, true);
            [command_buffer release];
        }
        [extras removeAllObjects];
        return GGML_STATUS_SUCCESS;
    }

    enum ggml_status result = GGML_STATUS_SUCCESS;
    for (size_t i = 0; i < count; ++i) {
        if (!main[i].obj) continue;
        ggml_metal_wait_if_pending(main[i].obj);
        if (ggml_metal_check_command_buffer(main[i].obj, (int) i, false, false) != GGML_STATUS_SUCCESS) {
            result = GGML_STATUS_FAILED;
        }
    }
    for (NSUInteger i = 0; i < extras.count; ++i) {
        id<MTLCommandBuffer> command_buffer = extras[i];
        ggml_metal_wait_if_pending(command_buffer);
        if (ggml_metal_check_command_buffer(command_buffer, (int) i, true, false) != GGML_STATUS_SUCCESS) {
            result = GGML_STATUS_FAILED;
        }
    }
    if (result != GGML_STATUS_SUCCESS) {
        ggml_metal_discard_command_buffers(main, count, extras);
    } else {
        for (NSUInteger i = 0; i < extras.count; ++i) {
            [extras[i] release];
        }
        [extras removeAllObjects];
    }
    return result;
}
