#import "context/ggml-metal-context-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(context) // Registry of fully initialized contexts, visited by ggml_metal_profile_window.
// Invariant: a context is linked only between the end of ggml_metal_init (success) and the start of
// ggml_metal_free, so the list never holds a failed/partial or freed context. The one mutex serializes
// list changes, arming, and context lifetime (init-register / unregister-free) against each other.
// It is NOT held across any GPU wait or dispatch; callers must quiesce the contexts before arming.
pthread_mutex_t g_registry_mutex = PTHREAD_MUTEX_INITIALIZER;
struct ggml_metal * g_registry_head = NULL;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));
    res->act_copy = ggml_metal_act_copy_init();

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    // TODO: would it be better to have one queue for the backend and one queue for the device?
    //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
    //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    if (queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    res->dev = dev;
    res->lib = ggml_metal_device_get_library(dev);
    if (res->lib == NULL) {
        GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
        GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

        res->lib = ggml_metal_library_init(dev);
        if (res->lib == NULL) {
            GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

            free(res);

            return NULL;
        }
    }

    res->ev_cpy = ggml_metal_device_event_init(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);
    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(dev);

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == nil;
    res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

    // P16.16 experiment flag, default off: GGML_METAL_CONCURRENCY_ALLOW_NONUMA=1 (literal "1" only; absent, "0" or
    // anything malformed is false) lets the existing concurrent encoder path run on a non-UMA GPU. It never changes
    // the UMA path and never overrides GGML_METAL_CONCURRENCY_DISABLE (presence of DISABLE keeps concurrency off).
    const bool concurrency_requested = res->use_concurrency;
    const char * allow_nonuma_env    = getenv("GGML_METAL_CONCURRENCY_ALLOW_NONUMA");
    const bool concurrency_allow_nonuma = allow_nonuma_env != NULL && strcmp(allow_nonuma_env, "1") == 0;
    const bool device_nonuma            = !profile->shared_memory;

    // MTLDispatchTypeConcurrent causes NaN on discrete (non-UMA) GPUs like AMD Radeon
    // The fence below is kept as is; only the explicit opt-in above steps past it.
    if (res->use_concurrency && device_nonuma) {
        if (concurrency_allow_nonuma) {
            GGML_LOG_WARN("%s: EXPERIMENTAL: GGML_METAL_CONCURRENCY_ALLOW_NONUMA=1 overrides the non-UMA concurrent dispatch "
                          "safety fence (known NaN on discrete GPUs); no hardware correctness or support is claimed\n", __func__);
        } else {
            GGML_LOG_WARN("%s: disabling concurrent dispatch (not supported on non-UMA devices)\n", __func__);
            res->use_concurrency = false;
        }
    }

    // One-shot trace so a runner can prove the override was (or was not) applied before generation
    GGML_LOG_INFO("%s: concurrency requested = %d, effective = %d, device_nonuma = %d, disable = %d, allow_nonuma = %d\n",
            __func__, concurrency_requested, res->use_concurrency, device_nonuma,
            getenv("GGML_METAL_CONCURRENCY_DISABLE") != NULL, concurrency_allow_nonuma);

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    // GGML_METAL_PROFILE: 0=off (default), 1=per-graph GPU wall time, 2=per-op GPU timing breakdown,
    // 3=per-dispatch timestamp sampling (async, uses MTLCounterSampleBuffer)
    // Setting the env var without a value (or empty) defaults to 1 for backward compatibility
    // Profile=2/3 runs for 2 graph computes (prompt + first decode), then drops to level 1
    {
        const char * val = getenv("GGML_METAL_PROFILE");
        res->profile   = val ? (val[0] ? atoi(val) : 1) : 0;
        res->profile_n = (res->profile >= 2) ? 2 : 0;
        res->profile_ctor = res->profile; // remember the request: the fallback below must not make a window arm this context
    }

    // init per-dispatch profiler for profile=3
    // if counter sampling is unsupported, profiler_init returns NULL and we fall back to profile=1
    res->profiler = NULL; GGML_METAL_PROBE_ONLY(res->probe_profiler = ggml_metal_probe_profiler_init(dev);)
    if (res->profile == 3) {
        res->profiler = ggml_metal_profiler_init(dev, 3);
        if (!res->profiler) {
            res->profile = 1; // fallback
        }
    }

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

    res->capture_next_compute = false;
    res->capture_started = false;
    res->capture_scope = nil;

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];
    res->get_pending  = [[NSMutableArray alloc] init];
    {
        // GGML_METAL_GET_DEFER=0: a discrete-card readback waits for its blit at once
        const char * defer_env = getenv("GGML_METAL_GET_DEFER");
        res->get_defer = !(defer_env && strcmp(defer_env, "0") == 0);
    }

    res->cmd_buf_last = nil;

    res->pipelines_ext = ggml_metal_pipelines_init();

    // register only now: every earlier return path left the context unregistered
    pthread_mutex_lock(&g_registry_mutex);
    res->registry_next = g_registry_head;
    g_registry_head = res;
    pthread_mutex_unlock(&g_registry_mutex);

    return res;
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    // unregister before the profiler/context are freed so ggml_metal_profile_window never sees a dangling context
    pthread_mutex_lock(&g_registry_mutex);
    for (struct ggml_metal ** link = &g_registry_head; *link; link = &(*link)->registry_next) {
        if (*link == ctx) {
            *link = ctx->registry_next;
            break;
        }
    }
    pthread_mutex_unlock(&g_registry_mutex);

    ggml_metal_profiler_free(ctx->profiler); GGML_METAL_PROBE_ONLY(ggml_metal_probe_profiler_finish(ctx->probe_profiler); ggml_metal_profiler_free(ctx->probe_profiler);)

    ggml_metal_transfer_stats_report();

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];
    [ctx->get_pending release];

    if (ctx->staging_kept) {
        [ctx->staging_kept release];
        ctx->staging_kept = nil;
    }

    ggml_metal_act_copy_free(ctx->act_copy);
    ctx->act_copy = NULL;

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

int ggml_metal_get_profile(ggml_metal_t ctx) {
    return ctx->profile;
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_set_act_precision(ggml_metal_t ctx, enum ggml_type type) {
    ggml_metal_act_copy_set_precision(ctx->act_copy, type);
}
