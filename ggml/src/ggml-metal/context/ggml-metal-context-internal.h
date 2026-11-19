#pragma once

#import "ggml-metal-context.h"

#import <mach/mach_time.h>
#import <pthread.h>

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-command-buffer-error.h"
#import "ggml-metal-ops.h"
#import "ggml-device-profile.h"
#import "act-copy/act-copy.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy; // for async copies

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;

    // convert-once state: the model's precision and the scratch copy (act-copy/)
    struct ggml_metal_act_copy * act_copy;

    // logits staging buffer kept across reads when the hw row says staging_keep (discrete cards)
    id<MTLBuffer> staging_kept;
    // discrete-card readbacks whose blit is committed; synchronize or the next graph compute waits and copies out
    NSMutableArray * get_pending;
    bool get_defer;
    bool use_concurrency;
    bool use_graph_optimize;
#ifdef GGML_METAL_TEST_OBSERVER
    ggml_backend_metal_test_observer_callback observer;
    void * observer_user_data;
#endif
    int profile;        // GGML_METAL_PROFILE: 0=off, 1=per-graph GPU timing, 2=per-op GPU timing, 3=per-dispatch timestamps
    int profile_n;      // number of graph computes left to profile at level 2/3 (then auto-drops to 1)
    uint64_t profile_cpu_start; // mach_absolute_time at graph_compute start (for CPU vs GPU comparison)

    // per-dispatch profiler (profile=3): uses MTLCounterSampleBuffer for GPU timestamps
    // in the normal async multi-CB path without serializing execution
    ggml_metal_profiler_t profiler;
#ifdef GGML_METAL_PROBES
    ggml_metal_profiler_t probe_profiler;
    bool probe_graph_active;
#endif

    // explicit profile window (ggml_metal_profile_window). All fields are zero after calloc, which keeps
    // the legacy GGML_METAL_PROFILE behavior byte-for-byte: window_enabled == false selects the original branches.
    int  profile_ctor;            // profile level requested at init (before the unsupported-counter fallback to 1);
                                  // a window may only arm a context that requested 3 AND still owns a profiler
    bool window_enabled;          // true once armed: graphs are profiled only while window_left > 0, profile stays 3
    int  window_left;             // graphs still to profile in the armed window; decremented once per reported graph
    bool window_graph_active;     // true between begin_graph (graph_compute) and its report (synchronize)
    struct ggml_metal * registry_next; // intrusive link in the g_registry list; protected by g_registry_mutex

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    bool capture_next_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};


// shared across the split files
extern pthread_mutex_t g_registry_mutex;
extern struct ggml_metal * g_registry_head;
void ggml_metal_get_pending_complete(ggml_metal_t ctx);
enum ggml_status ggml_metal_synchronize_checked(ggml_metal_t ctx);
enum ggml_status ggml_metal_graph_failed(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf, int index);
enum ggml_status ggml_metal_graph_compute_per_op(ggml_metal_t ctx, struct ggml_cgraph * gf);

#ifdef GGML_METAL_PROBES
ggml_metal_profiler_t ggml_metal_probe_profiler_init(ggml_metal_device_t dev);
bool ggml_metal_probe_profiler_begin(ggml_metal_profiler_t prof, int n_nodes);
bool ggml_metal_probe_profiler_finish(ggml_metal_profiler_t prof);
void ggml_metal_probe_transfer(id<MTLCommandBuffer> cb, const char * kind);
#endif
