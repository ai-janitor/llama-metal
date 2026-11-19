#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(device) //
// Per-dispatch GPU profiler (GGML_METAL_PROFILE=3)
//

#define PROFILER_MAX_CB          9   // 8 worker CBs + 1 main thread CB (matches GGML_METAL_MAX_COMMAND_BUFFERS + 1)
#define PROFILER_MAX_DISPATCHES  1024
#define PROFILER_SAMPLES_PER_DISPATCH 2

struct profiler_dispatch_record {
    int          cb_idx;
    int          sample_start;   // index into this CB's sample buffer
    int          sample_end;
    const char * op_name;        // ggml op (e.g. "MUL_MAT") — pointer to static string
    const char * pipeline_name;  // Metal kernel name — interned copy
    int          tg0, tg1, tg2;
    int          tptg0, tptg1, tptg2;
    int32_t      ne00, ne01;     // tensor shape for shape-attributed profiling
#ifdef GGML_METAL_PROBES
    uint16_t probe_op;
    uint8_t probe_family;
#endif
    uint64_t     gpu_ns;         // filled at resolve time
};

struct profiler_cb_state {
    id<MTLCounterSampleBuffer> sample_buffer;
    int next_sample_idx;
    int max_samples;
#ifdef GGML_METAL_PROBES
    id<MTLCommandBuffer> probe_cb;
    const char * probe_kernel;
    uint16_t probe_op;
    uint8_t probe_family;
#endif

    // per-CB dispatch records (thread-local to each encoding thread — no synchronization needed)
    struct profiler_dispatch_record records[PROFILER_MAX_DISPATCHES];
    int n_records;

    // per-CB string intern table for pipeline names (thread-local)
    char label_pool[256][128];
    int  n_labels;
};

struct ggml_metal_profiler {
    bool active;
#ifdef GGML_METAL_PROBES
    bool probe;
    enum ggml_metal_probe_gpu_mode probe_mode;
    MTLTimestamp probe_cpu0, probe_gpu0;
#endif

    // counter set reference (cached at init)
    id<MTLCounterSet> timestamp_set;
    id<MTLDevice>     device;

    // per-command-buffer sample state (each CB is encoded by a single thread)
    struct profiler_cb_state cb_states[PROFILER_MAX_CB];
    int n_cb_states;  // set atomically from begin_graph, read-only during encoding

    // overflow tracking (printed in report footer if nonzero)
    int dropped_dispatches; // dispatches that couldn't be sampled (per-CB record array full)
};

#ifdef GGML_METAL_PROBES
static void probe_cb_track(struct profiler_cb_state * cbs, id<MTLCommandBuffer> cb);
static void probe_tag(struct profiler_cb_state * cbs, const char * name);
static id<MTLComputeCommandEncoder> probe_border_encoder(id<MTLCommandBuffer> cb, MTLComputePassDescriptor * desc,
        ggml_metal_profiler_t prof, struct profiler_cb_state * cbs);
#endif

// Intern a pipeline name string into per-CB label pool (thread-safe: each CB has its own pool)
static const char * profiler_intern_label(struct profiler_cb_state * cbs, const char * str) {
    if (!str) return "unknown";
    for (int i = 0; i < cbs->n_labels; i++) {
        if (strcmp(cbs->label_pool[i], str) == 0) return cbs->label_pool[i];
    }
    if (cbs->n_labels < 256) {
        strncpy(cbs->label_pool[cbs->n_labels], str, 127);
        cbs->label_pool[cbs->n_labels][127] = '\0';
        return cbs->label_pool[cbs->n_labels++];
    }
    return "label_pool_full"; // stable sentinel
}

ggml_metal_profiler_t ggml_metal_profiler_init(ggml_metal_device_t dev, int profile_level) {
    if (profile_level != 3) return NULL;

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    // find timestamp counter set
    id<MTLCounterSet> ts_set = nil;
    for (id<MTLCounterSet> cs in [device counterSets]) {
        if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            ts_set = cs;
            break;
        }
    }
    if (!ts_set) {
        fprintf(stderr, "[METAL_PROFILE] profile=3: no timestamp counter set on %s, falling back to profile=1\n",
                device.name.UTF8String);
        return NULL;
    }

    if (![device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
        fprintf(stderr, "[METAL_PROFILE] profile=3: dispatch boundary sampling unsupported on %s, falling back to profile=1\n",
                device.name.UTF8String);
        return NULL;
    }

    ggml_metal_profiler_t prof = calloc(1, sizeof(struct ggml_metal_profiler));
    prof->active = true;
    prof->timestamp_set = ts_set;
    prof->device = device;

    fprintf(stderr, "[METAL_PROFILE] profile=3: per-dispatch timestamp profiling enabled on %s\n",
            device.name.UTF8String);

    return prof;
}

void ggml_metal_profiler_free(ggml_metal_profiler_t prof) {
    if (!prof) return;
    // MRC: release sample buffers (created via newCounterSampleBufferWithDescriptor: → +1 ownership)
    for (int i = 0; i < PROFILER_MAX_CB; i++) {
        if (prof->cb_states[i].sample_buffer) {
            [prof->cb_states[i].sample_buffer release];
            prof->cb_states[i].sample_buffer = nil;
        }
    }
    GGML_METAL_PROBE_ONLY(
        for (int i = 0; i < PROFILER_MAX_CB; ++i) {
            [prof->cb_states[i].probe_cb release];
        }
    )
    free(prof);
}

void ggml_metal_profiler_begin_graph(ggml_metal_profiler_t prof, int n_nodes) {
    if (!prof || !prof->active) return;
    (void)n_nodes;
    prof->dropped_dispatches = 0;
    for (int i = 0; i < PROFILER_MAX_CB; i++) {
        prof->cb_states[i].next_sample_idx = 0;
        prof->cb_states[i].n_records = 0;
        prof->cb_states[i].n_labels = 0;
    }
}

ggml_metal_encoder_t ggml_metal_encoder_init_profiled(
    ggml_metal_cmd_buf_t  cmd_buf_raw,
    bool                  concurrent,
    ggml_metal_profiler_t prof,
    int                   cb_idx
) {
    // if no profiler, fall through to standard encoder creation
    if (!prof || !prof->active || cb_idx < 0 || cb_idx >= PROFILER_MAX_CB) {
        return ggml_metal_encoder_init(cmd_buf_raw, concurrent);
    }

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    // allocate or reuse counter sample buffer for this CB
    struct profiler_cb_state * cbs = &prof->cb_states[cb_idx];
    GGML_METAL_PROBE_ONLY(
        if (prof->probe) {
            probe_cb_track(cbs, cmd_buf);
            if (prof->probe_mode == GGML_METAL_PROBE_BUFFER) {
                ggml_metal_encoder_t enc = ggml_metal_encoder_init(cmd_buf_raw, concurrent);
                enc->profiler = prof;
                enc->profiler_cb_idx = cb_idx;
                if (prof->n_cb_states <= cb_idx) {
                    prof->n_cb_states = cb_idx + 1;
                }
                return enc;
            }
        }
    )
    if (cbs->sample_buffer == nil) {
        MTLCounterSampleBufferDescriptor * desc = [[MTLCounterSampleBufferDescriptor alloc] init];
        desc.counterSet  = prof->timestamp_set;
        desc.sampleCount = PROFILER_MAX_DISPATCHES * PROFILER_SAMPLES_PER_DISPATCH;
        desc.storageMode = MTLStorageModeShared;
        desc.label       = @"ggml_profile3";

        NSError * err = nil;
        cbs->sample_buffer = [prof->device newCounterSampleBufferWithDescriptor:desc error:&err];
        [desc release]; // MRC: descriptor no longer needed after buffer creation

        if (!cbs->sample_buffer) {
            GGML_METAL_PROBE_ONLY(
                if (prof->probe) {
                    ggml_metal_probe_count(GGML_METAL_PROBE_ZONE_gpu_invalid, 1);
                    ggml_metal_encoder_t enc = ggml_metal_encoder_init(cmd_buf_raw, concurrent);
                    enc->profiler = prof;
                    enc->profiler_cb_idx = cb_idx;
                    if (prof->n_cb_states <= cb_idx) {
                        prof->n_cb_states = cb_idx + 1;
                    }
                    return enc;
                }
            )
            fprintf(stderr, "[METAL_PROFILE] failed to allocate sample buffer for cb %d: %s\n",
                    cb_idx, err.localizedDescription.UTF8String);
            return ggml_metal_encoder_init(cmd_buf_raw, concurrent);
        }
        cbs->max_samples = PROFILER_MAX_DISPATCHES * PROFILER_SAMPLES_PER_DISPATCH;
    }
    cbs->next_sample_idx = 0;

    // create encoder with sample buffer attachment via compute pass descriptor
    MTLComputePassDescriptor * pass_desc = [MTLComputePassDescriptor computePassDescriptor];
    pass_desc.sampleBufferAttachments[0].sampleBuffer = cbs->sample_buffer;
    pass_desc.sampleBufferAttachments[0].startOfEncoderSampleIndex = MTLCounterDontSample;
    pass_desc.sampleBufferAttachments[0].endOfEncoderSampleIndex   = MTLCounterDontSample;

    // note: computeCommandEncoderWithDescriptor does not support MTLDispatchTypeConcurrent,
    // so when profiling is active we always use serial dispatch. This is acceptable because
    // profile=3 is a debug mode and concurrency is already disabled on discrete AMD GPUs.
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));
    // Profiled encoders are always serial (see note above), so the actual mode is false.
    res->concurrent = false;
#ifdef GGML_METAL_PROBES
    res->obj = probe_border_encoder(cmd_buf, pass_desc, prof, cbs);
#else
    res->obj = [cmd_buf computeCommandEncoderWithDescriptor:pass_desc];
#endif
    [res->obj retain];

    // CRITICAL: store profiler state on the encoder so dispatch can route to profiled path
    res->profiler              = prof;
    res->profiler_cb_idx       = cb_idx;
    res->profiler_pipeline_name = NULL;
    res->profiler_op_name      = NULL;

    // track highest CB index seen — safe because main thread (highest cb_idx = n_cb)
    // runs first, before dispatch_apply launches worker threads (cb_idx = 0..n_cb-1)
    if (prof->n_cb_states <= cb_idx) {
        prof->n_cb_states = cb_idx + 1;
    }

    return res;
}

void ggml_metal_encoder_dispatch_profiled(
    ggml_metal_encoder_t  encoder,
    int tg0, int tg1, int tg2,
    int tptg0, int tptg1, int tptg2,
    ggml_metal_profiler_t prof,
    int                   cb_idx,
    const char *          op_name,
    const char *          pipeline_name
) {
    // fast path: no profiler → standard dispatch
    if (!prof || !prof->active || cb_idx < 0 || cb_idx >= PROFILER_MAX_CB) {
        [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2)
                     threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
        return;
    }

    struct profiler_cb_state * cbs = &prof->cb_states[cb_idx];
    GGML_METAL_PROBE_ONLY(
        if (prof->probe) {
            probe_tag(cbs, pipeline_name);
            if (prof->probe_mode != GGML_METAL_PROBE_DISPATCH ||
                    !cbs->sample_buffer || cbs->next_sample_idx + 2 > cbs->max_samples) {
                if (prof->probe_mode == GGML_METAL_PROBE_DISPATCH) {
                    __atomic_fetch_add(&prof->dropped_dispatches, 1, __ATOMIC_RELAXED);
                }
                [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2)
                             threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
                return;
            }
        }
    )
    if (!cbs->sample_buffer || cbs->next_sample_idx + 2 > cbs->max_samples) {
        // no sample buffer or overflow — dispatch without sampling
        [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2)
                     threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
        return;
    }

    if (cbs->n_records >= PROFILER_MAX_DISPATCHES) {
        // per-CB dispatch record array full — dispatch without sampling, count the drop
        __atomic_fetch_add(&prof->dropped_dispatches, 1, __ATOMIC_RELAXED);
        [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2)
                     threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
        return;
    }

    int start_idx = cbs->next_sample_idx++;
    int end_idx   = cbs->next_sample_idx++;

    // pre-dispatch sample
    [encoder->obj sampleCountersInBuffer:cbs->sample_buffer
                           atSampleIndex:(NSUInteger)start_idx
                             withBarrier:NO];

    // the actual dispatch
    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2)
                 threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];

    // post-dispatch sample
    [encoder->obj sampleCountersInBuffer:cbs->sample_buffer
                           atSampleIndex:(NSUInteger)end_idx
                             withBarrier:NO];

    // record metadata (per-CB — no thread contention)
    struct profiler_dispatch_record * rec = &cbs->records[cbs->n_records++];
    rec->cb_idx         = cb_idx;
    rec->sample_start   = start_idx;
    rec->sample_end     = end_idx;
    rec->op_name        = op_name;  // pointer to static ggml_op_name() string
    rec->pipeline_name  = profiler_intern_label(cbs, pipeline_name);
    rec->tg0 = tg0; rec->tg1 = tg1; rec->tg2 = tg2;
    rec->tptg0 = tptg0; rec->tptg1 = tptg1; rec->tptg2 = tptg2;
    rec->ne00 = encoder->profiler_ne00;
    rec->ne01 = encoder->profiler_ne01;
    rec->gpu_ns = 0;
    GGML_METAL_PROBE_ONLY(
        if (prof->probe) {
            rec->probe_family = ggml_metal_probe_family();
            rec->probe_op = ggml_metal_probe_op();
        }
    )
}

void ggml_metal_profiler_end_graph(ggml_metal_profiler_t prof) {
    if (!prof || !prof->active) return;

    // check if any CB has records
    int total_records = 0;
    for (int cb = 0; cb < prof->n_cb_states; cb++) {
        total_records += prof->cb_states[cb].n_records;
    }
    if (total_records == 0) return;

    typedef struct { uint64_t timestamp; } ts_result_t;

    // RESOLVE PHASE: decode timestamps from each CB's sample buffer
    // @autoreleasepool: NSData from resolveCounterRange: is autoreleased — drain here
    @autoreleasepool {
    for (int cb = 0; cb < prof->n_cb_states; cb++) {
        struct profiler_cb_state * cbs = &prof->cb_states[cb];
        if (!cbs->sample_buffer || cbs->next_sample_idx == 0 || cbs->n_records == 0) continue;

        NSData * resolved = nil;
        @try {
            resolved = [cbs->sample_buffer resolveCounterRange:NSMakeRange(0, cbs->next_sample_idx)];
        } @catch (NSException * e) {
            fprintf(stderr, "[METAL_PROFILE] resolve failed for cb %d: %s\n", cb, e.reason.UTF8String);
            continue;
        }
        if (!resolved) continue;

        const ts_result_t * ts = (const ts_result_t *)resolved.bytes;

        for (int i = 0; i < cbs->n_records; i++) {
            struct profiler_dispatch_record * rec = &cbs->records[i];
            uint64_t t0 = ts[rec->sample_start].timestamp;
            uint64_t t1 = ts[rec->sample_end].timestamp;
            if (t0 == ~0ULL || t1 == ~0ULL || t1 <= t0) {
                rec->gpu_ns = 0;
            } else {
                rec->gpu_ns = t1 - t0;
            }
        }
    }
    } // @autoreleasepool — drains NSData from resolveCounterRange

    // AGGREGATE PHASE: merge per-CB records and group by (op_name, pipeline_name, ne00)
    // ne00 bucketing allows shape-attributed profiling for matmul ops — different matrix
    // dimensions often want different kernel configurations on the same GPU.
    struct { const char * op; const char * pipe; int32_t ne00; int32_t ne01; uint64_t total_ns; int count; } aggs[256];
    int n_aggs = 0;
    uint64_t graph_total_ns = 0;
    int valid_dispatches = 0;
    int n_agg_overflow = 0;

    for (int cb = 0; cb < prof->n_cb_states; cb++) {
        struct profiler_cb_state * cbs = &prof->cb_states[cb];
        for (int i = 0; i < cbs->n_records; i++) {
            struct profiler_dispatch_record * rec = &cbs->records[i];
            if (rec->gpu_ns == 0) continue;

            graph_total_ns += rec->gpu_ns;
            valid_dispatches++;

            // bucket by (op, pipeline, ne00, ne01) — shape-attributed profiling
            // for non-matmul ops, ne00/ne01 are 0 so they aggregate as before
            int found = -1;
            for (int j = 0; j < n_aggs; j++) {
                if (aggs[j].op == rec->op_name &&
                    strcmp(aggs[j].pipe, rec->pipeline_name) == 0 &&
                    aggs[j].ne00 == rec->ne00 &&
                    aggs[j].ne01 == rec->ne01) {
                    found = j;
                    break;
                }
            }
            if (found >= 0) {
                aggs[found].total_ns += rec->gpu_ns;
                aggs[found].count++;
            } else if (n_aggs < 256) {
                aggs[n_aggs].op = rec->op_name;
                aggs[n_aggs].pipe = rec->pipeline_name;
                aggs[n_aggs].ne00 = rec->ne00;
                aggs[n_aggs].ne01 = rec->ne01;
                aggs[n_aggs].total_ns = rec->gpu_ns;
                aggs[n_aggs].count = 1;
                n_aggs++;
            } else {
                n_agg_overflow++;
            }
        }
    }

    // SORT descending by total_ns
    for (int i = 0; i < n_aggs - 1; i++) {
        for (int j = i + 1; j < n_aggs; j++) {
            if (aggs[j].total_ns > aggs[i].total_ns) {
                __typeof__(aggs[0]) tmp = aggs[i]; aggs[i] = aggs[j]; aggs[j] = tmp;
            }
        }
    }

    // REPORT
    int invalid = total_records - valid_dispatches;
    fprintf(stderr, "[METAL_PROFILE] dispatch breakdown (graph %.2f ms, %d dispatches", graph_total_ns / 1e6, total_records);
    if (invalid > 0) fprintf(stderr, ", %d invalid", invalid);
    if (prof->dropped_dispatches > 0) fprintf(stderr, ", %d dropped", prof->dropped_dispatches);
    if (n_agg_overflow > 0) fprintf(stderr, ", %d agg overflow", n_agg_overflow);
    fprintf(stderr, "):\n");

    for (int i = 0; i < n_aggs; i++) {
        double ms  = aggs[i].total_ns / 1e6;
        double pct = graph_total_ns > 0 ? (aggs[i].total_ns * 100.0 / graph_total_ns) : 0;
        double avg = aggs[i].count > 0 ? (aggs[i].total_ns / (double)aggs[i].count / 1e3) : 0;
        // show ne00xne01 shape when available (nonzero) — shape-attributed profiling
        if (aggs[i].ne00 > 0) {
            fprintf(stderr, "  %-24s %-32s [%5dx%-5d] %8.2f ms (%5.1f%%) x%-5d avg=%.1f us\n",
                    aggs[i].op, aggs[i].pipe, aggs[i].ne00, aggs[i].ne01, ms, pct, aggs[i].count, avg);
        } else {
            fprintf(stderr, "  %-24s %-32s              %8.2f ms (%5.1f%%) x%-5d avg=%.1f us\n",
                    aggs[i].op, aggs[i].pipe, ms, pct, aggs[i].count, avg);
        }
    }
}


#ifdef GGML_METAL_PROBES

ggml_metal_profiler_t ggml_metal_probe_profiler_init(ggml_metal_device_t dev) {
    if (!ggml_metal_probe_enabled()) {
        return NULL;
    }
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    ggml_metal_profiler_t prof = NULL;
    if (dev->probe_gpu_mode == GGML_METAL_PROBE_DISPATCH) {
        prof = ggml_metal_profiler_init(dev, 3);
    }
    if (!prof) {
        prof = calloc(1, sizeof(*prof));
        if (!prof) {
            return NULL;
        }
        prof->active = true;
        prof->device = device;
        if (dev->probe_gpu_mode == GGML_METAL_PROBE_ENCODER) {
            for (id<MTLCounterSet> set in device.counterSets) {
                if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                    prof->timestamp_set = set;
                    break;
                }
            }
        }
    }
    prof->probe = true;
    prof->probe_mode = prof->timestamp_set ? dev->probe_gpu_mode : GGML_METAL_PROBE_BUFFER;
    return prof;
}

static void probe_tag(struct profiler_cb_state * cbs, const char * name) {
    const uint8_t family = ggml_metal_probe_family();
    const uint16_t op = ggml_metal_probe_op();
    if (!cbs->probe_kernel) {
        cbs->probe_family = family;
        cbs->probe_op = op;
        cbs->probe_kernel = profiler_intern_label(cbs, name);
    } else if (cbs->probe_family != family || cbs->probe_op != op ||
               strcmp(cbs->probe_kernel, name) != 0) {
        cbs->probe_family = GGML_METAL_PROBE_FAMILY_context;
        cbs->probe_op = GGML_OP_NONE;
        cbs->probe_kernel = "mixed";
    }
}

static id<MTLComputeCommandEncoder> probe_border_encoder(id<MTLCommandBuffer> cb, MTLComputePassDescriptor * desc,
        ggml_metal_profiler_t prof, struct profiler_cb_state * cbs) {
    if (prof->probe && prof->probe_mode == GGML_METAL_PROBE_ENCODER) {
        desc.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0;
        desc.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
        cbs->next_sample_idx = 2;
    }
    return [cb computeCommandEncoderWithDescriptor:desc];
}

static void probe_cb_track(struct profiler_cb_state * cbs, id<MTLCommandBuffer> cb) {
    cbs->probe_cb = [cb retain];
    cbs->probe_kernel = NULL;
    cbs->probe_family = GGML_METAL_PROBE_FAMILY_context;
    cbs->probe_op = GGML_OP_NONE;
}

static uint64_t probe_host_ns(MTLTimestamp sample, MTLTimestamp cpu0, MTLTimestamp gpu0, double scale) {
    if (sample == MTLCounterErrorValue || sample == 0 || scale <= 0) {
        return 0;
    }
    const double delta = sample >= gpu0 ? (double) (sample - gpu0) : -(double) (gpu0 - sample);
    const double ns = (double) cpu0 + delta * scale;
    return ns > 0 && ns < (double) UINT64_MAX ? (uint64_t) ns : 0;
}

static void probe_resolve_cb(ggml_metal_profiler_t prof, struct profiler_cb_state * cbs, double scale) {
    const uint64_t device = prof->device.registryID;
    const double start = cbs->probe_cb.GPUStartTime;
    const double end = cbs->probe_cb.GPUEndTime;
    ggml_metal_probe_gpu(cbs->probe_family, GGML_METAL_PROBE_ZONE_gpu_buffer, cbs->probe_op, device,
            cbs->probe_kernel ? cbs->probe_kernel : "empty", (uint64_t) (start * 1e9), (uint64_t) (end * 1e9));
    if (!cbs->sample_buffer || !cbs->next_sample_idx) {
        return;
    }

    NSData * resolved = nil;
    @try {
        resolved = [cbs->sample_buffer resolveCounterRange:NSMakeRange(0, cbs->next_sample_idx)];
    } @catch (NSException * e) {
        (void) e;
    }
    if (resolved.length < (NSUInteger) cbs->next_sample_idx * sizeof(MTLCounterResultTimestamp)) {
        ggml_metal_probe_count(GGML_METAL_PROBE_ZONE_gpu_invalid, cbs->n_records > 0 ? (uint64_t) cbs->n_records : 1);
        return;
    }

    const MTLCounterResultTimestamp * samples = resolved.bytes;
    if (prof->probe_mode == GGML_METAL_PROBE_ENCODER) {
        ggml_metal_probe_gpu(cbs->probe_family, GGML_METAL_PROBE_ZONE_gpu_encoder, cbs->probe_op, device,
                cbs->probe_kernel,
                probe_host_ns(samples[0].timestamp, prof->probe_cpu0, prof->probe_gpu0, scale),
                probe_host_ns(samples[1].timestamp, prof->probe_cpu0, prof->probe_gpu0, scale));
        return;
    }
    for (int i = 0; i < cbs->n_records; ++i) {
        const struct profiler_dispatch_record * rec = &cbs->records[i];
        ggml_metal_probe_gpu(rec->probe_family, GGML_METAL_PROBE_ZONE_gpu_dispatch, rec->probe_op, device,
                rec->pipeline_name,
                probe_host_ns(samples[rec->sample_start].timestamp, prof->probe_cpu0, prof->probe_gpu0, scale),
                probe_host_ns(samples[rec->sample_end].timestamp, prof->probe_cpu0, prof->probe_gpu0, scale));
    }
}

bool ggml_metal_probe_profiler_finish(ggml_metal_profiler_t prof) {
    if (!prof) {
        return true;
    }
    bool any = false;
    for (int i = 0; i < prof->n_cb_states; ++i) {
        id<MTLCommandBuffer> cb = prof->cb_states[i].probe_cb;
        if (cb) {
            if (cb.status < MTLCommandBufferStatusCompleted) {
                return false;
            }
            any = true;
        }
    }
    if (!any) {
        return true;
    }

    double scale = 0;
    if (prof->timestamp_set) {
        MTLTimestamp cpu1 = 0, gpu1 = 0;
        [prof->device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
        if (cpu1 > prof->probe_cpu0 && gpu1 > prof->probe_gpu0 && prof->probe_gpu0 != 0) {
            scale = (double) (cpu1 - prof->probe_cpu0) / (double) (gpu1 - prof->probe_gpu0);
        }
    }
    @autoreleasepool {
        for (int i = 0; i < prof->n_cb_states; ++i) {
            struct profiler_cb_state * cbs = &prof->cb_states[i];
            if (cbs->probe_cb) {
                if (cbs->probe_cb.status == MTLCommandBufferStatusCompleted) {
                    probe_resolve_cb(prof, cbs, scale);
                } else {
                    ggml_metal_probe_count(GGML_METAL_PROBE_ZONE_gpu_invalid, cbs->n_records > 0 ? (uint64_t) cbs->n_records : 1);
                }
                [cbs->probe_cb release];
                cbs->probe_cb = nil;
            }
        }
    }
    ggml_metal_probe_count(GGML_METAL_PROBE_ZONE_gpu_dropped, prof->dropped_dispatches);
    return true;
}

bool ggml_metal_probe_profiler_begin(ggml_metal_profiler_t prof, int n_nodes) {
    if (!prof) {
        return false;
    }
    if (!ggml_metal_probe_profiler_finish(prof)) {
        ggml_metal_probe_count(GGML_METAL_PROBE_ZONE_gpu_dropped, 1);
        return false;
    }
    ggml_metal_profiler_begin_graph(prof, n_nodes);
    if (prof->timestamp_set) {
        [prof->device sampleTimestamps:&prof->probe_cpu0 gpuTimestamp:&prof->probe_gpu0];
    }
    return true;
}

void ggml_metal_probe_transfer(id<MTLCommandBuffer> cb, const char * kind) {
    if (!ggml_metal_probe_enabled()) {
        return;
    }
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (done.status == MTLCommandBufferStatusCompleted) {
            ggml_metal_probe_gpu(GGML_METAL_PROBE_FAMILY_memory, GGML_METAL_PROBE_ZONE_gpu_buffer, GGML_OP_NONE,
                    done.device.registryID, kind, (uint64_t) (done.GPUStartTime * 1e9), (uint64_t) (done.GPUEndTime * 1e9));
        } else {
            ggml_metal_probe_count(GGML_METAL_PROBE_ZONE_gpu_invalid, 1);
        }
    }];
}

#endif
