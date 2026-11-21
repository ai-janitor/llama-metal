#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(device) //
// MTLComputeCommandEncoder wrapper
//

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent) {
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));
    res->concurrent = concurrent;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    if (concurrent) {
        res->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
    } else {
        res->obj = [cmd_buf computeCommandEncoder];
    }

    [res->obj retain];

    // no profiling in standard path
    res->profiler = NULL;
    res->profiler_cb_idx = -1;
    res->profiler_pipeline_name = NULL;

    return res;
}

#ifdef GGML_METAL_TEST_OBSERVER
bool ggml_metal_encoder_is_concurrent(ggml_metal_encoder_t encoder) {
    return encoder->concurrent;
}
#endif

void ggml_metal_encoder_free(ggml_metal_encoder_t encoder) {
    [encoder->obj release];
    free(encoder);
}

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name) {
    [encoder->obj pushDebugGroup:[NSString stringWithCString:name encoding:NSUTF8StringEncoding]];
}

void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder) {
    [encoder->obj popDebugGroup];
}

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline) {
    [encoder->obj setComputePipelineState:pipeline.pipeline->obj];

    // P12.34: remember which pipeline (and which binary) the next dispatch will use; does not count
    memcpy(encoder->receipt_name, pipeline.pipeline->name,       sizeof(encoder->receipt_name));
    memcpy(encoder->receipt_sha,  pipeline.pipeline->binary_sha, sizeof(encoder->receipt_sha));
    encoder->launch_mem = 0;

    // capture pipeline name for profiler dispatch labels (if active)
    // uses the name stored on the pipeline struct at compile time (not Metal label,
    // which is unsupported on older AMD drivers like BronzeMtl on D700)
    if (encoder->profiler) {
        encoder->profiler_pipeline_name = pipeline.pipeline->name;
    }
}

void ggml_metal_encoder_set_bytes(ggml_metal_encoder_t encoder, void * data, size_t size, int idx) {
    [encoder->obj setBytes:data length:size atIndex:idx];
}

void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx) {
    [encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx];
}

size_t ggml_metal_buffer_id_length(struct ggml_metal_buffer_id buffer) {
    return buffer.metal ? (size_t) [(id<MTLBuffer>) buffer.metal length] : 0;
}

bool ggml_metal_buffer_get_gpu_address(struct ggml_metal_buffer_id buffer, uint64_t * address) {
    if (!buffer.metal || !address) {
        return false;
    }
    if (@available(macOS 13.0, *)) {
        const uint64_t base_address = [(id<MTLBuffer>) buffer.metal gpuAddress];
        if (base_address == 0 || buffer.offs > UINT64_MAX - base_address) {
            return false;
        }
        *address = base_address + buffer.offs;
        return true;
    }
    return false;
}

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx) {
    [encoder->obj setThreadgroupMemoryLength:size atIndex:idx];
    if (idx == 0) {
        encoder->launch_mem = size;
    }
}

// GGML_METAL_TEST_LAUNCHES set: one stderr line per launch (see ggml-metal-device.h).
static bool ggml_metal_launch_print_on(void) {
    static bool on;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        on = getenv("GGML_METAL_TEST_LAUNCHES") != NULL;
    });
    return on;
}

void ggml_metal_encoder_set_op_name(ggml_metal_encoder_t encoder, const char * op_name) {
    // stores op name for the next profiled dispatch
    // no-op when profiling is off (profiler field is NULL)
    encoder->profiler_op_name = op_name;
}

void ggml_metal_encoder_set_dispatch_shape(ggml_metal_encoder_t encoder, int32_t ne00, int32_t ne01) {
    // stores tensor shape for shape-attributed profiling
    // allows profile=3 to bucket dispatches by (op, kernel, ne00) instead of just (op, kernel)
    encoder->profiler_ne00 = ne00;
    encoder->profiler_ne01 = ne01;
}

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2) {
    if (ggml_metal_launch_print_on()) {
        fprintf(stderr, "LAUNCH %s grid=%d,%d,%d threads=%d,%d,%d mem=%zu\n",
                encoder->receipt_name, tg0, tg1, tg2, tptg0, tptg1, tptg2, encoder->launch_mem);
    }
    // P12.34: count this encoded dispatch exactly once, before the profiled/normal split below
    // (the profiled wrapper is only reached from here, so it must not record again)
    if (tg0 > 0 && tg1 > 0 && tg2 > 0) {
        ggml_metal_dispatch_receipt_record(encoder->receipt_name, encoder->receipt_sha, encoder->concurrent);
    }

    // when profiler is active on this encoder, route through the profiled dispatch path
    if (encoder->profiler) {
        ggml_metal_encoder_dispatch_profiled(
            encoder, tg0, tg1, tg2, tptg0, tptg1, tptg2,
            encoder->profiler, encoder->profiler_cb_idx,
            encoder->profiler_op_name ? encoder->profiler_op_name : "UNKNOWN_OP",
            encoder->profiler_pipeline_name ? encoder->profiler_pipeline_name : "unknown_kernel");
        return;
    }

    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
}

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder) {
    [encoder->obj endEncoding];
}

