#pragma once

#if defined(GGML_METAL_PROBES)

#include "ggml.h"

#define GGML_METAL_PROBE_FAMILIES(X) \
    X(unknown) X(cpu) X(context) X(device) X(memory) X(llama) X(sampling) X(ops) \
    X(conv) X(copy) X(fft) X(flash_attn) X(fusion) X(matmul) X(misc) X(norm) \
    X(reduction) X(rope) X(rows) X(softmax) X(spatial) X(ssm) X(unary)

#define GGML_METAL_PROBE_ZONES(X) \
    X(supports) X(pipeline) X(encode) X(setup) X(graph_build) X(graph_alloc) \
    X(inputs) X(submit) X(commit) X(wait) X(upload) X(readback) X(sample) X(decode) \
    X(token) X(compute) X(gpu_dispatch) X(gpu_encoder) X(gpu_buffer) X(gpu_invalid) X(gpu_dropped)

#define GGML_METAL_FAMILY_ID(name) GGML_METAL_PROBE_FAMILY_##name,
enum ggml_metal_probe_family { GGML_METAL_PROBE_FAMILIES(GGML_METAL_FAMILY_ID) };
#undef GGML_METAL_FAMILY_ID

#define GGML_METAL_ZONE_ID(name) GGML_METAL_PROBE_ZONE_##name,
enum ggml_metal_probe_zone { GGML_METAL_PROBE_ZONES(GGML_METAL_ZONE_ID) };
#undef GGML_METAL_ZONE_ID

enum ggml_metal_probe_gpu_mode {
    GGML_METAL_PROBE_BUFFER,
    GGML_METAL_PROBE_ENCODER,
    GGML_METAL_PROBE_DISPATCH,
};

struct ggml_metal_probe_scope {
    uint64_t start;
    uint32_t id;
    uint16_t previous_op;
    uint8_t previous_family;
};

#ifdef __cplusplus
extern "C" {
#endif

GGML_API bool ggml_metal_probe_enabled(void);
GGML_API uint16_t ggml_metal_probe_op(void);
GGML_API uint8_t ggml_metal_probe_family(void);
GGML_API struct ggml_metal_probe_scope ggml_metal_probe_begin(uint8_t family, uint8_t zone, uint16_t op);
GGML_API void ggml_metal_probe_end(struct ggml_metal_probe_scope * scope);
GGML_API void ggml_metal_probe_gpu(uint8_t family, uint8_t zone, uint16_t op, uint64_t device,
        const char * kernel, uint64_t start, uint64_t end);
GGML_API void ggml_metal_probe_count(uint8_t zone, uint64_t count);

#ifdef __cplusplus
}
#endif

#define GGML_METAL_PROBE_ONLY(...) __VA_ARGS__
#define GGML_METAL_FAMILY(name) enum { ggml_metal_probe_file_family = GGML_METAL_PROBE_FAMILY_##name };
#define GGML_METAL_ZONE_BEGIN_OP(zone, op) \
    struct ggml_metal_probe_scope ggml_metal_scope_##zone __attribute__((cleanup(ggml_metal_probe_end))) = \
        ggml_metal_probe_begin(ggml_metal_probe_file_family, GGML_METAL_PROBE_ZONE_##zone, (op))
#define GGML_METAL_ZONE_BEGIN(zone) GGML_METAL_ZONE_BEGIN_OP(zone, GGML_OP_NONE)
#define GGML_METAL_ZONE_END(zone) ggml_metal_probe_end(&ggml_metal_scope_##zone)

#else

#define GGML_METAL_PROBE_ONLY(...)
#define GGML_METAL_FAMILY(name)
#define GGML_METAL_ZONE_BEGIN_OP(zone, op)
#define GGML_METAL_ZONE_BEGIN(zone)
#define GGML_METAL_ZONE_END(zone)

#endif
