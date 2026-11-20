// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// Optional process-wide command-buffer failure report. Registration is serialized with graph execution;
// payload strings and encoder entries are borrowed for the duration of callback only.
struct ggml_backend_metal_gpu_error_encoder {
    const char * label;
    const char * state;
    const char * signposts;
};
struct ggml_backend_metal_gpu_error {
    int64_t code;
    const char * description;
    const struct ggml_backend_metal_gpu_error_encoder * encoders;
    size_t n_encoders;
};
typedef void (*ggml_backend_metal_gpu_error_callback)(const struct ggml_backend_metal_gpu_error *, void *);
GGML_BACKEND_API void ggml_backend_metal_set_gpu_error_callback(
        ggml_backend_metal_gpu_error_callback callback, void * user_data);

// Set model profile for kernel tuning. Call after loading GGUF metadata.
// Enables cache-aware NR0 selection based on model dimensions + hardware L1 size.
// If not called, dispatch uses hardcoded defaults (backward compatible).
GGML_BACKEND_API void ggml_backend_metal_set_model_profile(ggml_backend_t backend,
        uint32_t n_embd, uint32_t n_ff, uint32_t n_layer,
        uint32_t n_head, uint32_t n_head_kv);

// The model's activation precision (GGML_TYPE_BF16, GGML_TYPE_F16; GGML_TYPE_COUNT = not known, the default).
// Tensors between ops stay f32. Ops that read their activations more than once may read a copy at this precision,
// on the cards whose rows say so (ggml-metal/act-copy).
GGML_BACKEND_API void ggml_backend_metal_set_act_precision(ggml_backend_t backend, enum ggml_type type);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

// Opt-in profiling instrumentation: re-arm the per-dispatch profile (GGML_METAL_PROFILE=3) of every Metal context
// that requested it and has counter sampling support, so the next n_graphs graph computes are profiled, without
// reloading the model. Contexts at profile 0/1/2, or at 3 without counter support, are left untouched.
// n_graphs <= 0 returns 0 and changes nothing. Returns the number of contexts armed (0 if unsupported or none).
// If never called, behavior is unchanged. Graph-boundary control, not a hot-loop API: the caller must synchronize
// and serialize all Metal contexts before arming and must not race context execution.
GGML_BACKEND_API int ggml_backend_metal_profile_window(int n_graphs);

// Candidate-loader capability probe (P12.18). Returns 1 when this core can compile a registered
// kernel from a candidate metallib selected through the GGML_METAL_TEST_SLOT / _VARIANT /
// _METALLIB / _METALLIB_SHA environment variables. It denotes
// interface support only: it opens no Metal device, loads no library and is NOT proof that any
// candidate executed. A host that cannot find this symbol must treat the core as having no
// candidate loader. The host must synchronize and serialize its Metal contexts before changing
// the selector variables.
GGML_BACKEND_API int ggml_backend_metal_test_candidate_loader_version(void);

// Dispatch receipt (P12.34): proof that a kernel was actually ENCODED for dispatch, not merely selected or
// compiled. Writes a UTF-8 JSON array of {"pipeline":string,"binary_sha":string,"count":integer} (one entry per
// pipeline name + binary identity, count = encoded dispatches in the window) into buf. binary_sha is "builtin"
// for ordinary pipelines or the candidate metallib SHA-256 for a pipeline created from a selected candidate.
// Returns the required size in bytes INCLUDING the terminating NUL. buf == NULL or len == 0 only queries the size;
// len smaller than the required size returns the required size and writes nothing. Returns -1 on error.
// reset != 0 clears old records, starts a new recording window and returns "[]"; reset == 0 reads the current
// snapshot without clearing it. Recording is off until the first reset != 0, so a host that never calls this
// sees no change. Counts mean encoded dispatches only; they say nothing about whether the command buffer completed.
// Thread-safe; makes no GPU call.
GGML_BACKEND_API int ggml_backend_metal_test_dispatch_receipt(char * buf, size_t len, int reset);

// The work plans of the tiled matmul kernel compiled into this build (the rows of ggml-metal/data/matmul.tsv [tiles]),
// for a tool that measures them: their count, and the id of plan i (NULL outside 0 .. count - 1). Plan 0 is the default.
// has_weight: 1 when the tiled kernel has an entry for this weight type. None of them opens a device.
GGML_BACKEND_API int          ggml_backend_metal_tiled_plan_count(void);
GGML_BACKEND_API const char * ggml_backend_metal_tiled_plan_id(int i);
GGML_BACKEND_API int          ggml_backend_metal_tiled_has_weight(enum ggml_type type);
// The eleven values of plan i in the column order of the table (warp to pad), written to values[0 .. 10]: returns 11,
// or 0 outside the plans or when n < 11.
GGML_BACKEND_API int          ggml_backend_metal_tiled_plan_values(int i, int32_t * values, int n);
// The shape a tuned row records for a card: the wave width probed on this Metal device and the threadgroup memory it
// allows. Returns 1, or 0 when dev is NULL or its wave width has not been probed (no backend was initialized on it).
GGML_BACKEND_API int          ggml_backend_metal_tiled_card_shape(ggml_backend_dev_t dev, uint32_t * wave, uint32_t * group_bytes);

#ifdef GGML_METAL_TEST_OBSERVER
// Test-only observation API. It is absent from production builds.
struct ggml_backend_metal_test_tensor_range {
    uintptr_t metal_buffer_id;
    size_t    offset;
    size_t    length;
};

enum ggml_backend_metal_test_observer_stage {
    GGML_BACKEND_METAL_TEST_OBSERVER_CHECK,
    GGML_BACKEND_METAL_TEST_OBSERVER_ADD,
    GGML_BACKEND_METAL_TEST_OBSERVER_RESET,
};

struct ggml_backend_metal_test_observer_event {
    uintptr_t graph;
    uintptr_t node;
    int       node_index;
    enum ggml_backend_metal_test_observer_stage stage;
    bool      result;
    bool      barrier_issued;
    bool      concurrency_enabled;
};

typedef void (*ggml_backend_metal_test_observer_callback)(
        const struct ggml_backend_metal_test_observer_event * event,
        void * user_data);

GGML_BACKEND_API bool ggml_backend_metal_test_tensor_range(
        ggml_backend_t backend, const struct ggml_tensor * tensor,
        struct ggml_backend_metal_test_tensor_range * range);
GGML_BACKEND_API void ggml_backend_metal_test_set_observer(
        ggml_backend_t backend, ggml_backend_metal_test_observer_callback callback,
        void * user_data);
#endif

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

#ifdef __cplusplus
}
#endif
