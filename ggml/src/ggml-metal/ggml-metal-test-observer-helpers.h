#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef GGML_METAL_TEST_OBSERVER

static inline bool ggml_metal_test_observer_add_size(size_t lhs, size_t rhs, size_t * result) {
    if (lhs > SIZE_MAX - rhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

static inline bool ggml_metal_test_observer_mul_size(size_t lhs, size_t rhs, size_t * result) {
    if (lhs != 0 && rhs > SIZE_MAX / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

static inline bool ggml_metal_test_observer_checked_span(
        const int64_t ne[4], const size_t nb[4], size_t type_size, int64_t block_size, size_t * span) {
    if (!ne || !nb || !span || type_size == 0 || block_size <= 0) {
        return false;
    }
    // Validate every dimension first: a negative extent rejects even when an
    // earlier dimension is zero (an empty tensor still has legal extents).
    for (int i = 0; i < 4; ++i) {
        if (ne[i] < 0) {
            return false;
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (ne[i] == 0) {
            *span = 0;
            return true;
        }
    }

    size_t result = 0;
    if (block_size == 1) {
        result = type_size;
    } else {
        if (ne[0] % block_size != 0 || !ggml_metal_test_observer_mul_size((size_t) (ne[0] / block_size), nb[0], &result)) {
            return false;
        }
    }
    for (int i = block_size == 1 ? 0 : 1; i < 4; ++i) {
        size_t extent = 0;
        if (!ggml_metal_test_observer_mul_size((size_t) (ne[i] - 1), nb[i], &extent) ||
            !ggml_metal_test_observer_add_size(result, extent, &result)) {
            return false;
        }
    }
    *span = result;
    return true;
}

static inline bool ggml_metal_test_observer_checked_range(
        bool storage_valid, uintptr_t base, uintptr_t data, size_t buffer_size, size_t span) {
    return storage_valid && base != 0 && data >= base && span <= buffer_size && data - base <= buffer_size - span;
}

// Validate one Metal sub-buffer view without forming data - view_base until
// address ordering has made that unsigned subtraction defined for this lookup.
static inline bool ggml_metal_test_observer_checked_view_lookup(
        uintptr_t data, uintptr_t view_base, size_t view_size, size_t span, size_t * offset) {
    if (!offset || view_base == 0 || data < view_base || span > view_size) {
        return false;
    }

    const uintptr_t address_offset = data - view_base;
    // Keep the narrowing to Metal's size_t offset representable before it is
    // stored in ggml_metal_buffer_id.
    if (address_offset > SIZE_MAX) {
        return false;
    }

    const size_t view_offset = (size_t) address_offset;
    if (view_offset > view_size - span) {
        return false;
    }

    *offset = view_offset;
    return true;
}

static inline bool ggml_metal_test_observer_graph_index(
        const int * indices, size_t count, int local_index, int * graph_index) {
    if (!indices || !graph_index || local_index < 0 || (size_t) local_index >= count) {
        return false;
    }
    *graph_index = indices[local_index];
    return true;
}

static inline bool ggml_metal_test_observer_mapped_range(
        uintptr_t opaque_id, size_t mapped_offset, size_t span) {
    return opaque_id != 0 && mapped_offset <= SIZE_MAX - span;
}

// Two mapped ranges share a physical buffer only when both opaque ids are
// present and equal; distinct nonzero ids are different buffers.
static inline bool ggml_metal_test_observer_same_id(uintptr_t lhs, uintptr_t rhs) {
    return lhs != 0 && lhs == rhs;
}

// Portable record assembled by the real observe path before forwarding.
// Stage is carried as int so this header stays free of Metal/backend types;
// callers pass the ggml_backend_metal_test_observer_stage value unchanged.
struct ggml_metal_test_observer_record {
    uintptr_t graph;
    uintptr_t node;
    int       graph_index;
    int       stage;
    bool      result;
    bool      barrier_issued;
    bool      concurrency_enabled;
};

static inline bool ggml_metal_test_observer_assemble_record(
        uintptr_t graph, uintptr_t node, const int * indices, size_t count, int local_index,
        int stage, bool result, bool barrier_issued, bool actual_concurrent,
        struct ggml_metal_test_observer_record * record) {
    if (!record) {
        return false;
    }
    int graph_index = -1;
    if (!ggml_metal_test_observer_graph_index(indices, count, local_index, &graph_index)) {
        return false;
    }
    record->graph               = graph;
    record->node                = node;
    record->graph_index         = graph_index;
    record->stage               = stage;
    record->result              = result;
    record->barrier_issued      = barrier_issued;
    record->concurrency_enabled = actual_concurrent;
    return true;
}

#ifdef __cplusplus
template<typename Callback, typename Event>
static inline void ggml_metal_test_observer_forward(Callback callback, const Event * event, void * user_data) {
    if (callback) {
        callback(event, user_data);
    }
}
#endif

#endif
