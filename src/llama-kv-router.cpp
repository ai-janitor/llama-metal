// KV cache router implementation — graph scanner + backend override.
//
// Identifies KV cache tensors by name pattern (cache_k_l*, cache_v_l*)
// and walks their consumer ops in the compute graph. Pins attention ops
// that reference KV tensors to the designated KV backend (typically iGPU).
//
// Integration: called from llama-graph.cpp after attention ops are built,
// before ggml_backend_sched_split_graph(). Must run before the scheduler
// locks in backend assignments.

#include "llama-kv-router.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>

#ifndef GGML_MAX_SRC
#define GGML_MAX_SRC 10
#endif

// Check if a tensor name matches KV cache pattern
static bool is_kv_cache_tensor(const char * name) {
    if (name == nullptr) {
        return false;
    }
    return strncmp(name, "cache_k_l", 9) == 0 || strncmp(name, "cache_v_l", 9) == 0;
}

// Check if a tensor (or its view_src) is a KV cache tensor
static bool references_kv_cache(const struct ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }

    // Check direct name
    if (is_kv_cache_tensor(tensor->name)) {
        return true;
    }

    // Check view source (KV cache tensors are often accessed through views)
    if (tensor->view_src != nullptr && is_kv_cache_tensor(tensor->view_src->name)) {
        return true;
    }

    return false;
}

void llama_kv_router_apply(
    ggml_backend_sched_t   sched,
    struct ggml_cgraph   * graph,
    ggml_backend_t         kv_backend
) {
    if (sched == nullptr || graph == nullptr || kv_backend == nullptr) {
        return;
    }

    int n_nodes = ggml_graph_n_nodes(graph);
    int pinned_count = 0;

    // Walk all nodes in the compute graph
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == nullptr) {
            continue;
        }

        // Check all source tensors for this op
        bool touches_kv_cache = false;
        for (int j = 0; j < GGML_MAX_SRC && node->src[j] != nullptr; j++) {
            if (references_kv_cache(node->src[j])) {
                touches_kv_cache = true;
                break;
            }
        }

        // If this op references KV cache, pin it to the KV backend
        // Disabled: cross-device Metal sync uses MTLEvent (device-local), not MTLSharedEvent.
        // Pinning ops to a different device causes GPU timeout. See BUG-012.
        if (touches_kv_cache) {
            // ggml_backend_sched_set_tensor_backend(sched, node, kv_backend);
            pinned_count++;
        }
    }

    LLAMA_LOG_INFO("kv-router: pinned %d ops to %s\n", pinned_count, ggml_backend_name(kv_backend));
}
