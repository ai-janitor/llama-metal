// Test if ggml_cont(ggml_reshape_4d(g, 1, 1, H, 1)) preserves data correctly on Metal
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

int main() {
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) { fprintf(stderr, "No Metal\n"); return 1; }

    struct ggml_init_params params = { 256*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    const int H = 32;

    // Create gate as [H, 1, 1] (same as model)
    ggml_tensor * g_orig = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 1, 1);

    // Fused path: reshape to [1, 1, H, 1] then cont
    ggml_tensor * g_reshaped = ggml_reshape_4d(ctx, g_orig, 1, 1, H, 1);
    ggml_tensor * g_cont = ggml_cont(ctx, g_reshaped);

    // Build graph
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, g_cont);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Set known values: gate[h] = -0.1 * (h+1)
    std::vector<float> g_data(H);
    for (int h = 0; h < H; h++) g_data[h] = -0.1f * (h + 1);
    ggml_backend_tensor_set(g_orig, g_data.data(), 0, H * sizeof(float));

    // Run graph (computes the cont)
    ggml_backend_graph_compute(backend, graph);

    // Read back cont result
    std::vector<float> g_result(H);
    ggml_backend_tensor_get(g_cont, g_result.data(), 0, H * sizeof(float));

    printf("g_cont shape: [%lld, %lld, %lld, %lld]\n",
           g_cont->ne[0], g_cont->ne[1], g_cont->ne[2], g_cont->ne[3]);
    printf("g_cont strides: [%zu, %zu, %zu, %zu]\n",
           g_cont->nb[0], g_cont->nb[1], g_cont->nb[2], g_cont->nb[3]);

    printf("\nComparison (original vs cont'd):\n");
    float max_diff = 0;
    for (int h = 0; h < H; h++) {
        float diff = fabsf(g_data[h] - g_result[h]);
        if (diff > max_diff) max_diff = diff;
        printf("  h=%2d: orig=%.6f cont=%.6f diff=%.8f\n", h, g_data[h], g_result[h], diff);
    }
    printf("\nMax diff: %.10f\n", max_diff);
    printf(max_diff < 1e-6 ? "*** CONT WORKS ***\n" : "*** CONT IS BROKEN ***\n");

    // Also test: what value does the kernel see at src3[h]?
    // The cont result [1, 1, 32, 1] has data at sequential offsets.
    // src3[h] reads the h-th float. Does it read g_result[h]?
    // Since the cont makes it contiguous, src3[h] = g_result[h].
    // But let's verify by checking what element (0, 0, h, 0) maps to:
    printf("\nElement access test:\n");
    for (int h = 0; h < 4; h++) {
        // For [1, 1, 32, 1], element (0, 0, h, 0) at byte offset h * nb[2]
        size_t byte_off = h * g_cont->nb[2];
        printf("  (0,0,%d,0) byte_off=%zu float_idx=%zu value=%.6f (expected %.6f)\n",
               h, byte_off, byte_off/4, g_result[byte_off/4], g_data[h]);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
