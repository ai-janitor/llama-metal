// Standalone test for the fused gated delta-net Metal kernel
// Compiles against ggml and runs the kernel with known inputs
// Build: cmake --build build --target test-fused-kernel
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>

#define S 128
#define H 32  // Qwen3.5-4B has 32 value heads
#define T 1  // single token per call, but we'll call multiple times
#define B 1  // n_seqs

int main() {
    // Init backend
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "Failed to init Metal backend\n");
        return 1;
    }

    // Create context
    struct ggml_init_params params = {
        .mem_size   = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Create tensors
    // k, v, q: [S, H, T]
    struct ggml_tensor * k     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, T);
    struct ggml_tensor * v     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, T);
    struct ggml_tensor * q     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, T);
    // gate, beta: [1, 1, H, B]
    struct ggml_tensor * gate  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, H, B);
    struct ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, H, B);
    // state: [S, S, H, B]
    struct ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, B);

    // Create the op
    struct ggml_tensor * result = ggml_gated_delta_net(ctx, k, v, q, gate, beta, state, 1.0f);

    // Build graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    // Allocate tensors
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "Failed to allocate tensors\n");
        return 1;
    }

    // Set known input values (same as CPU test)
    srand(42);
    auto fill = [](ggml_backend_t be, ggml_tensor * t) {
        std::vector<float> data(ggml_nelements(t));
        for (size_t i = 0; i < data.size(); i++)
            data[i] = 0.01f * (rand() % 100 - 50);
        ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    };

    fill(backend, k);
    fill(backend, v);
    fill(backend, q);
    fill(backend, state);

    // gate: negative log-decay
    {
        std::vector<float> g(H * B);
        for (int i = 0; i < H * B; i++) g[i] = -0.1f * (1 + rand() % 10);
        ggml_backend_tensor_set(gate, g.data(), 0, g.size() * sizeof(float));
    }
    // beta: (0, 1) range
    {
        std::vector<float> b(H * B);
        for (int i = 0; i < H * B; i++) b[i] = 0.1f * (1 + rand() % 8);
        ggml_backend_tensor_set(beta, b.data(), 0, b.size() * sizeof(float));
    }

    // Read back inputs for CPU reference
    float k_data[S*H*T], v_data[S*H*T], q_data[S*H*T];
    float gate_data[H*B], beta_data[H*B];
    float state_data[S*S*H*B];
    ggml_backend_tensor_get(k, k_data, 0, sizeof(k_data));
    ggml_backend_tensor_get(v, v_data, 0, sizeof(v_data));
    ggml_backend_tensor_get(q, q_data, 0, sizeof(q_data));
    ggml_backend_tensor_get(gate, gate_data, 0, sizeof(gate_data));
    ggml_backend_tensor_get(beta, beta_data, 0, sizeof(beta_data));
    ggml_backend_tensor_get(state, state_data, 0, sizeof(state_data));

    printf("=== Inputs ===\n");
    printf("k[0..3]: %.4f %.4f %.4f %.4f\n", k_data[0], k_data[1], k_data[2], k_data[3]);
    printf("v[0..3]: %.4f %.4f %.4f %.4f\n", v_data[0], v_data[1], v_data[2], v_data[3]);
    printf("q[0..3]: %.4f %.4f %.4f %.4f\n", q_data[0], q_data[1], q_data[2], q_data[3]);
    printf("gate: %.4f %.4f\n", gate_data[0], gate_data[1]);
    printf("beta: %.4f %.4f\n", beta_data[0], beta_data[1]);
    printf("state[0..3]: %.4f %.4f %.4f %.4f\n", state_data[0], state_data[1], state_data[2], state_data[3]);

    // Run graph on Metal
    ggml_backend_graph_compute(backend, graph);

    // Read back results
    // result shape: [S*H, T + S*B, 1, 1]
    int result_size = S * H * (T + S * B);
    std::vector<float> result_data(result_size);
    ggml_backend_tensor_get(result, result_data.data(), 0, result_size * sizeof(float));

    // Extract output (first S*H*T floats) and state (next S*S*H*B floats)
    float* gpu_output = result_data.data();
    float* gpu_state = result_data.data() + S * H * T;

    printf("\n=== GPU Kernel Output ===\n");
    for (int h = 0; h < H; h++)
        for (int r = 0; r < S; r++)
            printf("  output[h=%d r=%d] = %.8f\n", h, r, gpu_output[h*S + r]);

    // CPU reference computation
    float cpu_state[S*S*H*B];
    float cpu_output[S*H*T];
    memcpy(cpu_state, state_data, sizeof(cpu_state));

    for (int head = 0; head < H; head++) {
        for (int t = 0; t < T; t++) {
            int gh_idx = t * H + head;

            // decay
            float decay = expf(gate_data[gh_idx]);
            for (int row = 0; row < S; row++)
                for (int col = 0; col < S; col++)
                    cpu_state[col + row*S + head*S*S] *= decay;

            // kv_mem
            for (int row = 0; row < S; row++) {
                float sk = 0;
                for (int col = 0; col < S; col++)
                    sk += cpu_state[col + row*S + head*S*S] * k_data[col + head*S + t*S*H];

                float delta = (v_data[row + head*S + t*S*H] - sk) * beta_data[gh_idx];

                for (int col = 0; col < S; col++)
                    cpu_state[col + row*S + head*S*S] += k_data[col + head*S + t*S*H] * delta;

                float y = 0;
                for (int col = 0; col < S; col++)
                    y += cpu_state[col + row*S + head*S*S] * q_data[col + head*S + t*S*H];

                cpu_output[t*S*H + head*S + row] = y * 1.0f;  // scale=1.0
            }
        }
    }

    printf("\n=== CPU Reference Output ===\n");
    for (int h = 0; h < H; h++)
        for (int r = 0; r < S; r++)
            printf("  output[h=%d r=%d] = %.8f\n", h, r, cpu_output[h*S + r]);

    printf("\n=== Comparison ===\n");
    float max_out_diff = 0, max_state_diff = 0;
    for (int i = 0; i < S*H*T; i++) {
        float diff = fabsf(gpu_output[i] - cpu_output[i]);
        if (diff > max_out_diff) max_out_diff = diff;
        if (diff > 1e-5f)
            printf("OUTPUT MISMATCH [%d]: gpu=%.8f cpu=%.8f diff=%.8f\n", i, gpu_output[i], cpu_output[i], diff);
    }
    for (int i = 0; i < S*S*H*B; i++) {
        float diff = fabsf(gpu_state[i] - cpu_state[i]);
        if (diff > max_state_diff) max_state_diff = diff;
        if (diff > 1e-5f)
            printf("STATE MISMATCH [%d]: gpu=%.8f cpu=%.8f diff=%.8f\n", i, gpu_state[i], cpu_state[i], diff);
    }

    printf("\nMax output diff: %.10f\n", max_out_diff);
    printf("Max state diff:  %.10f\n", max_state_diff);

    if (max_out_diff < 1e-5f && max_state_diff < 1e-5f)
        printf("\n*** KERNEL OUTPUT MATCHES CPU ***\n");
    else
        printf("\n*** KERNEL OUTPUT DIFFERS FROM CPU ***\n");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return 0;
}
