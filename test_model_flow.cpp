// Test the EXACT tensor flow that the model uses:
// 1. Create g as [H, 1, 1], reshape to [1,1,H,1], cont
// 2. Create state as [S, S*H, 1, 1], reshape to [S, S, H, 1]
// 3. Create k/v/q as [S, H, 1, 1], reshape to [S, H, 1]
// 4. Run the fused kernel
// 5. Compare with CPU reference
//
// This mimics the model's exact tensor creation pattern.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

#define S 128
#define H 32

int main() {
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) { fprintf(stderr, "No Metal\n"); return 1; }

    struct ggml_init_params params = { 256*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    // TEST A: create directly (like working test) — should work
    // TEST B: create through reshape+cont (like model) — might be broken
    // Toggle by setting USE_RESHAPE to 0 or 1
    #define USE_RESHAPE 0

    #if USE_RESHAPE
    // Create tensors through reshape chain (model pattern)
    ggml_tensor * q_4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, 1);
    ggml_tensor * k_4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, 1);
    ggml_tensor * v_4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, 1);
    ggml_tensor * q3 = ggml_reshape_3d(ctx, q_4d, S, H, 1);
    ggml_tensor * k3 = ggml_reshape_3d(ctx, k_4d, S, H, 1);
    ggml_tensor * v3 = ggml_reshape_3d(ctx, v_4d, S, H, 1);
    ggml_tensor * g_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 1, 1);
    ggml_tensor * gate = ggml_cont(ctx, ggml_reshape_4d(ctx, g_3d, 1, 1, H, 1));
    ggml_tensor * b_4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H, 1, 1, 1);
    ggml_tensor * beta = ggml_cont(ctx, ggml_reshape_4d(ctx, b_4d, 1, 1, H, 1));
    ggml_tensor * state_flat = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S*H, 1, 1);
    ggml_tensor * state = ggml_reshape_4d(ctx, state_flat, S, S, H, 1);
    #else
    // Create tensors directly (working pattern)
    ggml_tensor * k3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, 1);
    ggml_tensor * v3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, 1);
    ggml_tensor * q3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, 1);
    ggml_tensor * gate = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, H, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, H, 1);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    // aliases for setting data
    ggml_tensor * q_4d = q3;
    ggml_tensor * k_4d = k3;
    ggml_tensor * v_4d = v3;
    ggml_tensor * g_3d = gate;
    ggml_tensor * b_4d = beta;
    ggml_tensor * state_flat = state;
    #endif

    // Run the fused kernel
    ggml_tensor * result = ggml_gated_delta_net(ctx, k3, v3, q3, gate, beta, state, 1.0f);

    // Build graph
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    // Allocate
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Fill with random values
    srand(42);
    auto fill = [&](ggml_tensor * t) {
        std::vector<float> data(ggml_nelements(t));
        for (size_t i = 0; i < data.size(); i++)
            data[i] = 0.01f * (rand() % 100 - 50);
        ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    };
    fill(q_4d);
    fill(k_4d);
    fill(v_4d);
    fill(state_flat);
    {
        std::vector<float> g(H);
        for (int i = 0; i < H; i++) g[i] = -0.1f * (1 + rand() % 10);
        ggml_backend_tensor_set(g_3d, g.data(), 0, H * sizeof(float));
    }
    {
        std::vector<float> b(H);
        for (int i = 0; i < H; i++) b[i] = 0.1f * (1 + rand() % 8);
        ggml_backend_tensor_set(b_4d, b.data(), 0, H * sizeof(float));
    }

    // Read back inputs for CPU ref
    std::vector<float> k_data(S*H), v_data(S*H), q_data(S*H);
    std::vector<float> gate_data(H), beta_data(H);
    std::vector<float> state_data(S*S*H);
    ggml_backend_tensor_get(k_4d, k_data.data(), 0, S*H*sizeof(float));
    ggml_backend_tensor_get(v_4d, v_data.data(), 0, S*H*sizeof(float));
    ggml_backend_tensor_get(q_4d, q_data.data(), 0, S*H*sizeof(float));
    ggml_backend_tensor_get(g_3d, gate_data.data(), 0, H*sizeof(float));
    ggml_backend_tensor_get(b_4d, beta_data.data(), 0, H*sizeof(float));
    ggml_backend_tensor_get(state_flat, state_data.data(), 0, S*S*H*sizeof(float));

    // Verify gate/beta after cont
    {
        // Build a tiny graph just for the cont ops
        ggml_cgraph * g2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g2, gate);
        ggml_build_forward_expand(g2, beta);
        ggml_backend_graph_compute(backend, g2);

        std::vector<float> gate_after(H), beta_after(H);
        ggml_backend_tensor_get(gate, gate_after.data(), 0, H*sizeof(float));
        ggml_backend_tensor_get(beta, beta_after.data(), 0, H*sizeof(float));

        printf("Gate values (original vs after cont+reshape):\n");
        for (int h = 0; h < 4; h++)
            printf("  h=%d: orig=%.6f cont=%.6f\n", h, gate_data[h], gate_after[h]);
        printf("Beta values (original vs after cont+reshape):\n");
        for (int h = 0; h < 4; h++)
            printf("  h=%d: orig=%.6f cont=%.6f\n", h, beta_data[h], beta_after[h]);
    }

    // Compute main graph
    ggml_backend_graph_compute(backend, graph);

    // Read back GPU output
    int result_size = S * H * (1 + S);
    std::vector<float> result_data(result_size);
    ggml_backend_tensor_get(result, result_data.data(), 0, result_size * sizeof(float));
    float* gpu_output = result_data.data();

    // CPU reference
    std::vector<float> cpu_state(state_data);
    std::vector<float> cpu_output(S*H);

    for (int head = 0; head < H; head++) {
        float decay = expf(gate_data[head]);
        for (int row = 0; row < S; row++) {
            for (int col = 0; col < S; col++)
                cpu_state[col + row*S + head*S*S] *= decay;

            float sk = 0;
            for (int col = 0; col < S; col++)
                sk += cpu_state[col + row*S + head*S*S] * k_data[col + head*S];

            float delta = (v_data[row + head*S] - sk) * beta_data[head];

            for (int col = 0; col < S; col++)
                cpu_state[col + row*S + head*S*S] += k_data[col + head*S] * delta;

            float y = 0;
            for (int col = 0; col < S; col++)
                y += cpu_state[col + row*S + head*S*S] * q_data[col + head*S];

            cpu_output[head*S + row] = y;
        }
    }

    // Compare
    float max_diff = 0;
    int mismatches = 0;
    for (int i = 0; i < S*H; i++) {
        float diff = fabsf(gpu_output[i] - cpu_output[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f && mismatches < 10) {
            printf("MISMATCH [%d] (h=%d r=%d): gpu=%.8f cpu=%.8f diff=%.8f\n",
                   i, i/S, i%S, gpu_output[i], cpu_output[i], diff);
            mismatches++;
        }
    }
    printf("\nMax diff: %.10f  Mismatches: %d\n", max_diff, mismatches);
    printf(max_diff < 1e-4f ? "*** MODEL FLOW WORKS ***\n" : "*** MODEL FLOW BROKEN ***\n");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
