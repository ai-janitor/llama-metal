// Standalone test: compare fused vs non-fused delta-net math
// Compile: c++ -std=c++17 -o debug_delta_net debug_delta_net.cpp && ./debug_delta_net
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>

// Small dimensions for debugging
static const int S = 4;   // state dimension
static const int H = 2;   // heads
static const int T = 3;   // tokens

// State: [S, S, H] = state[col + row*S + head*S*S]
// k, v, q: [S, H, T] = x[s + h*S + t*S*H]
// gate, beta: [H] per token = g[t*H + h]

float state_init[S * S * H];
float k[S * H * T], v[S * H * T], q[S * H * T];
float gate[H * T];    // log(decay), <= 0
float beta_sig[H * T]; // already sigmoid'd

void init_test_data() {
    srand(42);
    for (int i = 0; i < S*S*H; i++) state_init[i] = 0.01f * (rand() % 100 - 50);
    for (int i = 0; i < S*H*T; i++) {
        k[i] = 0.01f * (rand() % 100 - 50);
        v[i] = 0.01f * (rand() % 100 - 50);
        q[i] = 0.01f * (rand() % 100 - 50);
    }
    for (int i = 0; i < H*T; i++) {
        gate[i] = -0.1f * (1 + rand() % 10);  // negative, log(decay)
        beta_sig[i] = 0.1f * (1 + rand() % 8); // (0, 1) range after sigmoid
    }
}

// Non-fused: step by step, matching ggml elementwise ops
void compute_nonfused(float* state, float* output) {
    for (int t = 0; t < T; t++) {
        for (int head = 0; head < H; head++) {
            int gh_idx = t * H + head;

            // Step 1: state *= exp(gate)
            float decay = expf(gate[gh_idx]);
            for (int row = 0; row < S; row++)
                for (int col = 0; col < S; col++)
                    state[col + row*S + head*S*S] *= decay;

            // Step 2: kv_mem[row] = sum_col(state[row,col] * k[col])
            float kv_mem[S];
            for (int row = 0; row < S; row++) {
                kv_mem[row] = 0.0f;
                for (int col = 0; col < S; col++)
                    kv_mem[row] += state[col + row*S + head*S*S] * k[col + head*S + t*S*H];
            }

            // Step 3: delta[row] = (v[row] - kv_mem[row]) * beta
            float delta[S];
            for (int row = 0; row < S; row++)
                delta[row] = (v[row + head*S + t*S*H] - kv_mem[row]) * beta_sig[gh_idx];

            // Step 4: state[row, col] += k[col] * delta[row]
            for (int row = 0; row < S; row++)
                for (int col = 0; col < S; col++)
                    state[col + row*S + head*S*S] += k[col + head*S + t*S*H] * delta[row];

            // Step 5: output[row] = sum_col(state[row,col] * q[col])
            for (int row = 0; row < S; row++) {
                float y = 0.0f;
                for (int col = 0; col < S; col++)
                    y += state[col + row*S + head*S*S] * q[col + head*S + t*S*H];
                output[row + head*S + t*S*H] = y;
            }
        }

        if (t == 0) {
            printf("[NON-FUSED] t=0 h=0 r=0:\n");
            printf("  gate=%.6f decay=%.6f\n", gate[0], expf(gate[0]));
            printf("  kv_mem[0]=computed above\n");
            printf("  state[0..3] after update: %.6f %.6f %.6f %.6f\n",
                   state[0], state[1], state[2], state[3]);
            printf("  output[0]=%.6f\n", output[0]);
        }
    }
}

// Fused: exact replica of the Metal kernel logic (single thread simulation)
void compute_fused(float* state, float* output) {
    // For each (row, head) — simulating one threadgroup
    for (int head = 0; head < H; head++) {
        for (int row = 0; row < S; row++) {
            // Load state row into "registers"
            float ls[S];
            for (int col = 0; col < S; col++)
                ls[col] = state[(head * S + row) * S + col];

            // Per-token recurrence
            for (int t = 0; t < T; t++) {
                int gh_idx = t * H + head;
                int k_head = head;  // no GQA
                int qk_off = (t * H + k_head) * S;

                // Step 1: decay
                float gate_log = gate[gh_idx];
                float decay = expf(gate_log);

                // Step 2: dot(state_row, k)
                float dot_state_k = 0.0f;
                for (int col = 0; col < S; col++) {
                    ls[col] *= decay;
                    dot_state_k += ls[col] * k[qk_off + col];
                }

                // Step 3: delta
                float v_val = v[gh_idx * S + row];
                float beta_val = beta_sig[gh_idx];
                float delta = (v_val - dot_state_k) * beta_val;

                // Step 4: state update + output
                float dot_state_q = 0.0f;
                for (int col = 0; col < S; col++) {
                    ls[col] += k[qk_off + col] * delta;
                    dot_state_q += ls[col] * q[qk_off + col];
                }

                output[t * S * H + head * S + row] = dot_state_q;

                if (head == 0 && row == 0 && t == 0) {
                    printf("[FUSED]     t=0 h=0 r=0:\n");
                    printf("  gate=%.6f decay=%.6f\n", gate_log, decay);
                    printf("  dot_state_k=%.6f v=%.6f beta=%.6f delta=%.6f\n",
                           dot_state_k, v_val, beta_val, delta);
                    printf("  state[0..3] after update: %.6f %.6f %.6f %.6f\n",
                           ls[0], ls[1], ls[2], ls[3]);
                    printf("  output[0]=%.6f\n", dot_state_q);
                }
            }

            // Write state back
            for (int col = 0; col < S; col++)
                state[(head * S + row) * S + col] = ls[col];
        }
    }
}

int main() {
    init_test_data();

    // Print initial values
    printf("=== Test Data ===\n");
    printf("state[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", state_init[i]);
    printf("\n");
    printf("k[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", k[i]);
    printf("\n");
    printf("v[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", v[i]);
    printf("\n");
    printf("q[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", q[i]);
    printf("\n");
    printf("gate[0..3]: ");
    for (int i = 0; i < H*T && i < 6; i++) printf("%.4f ", gate[i]);
    printf("\n");
    printf("beta[0..3]: ");
    for (int i = 0; i < H*T && i < 6; i++) printf("%.4f ", beta_sig[i]);
    printf("\n\n");

    // Run non-fused
    float state_nf[S*S*H], output_nf[S*H*T];
    memcpy(state_nf, state_init, sizeof(state_nf));
    memset(output_nf, 0, sizeof(output_nf));
    printf("=== Non-Fused ===\n");
    compute_nonfused(state_nf, output_nf);

    // Run fused
    float state_f[S*S*H], output_f[S*H*T];
    memcpy(state_f, state_init, sizeof(state_f));
    memset(output_f, 0, sizeof(output_f));
    printf("\n=== Fused ===\n");
    compute_fused(state_f, output_f);

    // Compare outputs
    printf("\n=== Comparison ===\n");
    float max_diff_out = 0, max_diff_state = 0;
    for (int i = 0; i < S*H*T; i++) {
        float diff = fabsf(output_nf[i] - output_f[i]);
        if (diff > max_diff_out) max_diff_out = diff;
        if (diff > 1e-5f) {
            printf("OUTPUT MISMATCH at [%d] (t=%d h=%d r=%d): nf=%.6f fused=%.6f diff=%.6f\n",
                   i, i / (S*H), (i % (S*H)) / S, i % S, output_nf[i], output_f[i], diff);
        }
    }
    for (int i = 0; i < S*S*H; i++) {
        float diff = fabsf(state_nf[i] - state_f[i]);
        if (diff > max_diff_state) max_diff_state = diff;
        if (diff > 1e-5f) {
            printf("STATE MISMATCH at [%d]: nf=%.6f fused=%.6f diff=%.6f\n",
                   i, state_nf[i], state_f[i], diff);
        }
    }

    printf("\nMax output diff: %.8f\n", max_diff_out);
    printf("Max state diff:  %.8f\n", max_diff_state);

    if (max_diff_out < 1e-5f && max_diff_state < 1e-5f) {
        printf("\n*** MATH IS IDENTICAL — bug is in GPU buffer management, not kernel logic ***\n");
    } else {
        printf("\n*** MATH DIFFERS — bug is in the kernel logic ***\n");
    }

    // Also print all outputs for detailed comparison
    printf("\n=== Full Output Comparison (t=0 only) ===\n");
    for (int h = 0; h < H; h++) {
        for (int r = 0; r < S; r++) {
            int idx = h * S + r;
            printf("  h=%d r=%d: nf=%.8f fused=%.8f diff=%.8f\n",
                   h, r, output_nf[idx], output_f[idx], output_nf[idx] - output_f[idx]);
        }
    }

    return 0;
}
