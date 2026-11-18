// MUL_MAT Q8_0 x F32 on the GPU backend with the activation at byte offset 0 of its tensor and as a view 4 bytes in,
// each compared byte for byte with the CPU backend. The values are chosen so every partial sum is exact on both.
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int K = 64; // columns: two Q8_0 blocks
static const int M = 8;  // rows

static std::string g_pipeline;

static void on_log(enum ggml_log_level, const char * text, void *) {
    const char * hit = std::strstr(text, "kernel_mul_mv_q8_0_f32");
    if (hit) {
        g_pipeline.assign(hit, std::strcspn(hit, " \t\n|"));
    }
}

static bool run(ggml_backend_t backend, size_t offset, const std::vector<float> & w, const std::vector<float> & x, std::vector<float> & y) {
    struct ggml_init_params params = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * a    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    struct ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K + 4);
    struct ggml_tensor * b    = ggml_view_2d(ctx, base, K, 1, K*sizeof(float), offset);
    struct ggml_tensor * out  = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * gf   = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    std::vector<uint8_t> q(ggml_nbytes(a));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), q.data(), 0, M, K, nullptr);
    std::vector<float> padded(K + 4, 0.0f);
    std::memcpy((char *) padded.data() + offset, x.data(), K*sizeof(float));
    ggml_backend_tensor_set(a, q.data(), 0, q.size());
    ggml_backend_tensor_set(base, padded.data(), 0, ggml_nbytes(base));
    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    y.resize(M);
    ggml_backend_tensor_get(out, y.data(), 0, M*sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_log_set(on_log, nullptr);
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) {
        std::fprintf(stderr, "FAIL no %s backend\n", gpu ? "CPU" : "GPU");
        return 1;
    }
    // Multiples of 1/128 with +-127/128 in every block give a block scale of exactly 1/128. The CPU backend quantizes
    // the activation to Q8_0 as well: integers with +-127 in every block survive that unchanged.
    std::vector<float> w(K*M), x(K);
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            const int v = c % 32 == 0 ? 127 : c % 32 == 1 ? -127 : ((r*37 + c*11) % 201) - 100;
            w[r*K + c] = v/128.0f;
        }
    }
    for (int c = 0; c < K; ++c) {
        x[c] = (float) (c % 32 == 0 ? 127 : c % 32 == 1 ? -127 : (c*7) % 201 - 100);
    }
    int failures = 0;
    for (const size_t offset : { (size_t) 0, (size_t) 4 }) {
        std::vector<float> on_gpu, on_cpu;
        const bool ok = run(gpu, offset, w, x, on_gpu) && run(cpu, offset, w, x, on_cpu);
        const bool equal = ok && std::memcmp(on_gpu.data(), on_cpu.data(), M*sizeof(float)) == 0;
        std::printf("CASE · offset %zu · pipeline %s · %s\n", offset, g_pipeline.empty() ? "none" : g_pipeline.c_str(), equal ? "equal" : "differ");
        if (!equal && ok) {
            for (int r = 0; r < M; ++r) {
                std::fprintf(stderr, "row %d gpu %.9g cpu %.9g\n", r, on_gpu[r], on_cpu[r]);
            }
        }
        failures += equal ? 0 : 1;
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return failures == 0 ? 0 : 1;
}
