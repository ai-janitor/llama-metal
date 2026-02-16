// Device-aware Flash Attention test
// Detects GPU vendor at runtime, picks test params that exercise device-specific dispatch paths:
//   - Apple Silicon: simdgroup_mm FA
//   - AMD discrete: scalar FA, single dispatch (fa_max_threadgroups=0)
//   - Intel iGPU:   scalar FA, chunked dispatch (fa_max_threadgroups=512)
//   - CPU:          reference implementation
//
// Usage:
//   GGML_METAL_DEVICES=2 ./build/bin/test-metal-device-fa
//
// BUG-005: Intel FA correctness failures
// FEAT-008: Intel FA chunked dispatch

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

// --- Device classification ---

enum device_vendor {
    VENDOR_CPU,
    VENDOR_APPLE,
    VENDOR_AMD,
    VENDOR_INTEL,
    VENDOR_OTHER,
};

struct device_info {
    device_vendor        vendor;
    std::string          name;
    std::string          desc;
    ggml_backend_dev_t   dev;
    bool                 has_unified_memory;
};

static device_vendor classify_device(const char * desc) {
    if (strstr(desc, "Apple") || strstr(desc, "apple"))  return VENDOR_APPLE;
    if (strstr(desc, "AMD")   || strstr(desc, "Radeon")) return VENDOR_AMD;
    if (strstr(desc, "Intel") || strstr(desc, "intel"))  return VENDOR_INTEL;
    return VENDOR_OTHER;
}

static const char * vendor_str(device_vendor v) {
    switch (v) {
        case VENDOR_CPU:    return "CPU";
        case VENDOR_APPLE:  return "Apple";
        case VENDOR_AMD:    return "AMD";
        case VENDOR_INTEL:  return "Intel";
        case VENDOR_OTHER:  return "Other";
    }
    return "Unknown";
}

// --- FA test parameters ---

struct fa_test_params {
    const char * name;
    int64_t hsk;    // K head size
    int64_t hsv;    // V head size
    int64_t nh;     // num KV heads
    int64_t nr;     // GQA ratio (repeat in dim 2)
    int64_t kv;     // KV sequence length
    int64_t nb;     // query batch size
    bool    mask;
    bool    causal;
};

// Tests that every device should pass
static const fa_test_params common_tests[] = {
    // basic: small, should work everywhere
    { "basic_small",    128, 128,  2, 1,   64,  1, true,  false },
    { "basic_medium",   128, 128,  2, 1,  256,  1, true,  false },
    // GQA: grouped-query attention (Qwen2.5 uses nr=6)
    { "gqa_qwen",       128, 128,  2, 6,  256,  1, true,  false },
    // causal mask: exercises -INF handling
    { "causal_small",   128, 128,  2, 1,  256,  4, true,  true  },
    { "causal_gqa",     128, 128,  2, 6,  256,  4, true,  true  },
    // head sizes: test different hsk values (BUG-005 shows hsk-dependent failures on Intel)
    { "hsk64",           64,  64,  4, 1,  256,  1, true,  false },
    { "hsk80",           80,  80,  4, 1,  256,  1, true,  false },
    { "hsk96",           96,  96,  4, 1,  256,  1, true,  false },
    { "hsk128",         128, 128,  4, 1,  256,  1, true,  false },
    // batch sizes: tg-like (nb=1) and pp-like (nb=32)
    { "batch_1",        128, 128,  2, 6,  256,  1, true,  false },
    { "batch_32",       128, 128,  2, 6,  256, 32, true,  false },
};
static const int n_common_tests = sizeof(common_tests) / sizeof(common_tests[0]);

// Intel-specific: test chunked dispatch (total threadgroups > 512)
// total_tg = nb * nh * nr, chunk triggers when > fa_max_threadgroups (512)
static const fa_test_params intel_chunk_tests[] = {
    // 64 * 12 * 1 = 768 threadgroups → 2 chunks of 512 + 256
    { "chunk_2",        128, 128, 12, 1,  128, 64, true,  false },
    // 32 * 2 * 6 = 384 threadgroups → below 512, single dispatch
    { "no_chunk",       128, 128,  2, 6,  128, 32, true,  false },
    // 48 * 12 * 1 = 576 threadgroups → 2 chunks (512 + 64), tests remainder
    { "chunk_remainder",128, 128, 12, 1,  128, 48, true,  false },
    // causal + chunked
    { "chunk_causal",   128, 128, 12, 1,  128, 64, true,  true  },
};
static const int n_intel_chunk_tests = sizeof(intel_chunk_tests) / sizeof(intel_chunk_tests[0]);

// --- Tensor helpers ---

static void init_tensor_uniform(ggml_tensor * t, float min_val = -1.0f, float max_val = 1.0f) {
    size_t n = ggml_nelements(t);
    size_t n_bytes = ggml_nbytes(t);

    // generate random float data
    std::vector<float> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = min_val + (max_val - min_val) * ((float)rand() / RAND_MAX);
    }

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, data.data(), 0, n_bytes);
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> data_f16(n);
        for (size_t i = 0; i < n; i++) {
            data_f16[i] = ggml_fp32_to_fp16(data[i]);
        }
        ggml_backend_tensor_set(t, data_f16.data(), 0, n_bytes);
    } else {
        // for quantized types, quantize from float
        std::vector<uint8_t> qdata(n_bytes);
        ggml_quantize_chunk(t->type, data.data(), qdata.data(), 0, 1, n, nullptr);
        ggml_backend_tensor_set(t, qdata.data(), 0, n_bytes);
    }
}

static void init_causal_mask(ggml_tensor * t) {
    const int64_t ne0 = t->ne[0]; // kv
    const int64_t ne1 = t->ne[1]; // nb
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];
    size_t n = ne0 * ne1 * ne2 * ne3;
    std::vector<ggml_fp16_t> data(n);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    const ggml_fp16_t zero    = ggml_fp32_to_fp16(0.0f);

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    int64_t idx = i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0 + i0;
                    data[idx] = (i0 <= i1) ? zero : neg_inf;
                }
            }
        }
    }
    ggml_backend_tensor_set(t, data.data(), 0, n * sizeof(ggml_fp16_t));
}

static void init_zero_mask(ggml_tensor * t) {
    // all-zero mask = all positions visible (no masking)
    size_t n = ggml_nelements(t);
    std::vector<ggml_fp16_t> data(n, ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(t, data.data(), 0, n * sizeof(ggml_fp16_t));
}

// --- Run a single FA test on one backend vs CPU ---

struct test_result {
    const char *  test_name;
    const char *  device_name;
    device_vendor vendor;
    bool          supported;
    bool          passed;
    float         max_err;
    int64_t       total_tg;   // total threadgroups dispatched
};

static test_result run_fa_test(
    ggml_backend_t gpu_backend,
    ggml_backend_t cpu_backend,
    const device_info & dinfo,
    const fa_test_params & p
) {
    test_result result = {};
    result.test_name   = p.name;
    result.device_name = dinfo.name.c_str();
    result.vendor      = dinfo.vendor;
    result.supported   = false;
    result.passed      = false;
    result.max_err     = 0.0f;
    result.total_tg    = p.nb * p.nh * p.nr;

    const int64_t hsk_padded = GGML_PAD(p.hsk, ggml_blck_size(GGML_TYPE_F16));
    const int64_t hsv_padded = GGML_PAD(p.hsv, ggml_blck_size(GGML_TYPE_F16));

    // --- Build graph ---
    ggml_init_params ctx_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /*.mem_base   =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ctx_params);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  hsk_padded, p.nb, p.nh * p.nr, 1);
    ggml_set_name(q, "q");

    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  hsk_padded, p.kv, p.nh, 1);
    ggml_set_name(k, "k");

    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  hsv_padded, p.kv, p.nh, 1);
    ggml_set_name(v, "v");

    ggml_tensor * m = NULL;
    if (p.mask) {
        m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, p.kv, p.nb, 1, 1);
        ggml_set_name(m, "m");
    }

    float scale = 1.0f / sqrtf((float)p.hsk);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_set_name(out, "out");

    // check if backend supports this op
    if (!ggml_backend_supports_op(gpu_backend, out)) {
        result.supported = false;
        ggml_free(ctx);
        return result;
    }
    result.supported = true;

    // --- GPU run ---
    ggml_cgraph * gf_gpu = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_gpu, out);

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx, gpu_backend);
    if (!buf_gpu) {
        printf("    [%s] %s: FAIL (alloc failed)\n", dinfo.name.c_str(), p.name);
        ggml_free(ctx);
        return result;
    }

    // seed RNG for reproducible data
    srand(42);

    // init tensors
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (strcmp(t->name, "m") == 0) {
            if (p.causal) {
                init_causal_mask(t);
            } else {
                init_zero_mask(t);
            }
        } else if (strcmp(t->name, "out") == 0) {
            // Fill output with sentinel value (NaN) to detect unwritten positions
            size_t n_out = ggml_nelements(t);
            std::vector<float> sentinel(n_out, std::nanf(""));
            ggml_backend_tensor_set(t, sentinel.data(), 0, n_out * sizeof(float));
        } else {
            init_tensor_uniform(t);
        }
    }

    // save input data for CPU run
    std::vector<std::vector<uint8_t>> input_data;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (strcmp(t->name, "out") == 0) {
            input_data.push_back({});
            continue;
        }
        std::vector<uint8_t> data(ggml_nbytes(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        input_data.push_back(data);
    }

    // compute on GPU - run 1
    ggml_status status = ggml_backend_graph_compute(gpu_backend, gf_gpu);
    if (status != GGML_STATUS_SUCCESS) {
        printf("    [%s] %s: FAIL (compute error %d)\n", dinfo.name.c_str(), p.name, status);
        ggml_backend_buffer_free(buf_gpu);
        ggml_free(ctx);
        return result;
    }

    // Wait for GPU to finish before reading results (critical for shared/UMA buffers)
    ggml_backend_synchronize(gpu_backend);

    // read GPU output
    size_t out_size = ggml_nbytes(out);
    std::vector<float> gpu_out(ggml_nelements(out));
    ggml_backend_tensor_get(out, gpu_out.data(), 0, out_size);

    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx);

    // --- CPU run (reference) ---
    ggml_context * ctx_cpu = ggml_init(ctx_params);

    ggml_tensor * q2 = ggml_new_tensor_4d(ctx_cpu, GGML_TYPE_F32,  hsk_padded, p.nb, p.nh * p.nr, 1);
    ggml_set_name(q2, "q");
    ggml_tensor * k2 = ggml_new_tensor_4d(ctx_cpu, GGML_TYPE_F16,  hsk_padded, p.kv, p.nh, 1);
    ggml_set_name(k2, "k");
    ggml_tensor * v2 = ggml_new_tensor_4d(ctx_cpu, GGML_TYPE_F16,  hsv_padded, p.kv, p.nh, 1);
    ggml_set_name(v2, "v");
    ggml_tensor * m2 = NULL;
    if (p.mask) {
        m2 = ggml_new_tensor_4d(ctx_cpu, GGML_TYPE_F16, p.kv, p.nb, 1, 1);
        ggml_set_name(m2, "m");
    }
    ggml_tensor * out2 = ggml_flash_attn_ext(ctx_cpu, q2, k2, v2, m2, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out2, GGML_PREC_F32);
    ggml_set_name(out2, "out");

    ggml_cgraph * gf_cpu = ggml_new_graph(ctx_cpu);
    ggml_build_forward_expand(gf_cpu, out2);

    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, cpu_backend);

    // restore same input data
    int idx = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_cpu); t; t = ggml_get_next_tensor(ctx_cpu, t)) {
        if (strcmp(t->name, "out") != 0 && idx < (int)input_data.size() && !input_data[idx].empty()) {
            ggml_backend_tensor_set(t, input_data[idx].data(), 0, ggml_nbytes(t));
        }
        idx++;
    }

    ggml_backend_graph_compute(cpu_backend, gf_cpu);

    std::vector<float> cpu_out(ggml_nelements(out2));
    ggml_backend_tensor_get(out2, cpu_out.data(), 0, ggml_nbytes(out2));

    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_cpu);

    // --- Compare ---
    float max_err = 0.0f;
    const float tol = 5e-3f; // tolerance
    int max_err_idx = 0;
    int n_zeros_gpu = 0;
    int n_nan_gpu = 0;

    for (size_t i = 0; i < gpu_out.size() && i < cpu_out.size(); i++) {
        if (gpu_out[i] == 0.0f) n_zeros_gpu++;
        if (isnan(gpu_out[i])) n_nan_gpu++;
        float err = fabsf(gpu_out[i] - cpu_out[i]);
        if (err > max_err) {
            max_err = err;
            max_err_idx = (int)i;
        }
    }

    result.max_err = max_err;
    result.passed  = (max_err < tol);

    // Debug output for failing tests
    if (!result.passed) {
        const int DV = (int)p.hsv;
        const int n_total = (int)gpu_out.size();
        printf("      DEBUG: n_elements=%d, n_zeros_gpu=%d, n_nan_gpu=%d\n", n_total, n_zeros_gpu, n_nan_gpu);
        printf("      DEBUG: max_err at idx=%d (query=%d, head=%d, dv=%d)\n",
            max_err_idx, max_err_idx / (DV * (int)(p.nh * p.nr)), (max_err_idx / DV) % (int)(p.nh * p.nr), max_err_idx % DV);
        printf("      DEBUG: gpu[%d]=%.6f, cpu[%d]=%.6f\n", max_err_idx, gpu_out[max_err_idx], max_err_idx, cpu_out[max_err_idx]);
        // Print first 8 values
        printf("      DEBUG: first 8 GPU: ");
        for (int i = 0; i < 8 && i < n_total; i++) printf("%.4f ", gpu_out[i]);
        printf("\n      DEBUG: first 8 CPU: ");
        for (int i = 0; i < 8 && i < n_total; i++) printf("%.4f ", cpu_out[i]);
        printf("\n");
        // Print values around max_err
        int start = std::max(0, max_err_idx - 2);
        int end = std::min(n_total, max_err_idx + 3);
        printf("      DEBUG: around max_err [%d..%d] GPU: ", start, end-1);
        for (int i = start; i < end; i++) printf("%.4f ", gpu_out[i]);
        printf("\n      DEBUG: around max_err [%d..%d] CPU: ", start, end-1);
        for (int i = start; i < end; i++) printf("%.4f ", cpu_out[i]);
        printf("\n");
    }

    return result;
}

// --- Timing benchmark ---

struct timing_result {
    const char *  test_name;
    const char *  device_name;
    device_vendor vendor;
    double        avg_time_us;
    double        throughput_gflops;
};

static timing_result benchmark_fa_kernel(
    ggml_backend_t backend,
    const device_info & dinfo,
    const fa_test_params & p,
    int warmup_iters = 10,
    int bench_iters = 100
) {
    timing_result result = {};
    result.test_name   = p.name;
    result.device_name = dinfo.name.c_str();
    result.vendor      = dinfo.vendor;
    result.avg_time_us = 0.0;
    result.throughput_gflops = 0.0;

    const int64_t hsk_padded = GGML_PAD(p.hsk, ggml_blck_size(GGML_TYPE_F16));
    const int64_t hsv_padded = GGML_PAD(p.hsv, ggml_blck_size(GGML_TYPE_F16));

    // --- Build graph ---
    ggml_init_params ctx_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /*.mem_base   =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ctx_params);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  hsk_padded, p.nb, p.nh * p.nr, 1);
    ggml_set_name(q, "q");

    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  hsk_padded, p.kv, p.nh, 1);
    ggml_set_name(k, "k");

    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  hsv_padded, p.kv, p.nh, 1);
    ggml_set_name(v, "v");

    ggml_tensor * m = NULL;
    if (p.mask) {
        m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, p.kv, p.nb, 1, 1);
        ggml_set_name(m, "m");
    }

    float scale = 1.0f / sqrtf((float)p.hsk);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_set_name(out, "out");

    // check if backend supports this op
    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return result;
    }

    // --- Allocate buffers ---
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("    [%s] %s: FAIL (alloc failed)\n", dinfo.name.c_str(), p.name);
        ggml_free(ctx);
        return result;
    }

    // seed RNG for reproducible data
    srand(42);

    // init tensors
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (strcmp(t->name, "m") == 0) {
            if (p.causal) {
                init_causal_mask(t);
            } else {
                init_zero_mask(t);
            }
        } else if (strcmp(t->name, "out") != 0) {
            init_tensor_uniform(t);
        }
    }

    // --- Warmup ---
    for (int i = 0; i < warmup_iters; i++) {
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
    }

    // --- Benchmark ---
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; i++) {
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> elapsed = end - start;
    result.avg_time_us = elapsed.count() / bench_iters;

    // Estimate FLOPs for Flash Attention
    // FA performs roughly 4*N*d*S^2 FLOPs where N=batch, d=head_size, S=seq_len
    // More accurately: QK^T (2*N*d*S), softmax (5*N*S), attention*V (2*N*d*S)
    // Approximate: 4*nb*nh*nr*hsk*kv FLOPs per forward pass
    const int64_t total_queries = p.nb * p.nh * p.nr;
    const int64_t flops_per_query = 4 * p.hsk * p.kv;  // simplified estimate
    const int64_t total_flops = total_queries * flops_per_query;
    result.throughput_gflops = (total_flops / result.avg_time_us) / 1000.0;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return result;
}

// --- Main ---

int main(int /*argc*/, char ** /*argv*/) {
    printf("=== Device-Aware Flash Attention Test ===\n");
    printf("Tests FA correctness per GPU vendor with device-specific dispatch params\n\n");

    ggml_backend_load_all();

    // enumerate all devices
    std::vector<device_info> devices;
    ggml_backend_dev_t cpu_dev = NULL;

    printf("Devices found:\n");
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);
        enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);

        size_t free_mem, total_mem;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);

        device_info di = {};
        di.name = name;
        di.desc = desc;
        di.dev  = dev;

        if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            di.vendor = VENDOR_CPU;
            cpu_dev = dev;
        } else if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            di.vendor = classify_device(desc);
            di.has_unified_memory = (type == GGML_BACKEND_DEVICE_TYPE_IGPU);
        } else {
            // skip accelerators (BLAS, etc)
            printf("  [%zu] %s: %s (accelerator, skipping)\n", i, name, desc);
            continue;
        }

        printf("  [%zu] %s: %s (%s, %zu MB)\n", i, name, desc, vendor_str(di.vendor), total_mem / 1024 / 1024);
        devices.push_back(di);
    }
    printf("\n");

    if (!cpu_dev) {
        printf("ERROR: No CPU backend found\n");
        return 1;
    }

    // init CPU backend for reference
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu_dev, NULL);
    if (!cpu_backend) {
        printf("ERROR: Failed to init CPU backend\n");
        return 1;
    }

    int total_tests = 0;
    int total_pass  = 0;
    int total_fail  = 0;
    int total_skip  = 0;

    for (const auto & di : devices) {
        if (di.vendor == VENDOR_CPU) continue; // skip CPU-vs-CPU

        printf("--- Testing: %s (%s) ---\n", di.name.c_str(), vendor_str(di.vendor));

        ggml_backend_t backend = ggml_backend_dev_init(di.dev, NULL);
        if (!backend) {
            printf("  ERROR: Failed to init backend\n");
            continue;
        }

        // 1. Common tests (all vendors)
        printf("  Common tests:\n");
        for (int t = 0; t < n_common_tests; t++) {
            const fa_test_params & p = common_tests[t];
            test_result r = run_fa_test(backend, cpu_backend, di, p);
            total_tests++;

            if (!r.supported) {
                printf("    %-20s SKIP (not supported)\n", p.name);
                total_skip++;
            } else if (r.passed) {
                printf("    %-20s PASS (max_err=%.6f, tg=%lld)\n", p.name, r.max_err, (long long)r.total_tg);
                total_pass++;
            } else {
                printf("    %-20s FAIL (max_err=%.6f, tg=%lld)\n", p.name, r.max_err, (long long)r.total_tg);
                total_fail++;
            }
        }

        // 2. Intel-specific chunked dispatch tests
        if (di.vendor == VENDOR_INTEL) {
            printf("  Intel chunked dispatch tests (fa_max_threadgroups=512):\n");
            for (int t = 0; t < n_intel_chunk_tests; t++) {
                const fa_test_params & p = intel_chunk_tests[t];
                test_result r = run_fa_test(backend, cpu_backend, di, p);
                total_tests++;

                int n_chunks = (r.total_tg > 512) ? (int)((r.total_tg + 511) / 512) : 1;

                if (!r.supported) {
                    printf("    %-20s SKIP (not supported)\n", p.name);
                    total_skip++;
                } else if (r.passed) {
                    printf("    %-20s PASS (max_err=%.6f, tg=%lld, chunks=%d)\n",
                           p.name, r.max_err, (long long)r.total_tg, n_chunks);
                    total_pass++;
                } else {
                    printf("    %-20s FAIL (max_err=%.6f, tg=%lld, chunks=%d)\n",
                           p.name, r.max_err, (long long)r.total_tg, n_chunks);
                    total_fail++;
                }
            }
        }

        printf("\n");
        ggml_backend_free(backend);
    }

    // Summary
    printf("=== Summary ===\n");
    printf("Total: %d  Pass: %d  Fail: %d  Skip: %d\n", total_tests, total_pass, total_fail, total_skip);

    // --- Timing Benchmarks (Intel only) ---
    if (total_fail == 0) {
        for (const auto & di : devices) {
            if (di.vendor != VENDOR_INTEL) continue;

            printf("\n=== Timing Benchmark: %s (%s) ===\n", di.name.c_str(), vendor_str(di.vendor));

            ggml_backend_t backend = ggml_backend_dev_init(di.dev, NULL);
            if (!backend) {
                printf("  ERROR: Failed to init backend\n");
                continue;
            }

            // Representative test case: dk=64, dv=64, KV=512, f16
            fa_test_params bench_params = {
                "bench_dk64_kv512",
                64,   // hsk (dk)
                64,   // hsv (dv)
                4,    // nh (number of heads)
                1,    // nr (GQA ratio)
                512,  // kv (sequence length)
                1,    // nb (batch size)
                true, // mask
                false // causal
            };

            const char * kernel_mode = getenv("GGML_METAL_FA_SCALAR") ? "SCALAR (forced)" : "TILED (default)";
            printf("  Mode: %s\n", kernel_mode);
            printf("  Test: dk=%lld, dv=%lld, nh=%lld, kv=%lld, nb=%lld\n",
                   (long long)bench_params.hsk, (long long)bench_params.hsv,
                   (long long)bench_params.nh, (long long)bench_params.kv,
                   (long long)bench_params.nb);

            timing_result tr = benchmark_fa_kernel(backend, di, bench_params, 10, 100);

            if (tr.avg_time_us > 0.0) {
                printf("  Average time: %.2f us\n", tr.avg_time_us);
                printf("  Throughput: %.2f GFLOPS\n", tr.throughput_gflops);
            } else {
                printf("  Benchmark failed (op not supported)\n");
            }

            ggml_backend_free(backend);
        }
    } else {
        printf("\nSkipping timing benchmarks due to correctness test failures.\n");
    }

    ggml_backend_free(cpu_backend);
    ggml_quantize_free();

    return total_fail > 0 ? 1 : 0;
}
