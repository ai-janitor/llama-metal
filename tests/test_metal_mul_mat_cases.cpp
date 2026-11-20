// MUL_MAT cases of the Metal backend against the CPU backend, on the first GPU device, which must be a Metal device.
// Run: test_metal_mul_mat_cases [tree | tiled | all] [routes]   (no set word, or all = both sets).
// routes: each case must also have run on the kernel family its set is about, read from the backend's dispatch
// receipt: a tree case on one matvec pipeline (kernel_mul_mv_...), a tiled case on one tiled pipeline
// (kernel_mul_mat_tiled_...). For cards that take those routes (the AMD cards); the pipeline is printed either way. Each case multiplies one random weight (fixed seed, converted to the case's type by ggml) by
// random f32 activations on both backends and passes when the normalized mean squared error of the Metal result
// against the CPU's is at most 5e-4, the bound llama.cpp's test-backend-ops uses for MUL_MAT.
//   tree:  one activation column on an f16 or bf16 weight whose input width sits at the steps of a backend that takes
//          one simdgroup per 128 input columns: 2 groups (256), 3 groups (257, 300, 384), 4 groups (385). A row sum
//          reduced by pairwise halving loses partial sums when the group count is 3.
//   tiled: batch shapes of the tiled kernel for q4_K, q5_K, q6_K, q8_0: 128 rows x 128 columns x width 512, and
//          200 x 130 x 768 (rows and columns that are not whole tiles).
// Prints one CASE line per case and a summary; exit 0 only when every case passes, 2 without a GPU device.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void fill(std::vector<float> & v, uint64_t seed) {
    uint64_t s = seed;
    for (float & x : v) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        x = (float) ((s >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
    }
}

// The pipelines dispatched since the receipt was last reset, "name xcount + ...".
static std::string dispatched(std::vector<std::string> * names) {
    const int need = ggml_backend_metal_test_dispatch_receipt(nullptr, 0, 0);
    std::string js(need > 0 ? (size_t) need : 1, '\0'), out;
    if (need <= 0 || ggml_backend_metal_test_dispatch_receipt(&js[0], js.size(), 0) != need) {
        return "no receipt";
    }
    for (size_t p = js.find('{'); p != std::string::npos; p = js.find('{', p + 1)) { // one object per pipeline; its strings hold no brace
        const size_t end = js.find('}', p);
        if (end == std::string::npos) {
            break;
        }
        const std::string obj = js.substr(p, end - p + 1);
        const size_t a = obj.find("\"pipeline\":\""), c = obj.find("\"count\":");
        if (a == std::string::npos || c == std::string::npos) {
            continue;
        }
        const std::string name = obj.substr(a + 12, obj.find('"', a + 12) - (a + 12));
        const long count = atol(obj.c_str() + c + 8);
        if (count > 0) {
            names->push_back(name);
            out += (out.empty() ? "" : " + ") + name + " x" + std::to_string(count);
        }
    }
    return out.empty() ? "nothing dispatched" : out;
}

// The m x n result of weight (type, m rows of k) times activations (n columns of k) on one backend; empty on failure.
static std::vector<float> multiply(ggml_backend_t backend, ggml_type type, int64_t m, int64_t n, int64_t k, const std::vector<uint8_t> & w, const std::vector<float> & x) {
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::vector<float> res;
    if (ggml_gallocr_alloc_graph(ga, gf)) {
        ggml_backend_tensor_set(a, w.data(), 0, w.size());
        ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS && ggml_nbytes(out) == (size_t) (m * n) * sizeof(float)) {
            res.resize((size_t) (m * n));
            ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        }
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return res;
}

struct mm_case {
    const char * set;
    ggml_type type;
    int64_t m, n, k;
};

int main(int argc, char ** argv) {
    std::string only = argc > 1 ? argv[1] : "";
    bool routes = argc > 2 && strcmp(argv[2], "routes") == 0;
    if (only == "routes" && argc == 2) {
        routes = true;
        only = "";
    }
    if (only == "all") {
        only = "";
    }
    if (argc > 3 || (argc == 3 && !routes) || (!only.empty() && only != "tree" && only != "tiled")) {
        printf("usage: test_metal_mul_mat_cases [tree | tiled | all] [routes]\n");
        return 2;
    }
    std::vector<mm_case> cases;
    for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_BF16 }) {
        for (int64_t k : { 256, 257, 300, 384, 385 }) {
            cases.push_back({ "tree", t, 512, 1, k });
        }
    }
    for (ggml_type t : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 }) {
        cases.push_back({ "tiled", t, 128, 128, 512 });
        cases.push_back({ "tiled", t, 200, 130, 768 });
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t gpu = dev != nullptr ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (gpu == nullptr || cpu == nullptr) {
        printf("no GPU device or no CPU backend\n");
        return 2;
    }
    if (!ggml_backend_is_metal(gpu)) {
        printf("the first GPU device (%s) is not a Metal device: this test is about the Metal backend\n", ggml_backend_dev_description(dev));
        return 2;
    }
    printf("device: %s (Metal)\n", ggml_backend_dev_description(dev));

    const double bound = 5e-4;
    int run = 0, failed = 0;
    for (const mm_case & c : cases) {
        if (!only.empty() && only != c.set) {
            continue;
        }
        run++;
        std::vector<float> wf((size_t) (c.m * c.k)), x((size_t) (c.n * c.k));
        fill(wf, 1);
        fill(x, 2);
        std::vector<uint8_t> w((size_t) c.m * ggml_row_size(c.type, c.k));
        const bool made = ggml_quantize_chunk(c.type, wf.data(), w.data(), 0, c.m, c.k, nullptr) == w.size();
        const std::vector<float> want = made ? multiply(cpu, c.type, c.m, c.n, c.k, w, x) : std::vector<float>();
        ggml_backend_metal_test_dispatch_receipt(nullptr, 0, 1); // a new recording window for this case
        const std::vector<float> got  = made ? multiply(gpu, c.type, c.m, c.n, c.k, w, x) : std::vector<float>();
        std::vector<std::string> names;
        const std::string ran = dispatched(&names);
        const char * family = strcmp(c.set, "tree") == 0 ? "kernel_mul_mv_" : "kernel_mul_mat_tiled_";
        const bool route_ok = !routes || (names.size() == 1 && names[0].rfind(family, 0) == 0);
        double err = 0, ref = 0;
        bool finite = !want.empty() && want.size() == got.size();
        for (size_t i = 0; finite && i < want.size(); ++i) {
            finite = std::isfinite(want[i]) && std::isfinite(got[i]);
            err += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
            ref += (double) want[i] * want[i];
        }
        const double nmse = finite && ref > 0 ? err / ref : -1;
        const bool ok = finite && ref > 0 && nmse <= bound && route_ok;
        failed += ok ? 0 : 1;
        printf("CASE %-5s %-4s m %lld n %lld k %lld · nmse %.3g · %s%s · %s\n", c.set, ggml_type_name(c.type), (long long) c.m, (long long) c.n, (long long) c.k, nmse, ran.c_str(),
               route_ok ? "" : " (not the route of this set)", ok ? "PASS" : "FAIL");
    }
    printf("mul_mat cases: %d run, %d failed (bound %.0e)\n", run, failed, bound);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return failed == 0 && run > 0 ? 0 : 1;
}
