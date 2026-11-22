// MUL_MAT cases of the Metal backend against the CPU backend, on the first GPU device, which must be a Metal device.
// Run: test_metal_mul_mat_cases [tree | tiled | all | w64] [routes]   (no set word, or all = tree and tiled; w64 only when named).
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
//   w64:   the width-64 fallback matvec kernels of eleven weights, forced by GGML_METAL_TEST_MV_GCN=1; on wave 64 each
//          case must run on kernel_mul_mv_gcn_<weight>_f32 once, another wave prints SKIP, an unknown wave exits 1.
// Prints one CASE line per case and a summary; exit 0 only when every case passes, 2 without a GPU device.
//   tails <results> [from <index>]: the tail cases of tests/mv-bodies.tsv (MV_BODIES names it; MV_TAILS_KIND=mat, id or
//          group runs one kind): one-column multiplies with spare row positions in the last threadgroup, each row of the
//          output against the CPU's (where the CPU would round the activation to another type, against the CPU on F32
//          weights dequantized by ggml from the same bytes), the weight together with the other tensors or alone at the end of its own buffer.
//          Appends START, one TAIL record per output and END to <results>; a case past 20 s is written TIMEOUT and exits 3.
//   strided: mxfp4, iq4_nl, iq4_xs weights as a 37-row view whose row stride is the packed row plus the block's
//          alignment; one STRIDED line each with the receipt and the output hash, for comparing builds.
//   tails-summary <results>: checks every case has its START and records, prints what did not pass and the coverage
//          of each row this card runs; exit 0 only with no GPU finding, no evidence error and every row covered.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

static void fill(std::vector<float> & v, uint64_t seed) {
    uint64_t s = seed;
    for (float & x : v) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        x = (float) ((s >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
    }
}

// The pipelines dispatched since the receipt was last reset, "name xcount + ...".
static std::string dispatched(std::vector<std::string> * names, std::vector<long> * counts) {
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
            counts->push_back(count);
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

// ---- tails: one-column matvecs whose last threadgroup has spare row positions (cases from tests/mv-bodies.tsv) ----

struct inv_row {
    std::string f[12]; // export file body rows case weight act k m placements cards reason
    const std::string & exp() const { return f[0]; }
};

struct tail_case {
    int index;
    size_t row;
    std::string kind;          // mat | id | group
    ggml_type w, a;
    int64_t k;
    std::vector<int64_t> m;    // one count, or the group's two members
    bool alone, partial;
};

static const double tail_row_bound = 1e-3; // a row fails above this share of the CPU vector's root mean square

static bool read_inventory(const char * path, std::vector<inv_row> * out) {
    FILE * fp = fopen(path, "r");
    if (fp == nullptr) {
        return false;
    }
    char line[4096];
    bool header = true;
    while (fgets(line, sizeof(line), fp)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        if (s.empty() || s[0] == '#') {
            continue;
        }
        if (header) {
            header = false;
            continue;
        }
        inv_row r;
        size_t p = 0;
        for (int i = 0; i < 12; ++i) {
            const size_t t = s.find('\t', p);
            r.f[i] = s.substr(p, t == std::string::npos ? std::string::npos : t - p);
            p = t == std::string::npos ? s.size() : t + 1;
        }
        out->push_back(r);
    }
    fclose(fp);
    return !out->empty();
}

static ggml_type type_by_name(const std::string & n) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (ggml_blck_size((ggml_type) t) > 0 && n == ggml_type_name((ggml_type) t)) {
            return (ggml_type) t;
        }
    }
    return GGML_TYPE_COUNT;
}

static int64_t tail_k(const std::string & cell, ggml_type t) {
    if (cell == "page" || cell == "page+2") {
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        int64_t k = 256;
        while (ggml_row_size(t, k) < page) {
            k += 256;
        }
        return cell == "page" ? k : k + 2;
    }
    return atoll(cell.c_str());
}

static std::vector<int64_t> split_counts(const std::string & s) {
    std::vector<int64_t> v;
    for (size_t p = 0; p < s.size();) {
        size_t c = s.find(',', p);
        v.push_back(atoll(s.substr(p, c == std::string::npos ? std::string::npos : c - p).c_str()));
        p = c == std::string::npos ? s.size() : c + 1;
    }
    return v;
}

// Every case of the inventory, in its order; kind "all" or one case kind.
static std::vector<tail_case> tail_cases(const std::vector<inv_row> & inv, const std::string & kind) {
    std::vector<tail_case> out;
    int index = 0;
    for (size_t i = 0; i < inv.size(); ++i) {
        const inv_row & r = inv[i];
        const std::string & c = r.f[4];
        if (c == "none") {
            continue;
        }
        const ggml_type w = type_by_name(r.f[5]), a = type_by_name(r.f[6]);
        const size_t slash = r.f[8].find('/');
        const std::vector<int64_t> part = split_counts(r.f[8].substr(0, slash));
        const int64_t ctrl = slash == std::string::npos ? 0 : atoll(r.f[8].c_str() + slash + 1);
        const std::string ks = r.f[7];
        const size_t ksl = ks.find('/');
        const std::string k_together = ks.substr(0, ksl), k_alone = ksl == std::string::npos ? ks : ks.substr(ksl + 1);
        auto push = [&](std::vector<int64_t> m, bool alone, bool partial, const std::string & kcell) {
            tail_case t = { index++, i, c, w, a, tail_k(kcell, w), m, alone, partial };
            if (kind == "all" || kind == c) {
                out.push_back(t);
            }
        };
        if (c == "group") {
            push({ part[0], ctrl }, false, true, k_together);
            continue;
        }
        const std::string & pl = r.f[9];
        for (int alone = 0; alone < 2; ++alone) {
            if ((alone == 0 && pl == "alone") || (alone == 1 && pl == "together") || (c == "id" && alone == 0)) {
                continue;
            }
            for (int64_t m : part) {
                push({ m }, alone == 1, true, alone ? k_alone : k_together);
            }
            push({ ctrl }, alone == 1, false, alone ? k_alone : k_together);
        }
    }
    return out;
}

static bool card_named(const std::string & cards, const std::string & device) {
    if (cards == "all") {
        return true;
    }
    for (size_t p = 0; p < cards.size();) {
        size_t c = cards.find(',', p);
        const std::string part = cards.substr(p, c == std::string::npos ? std::string::npos : c - p);
        if (!part.empty() && device.find(part) != std::string::npos) {
            return true;
        }
        p = c == std::string::npos ? cards.size() : c + 1;
    }
    return false;
}

// The pipeline name is the export, alone or followed by _<key>=...
static bool is_export(const std::string & name, const std::string & exp) {
    if (name.compare(0, exp.size(), exp) != 0) {
        return false;
    }
    if (name.size() == exp.size()) {
        return true;
    }
    if (name[exp.size()] != '_') {
        return false;
    }
    for (size_t i = exp.size() + 1; i < name.size(); ++i) {
        if (name[i] == '=') {
            return i > exp.size() + 1;
        }
        if (!isalnum((unsigned char) name[i])) {
            return false;
        }
    }
    return false;
}

struct place_info {
    size_t offset = 0, bytes = 0, buffer = 0, rounded = 0, stride = 0;
};

struct graph_out {
    bool unsupported = false, ok = false;
    std::vector<std::vector<float>> outs;
    std::vector<place_info> place;
};

static std::vector<uint8_t> to_type(ggml_type t, const std::vector<float> & f, int64_t k) {
    const int64_t rows = (int64_t) f.size() / k;
    std::vector<uint8_t> b((size_t) rows * ggml_row_size(t, k));
    if (t == GGML_TYPE_F32) {
        memcpy(b.data(), f.data(), b.size());
    } else if (t == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) b.data(), (int64_t) f.size());
    } else if (t == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) b.data(), (int64_t) f.size());
    } else {
        std::vector<float> imatrix((size_t) k, 1.0f);
        if (ggml_quantize_chunk(t, f.data(), b.data(), 0, rows, k, ggml_quantize_requires_imatrix(t) ? imatrix.data() : nullptr) != b.size()) {
            b.clear();
        }
    }
    return b;
}

// A block format's weights as F32 by ggml's own dequantization of the same bytes; empty when it fails or is not finite.
static std::vector<uint8_t> dequantized(ggml_type t, const std::vector<uint8_t> & b, int64_t k) {
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    const size_t rs = ggml_row_size(t, k), rows = b.size() / rs;
    std::vector<float> f(rows * (size_t) k);
    if (tr->to_float == nullptr) {
        return {};
    }
    for (size_t r = 0; r < rows; ++r) {
        tr->to_float(b.data() + r * rs, f.data() + r * (size_t) k, k);
    }
    for (float v : f) {
        if (!std::isfinite(v)) {
            return {};
        }
    }
    std::vector<uint8_t> out(f.size() * sizeof(float));
    memcpy(out.data(), f.data(), out.size());
    return out;
}

// The CPU rounds the activation to its vec_dot_type before it multiplies; where that is not the activation's own type,
// the reference is the CPU on F32 weights dequantized from the same bytes.
static bool cpu_rounds_activation(ggml_type w, ggml_type a) {
    return ggml_get_type_traits_cpu(w)->vec_dot_type != a;
}

// One case's graph on one backend: w holds one weight per member (mat: 1, group: 2) or the experts' planes (id).
static graph_out tail_graph(ggml_backend_t be, const tail_case & c, const std::vector<std::vector<uint8_t>> & w, const std::vector<uint8_t> & x, int32_t expert, bool alone) {
    graph_out g;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_context * cw  = alone ? ggml_init(ip) : ctx;
    std::vector<ggml_tensor *> ws, outs;
    ggml_tensor * b   = nullptr;
    ggml_tensor * ids = nullptr;
    if (c.kind == "id") {
        ws.push_back(ggml_new_tensor_3d(cw, c.w, c.k, c.m[0], 2));
        b   = ggml_new_tensor_3d(ctx, c.a, c.k, 1, 1);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
        outs.push_back(ggml_mul_mat_id(ctx, ws[0], b, ids));
    } else {
        b = ggml_new_tensor_2d(ctx, c.a, c.k, 1);
        for (int64_t m : c.m) {
            if (c.kind == "mat" && !ws.empty()) {
                break;
            }
            ws.push_back(ggml_new_tensor_2d(cw, c.w, c.k, m));
            outs.push_back(ggml_mul_mat(ctx, ws.back(), b));
        }
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * o : outs) {
        ggml_set_output(o);
        ggml_build_forward_expand(gf, o);
    }
    for (ggml_tensor * o : outs) {
        if (!ggml_backend_supports_op(be, o)) {
            g.unsupported = true;
        }
    }
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_gallocr_t ga = nullptr;
    if (!g.unsupported) {
        if (alone) {
            wbuf = ggml_backend_alloc_ctx_tensors(cw, be);
        }
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if ((!alone || wbuf != nullptr) && ggml_gallocr_alloc_graph(ga, gf)) {
            for (size_t i = 0; i < ws.size(); ++i) {
                ggml_backend_tensor_set(ws[i], w[i].data(), 0, w[i].size());
            }
            ggml_backend_tensor_set(b, x.data(), 0, x.size());
            if (ids != nullptr) {
                ggml_backend_tensor_set(ids, &expert, 0, sizeof(expert));
            }
            const size_t page = (size_t) sysconf(_SC_PAGESIZE);
            for (ggml_tensor * t : ws) {
                place_info p;
                const char * base = (const char *) ggml_backend_buffer_get_base(t->buffer);
                p.offset  = (size_t) ((const char *) t->data - base);
                p.bytes   = ggml_nbytes(t);
                p.buffer  = ggml_backend_buffer_get_size(t->buffer);
                p.rounded = (p.buffer + page - 1) / page * page;
                p.stride  = t->nb[1];
                g.place.push_back(p);
            }
            if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
                g.ok = true;
                for (ggml_tensor * o : outs) {
                    std::vector<float> r((size_t) ggml_nelements(o));
                    if (ggml_nbytes(o) != r.size() * sizeof(float)) {
                        g.ok = false;
                    } else {
                        ggml_backend_tensor_get(o, r.data(), 0, ggml_nbytes(o));
                    }
                    g.outs.push_back(r);
                }
            }
        }
    }
    if (ga != nullptr) {
        ggml_gallocr_free(ga);
    }
    if (wbuf != nullptr) {
        ggml_backend_buffer_free(wbuf);
    }
    if (alone) {
        ggml_free(cw);
    }
    ggml_free(ctx);
    return g;
}

static double rms_of(const std::vector<float> & v) {
    double s = 0;
    for (float x : v) {
        s += (double) x * x;
    }
    return v.empty() ? 0 : sqrt(s / v.size());
}

// The rows of the last threadgroups (the last min(m, 16)) are far enough from zero and from each other to be told apart.
static bool tail_rows_distinct(const std::vector<float> & v) {
    const double r = rms_of(v);
    if (!(r > 0) || !std::isfinite(r)) {
        return false;
    }
    const size_t n = std::min<size_t>(v.size(), 16), s = v.size() - n;
    for (size_t i = s; i < v.size(); ++i) {
        if (fabs(v[i]) < 0.05 * r) {
            return false;
        }
        for (size_t j = i + 1; j < v.size(); ++j) {
            if (fabs((double) v[i] - v[j]) < 0.01 * r) {
                return false;
            }
        }
    }
    return true;
}

static uint64_t fnv_floats(const std::vector<float> & v) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * p = (const uint8_t *) v.data();
    for (size_t i = 0; i < v.size() * sizeof(float); ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static FILE * tail_file = nullptr;

static void tail_line(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(tail_file, fmt, ap);
    va_end(ap);
    fflush(tail_file);
    fsync(fileno(tail_file));
}

static std::string receipt_names(std::vector<std::string> * names) {
    std::vector<long> counts;
    dispatched(names, &counts);
    std::string s;
    for (const std::string & n : *names) {
        s += (s.empty() ? "" : "+") + n;
    }
    return s.empty() ? "unknown" : s;
}

static const char * route_of(const std::vector<std::string> & names, const std::string & exp, bool named) {
    if (names.size() != 1) {
        return "other";
    }
    const bool hit = is_export(names[0], exp);
    return named ? (hit ? "ok" : "other") : (hit ? "unlisted" : "othercard");
}

static void tail_record(const tail_case & c, const inv_row & r, size_t member, const place_info & p, const std::string & exp_names,
                        double nmse, const char * hash, const char * worst, const char * ref, const char * cpuq, const char * route, const char * result, const char * note) {
    tail_line("TAIL index=%d member=%zu/%zu case=%s row=%s type=%s k=%lld m=%lld placement=%s offset=%zu bytes=%zu buffer=%zu rounded=%zu export=%s nmse=%.3g hash=%s worst=%s ref=%s cpuq=%s route=%s result=%s%s%s\n",
              c.index, member + 1, c.kind == "group" ? (size_t) 2 : (size_t) 1, c.kind.c_str(), r.exp().c_str(), ggml_type_name(c.w), (long long) c.k,
              (long long) c.m[c.kind == "group" ? member : 0], c.alone ? "alone" : "together", p.offset, p.bytes, p.buffer, p.rounded, exp_names.c_str(), nmse, hash, worst,
              ref, cpuq, route, result, note[0] ? " note=" : "", note);
}

// The Metal name of the system default device, which the backend opens as its first device (device 0); empty when
// this file is not compiled as Objective-C++ (check-mv-tails compiles it so).
static std::string metal_default_name() {
#ifdef __OBJC__
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        return d != nil ? std::string([[d name] UTF8String]) : std::string();
    }
#else
    return std::string();
#endif
}

// MV_TAILS_KIND: all or one case kind; anything else, or a kind that selects no case, is refused.
static bool tail_kind_ok(const std::string & kind, const std::vector<tail_case> & cases, const char * who) {
    if (kind != "all" && kind != "mat" && kind != "id" && kind != "group") {
        printf("%s: REFUSED MV_TAILS_KIND %s: all, mat, id or group\n", who, kind.c_str());
        return false;
    }
    if (cases.empty()) {
        printf("%s: REFUSED MV_TAILS_KIND %s: the inventory gives no case of this kind\n", who, kind.c_str());
        return false;
    }
    return true;
}

static int run_tails(const std::string & path, int from) {
    const char * inv_path = std::getenv("MV_BODIES");
    const char * kind_env = std::getenv("MV_TAILS_KIND");
    const std::string kind = kind_env != nullptr && kind_env[0] ? kind_env : "all";
    std::vector<inv_row> inv;
    if (inv_path == nullptr || !read_inventory(inv_path, &inv)) {
        printf("tails: MV_BODIES must name the inventory (tests/mv-bodies.tsv)\n");
        return 2;
    }
    const std::vector<tail_case> cases = tail_cases(inv, kind);
    if (!tail_kind_ok(kind, cases, "tails")) {
        return 2;
    }
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t gpu = dev != nullptr ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (gpu == nullptr || cpu == nullptr || !ggml_backend_is_metal(gpu)) {
        printf("tails: no Metal GPU device or no CPU backend\n");
        return 2;
    }
    const std::string device = metal_default_name();
    if (device.empty()) {
        printf("tails: no card name: build this file as Objective-C++ (make -C tests check-mv-tails)\n");
        return 2;
    }
    tail_file = fopen(path.c_str(), "a");
    if (tail_file == nullptr) {
        printf("tails: cannot open %s\n", path.c_str());
        return 2;
    }
    tail_line("DEVICE %s\n", device.c_str());
    for (const tail_case & c : cases) {
        if (c.index < from) {
            continue;
        }
        const inv_row & r = inv[c.row];
        const bool named = card_named(r.f[10], device);
        tail_line("START index=%d row=%s case=%s type=%s k=%lld m=%lld placement=%s\n", c.index, r.exp().c_str(), c.kind.c_str(), ggml_type_name(c.w),
                  (long long) c.k, (long long) c.m[0], c.alone ? "alone" : "together");
        fprintf(stderr, "START index=%d %s\n", c.index, r.exp().c_str());
        const size_t members = c.kind == "group" ? 2 : 1;
        const place_info none_place;
        if (c.w == GGML_TYPE_COUNT || c.a == GGML_TYPE_COUNT || c.k % ggml_blck_size(c.w) != 0) {
            for (size_t i = 0; i < members; ++i) {
                tail_record(c, r, i, none_place, "unknown", -1, "-", "-", "-", "-", "-", "FIXTURE", "type or k");
            }
            fprintf(stderr, "DONE index=%d\n", c.index);
            continue;
        }
        // the weights and the CPU result: the seed moves until the tail rows can be told apart
        std::vector<std::vector<uint8_t>> w;
        std::vector<uint8_t> x;
        // the reference multiplies ggml's F32 dequantization of the same bytes where the CPU would round the activation
        const bool blk = cpu_rounds_activation(c.w, c.a);
        const char * ref_kind = blk ? "f32w" : "cpu";
        tail_case cref = c;
        if (blk) {
            cref.w = GGML_TYPE_F32;
        }
        std::vector<std::vector<uint8_t>> wref;
        graph_out want, want0, wantq;
        bool fixture = false, made = true;
        std::vector<float> xf((size_t) c.k);
        fill(xf, 7);
        x = to_type(c.a, xf, c.k);
        for (int attempt = 0; attempt < 16 && !fixture; ++attempt) {
            w.clear();
            made = !x.empty();
            const int planes = c.kind == "group" ? 2 : c.kind == "id" ? 2 : 1;
            std::vector<float> all;
            for (int p = 0; p < planes; ++p) {
                const int64_t rows = c.kind == "group" ? c.m[p] : c.m[0];
                std::vector<float> f((size_t) (rows * c.k));
                fill(f, 1000003ull * (uint64_t) (c.index + 1) + 101ull * (uint64_t) attempt + (uint64_t) p);
                if (c.kind == "id") {
                    all.insert(all.end(), f.begin(), f.end());
                } else {
                    w.push_back(to_type(c.w, f, c.k));
                    made = made && !w.back().empty();
                }
            }
            if (c.kind == "id") {
                w.push_back(to_type(c.w, all, c.k));
                made = made && !w.back().empty();
            }
            wref = w;
            if (blk) {
                for (std::vector<uint8_t> & b : wref) {
                    b = made ? dequantized(c.w, b, c.k) : std::vector<uint8_t>();
                    made = made && !b.empty();
                }
            }
            if (!made) {
                break;
            }
            want = tail_graph(cpu, cref, wref, x, 1, false);
            if (want.unsupported || !want.ok) {
                break;
            }
            fixture = true;
            for (const std::vector<float> & o : want.outs) {
                fixture = fixture && tail_rows_distinct(o);
            }
        }
        graph_out got;
        std::string names_s = "unknown";
        std::vector<std::string> names;
        if (made && fixture && !want.unsupported && want.ok) {
            if (c.kind == "id") {
                want0 = tail_graph(cpu, cref, wref, x, 0, false);
            }
            if (blk) {
                wantq = tail_graph(cpu, c, w, x, 1, false);
            }
            ggml_backend_metal_test_dispatch_receipt(nullptr, 0, 1);
            std::mutex mu;
            std::condition_variable cv;
            bool done = false;
            std::thread watchdog([&]() {
                std::unique_lock<std::mutex> lk(mu);
                if (!cv.wait_for(lk, std::chrono::seconds(20), [&] { return done; })) {
                    std::vector<std::string> n;
                    const std::string s = receipt_names(&n);
                    for (size_t i = 0; i < members; ++i) {
                        tail_record(c, r, i, none_place, s, -1, "-", "-", ref_kind, "-", route_of(n, r.exp(), named), "TIMEOUT", "");
                    }
                    fprintf(stderr, "TIMEOUT index=%d\n", c.index);
                    _exit(3);
                }
            });
            got = tail_graph(gpu, c, w, x, 1, c.alone);
            {
                std::lock_guard<std::mutex> lk(mu);
                done = true;
            }
            cv.notify_one();
            watchdog.join();
            names_s = receipt_names(&names);
        }
        const char * route = route_of(names, r.exp(), named);
        for (size_t i = 0; i < members; ++i) {
            const place_info & p = i < got.place.size() ? got.place[i] : none_place;
            std::string note;
            const char * result = nullptr;
            if (!made || (!want.unsupported && want.ok && !fixture)) {
                result = "FIXTURE";
                note = made ? "tail rows not distinct in 16 seeds" : "quantize or dequantize";
            } else if (want.unsupported || got.unsupported) {
                result = "NOT-RUN";
                note = want.unsupported ? "cpu" : "gpu";
            } else if (!want.ok) {
                result = "FIXTURE";
                note = "cpu compute";
            } else if (!got.ok || got.outs.size() != members || got.outs[i].size() != want.outs[i].size()) {
                result = "ERROR";
            }
            if (result == nullptr && c.alone && !(p.offset == 0 && p.rounded - p.bytes < p.stride)) {
                result = "FIXTURE";
                note = "weight is not at the end of its allocation";
            }
            double nmse = -1;
            char hash[32] = "-", worst[96] = "-", cpuq[32] = "-";
            if (blk && wantq.ok && i < wantq.outs.size() && i < want.outs.size() && wantq.outs[i].size() == want.outs[i].size()) {
                double d = 0;
                for (size_t j = 0; j < want.outs[i].size(); ++j) {
                    d = std::max(d, fabs((double) want.outs[i][j] - wantq.outs[i][j]));
                }
                snprintf(cpuq, sizeof(cpuq), "%.3g", d); // information only: the CPU's own multiply on the quantized weight
            }
            if (result == nullptr) {
                const std::vector<float> & a = got.outs[i], & e = want.outs[i];
                const double rms = rms_of(e);
                double err = 0, ref = 0, wd = -1;
                size_t wi = 0;
                bool bad = !(rms > 0) || !std::isfinite(rms);
                for (size_t j = 0; j < a.size(); ++j) {
                    const double d = fabs((double) a[j] - e[j]);
                    if (!std::isfinite(a[j]) || !(d <= tail_row_bound * rms)) {
                        bad = true;
                    }
                    if (!(d <= wd)) {
                        wd = d;
                        wi = j;
                    }
                    err += d * d;
                    ref += (double) e[j] * e[j];
                }
                nmse = ref > 0 ? err / ref : -1;
                snprintf(hash, sizeof(hash), "%016llx", (unsigned long long) fnv_floats(a));
                snprintf(worst, sizeof(worst), "%zu:%.7g:%.7g", wi, a.empty() ? 0.0 : (double) a[wi], e.empty() ? 0.0 : (double) e[wi]);
                if (bad) {
                    result = "FAIL";
                    if (c.kind == "id" && want0.ok && want0.outs.size() == 1 && want0.outs[0].size() == a.size()) {
                        bool expert0 = true;
                        const double r0 = rms_of(want0.outs[0]);
                        for (size_t j = 0; j < a.size(); ++j) {
                            expert0 = expert0 && fabs((double) a[j] - want0.outs[0][j]) <= tail_row_bound * r0;
                        }
                        if (expert0) {
                            note = "wrong expert";
                        }
                    }
                } else {
                    result = strcmp(route, "ok") == 0 ? "PASS" : strcmp(route, "unlisted") == 0 ? "CARD-UNLISTED" : strcmp(route, "othercard") == 0 ? "OTHER-CARD" : "ROUTE-OTHER";
                }
            }
            tail_record(c, r, i, p, names_s, nmse, hash, worst, ref_kind, cpuq, route, result, note.c_str());
        }
        fprintf(stderr, "DONE index=%d\n", c.index);
    }
    tail_line("END\n");
    fclose(tail_file);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return 0;
}

// Which result a case takes from its members and its markers: GPU findings first, then evidence errors.
static int tail_rank(const std::string & r) {
    if (r.empty()) {
        return 0;
    }
    if (r == "PASS") {
        return 1;
    }
    if (r == "OTHER-CARD") {
        return 2;
    }
    if (r == "NOT-RUN") {
        return 3;
    }
    if (r == "CRASH" || r == "FAIL" || r == "TIMEOUT" || r == "ERROR") {
        return 5;
    }
    return 4; // ROUTE-OTHER, CARD-UNLISTED, FIXTURE, GAP, DUPLICATE
}

static std::string field_of(const std::string & line, const char * key) {
    const std::string k = std::string(" ") + key + "=";
    const size_t p = line.find(k);
    if (p == std::string::npos) {
        return "";
    }
    const size_t s = p + k.size(), e = line.find(' ', s);
    return line.substr(s, e == std::string::npos ? std::string::npos : e - s);
}

static int run_tails_summary(const std::string & path) {
    const char * inv_path = std::getenv("MV_BODIES");
    const char * kind_env = std::getenv("MV_TAILS_KIND");
    const std::string kind = kind_env != nullptr && kind_env[0] ? kind_env : "all";
    std::vector<inv_row> inv;
    if (inv_path == nullptr || !read_inventory(inv_path, &inv)) {
        printf("tails-summary: MV_BODIES must name the inventory (tests/mv-bodies.tsv)\n");
        return 2;
    }
    const std::vector<tail_case> cases = tail_cases(inv, kind);
    if (!tail_kind_ok(kind, cases, "tails-summary")) {
        return 2;
    }
    FILE * fp = fopen(path.c_str(), "r");
    if (fp == nullptr) {
        printf("tails-summary: cannot open %s\n", path.c_str());
        return 2;
    }
    std::string device, last;
    std::map<int, int> starts;
    std::map<int, std::vector<std::string>> recs;
    char buf[8192];
    while (fgets(buf, sizeof(buf), fp)) {
        std::string s(buf);
        while (!s.empty() && s.back() == '\n') {
            s.pop_back();
        }
        if (s.empty()) {
            continue;
        }
        last = s;
        if (s.rfind("DEVICE ", 0) == 0) {
            device = s.substr(7);
        } else if (s.rfind("START ", 0) == 0) {
            starts[atoi(field_of(s, "index").c_str())]++;
        } else if (s.rfind("TAIL ", 0) == 0) {
            recs[atoi(field_of(s, "index").c_str())].push_back(s);
        }
    }
    fclose(fp);
    int p = 0, g = 0, e = 0, n = 0, o = 0, uncovered = 0;
    std::map<int, std::string> case_result; // the case's worst member: PASS, OTHER-CARD, NOT-RUN, an evidence error or a GPU finding
    std::map<int, bool> expected;
    for (const tail_case & c : cases) {
        expected[c.index] = true;
        const size_t members = c.kind == "group" ? 2 : 1;
        const int st = starts.count(c.index) ? starts[c.index] : 0;
        const std::vector<std::string> & rs = recs[c.index];
        // One record per member, each named member=<i>/<members>.
        std::map<std::string, int> per_member;
        for (const std::string & l : rs) {
            per_member[field_of(l, "member")]++;
        }
        size_t named = 0;
        bool doubled = false;
        for (size_t i = 1; i <= members; ++i) {
            const auto it = per_member.find(std::to_string(i) + "/" + std::to_string(members));
            named += it != per_member.end() ? 1 : 0;
            doubled = doubled || (it != per_member.end() && it->second > 1);
        }
        std::string res;
        if (st == 0) {
            res = "GAP";
            printf("GAP index=%d row=%s\n", c.index, inv[c.row].exp().c_str());
        } else if (st > 1 || rs.size() > members || doubled) {
            res = "DUPLICATE";
            printf("DUPLICATE index=%d row=%s\n", c.index, inv[c.row].exp().c_str());
        } else if (named < members) {
            res = "CRASH";
            printf("CRASH index=%d row=%s records=%zu/%zu%s\n", c.index, inv[c.row].exp().c_str(), named, members, rs.empty() ? "" : " (incomplete)");
        }
        for (const std::string & l : rs) {
            const std::string r = field_of(l, "result"), route = field_of(l, "route");
            if (r != "PASS") {
                printf("%s\n", l.c_str());
            }
            if ((r == "FAIL" || r == "TIMEOUT" || r == "ERROR") && (route == "other" || route == "unlisted")) {
                e++; // the pipeline that ran is not the row's on a card the row names, or is the row's on one it does not
                printf("EVIDENCE index=%d route=%s on a %s record\n", c.index, route.c_str(), r.c_str());
            }
            if (tail_rank(r) > tail_rank(res)) {
                res = r;
            }
        }
        case_result[c.index] = res;
        if (res == "PASS") {
            p++;
        } else if (tail_rank(res) >= tail_rank("CRASH")) {
            g++;
        } else if (res == "NOT-RUN") {
            n++;
            e++; // no inventory row exempts a case, so a skipped case is unexplained
            printf("EVIDENCE index=%d NOT-RUN without an inventory exemption\n", c.index);
        } else if (res == "OTHER-CARD") {
            o++;
        } else {
            e++;
        }
    }
    for (const auto & kv : starts) {
        if (!expected.count(kv.first)) {
            printf("STRAY index=%d\n", kv.first);
            e++;
        }
    }
    if (last != "END") {
        printf("NO-END: the results file does not end with END\n");
        e++;
    }
    for (size_t i = 0; i < inv.size(); ++i) {
        const inv_row & r = inv[i];
        if (r.f[4] == "none") {
            printf("EXEMPT %s %s\n", r.exp().c_str(), r.f[11].c_str());
            continue;
        }
        if ((kind != "all" && kind != r.f[4]) || !card_named(r.f[10], device)) {
            continue;
        }
        bool all_named = true, any = false, attempted = false;
        for (const tail_case & c : cases) {
            if (c.row != i || !c.partial) {
                continue;
            }
            any = true;
            const std::vector<std::string> & rs = recs[c.index];
            const std::string first = std::string("1/") + (c.kind == "group" ? "2" : "1"); // member 1 holds the partial rows
            const auto rec = std::find_if(rs.begin(), rs.end(), [&](const std::string & l) { return field_of(l, "member") == first; });
            const bool hit = rec != rs.end() && is_export(field_of(*rec, "export"), r.exp());
            if (!hit) {
                all_named = false;
                const std::string cr = case_result[c.index];
                attempted = attempted || cr == "TIMEOUT" || cr == "CRASH";
            }
        }
        const char * v = any && all_named ? "COVERED" : attempted ? "ATTEMPTED" : "NOT-COVERED";
        if (strcmp(v, "COVERED") != 0) {
            uncovered++;
        }
        printf("%s %s\n", v, r.exp().c_str());
    }
    printf("mv tails: %zu cases, %d pass, %d GPU findings, %d evidence errors, %d not run\n", cases.size(), p, g, e, n);
    printf("mv tails other card: %d (byte-compared between revisions, neither pass nor error); rows not covered or attempted: %d; device %s\n", o, uncovered, device.c_str());
    return g == 0 && e == 0 && uncovered == 0 ? 0 : 1;
}

// ---- strided: weights addressed by block count, as a row view whose stride is not a whole number of blocks ----
// Prints one STRIDED line per format with the receipt and the output hash; two builds compare the hashes (no CPU reference).
static int run_strided() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t gpu = dev != nullptr ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (gpu == nullptr || !ggml_backend_is_metal(gpu)) {
        printf("strided: no Metal GPU device\n");
        return 2;
    }
    const int64_t m = 37, k = 512;
    const char * plant = std::getenv("MV_STRIDED_PLANT"); // route: expect a pipeline no build has (the harness control)
    const bool plant_route = plant != nullptr && strcmp(plant, "route") == 0;
    int bad = 0;
    for (ggml_type t : { GGML_TYPE_MXFP4, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS }) {
        const size_t packed = ggml_row_size(t, k), bs = ggml_type_size(t);
        const size_t a = t == GGML_TYPE_MXFP4 ? 4 : 2; // the block's alignment: 2 when it holds a half, else 4
        const size_t stride = packed + a;
        std::vector<float> wf((size_t) (m * k)), x((size_t) k);
        fill(wf, 3);
        fill(x, 4);
        std::vector<uint8_t> w((size_t) m * packed);
        const bool made = ggml_quantize_chunk(t, wf.data(), w.data(), 0, m, k, nullptr) == w.size();
        const int64_t blocks = (int64_t) ((stride * m + bs - 1) / bs);
        std::vector<uint8_t> base((size_t) blocks * bs, 0);
        for (int64_t r = 0; r < m; ++r) {
            memcpy(base.data() + r * stride, w.data() + r * packed, packed);
        }
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * full = ggml_new_tensor_1d(ctx, t, blocks * ggml_blck_size(t));
        ggml_tensor * view = ggml_view_2d(ctx, full, k, m, stride, 0);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
        ggml_tensor * out = ggml_mul_mat(ctx, view, b);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
        std::vector<float> res;
        std::string ran = "not computed";
        std::vector<std::string> names;
        std::vector<long> counts;
        if (made && ggml_backend_supports_op(gpu, out) && ggml_gallocr_alloc_graph(ga, gf)) {
            ggml_backend_tensor_set(full, base.data(), 0, base.size());
            ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
            ggml_backend_metal_test_dispatch_receipt(nullptr, 0, 1);
            if (ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS) {
                res.resize((size_t) m);
                ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
            }
            ran = dispatched(&names, &counts);
        }
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        // The case shows the bounded address only on its one export, once; another route or none is NOT-RUN.
        const std::string want = plant_route ? "kernel_mul_mv_none_f32" : std::string("kernel_mul_mv_") + ggml_type_name(t) + "_f32";
        const bool on_route = names.size() == 1 && counts[0] == 1 && is_export(names[0], want);
        if (res.empty() || !on_route) {
            bad++;
            printf("STRIDED %s m=%lld k=%lld stride=%zu (packed %zu + %zu) · %s · NOT-RUN (%s, want %s)\n", ggml_type_name(t), (long long) m, (long long) k, stride, packed, a, ran.c_str(),
                   res.empty() ? "no output" : "route", want.c_str());
            continue;
        }
        const size_t p = names[0].find("_nr0=");
        const int nr0 = p == std::string::npos ? 0 : atoi(names[0].c_str() + p + 5);
        printf("STRIDED %s m=%lld k=%lld stride=%zu (packed %zu + %zu) · %s · nr0=%d%s · hash %016llx\n", ggml_type_name(t), (long long) m, (long long) k, stride, packed, a, ran.c_str(),
               nr0, nr0 == 1 ? " (no partial simdgroup holds a valid row: the hash shows nothing)" : "", (unsigned long long) fnv_floats(res));
    }
    return bad ? 1 : 0;
}

struct mm_case {
    const char * set;
    ggml_type type;
    int64_t m, n, k;
};

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "tails") == 0 && (argc == 3 || (argc == 5 && strcmp(argv[3], "from") == 0))) {
        return run_tails(argv[2], argc == 5 ? atoi(argv[4]) : 0);
    }
    if (argc == 2 && strcmp(argv[1], "strided") == 0) {
        return run_strided();
    }
    if (argc == 3 && strcmp(argv[1], "tails-summary") == 0) {
        return run_tails_summary(argv[2]);
    }
    std::string only = argc > 1 ? argv[1] : "";
    bool routes = argc > 2 && strcmp(argv[2], "routes") == 0;
    if (only == "routes" && argc == 2) {
        routes = true;
        only = "";
    }
    if (only == "all") {
        only = "";
    }
    if (argc > 3 || (argc == 3 && !routes) || (!only.empty() && only != "tree" && only != "tiled" && only != "w64")) {
        printf("usage: test_metal_mul_mat_cases [tree | tiled | all | w64] [routes] | tails <results> [from <index>] | tails-summary <results> | strided\n");
        return 2;
    }
    if (only == "w64" && std::getenv("GGML_METAL_TEST_MV_GCN") == nullptr) {
        printf("w64 needs GGML_METAL_TEST_MV_GCN=1\n");
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
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                         GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
        cases.push_back({ "w64", t, 16, 1, 256 });
        cases.push_back({ "w64", t, 64, 1, 2304 });
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

    uint32_t wave = 0, group_bytes = 0;
    if (only == "w64" && ggml_backend_metal_tiled_card_shape(dev, &wave, &group_bytes) == 0) {
        printf("w64: the card's wave is unknown\n");
        return 1;
    }
    const bool w64_route = only == "w64" && wave == 64;

    const double bound = 5e-4;
    int run = 0, failed = 0;
    for (const mm_case & c : cases) {
        if (only.empty() ? strcmp(c.set, "w64") == 0 : only != c.set) {
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
        std::vector<long> counts;
        const std::string ran = dispatched(&names, &counts);
        bool route_ok = true;
        if (strcmp(c.set, "w64") == 0) {
            const std::string want_name = std::string("kernel_mul_mv_gcn_") + ggml_type_name(c.type) + "_f32";
            route_ok = !w64_route || (names.size() == 1 && names[0] == want_name && counts[0] == 1);
        } else {
            const char * family = strcmp(c.set, "tree") == 0 ? "kernel_mul_mv_" : "kernel_mul_mat_tiled_";
            route_ok = !routes || (names.size() == 1 && names[0].rfind(family, 0) == 0);
        }
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
    if (only == "w64" && !w64_route && failed == 0) {
        printf("SKIP w64: wave %u, this card takes no width-64 fallback kernel\n", wave);
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return failed == 0 && run > 0 ? 0 : 1;
}
