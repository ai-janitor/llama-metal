// Metal operators against the CPU backend on the first GPU device (must be Metal), one CASE line per case.
// Run: test_metal_op_cases [fold | convt | argmax | attn | tri | mmid]; EXPECT_RED="<ids>" names cases that must give their predicted red value.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static void fill(std::vector<float> & v, uint64_t seed) {
    uint64_t s = seed;
    for (float & x : v) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        x = (float) ((s >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
    }
}

static std::vector<float> rnd(size_t n, uint64_t seed) {
    std::vector<float> v(n);
    fill(v, seed);
    return v;
}

// pipeline key -> exported function, from the backend's "compiling pipeline" log lines
static std::map<std::string, std::string> g_base;

static void log_cb(enum ggml_log_level level, const char * text, void * ud) {
    (void) ud;
    const char * p = strstr(text, "compiling pipeline:");
    if (p) {
        const char * b = strstr(p, "base = '");
        const char * n = strstr(p, "name = '");
        if (b && n) {
            b += 8;
            n += 8;
            const char * be = strchr(b, '\'');
            const char * ne = strchr(n, '\'');
            if (be && ne) {
                g_base[std::string(n, ne - n)] = std::string(b, be - b);
            }
        }
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static std::string base_of(const std::string & key) {
    auto it = g_base.find(key);
    return it == g_base.end() ? key : it->second;
}

// The pipelines dispatched since the receipt was last reset, by key.
static std::map<std::string, long> dispatched() {
    std::map<std::string, long> keys;
    const int need = ggml_backend_metal_test_dispatch_receipt(nullptr, 0, 0);
    std::string js(need > 0 ? (size_t) need : 1, '\0');
    if (need <= 0 || ggml_backend_metal_test_dispatch_receipt(&js[0], js.size(), 0) != need) {
        return keys;
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
        const long count = atol(obj.c_str() + c + 8);
        if (count > 0) {
            keys[obj.substr(a + 12, obj.find('"', a + 12) - (a + 12))] += count;
        }
    }
    return keys;
}

// One input: its shape, type and values (floats are converted to the type; I32 takes ints).
struct input {
    ggml_type type;
    int64_t ne[4];
    std::vector<float> f;
    std::vector<int32_t> i;
};

struct op_case {
    std::string set, id, exported;
    bool key_nsg_sr;                       // the receipt key must be exported + "_nsg=<n>_sr=<0|1>"
    std::vector<input> in;
    std::function<ggml_tensor *(ggml_context *, std::vector<ggml_tensor *> &)> make;
    std::vector<float> prefill;            // the output's values before the compute, empty = none
    // judge(cpu, metal, detail): the case's own rule; red(metal, cpu, detail): the predicted red result
    std::function<bool(const std::vector<uint8_t> &, const std::vector<uint8_t> &, std::string &)> judge, red;
};

static std::vector<uint8_t> to_bytes(const input & x) {
    int64_t n = x.ne[0] * x.ne[1] * x.ne[2] * x.ne[3];
    std::vector<uint8_t> b(ggml_row_size(x.type, x.ne[0]) * (size_t) (n / x.ne[0]));
    if (x.type == GGML_TYPE_I32) {
        memcpy(b.data(), x.i.data(), b.size());
    } else if (x.type == GGML_TYPE_F32) {
        memcpy(b.data(), x.f.data(), b.size());
    } else if (x.type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(x.f.data(), (ggml_fp16_t *) b.data(), n);
    } else {
        ggml_quantize_chunk(x.type, x.f.data(), b.data(), 0, n / x.ne[0], x.ne[0], nullptr);
    }
    return b;
}

// The output bytes of the case on one backend; empty when it did not compute. Resets the receipt before the compute.
static std::vector<uint8_t> compute(ggml_backend_t backend, const op_case & c, bool receipt) {
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> ts;
    for (const input & x : c.in) {
        ggml_tensor * t = ggml_new_tensor(ctx, x.type, 4, x.ne);
        ggml_set_input(t);
        ts.push_back(t);
    }
    ggml_tensor * out = c.make(ctx, ts);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    std::vector<uint8_t> res;
    bool ok = true;
    for (int i = 0; ok && i < ggml_graph_n_nodes(gf); ++i) {
        ok = ggml_backend_supports_op(backend, ggml_graph_node(gf, i));
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (ok && ggml_gallocr_alloc_graph(ga, gf)) {
        for (size_t i = 0; i < ts.size(); ++i) {
            const std::vector<uint8_t> b = to_bytes(c.in[i]);
            ggml_backend_tensor_set(ts[i], b.data(), 0, b.size());
        }
        if (!c.prefill.empty()) {
            ggml_backend_tensor_set(out, c.prefill.data(), 0, ggml_nbytes(out));
        }
        if (receipt) {
            ggml_backend_metal_test_dispatch_receipt(nullptr, 0, 1); // a new recording window for this case
        }
        if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
            res.resize(ggml_nbytes(out));
            ggml_backend_tensor_get(out, res.data(), 0, res.size());
        }
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return res;
}

static std::vector<float> as_f32(const std::vector<uint8_t> & b) {
    std::vector<float> v(b.size() / sizeof(float));
    memcpy(v.data(), b.data(), v.size() * sizeof(float));
    return v;
}

static std::vector<int32_t> as_i32(const std::vector<uint8_t> & b) {
    std::vector<int32_t> v(b.size() / sizeof(int32_t));
    memcpy(v.data(), b.data(), v.size() * sizeof(int32_t));
    return v;
}

static long first_nonfinite(const std::vector<float> & v) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v[i])) {
            return (long) i;
        }
    }
    return -1;
}

// normalized mean squared error over the elements whose index keep() accepts; -1 when either side is not finite
static double nmse(const std::vector<float> & want, const std::vector<float> & got, const std::function<bool(size_t)> & keep = nullptr) {
    if (want.size() != got.size() || want.empty()) {
        return -1;
    }
    double err = 0, ref = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (keep && !keep(i)) {
            continue;
        }
        if (!std::isfinite(want[i]) || !std::isfinite(got[i])) {
            return -1;
        }
        err += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
        ref += (double) want[i] * want[i];
    }
    return ref > 0 ? err / ref : (err == 0 ? 0 : -1);
}

static std::string fmt(const char * f, double v) {
    char s[64];
    snprintf(s, sizeof(s), f, v);
    return s;
}

static std::function<bool(const std::vector<uint8_t> &, const std::vector<uint8_t> &, std::string &)> cpu_bound(double bound) {
    return [=](const std::vector<uint8_t> & cpu, const std::vector<uint8_t> & gpu, std::string & d) {
        const double e = nmse(as_f32(cpu), as_f32(gpu));
        d = "nmse " + fmt("%.3g", e) + " (bound " + fmt("%.0e", bound) + ")";
        return e >= 0 && e <= bound;
    };
}

static std::vector<op_case> case_table() {
    std::vector<op_case> cs;
    const ggml_type F32 = GGML_TYPE_F32, F16 = GGML_TYPE_F16, I32 = GGML_TYPE_I32;

    // fold: frames [64, n_codes], window [64], n_hop 16; case 5 has a zero window value under a frame sample of 1.0
    struct fold_c { int id, n_codes, n_hop; };
    for (fold_c f : { fold_c{ 1, 8, 16 }, fold_c{ 2, 61, 16 }, fold_c{ 3, 62, 16 }, fold_c{ 4, 200, 16 }, fold_c{ 5, 4, 64 } }) {
        const int n_fft = 64, n_out = f.n_hop * (f.n_codes - 1) + n_fft;
        std::vector<float> frames = rnd((size_t) (n_fft * f.n_codes), 100 + f.id), window = rnd(n_fft, 200 + f.id);
        for (float & w : window) {
            w = 0.25f + 0.375f * (w + 1.0f);
        }
        op_case c = { "fold", "fold." + std::to_string(f.id), "kernel_fold_f32", false,
            { { F32, { n_fft, f.n_codes, 1, 1 }, frames, {} }, { F32, { n_fft, 1, 1, 1 }, window, {} } },
            [=](ggml_context * ctx, std::vector<ggml_tensor *> & t) { return ggml_fold(ctx, t[0], t[1], n_out, f.n_hop, 0); },
            {}, cpu_bound(1e-6), nullptr };
        if (f.id == 5) {
            c.in[1].f[0] = 0.0f;
            for (int l = 0; l < f.n_codes; ++l) {
                c.in[0].f[(size_t) (l * n_fft)] = 1.0f;
            }
            // samples 0, 64, 128, 192: one frame adds 1.0 and the envelope is zero there
            auto at_zero = [](size_t i) { return i % 64 == 0; };
            c.judge = [=](const std::vector<uint8_t> & cpu, const std::vector<uint8_t> & gpu, std::string & d) {
                const std::vector<float> w = as_f32(cpu), g = as_f32(gpu);
                bool ones = !w.empty() && w.size() == g.size();
                for (size_t i = 0; ones && i < w.size(); i += 64) {
                    ones = w[i] == 1.0f && g[i] == 1.0f;
                }
                const double e = nmse(w, g);
                d = "samples 0,64,128,192 " + std::string(ones ? "1.0 on both" : "not 1.0 on both") + " · nmse " + fmt("%.3g", e) + " (bound 1e-06)";
                return ones && e >= 0 && e <= 1e-6;
            };
            c.red = [=](const std::vector<uint8_t> & cpu, const std::vector<uint8_t> & gpu, std::string & d) {
                const std::vector<float> w = as_f32(cpu), g = as_f32(gpu);
                bool zeros = !w.empty() && w.size() == g.size();
                for (size_t i = 0; zeros && i < g.size(); i += 64) {
                    zeros = g[i] == 0.0f && w[i] == 1.0f;
                }
                const double e = nmse(w, g, [=](size_t i) { return !at_zero(i); });
                d = "predicted red: samples 0,64,128,192 " + std::string(zeros ? "0.0 (CPU 1.0)" : "not 0.0") + " · nmse of the others " + fmt("%.3g", e);
                return zeros && e >= 0 && e <= 1e-6;
            };
        }
        cs.push_back(c);
    }

    // convt: kernel [K, OC, IC] f32 or f16, data [IL, IC]; a gap (s0 > K), one input, a zero stride
    struct convt_c { int il, k, s0, ic, oc; };
    for (ggml_type kt : { F32, F16 }) {
        for (convt_c v : { convt_c{ 64, 4, 2, 3, 2 }, convt_c{ 300, 16, 8, 2, 2 }, convt_c{ 40, 3, 5, 2, 2 }, convt_c{ 1, 4, 1, 1, 1 }, convt_c{ 3, 4, 0, 2, 1 } }) {
            const std::string id = "convt." + std::string(ggml_type_name(kt)) + ".il" + std::to_string(v.il) + "k" + std::to_string(v.k) + "s" + std::to_string(v.s0) +
                                   "ic" + std::to_string(v.ic) + "oc" + std::to_string(v.oc);
            const uint64_t seed = 300 + 10 * (uint64_t) cs.size();
            cs.push_back({ "convt", id, "kernel_conv_transpose_1d_" + std::string(ggml_type_name(kt)) + "_f32", false,
                { { kt, { v.k, v.oc, v.ic, 1 }, rnd((size_t) (v.k * v.oc * v.ic), seed), {} }, { F32, { v.il, v.ic, 1, 1 }, rnd((size_t) (v.il * v.ic), seed + 1), {} } },
                [=](ggml_context * ctx, std::vector<ggml_tensor *> & t) { return ggml_conv_transpose_1d(ctx, t[0], t[1], v.s0, 0, 1); },
                {}, cpu_bound(kt == F16 ? 1e-5 : 1e-6), nullptr });
        }
    }

    // argmax: rows of 1024; 3 rows take kernel_argmax_f32, 1 row kernel_argmax_f32_w4. A tie expects the lowest index.
    struct am_c { const char * name; int a, b; bool ninf; };
    for (int rows : { 3, 1 }) {
        for (am_c v : { am_c{ "distinct", -1, -1, false }, am_c{ "tie512", 300, 812, false }, am_c{ "tie1", 300, 301, false },
                        am_c{ "ninf", -1, -1, true }, am_c{ "first", 0, -1, false }, am_c{ "last", 1023, -1, false } }) {
            std::vector<float> row(1024);
            for (int i = 0; i < 1024; ++i) {
                row[i] = v.ninf ? -INFINITY : (float) ((i * 389) % 1024) / 1024.0f - 0.5f; // distinct values
            }
            int want = (int) (std::max_element(row.begin(), row.end()) - row.begin());
            if (v.a >= 0) {
                row[v.a] = 2.0f;
                want = v.a;
            }
            if (v.b >= 0) {
                row[v.b] = 2.0f;
            }
            if (v.ninf) {
                want = 1023;
            }
            // predicted red of the kernels before the tie rule: the higher tied index on the 3-row kernel; -1 or INT32_MAX for -inf
            const int pred = rows == 3 ? (v.ninf ? -1 : v.b) : (v.ninf ? 0x7fffffff : -2);
            std::vector<float> x;
            for (int r = 0; r < rows; ++r) {
                x.insert(x.end(), row.begin(), row.end());
            }
            auto line = [=](const std::vector<uint8_t> & cpu, const std::vector<uint8_t> & gpu, std::string & d, int expect) {
                const std::vector<int32_t> w = as_i32(cpu), g = as_i32(gpu);
                bool ok = (int) g.size() == rows && (int) w.size() == rows;
                std::string ks;
                for (int r = 0; ok && r < rows; ++r) {
                    ok = g[r] == expect;
                    ks += (r ? "," : "") + std::to_string(g[r]);
                }
                d = "expected " + std::to_string(expect) + " · CPU " + (w.empty() ? std::string("-") : std::to_string(w[0])) + " · kernel " + (ks.empty() ? "-" : ks);
                return ok;
            };
            op_case c = { "argmax", "argmax.rows" + std::to_string(rows) + "." + v.name, rows == 3 ? "kernel_argmax_f32" : "kernel_argmax_f32_w4", false,
                { { F32, { 1024, rows, 1, 1 }, x, {} } },
                [](ggml_context * ctx, std::vector<ggml_tensor *> & t) { return ggml_argmax(ctx, t[0]); },
                {}, nullptr, nullptr };
            c.judge = [=](const std::vector<uint8_t> & cpu, const std::vector<uint8_t> & gpu, std::string & d) { return line(cpu, gpu, d, want); };
            if (pred != -2 && (v.ninf || std::string(v.name) == "tie1")) {
                c.red = [=](const std::vector<uint8_t> & cpu, const std::vector<uint8_t> & gpu, std::string & d) {
                    const bool r = line(cpu, gpu, d, pred);
                    d = "predicted red: " + d;
                    return r;
                };
            }
            cs.push_back(c);
        }
    }

    // attn: Q f32, K and V f16, head 256, 64 cache rows, 32 queries; the queries are the last 32 positions of the cache
    struct attn_c { const char * name; int heads, mask; float max_bias; };
    for (attn_c v : { attn_c{ "nomask", 4, 0, 0.0f }, attn_c{ "causal", 4, 1, 0.0f }, attn_c{ "causal_rowinf", 4, 2, 0.0f }, attn_c{ "bias", 4, 3, 0.0f },
                      attn_c{ "causal_alibi8", 4, 1, 8.0f }, attn_c{ "causal_alibi1024_h1", 1, 1, 1024.0f } }) {
        const int d = 256, nkv = 64, nq = 32, nh = v.heads, inf_row = 5;
        const uint64_t seed = 500 + 10 * (uint64_t) cs.size();
        op_case c = { "attn", std::string("attn.") + v.name, "kernel_flash_attn_ext_scalar_dk256_dv256", false,
            { { F32, { d, nq, nh, 1 }, rnd((size_t) (d * nq * nh), seed), {} }, { F16, { d, nkv, nh, 1 }, rnd((size_t) (d * nkv * nh), seed + 1), {} },
              { F16, { d, nkv, nh, 1 }, rnd((size_t) (d * nkv * nh), seed + 2), {} } },
            nullptr, {}, nullptr, nullptr };
        if (v.mask) {
            std::vector<float> m((size_t) (nkv * nq));
            const std::vector<float> r = rnd(m.size(), seed + 3);
            for (int q = 0; q < nq; ++q) {
                for (int j = 0; j < nkv; ++j) {
                    const bool above = j > q + (nkv - nq);
                    float x = v.mask == 3 ? r[(size_t) (q * nkv + j)] : (above ? -INFINITY : 0.0f);
                    if (v.mask == 2 && q == inf_row) {
                        x = -INFINITY;
                    }
                    m[(size_t) (q * nkv + j)] = x;
                }
            }
            c.in.push_back({ F16, { nkv, nq, 1, 1 }, m, {} });
        }
        const bool has_mask = v.mask != 0;
        const float max_bias = v.max_bias;
        c.make = [=](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
            return ggml_flash_attn_ext(ctx, t[0], t[1], t[2], has_mask ? t[3] : nullptr, 1.0f / 16.0f, max_bias, 0.0f);
        };
        const bool row_inf = v.mask == 2;
        c.judge = [=](const std::vector<uint8_t> & cpu, const std::vector<uint8_t> & gpu, std::string & dd) {
            const std::vector<float> w = as_f32(cpu), g = as_f32(gpu);
            bool ok = true;
            if (row_inf) { // output [d, heads, queries]: every element of the all-masked query is a finite zero
                for (int h = 0; h < nh; ++h) {
                    for (int i = 0; i < d; ++i) {
                        const float x = g.size() == w.size() ? g[(size_t) ((inf_row * nh + h) * d + i)] : 1.0f;
                        ok = ok && x == 0.0f;
                    }
                }
                dd = std::string("masked query ") + (ok ? "zero" : "NOT zero") + " · ";
            }
            const long wn = first_nonfinite(w), gn = first_nonfinite(g);
            if (wn >= 0) { // no CPU comparison: the case is counted apart
                dd += "CPU not finite at " + std::to_string(wn) + " · Metal first non-finite " + std::to_string(gn);
                return ok && gn < 0;
            }
            const double e = nmse(w, g);
            dd += "nmse " + fmt("%.3g", e) + " (bound 5e-04)";
            return ok && e >= 0 && e <= 5e-4;
        };
        cs.push_back(c);
    }

    // tri: SOLVE_TRI, a lower-triangular [N, N] system (the upper part holds unused values), right-hand side [K, N]
    for (int n : { 5, 33, 64, 100 }) {
        for (int k : { 1, 3 }) {
            const uint64_t seed = 700 + 10 * (uint64_t) cs.size();
            std::vector<float> a = rnd((size_t) (n * n), seed);
            for (int r = 0; r < n; ++r) {
                for (int j = 0; j < r; ++j) {
                    a[(size_t) (r * n + j)] *= 0.1f;
                }
                a[(size_t) (r * n + r)] = 1.5f + 0.5f * a[(size_t) (r * n + r)];
            }
            cs.push_back({ "tri", "tri.n" + std::to_string(n) + "k" + std::to_string(k), "kernel_solve_tri_f32", false,
                { { F32, { n, n, 1, 1 }, a, {} }, { F32, { k, n, 1, 1 }, rnd((size_t) (k * n), seed + 1), {} } },
                [](ggml_context * ctx, std::vector<ggml_tensor *> & t) { return ggml_solve_tri(ctx, t[0], t[1], true, true, false); },
                {}, cpu_bound(1e-5), nullptr });
        }
    }

    // mmid: 4 experts of 512 rows x width 256, one token, slots (1, -1); the output is filled with 1.0 first
    for (ggml_type wt : { F32, GGML_TYPE_Q8_0 }) {
        const int k = 256, m = 512, ne = 4;
        const uint64_t seed = 900 + 10 * (uint64_t) cs.size();
        op_case c = { "mmid", "mmid." + std::string(ggml_type_name(wt)), wt == F32 ? "kernel_mul_mv_id_f32_f32_4" : "kernel_mul_mv_id_q8_0_f32", true,
            { { wt, { k, m, ne, 1 }, rnd((size_t) (k * m * ne), seed), {} }, { F32, { k, 1, 1, 1 }, rnd((size_t) k, seed + 1), {} },
              { I32, { 2, 1, 1, 1 }, {}, { 1, -1 } } },
            [](ggml_context * ctx, std::vector<ggml_tensor *> & t) { return ggml_mul_mat_id(ctx, t[0], t[1], t[2]); },
            std::vector<float>((size_t) (2 * m), 1.0f), nullptr, nullptr };
        c.judge = [=](const std::vector<uint8_t> & cpu, const std::vector<uint8_t> & gpu, std::string & d) {
            const std::vector<float> w = as_f32(cpu), g = as_f32(gpu);
            if (w.size() != (size_t) (2 * m) || g.size() != w.size()) {
                d = "wrong output size";
                return false;
            }
            int nonzero = 0;
            for (int i = m; i < 2 * m; ++i) {
                nonzero += g[(size_t) i] == 0.0f ? 0 : 1;
            }
            const double e = nmse(w, g, [=](size_t i) { return i < (size_t) m; });
            d = "slot of id -1: " + std::to_string(nonzero) + " of 512 not 0.0 · slot of id 1: nmse " + fmt("%.3g", e) + " (bound 5e-04)";
            return nonzero == 0 && e >= 0 && e <= 5e-4;
        };
        cs.push_back(c);
    }
    return cs;
}

static uint64_t fnv1a(const std::vector<uint8_t> & b) {
    uint64_t h = 14695981039346656037ull;
    for (uint8_t x : b) {
        h ^= x;
        h *= 1099511628211ull;
    }
    return h;
}

// key == exported + "_nsg=<digits>_sr=<0|1>"
static bool nsg_sr_key(const std::string & key, const std::string & exported) {
    const std::string p = exported + "_nsg=";
    if (key.rfind(p, 0) != 0) {
        return false;
    }
    size_t i = p.size(), j = i;
    while (j < key.size() && key[j] >= '0' && key[j] <= '9') {
        j++;
    }
    return j > i && (key.compare(j, std::string::npos, "_sr=0") == 0 || key.compare(j, std::string::npos, "_sr=1") == 0);
}

int main(int argc, char ** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    const std::set<std::string> sets = { "fold", "convt", "argmax", "attn", "tri", "mmid" };
    if (argc > 2 || (!only.empty() && !sets.count(only))) {
        printf("usage: test_metal_op_cases [fold | convt | argmax | attn | tri | mmid]\n");
        return 2;
    }
    std::vector<op_case> cases = case_table();
    std::set<std::string> red;
    {
        std::istringstream ss(std::getenv("EXPECT_RED") ? std::getenv("EXPECT_RED") : "");
        for (std::string id; ss >> id;) {
            red.insert(id);
        }
        for (const std::string & id : red) {
            auto it = std::find_if(cases.begin(), cases.end(), [&](const op_case & c) { return c.id == id; });
            if (it == cases.end() || !it->red) {
                printf("REFUSED EXPECT_RED %s: no case with a predicted red result\n", id.c_str());
                return 1;
            }
        }
    }

    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t gpu = dev != nullptr ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (gpu == nullptr || cpu == nullptr || !ggml_backend_is_metal(gpu)) {
        printf("the first GPU device is not a Metal device, or no CPU backend: this test is about the Metal backend\n");
        return 2;
    }
    printf("device: %s (Metal)\n", ggml_backend_dev_description(dev));

    int failed = 0;
    if (only.empty() || only == "convt") {
        // the bounds of the transposed convolution's loop, in 64-bit as in the kernel, at a stride past 32 bits of headroom
        const int64_t il = 2, kk = 1, s0 = 2147483646, j = 2147483646;
        const int64_t i_min = (s0 > 0 && j - kk + 1 > 0) ? (j - kk + s0) / s0 : 0;
        const int64_t i_max = s0 > 0 ? std::min(j / s0, il - 1) : il - 1;
        const bool ok = i_min == 1 && i_max == 1;
        failed += ok ? 0 : 1;
        printf("WITNESS convt bounds IL 2 K 1 s0 2147483646 j 2147483646: [%lld, %lld] · %s\n", (long long) i_min, (long long) i_max, ok ? "PASS" : "FAIL");
    }
    int run = 0, passed = 0, reds = 0, cpu_nf = 0;
    for (const op_case & c : cases) {
        if (!only.empty() && c.set != only) {
            continue;
        }
        run++;
        const std::vector<uint8_t> want = compute(cpu, c, false);
        const std::vector<uint8_t> got  = compute(gpu, c, true);
        const std::map<std::string, long> keys = dispatched();
        bool route = false;
        std::string rec;
        for (const auto & kv : keys) {
            rec += (rec.empty() ? "" : " + ") + kv.first + " x" + std::to_string(kv.second);
            route = route || (c.key_nsg_sr ? nsg_sr_key(kv.first, c.exported) : base_of(kv.first) == c.exported);
        }
        const bool computed = !want.empty() && !got.empty();
        std::string detail = computed ? "" : (want.empty() ? "CPU did not compute" : "Metal did not compute");
        bool finite = true;
        if (computed && c.set != "argmax") {
            finite = first_nonfinite(as_f32(got)) < 0;
        }
        const bool expect_red = red.count(c.id) > 0;
        bool ok = false;
        if (computed && route) {
            std::string d;
            ok = expect_red ? c.red(want, got, d) : (finite && c.judge(want, got, d));
            detail = d + (finite ? "" : " · Metal not finite");
        }
        const bool nocpu = ok && detail.find("CPU not finite") != std::string::npos;
        const char * verdict = ok ? (expect_red ? "RED (as predicted)" : nocpu ? "CPU NOT FINITE" : "PASS") : "FAIL";
        passed += ok && !expect_red && !nocpu ? 1 : 0;
        cpu_nf += nocpu ? 1 : 0;
        reds += ok && expect_red ? 1 : 0;
        failed += ok ? 0 : 1;
        char hx[17];
        snprintf(hx, sizeof(hx), "%016llx", (unsigned long long) fnv1a(got));
        printf("CASE %s · %s · %s%s · hash %s · %s · %s\n", c.id.c_str(), c.exported.c_str(), rec.empty() ? "nothing dispatched" : rec.c_str(),
               route ? "" : " (not the export of this case)", got.empty() ? "-" : hx, detail.c_str(), verdict);
    }
    printf("op cases: %d run, %d passed, %d red as predicted, %d CPU not finite, %d failed\n", run, passed, reds, cpu_nf, failed);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return failed == 0 && run > 0 ? 0 : 1;
}
