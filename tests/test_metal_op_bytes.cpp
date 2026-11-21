// Output bytes of Metal operators, one line per case, for comparing two builds of the backend (no CPU reference).
// Run: test_metal_op_bytes [flip <case id> | dup <case id>]
//   flip: inverts the lowest bit of the first output byte of that case before it is hashed (negative control).
//   dup: appends a copy of that case to the table, which must then be refused (control of the duplicate check).
// Each case builds one graph on the first GPU device (must be Metal); every input is filled from a fixed seed
// (16 * case index + tensor index). Line: <case id> TAB <expected entry> TAB <dispatch receipt> TAB <FNV-1a 64 of the
// bytes of every output tensor, in order>. A case ran its entry when the exported function of one receipt key is the
// expected entry (the plan suffix "_lay_..." of a tiled entry allowed) and no other key of the case's family is present;
// otherwise the case is listed ROUTE-OTHER. The exported function of a key is read from the backend's
// "compiling pipeline: base = ..., name = ..." debug log line. Summary: COVERED <entry> per entry a case ran,
// a case ran (its expected entry only), COMPILES <n> (first pipeline compiles of this process), RETRIES <n> (simd-width recompiles).
// Exit 0 when every case computed, 1 on a duplicate id, an empty receipt or a failed compute, 2 without a Metal device.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

static void fill(std::vector<float> & v, uint64_t seed) {
    uint64_t s = seed;
    for (float & x : v) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        x = (float) ((s >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
    }
}

// pipeline name -> exported function, and the compile counts, from the backend's debug log
static std::map<std::string, std::string> g_base;
static int g_compiles = 0, g_retries = 0;

static void log_cb(enum ggml_log_level level, const char * text, void * ud) {
    (void) ud;
    const char * p = strstr(text, "compiling pipeline:");
    if (p) {
        g_compiles++;
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
    if (strstr(text, "simd mismatch")) {
        g_retries++;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct receipt_key {
    std::string name;
    long count;
};

// The pipelines dispatched since the receipt was last reset, sorted by name; empty when there is no receipt.
static std::vector<receipt_key> dispatched() {
    std::vector<receipt_key> keys;
    const int need = ggml_backend_metal_test_dispatch_receipt(nullptr, 0, 0);
    std::string js(need > 0 ? (size_t) need : 1, '\0');
    if (need <= 0 || ggml_backend_metal_test_dispatch_receipt(&js[0], js.size(), 0) != need) {
        return keys;
    }
    std::map<std::string, long> by_name;
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
            by_name[obj.substr(a + 12, obj.find('"', a + 12) - (a + 12))] += count;
        }
    }
    for (const auto & kv : by_name) {
        keys.push_back({ kv.first, kv.second });
    }
    return keys;
}

static std::string base_of(const std::string & key) {
    auto it = g_base.find(key);
    return it == g_base.end() ? key : it->second;
}

static bool is_entry(const std::string & base, const std::string & expected) {
    return base == expected || base.rfind(expected + "_lay_", 0) == 0;
}

// One case's graph: inputs made through in(), outputs listed in out().
struct graph_builder {
    ggml_context * ctx;
    std::vector<ggml_tensor *> inputs;
    std::vector<std::vector<int32_t>> ints; // values of an I32 input, by input index; empty = random
    std::vector<ggml_tensor *> outputs;
    ggml_tensor * in(ggml_type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        ggml_set_input(t);
        inputs.push_back(t);
        ints.emplace_back();
        return t;
    }
    ggml_tensor * in_i32(const std::vector<int32_t> & v) {
        ggml_tensor * t = in(GGML_TYPE_I32, (int64_t) v.size());
        ints.back() = v;
        return t;
    }
    // a weight whose rows start 2 bytes off a 4-byte boundary in its buffer
    ggml_tensor * in_unaligned(ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * base = ggml_new_tensor_2d(ctx, type, ne0, ne1 + 1);
        ggml_set_input(base);
        ggml_tensor * t = ggml_view_2d(ctx, base, ne0, ne1, ggml_row_size(type, ne0), 2);
        inputs.push_back(t);
        ints.emplace_back();
        return t;
    }
    void out(ggml_tensor * t) {
        ggml_set_output(t);
        outputs.push_back(t);
    }
};

struct op_case {
    std::string id, expected, family;
    std::function<void(graph_builder &)> make;
};

static void set_input(ggml_tensor * t, const std::vector<int32_t> & ints, uint64_t seed) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> v(ints);
        if (v.empty()) {
            std::vector<float> f((size_t) n);
            fill(f, seed);
            for (float x : f) v.push_back((int32_t) ((x + 1.0f) * 32.0f));
        }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        return;
    }
    std::vector<float> f((size_t) n);
    fill(f, seed);
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(bytes.data(), f.data(), bytes.size());
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) bytes.data(), n);
    } else if (t->type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) bytes.data(), n);
    } else {
        const int64_t row = t->ne[0], rows = n / row;
        std::vector<float> imatrix((size_t) row, 1.0f);
        ggml_quantize_chunk(t->type, f.data(), bytes.data(), 0, rows, row, ggml_quantize_requires_imatrix(t->type) ? imatrix.data() : nullptr);
    }
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

static uint64_t fnv1a(uint64_t h, const uint8_t * p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static std::vector<op_case> case_table() {
    std::vector<op_case> cs;
    auto mm = [](ggml_type t0, ggml_type t1, int64_t m, int64_t n, int64_t k) {
        return [=](graph_builder & g) { g.out(ggml_mul_mat(g.ctx, g.in(t0, k, m), g.in(t1, k, n))); };
    };
    const std::string mv = "kernel_mul_mv_";
    const ggml_type F32 = GGML_TYPE_F32, F16 = GGML_TYPE_F16;
    const ggml_type kq[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                             GGML_TYPE_Q4_K, GGML_TYPE_Q3_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K };
    // A: one activation column, 512 rows x width 512
    for (auto p : std::vector<std::pair<ggml_type, ggml_type>>{ { F32, F32 }, { F16, F32 }, { F16, F16 }, { F32, F16 } }) {
        const std::string tn = std::string(ggml_type_name(p.first)) + "_" + ggml_type_name(p.second);
        cs.push_back({ "A.mv." + tn, mv + "gcn_" + tn, mv, mm(p.first, p.second, 512, 1, 512) });
    }
    for (ggml_type t : kq) {
        const std::string tn = std::string(ggml_type_name(t)) + "_f32";
        cs.push_back({ "A.mv." + tn, mv + "gcn_" + tn, mv, mm(t, F32, 512, 1, 512) });
    }
    // K, N: the same quantized cases on the matvec entries of cards that do not take the gcn entries
    for (ggml_type t : kq) {
        const std::string tn = std::string(ggml_type_name(t)) + "_f32";
        cs.push_back({ "K.mv." + tn, mv + tn, mv, mm(t, F32, 512, 1, 512) });
    }
    // J: 2 to 5 activation columns
    for (ggml_type t : { F32, F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_MXFP4,
                         GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
        for (int n = 2; n <= 5; ++n) {
            const std::string tn = std::string(ggml_type_name(t)) + "_f32_r1_" + std::to_string(n);
            cs.push_back({ "J.mv." + tn, mv + "ext_" + tn, mv, mm(t, F32, 512, n, 512) });
        }
    }
    // M: the tiled kernel, 24 weight types at two shapes; f16, f32, bf16 also at width 513; the q4_K w4 loader; the activation-copy entries
    const std::string tl = "kernel_mul_mat_tiled_";
    const ggml_type tiled_types[] = { F16, GGML_TYPE_MXFP4, F32, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS };
    struct shape { int64_t m, n, k; };
    for (ggml_type t : tiled_types) {
        for (shape s : { shape{ 128, 128, 512 }, shape{ 200, 130, 768 } }) {
            const std::string sz = std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k);
            cs.push_back({ "M.mm." + std::string(ggml_type_name(t)) + "." + sz, tl + ggml_type_name(t), tl, mm(t, F32, s.m, s.n, s.k) });
            if (t == GGML_TYPE_Q4_K) {
                cs.push_back({ "M.mm.q4_K_w4." + sz, tl + "q4_K_w4", tl, mm(t, F32, s.m, s.n, s.k) });
            }
        }
    }
    // a width that is not a multiple of 128 keeps these formats off the small-batch kernel
    for (ggml_type t : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL }) {
        for (shape s : { shape{ 128, 128, 544 }, shape{ 200, 130, 544 } }) {
            const std::string sz = std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k);
            cs.push_back({ "M.mm." + std::string(ggml_type_name(t)) + "." + sz, tl + ggml_type_name(t), tl, mm(t, F32, s.m, s.n, s.k) });
        }
    }
    // q4_K rows off a 4-byte boundary take the byte entry instead of the word entry
    cs.push_back({ "M.mm.q4_K.unaligned.128x128x512", tl + "q4_K", tl,
        [=](graph_builder & g) { g.out(ggml_mul_mat(g.ctx, g.in_unaligned(GGML_TYPE_Q4_K, 512, 128), g.in(F32, 512, 128))); } });
    for (ggml_type t : { F16, F32, GGML_TYPE_BF16 }) {
        cs.push_back({ "M.mm." + std::string(ggml_type_name(t)) + ".128x128x513", tl + ggml_type_name(t), tl, mm(t, F32, 128, 128, 513) });
    }
    for (ggml_type t : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
        cs.push_back({ "M.mm." + std::string(ggml_type_name(t)) + "_abf16.128x128x512", tl + ggml_type_name(t) + "_abf16", tl, mm(t, F32, 128, 128, 512) });
    }
    // B: row softmax
    for (int64_t w : { 37, 128 }) {
        for (int mask : { 0, 1, 2 }) {
            for (float scale : { 1.0f, 0.5f }) {
                for (float bias : { 0.0f, 8.0f }) {
                    if (mask == 0 && bias != 0.0f) {
                        continue; // ggml asks for a mask with a bias
                    }
                    const ggml_type mt = mask == 2 ? F16 : F32;
                    const std::string id = "B.soft_max.w" + std::to_string(w) + (mask == 0 ? ".nomask" : mask == 1 ? ".f32mask" : ".f16mask") +
                                           ".s" + (scale == 1.0f ? "1" : "0.5") + ".b" + (bias == 0.0f ? "0" : "8");
                    cs.push_back({ id, "kernel_soft_max_" + std::string(ggml_type_name(mt)) + (w % 4 == 0 ? "_4" : ""), "kernel_soft_max_",
                        [=](graph_builder & g) {
                            ggml_tensor * a = g.in(F32, w, 4, 2);
                            ggml_tensor * m = mask ? g.in(mt, w, 4) : nullptr;
                            g.out(ggml_soft_max_ext(g.ctx, a, m, scale, bias));
                        } });
                }
            }
        }
    }
    // C: pooling with padding 1 (edge windows)
    for (int avg : { 0, 1 }) {
        const ggml_op_pool op = avg ? GGML_OP_POOL_AVG : GGML_OP_POOL_MAX;
        const std::string on = avg ? "avg" : "max";
        cs.push_back({ "C.pool_2d." + on, "kernel_pool_2d_" + on + "_f32", "kernel_pool_",
            [=](graph_builder & g) { g.out(ggml_pool_2d(g.ctx, g.in(F32, 9, 7, 3), op, 3, 3, 2, 2, 1, 1)); } });
        cs.push_back({ "C.pool_1d." + on, "kernel_pool_1d_" + on + "_f32", "kernel_pool_",
            [=](graph_builder & g) { g.out(ggml_pool_1d(g.ctx, g.in(F32, 11, 4), op, 3, 2, 1)); } });
    }
    // D: rotary position transform; 64-wide heads, 4 heads, 5 tokens
    for (int mode : { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_MROPE, GGML_ROPE_TYPE_IMROPE, GGML_ROPE_TYPE_VISION }) {
        for (ggml_type t : { F32, F16 }) {
            for (int yarn : { 0, 1 }) {
                const std::string mn = mode == GGML_ROPE_TYPE_NORMAL ? "norm" : mode == GGML_ROPE_TYPE_NEOX ? "neox" :
                                       mode == GGML_ROPE_TYPE_MROPE ? "mrope" : mode == GGML_ROPE_TYPE_IMROPE ? "imrope" : "vision";
                const std::string kn = mode == GGML_ROPE_TYPE_NORMAL ? "norm" : mode == GGML_ROPE_TYPE_NEOX ? "neox" :
                                       mode == GGML_ROPE_TYPE_VISION ? "vision" : "multi";
                const float fs = yarn ? 0.5f : 1.0f, ef = yarn ? 1.0f : 0.0f;
                cs.push_back({ "D.rope." + mn + "." + ggml_type_name(t) + (yarn ? ".yarn" : ".plain"), "kernel_rope_" + kn + "_" + ggml_type_name(t), "kernel_rope_",
                    [=](graph_builder & g) {
                        const int n_tok = 5;
                        ggml_tensor * a = g.in(t, 64, 4, n_tok);
                        if (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX) {
                            std::vector<int32_t> pos;
                            for (int i = 0; i < n_tok; ++i) pos.push_back(3 * i + 1);
                            g.out(ggml_rope_ext(g.ctx, a, g.in_i32(pos), nullptr, 48, mode, 4096, 10000.0f, fs, ef, 1.0f, 32.0f, 1.0f));
                        } else {
                            std::vector<int32_t> pos;
                            for (int i = 0; i < 4 * n_tok; ++i) pos.push_back(2 * i + 1);
                            int sec_m[4] = { 8, 8, 8, 0 };
                            int sec_v[4] = { 16, 16, 0, 0 };
                            const bool vis = mode == GGML_ROPE_TYPE_VISION;
                            g.out(ggml_rope_multi(g.ctx, a, g.in_i32(pos), nullptr, vis ? 32 : 48, vis ? sec_v : sec_m, mode, 4096, 10000.0f, fs, ef, 1.0f, 32.0f, 1.0f));
                        }
                    } });
            }
        }
    }
    // E: the gated-linear-unit operators, width 130
    for (int e = 0; e < 7; ++e) {
        const char * names[] = { "reglu", "geglu", "swiglu", "swiglu_oai", "geglu_erf", "geglu_quick", "swiglu_clamp" };
        cs.push_back({ std::string("E.glu.") + names[e], std::string("kernel_") + names[e] + "_f32", std::string("kernel_") + names[e] + "_f32",
            [=](graph_builder & g) {
                ggml_tensor * a = g.in(F32, 130, 3);
                ggml_tensor * b = g.in(F32, 130, 3);
                ggml_tensor * r;
                switch (e) {
                    case 0:  r = ggml_glu_split(g.ctx, a, b, GGML_GLU_OP_REGLU); break;
                    case 1:  r = ggml_glu_split(g.ctx, a, b, GGML_GLU_OP_GEGLU); break;
                    case 2:  r = ggml_glu_split(g.ctx, a, b, GGML_GLU_OP_SWIGLU); break;
                    case 3:  r = ggml_swiglu_oai(g.ctx, a, b, 1.702f, 0.5f); break;
                    case 4:  r = ggml_glu_split(g.ctx, a, b, GGML_GLU_OP_GEGLU_ERF); break;
                    case 5:  r = ggml_glu_split(g.ctx, a, b, GGML_GLU_OP_GEGLU_QUICK); break;
                    default: r = ggml_swiglu_clamp(g.ctx, a, b, 0.5f); break;
                }
                g.out(r);
            } });
    }
    // F: centered and uncentered norms at widths that are not a power of two, alone and in the fused chains; the small fused norms
    for (int rms : { 0, 1 }) {
        for (int64_t w : { 511, 512, 640 }) {
            for (int chain : { 0, 1, 2 }) {
                const std::string on = rms ? "rms_norm" : "norm";
                const std::string cn = chain == 0 ? "" : chain == 1 ? "_mul" : "_mul_add";
                cs.push_back({ "F." + on + cn + ".w" + std::to_string(w), "kernel_" + on + cn + "_f32" + (w % 4 == 0 ? "_4" : ""), "kernel_" + on + "_",
                    [=](graph_builder & g) {
                        ggml_tensor * x = g.in(F32, w, 3);
                        ggml_tensor * r = rms ? ggml_rms_norm(g.ctx, x, 1e-6f) : ggml_norm(g.ctx, x, 1e-5f);
                        if (chain >= 1) r = ggml_mul(g.ctx, r, g.in(F32, w));
                        if (chain >= 2) r = ggml_add(g.ctx, r, g.in(F32, w));
                        g.out(r);
                    } });
            }
        }
    }
    for (int64_t w : { 127, 128 }) {
        const std::string sfx = w % 4 == 0 ? "_4" : "";
        cs.push_back({ "F.small_gated_rms_norm.w" + std::to_string(w), "kernel_small_gated_rms_norm_f32" + sfx, "kernel_small_",
            [=](graph_builder & g) {
                ggml_tensor * x = g.in(F32, w, 3);
                ggml_tensor * gate = g.in(F32, w, 3);
                ggml_tensor * r = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, 1e-6f), g.in(F32, w));
                g.out(ggml_mul(g.ctx, r, ggml_silu(g.ctx, gate)));
            } });
        cs.push_back({ "F.small_split_l2.w" + std::to_string(w), "kernel_small_split_l2_f32" + sfx, "kernel_small_",
            [=](graph_builder & g) {
                ggml_tensor * q = g.in(F32, w, 3);
                ggml_tensor * k = g.in(F32, w, 2);
                ggml_tensor * sq = ggml_scale(g.ctx, ggml_rms_norm(g.ctx, q, 1e-6f), 0.25f);
                ggml_tensor * sk = ggml_scale(g.ctx, ggml_rms_norm(g.ctx, k, 1e-6f), 0.5f);
                g.out(sq);
                g.out(sk);
            } });
    }
    cs.push_back({ "F.small_add_norm.w256", "kernel_small_add_norm_f32_4", "kernel_small_",
        [=](graph_builder & g) {
            ggml_tensor * sum = ggml_add(g.ctx, g.in(F32, 256), g.in(F32, 256));
            ggml_tensor * r = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, sum, 1e-6f), g.in(F32, 256));
            g.out(sum);
            g.out(r);
        } });
    // G: short convolution of the state-space layer, kernel widths 4 and 3, one token and a batch; the state written too
    for (int64_t dc : { 4, 3 }) {
        for (int64_t nt : { 1, 8 }) {
            for (int ext : { 0, 1 }) {
                const std::string ex = std::string("kernel_ssm_conv_f32_f32") + (nt > 1 ? "_batched" : "") + (dc % 4 == 0 ? "_4" : "");
                cs.push_back({ "G.ssm_conv.d" + std::to_string(dc) + ".t" + std::to_string(nt) + (ext ? ".state" : ""), ex, "kernel_ssm_conv_",
                    [=](graph_builder & g) {
                        ggml_tensor * sx = g.in(F32, dc - 1 + nt, 64, 1);
                        ggml_tensor * c = g.in(F32, dc, 64);
                        if (ext) {
                            ggml_tensor * st = g.in(F32, dc - 1, 64, 1);
                            g.out(ggml_ssm_conv_ext(g.ctx, sx, c, st));
                            g.out(st);
                        } else {
                            g.out(ggml_ssm_conv(g.ctx, sx, c));
                        }
                    } });
            }
        }
    }
    // H: the two dsv4 pre operators
    cs.push_back({ "H.dsv4_hc_pre", "kernel_dsv4_hc_pre_f32", "kernel_dsv4_hc_pre",
        [=](graph_builder & g) { g.out(ggml_dsv4_hc_pre(g.ctx, g.in(F32, 96, 4, 3), g.in(F32, 4, 3))); } });
    cs.push_back({ "H.dsv4_hc_pre_gated", "kernel_dsv4_hc_pre_gated_f32", "kernel_dsv4_hc_pre",
        [=](graph_builder & g) { g.out(ggml_dsv4_hc_pre_gated(g.ctx, g.in(F32, 96, 4, 3), g.in(F32, 96, 4, 3), 0.5f)); } });
    // O: attention on the scalar route, 32 queries (a large batch), 64 cache rows, 2 heads
    struct fa_pair { ggml_type kv; int dk, dv; std::string expected; };
    std::vector<fa_pair> fa;
    for (int d : { 32, 40, 48, 64, 72, 80, 96, 112, 128, 192, 256, 576 }) {
        fa.push_back({ F16, d, d, "kernel_flash_attn_ext_scalar_dk" + std::to_string(d) + "_dv" + std::to_string(d) });
    }
    for (ggml_type t : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
        for (int d : { 64, 128, 256 }) {
            fa.push_back({ t, d, d, "kernel_flash_attn_ext_scalar_" + std::string(ggml_type_name(t)) + "_dk" + std::to_string(d) + "_dv" + std::to_string(d) });
        }
    }
    for (auto p : std::vector<std::pair<int, int>>{ { 32, 32 }, { 40, 40 }, { 48, 48 }, { 64, 64 }, { 72, 72 }, { 80, 80 }, { 96, 96 }, { 112, 112 },
             { 128, 128 }, { 192, 192 }, { 256, 256 }, { 512, 512 }, { 576, 576 }, { 96, 64 }, { 128, 96 }, { 192, 128 }, { 320, 256 }, { 576, 512 } }) {
        fa.push_back({ F32, p.first, p.second, "kernel_flash_attn_ext_scalar_kv_dk" + std::to_string(p.first) + "_dv" + std::to_string(p.second) });
    }
    fa.push_back({ GGML_TYPE_Q4_0, 32, 32, "kernel_flash_attn_ext_scalar_kv_dk32_dv32" });
    for (const fa_pair & p : fa) {
        for (int masked : { 0, 1 }) {
            const std::string id = "O.fa." + std::string(ggml_type_name(p.kv)) + ".dk" + std::to_string(p.dk) + ".dv" + std::to_string(p.dv) + (masked ? ".mask" : ".nomask");
            cs.push_back({ id, p.expected, "kernel_flash_attn_ext_scalar",
                [=](graph_builder & g) {
                    const int nq = 32, nkv = 64, nh = 2;
                    ggml_tensor * q = g.in(F32, p.dk, nq, nh);
                    ggml_tensor * k = g.in(p.kv, p.dk, nkv, nh);
                    ggml_tensor * v = g.in(p.kv, p.dv, nkv, nh);
                    ggml_tensor * m = masked ? g.in(F16, nkv, nq) : nullptr;
                    g.out(ggml_flash_attn_ext(g.ctx, q, k, v, m, 1.0f / sqrtf((float) p.dk), 0.0f, 0.0f));
                } });
        }
    }
    return cs;
}

int main(int argc, char ** argv) {
    std::string flip, dup;
    if (argc == 3 && strcmp(argv[1], "flip") == 0) {
        flip = argv[2];
    } else if (argc == 3 && strcmp(argv[1], "dup") == 0) {
        dup = argv[2];
    } else if (argc != 1) {
        printf("usage: test_metal_op_bytes [flip <case id> | dup <case id>]\n");
        return 2;
    }
    std::vector<op_case> cases = case_table();
    if (!dup.empty()) {
        for (size_t i = 0; i < cases.size(); ++i) {
            if (cases[i].id == dup) {
                cases.push_back(cases[i]);
                break;
            }
        }
    }
    {
        std::set<std::string> ids;
        for (const op_case & c : cases) {
            if (!ids.insert(c.id).second) {
                printf("REFUSED duplicate case id %s\n", c.id.c_str());
                return 1;
            }
        }
        if ((!flip.empty() && !ids.count(flip)) || (!dup.empty() && !ids.count(dup))) {
            printf("REFUSED no case %s\n", (flip.empty() ? dup : flip).c_str());
            return 1;
        }
    }

    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t gpu = dev != nullptr ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (gpu == nullptr || !ggml_backend_is_metal(gpu)) {
        printf("the first GPU device is not a Metal device: this test is about the Metal backend\n");
        return 2;
    }
    printf("device\t%s\n", ggml_backend_dev_description(dev));

    int bad = 0;
    std::set<std::string> covered;
    std::vector<std::string> other;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const op_case & c = cases[ci];
        ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        graph_builder g = { ggml_init(ip), {}, {}, {} };
        c.make(g);
        ggml_cgraph * gf = ggml_new_graph(g.ctx);
        for (ggml_tensor * t : g.outputs) {
            ggml_build_forward_expand(gf, t);
        }
        std::string unsupported;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (!ggml_backend_supports_op(gpu, ggml_graph_node(gf, i))) {
                unsupported = ggml_op_desc(ggml_graph_node(gf, i));
                break;
            }
        }
        std::string receipt, hash;
        bool ran = false;
        if (!unsupported.empty()) {
            receipt = "unsupported " + unsupported;
            hash = "-";
        } else {
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
            bool ok = ggml_gallocr_alloc_graph(ga, gf);
            for (size_t i = 0; ok && i < g.inputs.size(); ++i) {
                set_input(g.inputs[i], g.ints[i], 16 * (uint64_t) ci + i);
            }
            ggml_backend_metal_test_dispatch_receipt(nullptr, 0, 1); // a new recording window for this case
            ok = ok && ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
            const std::vector<receipt_key> keys = dispatched();
            uint64_t h = 14695981039346656037ull;
            for (size_t i = 0; ok && i < g.outputs.size(); ++i) {
                std::vector<uint8_t> bytes(ggml_nbytes(g.outputs[i]));
                ggml_backend_tensor_get(g.outputs[i], bytes.data(), 0, bytes.size());
                if (i == 0 && c.id == flip && !bytes.empty()) {
                    bytes[0] ^= 1;
                }
                h = fnv1a(h, bytes.data(), bytes.size());
            }
            ggml_gallocr_free(ga);
            if (!ok) {
                printf("FAILED %s: the graph did not compute\n", c.id.c_str());
                bad++;
            }
            if (keys.empty()) {
                printf("FAILED %s: empty dispatch receipt\n", c.id.c_str());
                bad++;
            }
            int hits = 0, others = 0;
            for (const receipt_key & k : keys) {
                const std::string base = base_of(k.name);
                receipt += (receipt.empty() ? "" : " + ") + k.name + " x" + std::to_string(k.count);
                if (is_entry(base, c.expected)) {
                    hits++;
                } else if (base.rfind(c.family, 0) == 0) {
                    others++;
                }
            }
            ran = hits > 0 && others == 0;
            if (ran) {
                covered.insert(c.expected);
            }
            char hx[17];
            snprintf(hx, sizeof(hx), "%016llx", (unsigned long long) h);
            hash = hx;
        }
        if (!ran) {
            other.push_back(c.id);
        }
        printf("%s\t%s\t%s\t%s\n", c.id.c_str(), c.expected.c_str(), receipt.c_str(), hash.c_str());
        ggml_free(g.ctx);
    }
    for (const std::string & id : other) {
        printf("ROUTE-OTHER %s\n", id.c_str());
    }
    for (const std::string & e : covered) {
        printf("COVERED %s\n", e.c_str());
    }
    printf("COMPILES %d\n", g_compiles);
    printf("RETRIES %d\n", g_retries);
    printf("op bytes: %zu cases, %zu route-other, %d failed\n", cases.size(), other.size(), bad);
    ggml_backend_free(gpu);
    return bad == 0 ? 0 : 1;
}
