// llama-metal-tiled-sweep: measures every compiled work plan of the tiled matmul kernel on this machine's Metal card
// and prints the rows to add to section [tiles] of ggml/src/ggml-metal/data/matmul.tsv. No model is needed.
//
// Per weight type: one random weight (fixed seed, ROWS x COLS, quantized once by ggml) and random activations. A run
// is a child process of this program that multiplies at each column count (--tokens) and prints, per count, the median
// time, a digest of the output bytes (FNV-1a, 64 bits) and the pipeline the dispatch receipt shows. A round is the
// card's own run (nothing forced: the plan its rows give it) and then one run per compiled plan, the default plan
// included, forced with GGML_METAL_TEST_TILED_PLAN=<weight>:<plan id>.
//
// Each column count is judged on its own (tiled-rows.h, tiled_judge_count): the bar is the plan the card dispatched
// there with nothing forced; a plan with an equal digest that is at least GAIN faster than that bar gets a row for
// that count (plan_cols = the count). Otherwise the count gets a KEEP line (the card's plan stays, with both times) or
// a NOROW line with its reason (not on the tiled kernel, a failure, digests differ). A failure ends one plan at one
// count. A row is never for a range and counts are never merged; a count that was not measured keeps the card's plan.
// The rows are candidates: one goes into the table once prompt speed on a model shows its gain.
//
// A run whose child failed as a whole (a forced plan the card cannot run ends the process) is run again one column
// count per child, so the failure costs that count only.
//
// Limits: one synthetic weight (ROWS x COLS) per type, one multiply at a time. The time is the host's stopwatch around
// the multiply and the wait for it: it holds the encoding, the synchronization and the dispatch receipt's recording in
// every run, and the hook's counters in the forced runs. Near the GAIN line that matters; prompt speed on a model decides.
//
// Mode --matvec does the same for the matvec work plan (rows per simdgroup, columns, loop form, prepared headers) and
// prints rows for section [matvec_shapes]: see the comment above mv_main below.
//
// Exit code: 0 = every count has a row or a KEEP and every plan was judged at every count, 3 = the sweep finished and
// some count has a NOROW or some plan was not judged at a count (rows are for the rest; every count a NOROW is still
// 3), 2 = bad arguments, 1 = no run sent a time and a digest at any count (no Metal device, every allocation failed),
// or one compiled plan only.
//
// For the check of this tool only: LLAMA_METAL_TILED_SWEEP_TEST_ABORT=<plan id>:<columns> ends the child that is forced
// to that plan when it reaches that column count, the way a plan the card cannot run does.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "gguf.h"

#include "mv-row.h"
#include "tiled-rows.h"

#include <mach-o/dyld.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

static const int64_t ROWS = 4096;   // weight rows
static const int64_t COLS = 2560;   // weight columns
static const int     REPS = 9;      // timed multiplies per column count in a child, after one untimed
static const double  GAIN = 0.03;   // a row needs this much less time than the plan the card runs at that column count

static void fill(std::vector<float> & v, uint64_t seed) {
    uint64_t s = seed;
    for (float & x : v) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        x = (float) ((s >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
    }
}

static uint64_t fnv(const uint8_t * p, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static ggml_type type_by_name(const std::string & name) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * n = ggml_type_name((ggml_type) i);
        if (n != nullptr && name == n && ggml_blck_size((ggml_type) i) > 0 && ggml_type_size((ggml_type) i) > 0) {
            return (ggml_type) i;
        }
    }
    return GGML_TYPE_COUNT;
}

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        const size_t b = s.find(sep, a);
        const std::string cell = s.substr(a, b == std::string::npos ? std::string::npos : b - a);
        if (!cell.empty()) out.push_back(cell);
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}

// The value of key= in a line of space-separated key=value cells ("" when absent).
static std::string cell(const std::string & line, const char * key) {
    const std::string k = std::string(" ") + key + "=";
    const size_t p = (" " + line).find(k);
    if (p == std::string::npos) return "";
    const size_t a = p + k.size() - 1;
    const size_t e = line.find(' ', a);
    return line.substr(a, e == std::string::npos ? std::string::npos : e - a);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

// ---- child: one plan (from the environment, or the card's own when it is unset), one weight file, every column count ----

// The pipelines of the dispatch receipt's window as lines "SWEEP_ENTRY n=<n> pipeline=<name> count=<encoded dispatches>".
static bool print_entries(int64_t n) {
    const int need = ggml_backend_metal_test_dispatch_receipt(nullptr, 0, 0);
    std::string js(need > 0 ? (size_t) need : 1, '\0');
    if (need <= 0 || ggml_backend_metal_test_dispatch_receipt(&js[0], js.size(), 0) != need) return false;
    // one JSON object per pipeline; its keys in any order
    for (size_t p = js.find('{'); p != std::string::npos; p = js.find('{', p + 1)) {
        const size_t end = js.find('}', p);
        if (end == std::string::npos) return false;
        const std::string obj = js.substr(p, end - p + 1);
        const size_t np = obj.find("\"pipeline\":\""), cp = obj.find("\"count\":");
        if (np == std::string::npos || cp == std::string::npos) return false;
        const size_t s = np + 12, e = obj.find('"', s);
        if (e == std::string::npos) return false;
        printf("SWEEP_ENTRY n=%lld pipeline=%s count=%lld\n", (long long) n, obj.substr(s, e - s).c_str(), atoll(obj.c_str() + cp + 8));
    }
    return true;
}

static int child_main(const char * type_name, const char * weight_path, const char * tokens_csv) {
    const ggml_type type = type_by_name(type_name);
    if (type == GGML_TYPE_COUNT) { printf("SWEEP_FAIL bad weight type\n"); return 1; }
    const size_t wb = (size_t) ROWS * ggml_row_size(type, COLS);
    std::vector<uint8_t> wq(wb);
    FILE * f = fopen(weight_path, "rb");
    if (f == nullptr || fread(wq.data(), 1, wb, f) != wb) { printf("SWEEP_FAIL cannot read the weight file\n"); return 1; }
    fclose(f);

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t backend = dev != nullptr ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (backend == nullptr || !ggml_backend_is_metal(backend)) { printf("SWEEP_FAIL no Metal device\n"); return 1; }
    // the device's name is "MTL<index>:<name as Metal reports it>"; the index tells two cards of one name apart
    printf("SWEEP_DEVICE %s\n", ggml_backend_dev_name(dev));
    uint32_t wave = 0, group_bytes = 0;
    if (ggml_backend_metal_tiled_card_shape(dev, &wave, &group_bytes) != 0) {
        printf("SWEEP_SHAPE wave=%u group_bytes=%u\n", wave, group_bytes);
    }

    const char * test_abort = getenv("LLAMA_METAL_TILED_SWEEP_TEST_ABORT");
    const char * hook       = getenv("GGML_METAL_TEST_TILED_PLAN");
    // a failure at one count is said for that count and the next count still runs
    for (const std::string & tok : split(tokens_csv, ',')) {
        const int64_t n = atoll(tok.c_str());
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a   = ggml_new_tensor_2d(ctx, type, COLS, ROWS);
        ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, COLS, n);
        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_cgraph * gf  = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        const char * bad = nullptr;
        if (test_abort != nullptr && hook != nullptr && std::string(hook) + ":" + std::to_string(n) == std::string(type_name) + ":" + test_abort) {
            printf("SWEEP_FAIL the check's abort at %lld columns\n", (long long) n);
            fflush(stdout);
            abort();
        }
        if (!ggml_gallocr_alloc_graph(ga, gf)) bad = "graph allocation";
        std::vector<double> ms;
        if (bad == nullptr) {
            std::vector<float> xf((size_t) (n * COLS));
            fill(xf, 2);
            ggml_backend_tensor_set(a, wq.data(), 0, wb);
            ggml_backend_tensor_set(b, xf.data(), 0, xf.size() * sizeof(float));
            // the receipt's window is this count's multiplies: which pipeline ran them
            if (ggml_backend_metal_test_dispatch_receipt(nullptr, 0, 1) < 0) bad = "dispatch receipt reset";
        }
        for (int r = 0; bad == nullptr && r <= REPS; ++r) { // r = 0 compiles the kernel and is not timed
            const int64_t t0 = ggml_time_us();
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { bad = "compute"; break; }
            ggml_backend_synchronize(backend);
            if (r > 0) ms.push_back((double) (ggml_time_us() - t0) / 1000.0);
        }
        if (bad == nullptr && !print_entries(n)) bad = "dispatch receipt read";
        if (bad != nullptr) {
            printf("SWEEP_FAIL_AT n=%lld %s\n", (long long) n, bad);
        } else {
            const size_t nb = ggml_nbytes(out);
            std::vector<uint8_t> res(nb);
            ggml_backend_tensor_get(out, res.data(), 0, nb);
            printf("SWEEP_RESULT n=%lld ms=%.4f bytes=%zu fnv=%016llx\n", (long long) n, median(ms), nb, (unsigned long long) fnv(res.data(), nb));
        }
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }
    fflush(stdout);
    ggml_backend_free(backend);
    return 0; // the library prints the hook's receipt lines at exit
}

// ---- parent ----

// The name Metal reports, from the device's "MTL<index>:<name>".
static std::string card_name(const std::string & device) {
    const size_t colon = device.find(':');
    return colon == std::string::npos ? device : device.substr(colon + 1);
}

struct run {
    bool ok = false;
    std::string why;                      // when !ok
    std::string device;                   // "MTL<index>:<name>"
    std::string wave, group_bytes;        // the card's shape, "" when the child did not report it
    std::map<int64_t, double> ms;         // per column count
    std::map<int64_t, std::string> hash;
    std::map<int64_t, std::set<std::string>> entries; // the pipelines the dispatch receipt holds for the multiply at this count
    std::map<int64_t, std::string> fail;  // a failure at this count only
};

static std::string shell_quote(const std::string & s) {
    std::string q = "'";
    for (char c : s) { if (c == '\'') q += "'\\''"; else q += c; }
    return q + "'";
}

// plan "" = nothing forced: the card's own run.
static run run_child(const std::string & self, const std::string & type, const std::string & weight_path, const std::string & plan,
                     const std::string & tokens_csv) {
    run r;
    if (plan.empty()) unsetenv("GGML_METAL_TEST_TILED_PLAN");
    else              setenv("GGML_METAL_TEST_TILED_PLAN", (type + ":" + plan).c_str(), 1);
    const std::string cmd = shell_quote(self) + " --child " + shell_quote(type) + " " + shell_quote(weight_path) + " " + shell_quote(tokens_csv) + " 2>&1";
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) { unsetenv("GGML_METAL_TEST_TILED_PLAN"); r.why = "could not start the child process"; return r; }
    char buf[4096];
    std::string receipt, last;
    while (fgets(buf, sizeof(buf), p) != nullptr) {
        std::string line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line.rfind("SWEEP_DEVICE ", 0) == 0) {
            r.device = line.substr(13);
        } else if (line.rfind("SWEEP_SHAPE ", 0) == 0) {
            r.wave        = cell(line, "wave");
            r.group_bytes = cell(line, "group_bytes");
        } else if (line.rfind("SWEEP_RESULT ", 0) == 0) {
            const int64_t n = atoll(cell(line, "n").c_str());
            r.ms[n]   = atof(cell(line, "ms").c_str());
            r.hash[n] = cell(line, "fnv") + "/" + cell(line, "bytes");
        } else if (line.rfind("SWEEP_ENTRY ", 0) == 0) {
            if (!cell(line, "pipeline").empty()) r.entries[atoll(cell(line, "n").c_str())].insert(cell(line, "pipeline"));
        } else if (line.rfind("SWEEP_FAIL_AT ", 0) == 0) {
            r.fail[atoll(cell(line, "n").c_str())] = line.substr(line.find(' ', 14) == std::string::npos ? 14 : line.find(' ', 14) + 1);
        } else if (line.rfind("TILED_TEST_PLAN ", 0) == 0) {
            receipt = line;
        } else if (line.find("GGML_METAL_TEST_TILED_PLAN") != std::string::npos || line.rfind("SWEEP_FAIL ", 0) == 0) {
            if (r.why.empty()) r.why = line;
        }
        if (!line.empty()) last = line;
    }
    const int status = pclose(p);
    unsetenv("GGML_METAL_TEST_TILED_PLAN");
    if (status != 0) {
        if (r.why.empty()) r.why = "the child ended with status " + std::to_string(status) + (last.empty() ? "" : ": " + last);
        return r;
    }
    if (r.device.empty()) { r.why = "the child reported no device"; return r; }
    if (!plan.empty() && (receipt.empty() || cell(receipt, "plan") != plan || atol(cell(receipt, "on_plan").c_str()) < 1)) {
        r.why = "the receipt does not show a multiply on plan " + plan + (receipt.empty() ? " (no TILED_TEST_PLAN line)" : ": " + receipt);
        return r;
    }
    r.ok = true;
    return r;
}


// ---- mode --matvec ----
//
// Measures matvec work plans on this machine's Metal card and prints the rows to add to section [matvec_shapes] of
// ggml/src/ggml-metal/data/matmul.tsv. A shape is a kind (single | group), a weight type and the weight's k x m; the
// shapes come from --shapes, from the 2-D weights of --model (single only: which weights share a launch is the graph's),
// or, with neither, from the distinct shapes already in [matvec_shapes]. Only the weight types whose [formats] row
// says shape_rows yes are taken. Per shape the candidates are every nr0 of 1, 2, 4, 8, with prepared headers off and
// (q4_K) on; with --cols also 2, 3 and 4 activation columns in each loop form (--columns names the counts). A candidate the plan check does not
// admit is reported with the condition and is no failure.
//
// Each candidate runs in a child process with GGML_METAL_TEST_MV_PLAN (the plan is forced and checked there), beside
// a run of the default: the same multiply with the card's rows switched off (one column: GGML_METAL_KQ_HW_ROWS=0;
// 2 to 4 columns: GGML_METAL_KQ_COLS=0, the card's launch without a column row). Every run prints its dispatch
// receipts (GGML_METAL_TEST_MV_RECEIPT=1); a forced run whose receipts are not exactly the forced kind, columns, rows,
// form and prep is refused. The time compared is GPU time: the probe trace of the run (a build with
// -DGGML_METAL_PROBES=ON, ZR_ON=1): a child works the multiply 24 times back to back per graph run, as decoding does,
// and the GPU intervals of a run are summed and divided by 24, median over the timed runs; host
// time is printed and not used. The intervals timed must be the forced pipeline's, one per multiply; the default's are
// whatever kernels the card's route launches (they are printed). A row is proposed, per shape, column count and prep,
// for the fastest candidate whose output bytes equal the default's and whose GPU time is at least GAIN less; where
// the card launches a row's plan for them already (read from the receipt of one more run, rows on and nothing
// forced), only when it is also GAIN faster than that plan (KEEP line otherwise). A default run must be one kernel,
// launched once per multiply (or per column, or per member of a pair), with as many receipts as intervals.
// A proposed row is a candidate: it goes into the table once a decode run shows the gain in token time.
// --force <hook value> runs that one plan and prints its receipt: exit 0 when it ran as forced, 4 when the check
// refused it (the condition is printed), 1 otherwise.

static const int MV_REPS  = 9;  // timed graph runs per child, after one untimed
static const int MV_BURST = 24; // times the multiply stands in the graph of one run

struct mv_shape {
    int kind = 0; // 0 single, 1 group
    std::string weight;
    int64_t k = 0, m = 0;
    bool operator<(const mv_shape & o) const {
        if (kind != o.kind) return kind < o.kind;
        if (weight != o.weight) return weight < o.weight;
        if (k != o.k) return k < o.k;
        return m < o.m;
    }
};

struct mv_cand {
    int cols = 1, prep = 0, nr0 = 2;
    std::string form = "-";
};

static int mv_form_number(const std::string & form) {
    if (form == "-" || form == "a") return 0;
    if (form == "w")   return 1;
    if (form == "ap")  return 2;
    if (form == "wp")  return 3;
    if (form == "ha")  return 4;
    if (form == "hap") return 6;
    return -1;
}

static std::string mv_hook(const mv_shape & s, const mv_cand & c) {
    return std::string(s.kind == 1 ? "group" : "single") + ":" + s.weight + ":" + std::to_string(s.k) + ":" + std::to_string(s.m) + ":" + std::to_string(c.cols) + ":" +
           (c.prep ? "on" : "off") + ":" + std::to_string(c.nr0) + ":" + c.form;
}

// The weights of one shape as the children read them: `count` matrices of m rows, one after the other. Up to 256
// rows of each are quantized from random values; the rows after them repeat those (a large matrix costs no more).
static bool mv_write_weights(const std::string & path, ggml_type type, int64_t k, int64_t m, int count) {
    const size_t row = ggml_row_size(type, k);
    const int64_t distinct = std::min<int64_t>(m, 256);
    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    bool ok = true;
    for (int w = 0; w < count && ok; ++w) {
        std::vector<float> wf((size_t) (distinct * k));
        fill(wf, 1 + 2 * (uint64_t) w);
        std::vector<uint8_t> wq((size_t) distinct * row);
        std::vector<float> importance;
        if (ggml_quantize_requires_imatrix(type)) importance.assign((size_t) k, 1.0f);
        ok = ggml_quantize_chunk(type, wf.data(), wq.data(), 0, distinct, k, importance.empty() ? nullptr : importance.data()) == wq.size();
        for (int64_t r = 0; r < m && ok; r += distinct) {
            const size_t n = (size_t) std::min<int64_t>(distinct, m - r) * row;
            ok = fwrite(wq.data(), 1, n, f) == n;
        }
    }
    return fclose(f) == 0 && ok;
}

// child: the multiply (single: one weight; group: two weights of one activation) MV_BURST times in one graph, run MV_REPS + 1 times
static int mv_child_main(const char * kind_word, const char * type_name, const char * k_text, const char * m_text, const char * cols_text, const char * weight_path) {
    const bool group = strcmp(kind_word, "group") == 0;
    const ggml_type type = type_by_name(type_name);
    const int64_t k = atoll(k_text), m = atoll(m_text), cols = atoll(cols_text);
    if (type == GGML_TYPE_COUNT || k < 1 || m < 1 || cols < 1 || cols > 4 || k % ggml_blck_size(type) != 0) { printf("SWEEP_FAIL bad shape\n"); return 1; }
    const size_t wb = (size_t) m * ggml_row_size(type, k);
    const int n_w = group ? 2 : 1;
    std::vector<uint8_t> wq(wb * n_w);
    FILE * f = fopen(weight_path, "rb");
    if (f == nullptr || fread(wq.data(), 1, wq.size(), f) != wq.size()) { printf("SWEEP_FAIL cannot read the weight file\n"); return 1; }
    fclose(f);

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t backend = dev != nullptr ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (backend == nullptr || !ggml_backend_is_metal(backend)) { printf("SWEEP_FAIL no Metal device\n"); return 1; }
    printf("SWEEP_DEVICE %s\n", ggml_backend_dev_name(dev));
    uint32_t wave = 0, group_bytes = 0;
    if (ggml_backend_metal_tiled_card_shape(dev, &wave, &group_bytes) != 0) {
        printf("SWEEP_SHAPE wave=%u group_bytes=%u\n", wave, group_bytes);
    }

    // the weights in a buffer of their own, one after the other, as a model file holds them. The multiply stands
    // MV_BURST times in the graph, each with an activation tensor of its own (matvecs of one activation would be
    // taken into one grouped launch), so the card works them back to back as it does while decoding.
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (MV_BURST * 4 + 16) + ggml_graph_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(ip);
    ggml_context * ctx  = ggml_init(ip);
    ggml_tensor * a[2] = { nullptr, nullptr };
    std::vector<ggml_tensor *> acts, outs;
    for (int i = 0; i < n_w; ++i) a[i] = ggml_new_tensor_2d(wctx, type, k, m);
    // a kernel loads every weight row of a threadgroup before the output bound (up to 8 rows x 2 simdgroups): rows of
    // zeros stand after the last weight, as the next tensor does in a model file, so no plan reads past the buffer
    ggml_tensor * pad = ggml_new_tensor_2d(wctx, type, k, 16);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (int r = 0; r < MV_BURST; ++r) {
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, cols);
        ggml_set_output(b); // the allocator keeps only outputs: no result may land on an activation still to be read
        acts.push_back(b);
        for (int i = 0; i < n_w; ++i) {
            ggml_tensor * out = ggml_mul_mat(ctx, a[i], b);
            ggml_set_output(out);
            outs.push_back(out);
        }
    }
    for (ggml_tensor * out : outs) ggml_build_forward_expand(gf, out);
    if (ggml_backend_alloc_ctx_tensors(wctx, backend) == nullptr) { printf("SWEEP_FAIL weight buffer allocation\n"); return 1; }
    // the zero rows guard a weight only from the same buffer, after it: an allocation the backend split over buffers
    // (the weights are over its buffer limit) leaves a weight without them, and is refused
    for (int i = 0; i < n_w; ++i) {
        if (a[i]->buffer != pad->buffer || (const uint8_t *) a[i]->data + wb > (const uint8_t *) pad->data) {
            printf("SWEEP_FAIL the weights and the zero rows after them are not in one buffer (the weight is over the backend's buffer limit): no plan is measured on it\n");
            return 1;
        }
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("SWEEP_FAIL graph allocation\n"); return 1; }
    for (int i = 0; i < n_w; ++i) ggml_backend_tensor_set(a[i], wq.data() + wb * i, 0, wb);
    {
        const std::vector<uint8_t> zeros(ggml_nbytes(pad), 0);
        ggml_backend_tensor_set(pad, zeros.data(), 0, zeros.size());
    }
    std::vector<float> xf((size_t) (cols * k));
    for (int r = 0; r < MV_BURST; ++r) {
        fill(xf, 2 + (uint64_t) r);
        ggml_backend_tensor_set(acts[(size_t) r], xf.data(), 0, xf.size() * sizeof(float));
    }

    std::vector<double> ms;
    for (int r = 0; r <= MV_REPS; ++r) { // r = 0 compiles the kernel and is not timed
        const int64_t t0 = ggml_time_us();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("SWEEP_FAIL compute\n"); return 1; }
        ggml_backend_synchronize(backend);
        if (r > 0) ms.push_back((double) (ggml_time_us() - t0) / 1000.0 / MV_BURST);
    }
    const size_t nb = ggml_nbytes(outs[0]);
    std::vector<uint8_t> res(nb);
    uint64_t digest = 14695981039346656037ull;
    for (ggml_tensor * out : outs) { // every result of the burst, chained
        ggml_backend_tensor_get(out, res.data(), 0, nb);
        digest = (digest ^ fnv(res.data(), nb)) * 1099511628211ull;
    }
    printf("SWEEP_RESULT n=%lld ms=%.4f bytes=%zu fnv=%016llx\n", (long long) cols, median(ms), nb * outs.size(), (unsigned long long) digest);
    fflush(stdout);
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return 0; // the probe trace is written at exit
}

struct mv_run {
    bool ok = false;
    bool refused = false;                // the plan check refused the forced plan (why holds the condition)
    std::string why;
    std::string device, wave, group_bytes;
    std::string digest;
    double host_ms = 0, gpu_ms = 0;
    std::string gpu_level;               // the probe zone the GPU time was read from
    std::map<std::string, int> finals;   // the final= part of every receipt of the run, and how many launches printed it
    std::map<std::string, int> kernels;  // the kernels of the run's GPU dispatch intervals, and how many intervals each has
    std::string th_max;                  // thread limit of the receipt's pipeline, "" when the log does not show it
};

// GPU time of one multiply from the probe trace of a child: the dispatch intervals (zone gpu_dispatch: the card's own
// timestamps around each kernel launch), in start order, split evenly over the `computes` graph runs. Every run must
// hold the same kernels the same number of times. The first run is dropped; each of the others is the sum of its
// intervals over the MV_BURST multiplies of the graph; the median of those, in ms. kernels: every interval's kernel
// name and its count over all runs.
static bool mv_gpu_ms(const std::string & path, int computes, double * ms, std::map<std::string, int> * kernels, std::string * err) {
    std::ifstream in(path);
    if (!in) { *err = "no probe trace was written (the build needs -DGGML_METAL_PROBES=ON)"; return false; }
    std::map<long long, std::string> zones, names;
    bool versioned = false;
    struct mark { long long id, stamp, value, thread; };
    std::vector<mark> marks;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("# ggml_metal_probe v1", 0) == 0) { versioned = true; continue; }
        const std::vector<std::string> f = split(line, '\t');
        if (f.empty() || f[0][0] == '#' || f[0] == "S") continue;
        if (f[0] == "Z" && f.size() >= 3) { zones[atoll(f[1].c_str())] = f[2]; continue; }
        if (f[0] == "K" && f.size() >= 4) { names[atoll(f[1].c_str())] = f[3]; continue; }
        if (f[0] == "F" || f[0] == "O" || f[0] == "K") continue;
        if (f.size() == 4) marks.push_back({ atoll(f[0].c_str()), atoll(f[1].c_str()), atoll(f[2].c_str()), atoll(f[3].c_str()) });
    }
    if (!versioned) { *err = "the probe trace is not a ggml_metal_probe v1 dump"; return false; }
    const long long END = 1ll << 31;
    struct span { long long start, length; std::string kernel; };
    std::map<std::string, std::vector<long long>> open;
    std::vector<span> spans;
    for (const mark & r : marks) {
        const long long id = r.id & ~END;
        const auto z = zones.find((id >> 16) & 255);
        if (z == zones.end() || z->second != "gpu_dispatch") continue;
        const std::string key = std::to_string(id) + "/" + std::to_string(r.value) + "/" + std::to_string(r.thread);
        if (!(r.id & END)) { open[key].push_back(r.stamp); continue; }
        if (open[key].empty()) continue;
        const long long start = open[key].back();
        open[key].pop_back();
        const auto name = names.find(r.value);
        if (r.stamp > start) spans.push_back({ start, r.stamp - start, name != names.end() ? name->second : "unknown" });
    }
    if (spans.empty()) { *err = "the probe trace holds no dispatch interval (the card's row of data/device.tsv [devices] needs probe_dispatch on)"; return false; }
    if (spans.size() % (size_t) computes != 0) {
        *err = std::to_string(spans.size()) + " dispatch intervals do not divide over " + std::to_string(computes) + " graph runs";
        return false;
    }
    std::sort(spans.begin(), spans.end(), [](const span & x, const span & y) { return x.start < y.start; });
    const size_t per = spans.size() / (size_t) computes;
    std::vector<double> each;
    std::map<std::string, int> first;
    for (int c = 0; c < computes; ++c) {
        long long sum = 0;
        std::map<std::string, int> here;
        for (size_t i = 0; i < per; ++i) {
            const span & one = spans[(size_t) c * per + i];
            sum += one.length;
            here[one.kernel]++;
            (*kernels)[one.kernel]++;
        }
        if (c == 0) first = here;
        else if (here != first) { *err = "the graph runs did not launch the same kernels the same number of times"; return false; }
        if (c > 0) each.push_back((double) sum / 1e6 / MV_BURST);
    }
    *ms = median(each);
    return true;
}

// One child. hook "" = no forced plan; env_off = the switch that turns the card's rows off for a default run, or "".
static mv_run mv_run_child(const std::string & self, const mv_shape & s, int cols, const std::string & weight_path, const std::string & trace_path,
                           const std::string & hook, const std::string & env_off) {
    mv_run r;
    unlink(trace_path.c_str());
    if (hook.empty()) unsetenv("GGML_METAL_TEST_MV_PLAN"); else setenv("GGML_METAL_TEST_MV_PLAN", hook.c_str(), 1);
    if (!env_off.empty()) setenv(env_off.c_str(), "0", 1);
    setenv("GGML_METAL_TEST_MV_RECEIPT", "1", 1);
    setenv("ZR_ON", "1", 1);
    setenv("ZR_OUT", trace_path.c_str(), 1);
    const std::string cmd = shell_quote(self) + " --child-mv " + (s.kind == 1 ? "group " : "single ") + shell_quote(s.weight) + " " + std::to_string(s.k) + " " + std::to_string(s.m) + " " +
                            std::to_string(cols) + " " + shell_quote(weight_path) + " 2>&1";
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) { r.why = "could not start the child process"; return r; }
    char buf[4096];
    std::string last;
    std::vector<std::string> loaded;
    while (fgets(buf, sizeof(buf), p) != nullptr) {
        std::string line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line.rfind("SWEEP_DEVICE ", 0) == 0) {
            r.device = line.substr(13);
        } else if (line.rfind("SWEEP_SHAPE ", 0) == 0) {
            r.wave        = cell(line, "wave");
            r.group_bytes = cell(line, "group_bytes");
        } else if (line.rfind("SWEEP_RESULT ", 0) == 0) {
            r.host_ms = atof(cell(line, "ms").c_str());
            r.digest  = cell(line, "fnv") + "/" + cell(line, "bytes");
        } else if (line.rfind("MV_RECEIPT ", 0) == 0) {
            const size_t at = line.find(" final=");
            if (at != std::string::npos) r.finals[line.substr(at + 7)]++;
        } else if (line.find("GGML_METAL_TEST_MV_PLAN:") != std::string::npos) {
            r.refused = true;
            r.why = line.substr(line.find("GGML_METAL_TEST_MV_PLAN:") + 25);
        } else if (line.rfind("SWEEP_FAIL ", 0) == 0) {
            if (r.why.empty()) r.why = line;
        } else if (line.find("th_max") != std::string::npos) {
            loaded.push_back(line);
        }
        if (!line.empty()) last = line;
    }
    const int status = pclose(p);
    unsetenv("GGML_METAL_TEST_MV_PLAN");
    if (!env_off.empty()) unsetenv(env_off.c_str());
    unsetenv("ZR_ON");
    unsetenv("ZR_OUT");
    if (status != 0 || r.refused) {
        if (r.why.empty()) r.why = "the child ended with status " + std::to_string(status) + (last.empty() ? "" : ": " + last);
        unlink(trace_path.c_str());
        return r;
    }
    if (r.digest.empty()) { r.why = "the child printed no result"; unlink(trace_path.c_str()); return r; }
    std::string err;
    r.gpu_level = "gpu_dispatch";
    if (!mv_gpu_ms(trace_path, MV_REPS + 1, &r.gpu_ms, &r.kernels, &err)) { r.why = err; unlink(trace_path.c_str()); return r; }
    unlink(trace_path.c_str());
    if (r.finals.size() == 1) {
        const std::string & fin = r.finals.begin()->first;
        const std::string pipeline = fin.substr(fin.rfind(':') + 1);
        for (const std::string & l : loaded) {
            const size_t t = l.find("th_max");
            if (l.find(pipeline) == std::string::npos || t == std::string::npos) continue;
            size_t a = t + 6;
            while (a < l.size() && (l[a] == ' ' || l[a] == '=' || l[a] == ':')) ++a;
            size_t e = a;
            while (e < l.size() && l[e] >= '0' && l[e] <= '9') ++e;
            if (e > a) r.th_max = l.substr(a, e - a);
        }
    }
    r.ok = true;
    return r;
}

// "kind:cols:nr0:form:prep:" as a receipt's final= part starts for this candidate
static std::string mv_final_prefix(const mv_shape & s, const mv_cand & c) {
    return std::to_string(s.kind) + ":" + std::to_string(c.cols) + ":" + std::to_string(c.nr0) + ":" + std::to_string(mv_form_number(c.form)) + ":" + std::to_string(c.prep) + ":";
}

// "" when the run is exactly the forced plan: one receipt text, of the forced kind, columns, rows, form and prep, printed
// once per multiply, and the GPU intervals timed are that receipt's pipeline, one per multiply. Else what differs.
static std::string mv_not_forced(const mv_run & r, const mv_shape & s, const mv_cand & c) {
    const int launches = MV_BURST * (MV_REPS + 1);
    std::string all;
    for (const auto & one : r.finals) all += (all.empty() ? "" : " + ") + one.first + " x" + std::to_string(one.second);
    if (r.finals.size() != 1 || r.finals.begin()->first.rfind(mv_final_prefix(s, c), 0) != 0) {
        return "the receipt is not the forced plan: " + (all.empty() ? std::string("no MV_RECEIPT line") : all);
    }
    if (r.finals.begin()->second != launches) return "the forced plan was launched " + std::to_string(r.finals.begin()->second) + " times, not " + std::to_string(launches);
    const std::string & fin = r.finals.begin()->first;
    const std::string pipeline = fin.substr(mv_final_prefix(s, c).size());
    std::string timed;
    for (const auto & one : r.kernels) timed += (timed.empty() ? "" : " + ") + one.first + " x" + std::to_string(one.second);
    if (r.kernels.size() != 1 || r.kernels.begin()->first != pipeline || r.kernels.begin()->second != launches) {
        return "the GPU intervals timed are not the forced pipeline, one per multiply (" + pipeline + " x" + std::to_string(launches) + "): " + timed;
    }
    return "";
}

// the kernels a run's GPU time was read from, "name xcount + ..."
static std::string mv_kernels(const mv_run & r) {
    std::string timed;
    for (const auto & one : r.kernels) timed += (timed.empty() ? "" : " + ") + one.first + " x" + std::to_string(one.second / (MV_REPS + 1)) + " per run";
    return timed;
}

// the pipeline of a receipt's final= part (kind:cols:nr0:form:prep:pipeline)
static std::string mv_final_pipeline(const std::string & fin) {
    size_t at = 0;
    for (int i = 0; i < 5 && at != std::string::npos; ++i) at = fin.find(':', at == 0 && i == 0 ? 0 : at + 1);
    return at == std::string::npos ? "" : fin.substr(at + 1);
}

// "" when the GPU intervals of a default run are whole: one kernel, launched once per multiply of the graph (a pair
// the card does not group: once per member; a batch the card works one column at a time: once per column), and, where
// the route prints matvec receipts, one receipt per interval, each naming that kernel. Else what is off. The
// small-batch route prints no receipt: its intervals are held to the multiply count only.
static std::string mv_base_off(const mv_run & r, const mv_shape & s, int cols) {
    const int runs = MV_REPS + 1;
    std::string timed;
    for (const auto & one : r.kernels) timed += (timed.empty() ? "" : " + ") + one.first + " x" + std::to_string(one.second);
    if (r.kernels.size() != 1) return "the default's GPU intervals are not of one kernel: " + timed;
    const std::string & kernel = r.kernels.begin()->first;
    const int total = r.kernels.begin()->second;
    const int once = MV_BURST * runs;
    const bool whole = total == once || (s.kind == 1 ? total == 2 * once : total == cols * once);
    if (!whole) return "the default's GPU intervals are not one per multiply (" + std::to_string(once) + ", or that times the " + (s.kind == 1 ? "members" : "columns") + "): " + timed;
    int receipts = 0;
    for (const auto & one : r.finals) {
        if (mv_final_pipeline(one.first) != kernel) return "a receipt of the default names another pipeline than the intervals timed (" + kernel + "): " + one.first;
        receipts += one.second;
    }
    if (!r.finals.empty() && receipts != total) return "the default printed " + std::to_string(receipts) + " receipts for " + std::to_string(total) + " GPU intervals";
    return "";
}

static const char * mv_form_word(int cols, int form) {
    if (cols == 1) return "-";
    return form == 0 ? "a" : form == 1 ? "w" : form == 2 ? "ap" : form == 3 ? "wp" : form == 4 ? "ha" : form == 6 ? "hap" : "?";
}

static bool mv_parse_shape(const std::string & text, mv_shape * s) {
    const std::vector<std::string> f = split(text, ':');
    if (f.size() != 4 || (f[0] != "single" && f[0] != "group")) return false;
    s->kind   = f[0] == "group" ? 1 : 0;
    s->weight = f[1];
    s->k      = atoll(f[2].c_str());
    s->m      = atoll(f[3].c_str());
    const ggml_type t = type_by_name(s->weight);
    return t != GGML_TYPE_COUNT && s->k > 0 && s->m > 0 && s->k % ggml_blck_size(t) == 0;
}

static void mv_usage(void) {
    printf("usage: llama-metal-tiled-sweep --matvec [--model FILE.gguf | --shapes KIND:WEIGHT:K:M,...] [--cols | --columns 1,2,3,4] [--nr0 1,2,4,8]\n"
           "                                         [--prep on|off|both] [--forms a,w,ap,wp,ha,hap] [--rounds 3]\n"
           "       llama-metal-tiled-sweep --matvec --force KIND:WEIGHT:K:M:COLS:PREP:NR0:FORM\n"
           "  --model    take the shapes of the model's 2-D weights (single matvecs)\n"
           "  --shapes   take these shapes; KIND is single or group. With neither: the shapes of [matvec_shapes]\n"
           "  --cols     also measure 2 to 4 activation columns in one launch (single shapes); --columns names the counts\n"
           "  --nr0 --prep --forms   narrow the candidates\n"
           "  --force    run one forced plan and print its receipt (exit 4 when the plan check refuses it)\n"
           "  needs a build with -DGGML_METAL_PROBES=ON: the time compared is GPU time from the probe trace\n");
}

static int mv_main(int argc, char ** argv) {
    std::string model, shapes_csv, nr0_csv = "1,2,4,8", prep_word = "both", columns_csv = "1", forms_csv = "a,w,ap,wp,ha,hap", force;
    int rounds = 3;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc)        model = argv[++i];
        else if (a == "--shapes" && i + 1 < argc)  shapes_csv = argv[++i];
        else if (a == "--cols")                    columns_csv = "1,2,3,4";
        else if (a == "--nr0" && i + 1 < argc)     nr0_csv = argv[++i];
        else if (a == "--prep" && i + 1 < argc)    prep_word = argv[++i];
        else if (a == "--columns" && i + 1 < argc) columns_csv = argv[++i];
        else if (a == "--forms" && i + 1 < argc)   forms_csv = argv[++i];
        else if (a == "--rounds" && i + 1 < argc)  rounds = atoi(argv[++i]);
        else if (a == "--force" && i + 1 < argc)   force = argv[++i];
        else { mv_usage(); return 2; }
    }
    if (rounds < 1 || rounds > 99 || (prep_word != "on" && prep_word != "off" && prep_word != "both") || (!model.empty() && !shapes_csv.empty())) { mv_usage(); return 2; }
    for (const std::string & v : split(nr0_csv, ','))     if (v != "1" && v != "2" && v != "4" && v != "8") { mv_usage(); return 2; }
    for (const std::string & v : split(columns_csv, ',')) if (v != "1" && v != "2" && v != "3" && v != "4") { mv_usage(); return 2; }
    for (const std::string & v : split(forms_csv, ','))   if (v == "-" || mv_form_number(v) < 0) { mv_usage(); return 2; }

    char self[4096];
    uint32_t self_size = sizeof(self);
    if (_NSGetExecutablePath(self, &self_size) != 0) { printf("cannot find this program's own path\n"); return 1; }
    const char * tmp = getenv("TMPDIR");
    const std::string stem = std::string(tmp != nullptr && tmp[0] != '\0' ? tmp : "/tmp") + "/llama-metal-tiled-sweep-" + std::to_string((long) getpid());
    const std::string weight_path = stem + ".bin", trace_path = stem + ".trace";

    if (!force.empty()) {
        const std::vector<std::string> f = split(force, ':');
        mv_shape s;
        mv_cand c;
        if (f.size() != 8 || !mv_parse_shape(f[0] + ":" + f[1] + ":" + f[2] + ":" + f[3], &s) || atoi(f[4].c_str()) < 1 || atoi(f[4].c_str()) > 4) { mv_usage(); return 2; }
        c.cols = atoi(f[4].c_str());
        c.prep = f[5] == "on" ? 1 : 0;
        c.nr0  = atoi(f[6].c_str());
        c.form = f[7];
        const ggml_type type = type_by_name(s.weight);
        if (!mv_write_weights(weight_path, type, s.k, s.m, s.kind == 1 ? 2 : 1)) { printf("cannot write %s\n", weight_path.c_str()); return 1; }
        const mv_run r = mv_run_child(self, s, c.cols, weight_path, trace_path, force, "");
        unlink(weight_path.c_str());
        if (r.refused) { printf("FORCE %s refused: %s\n", force.c_str(), r.why.c_str()); return 4; }
        if (!r.ok) { printf("FORCE %s failed: %s\n", force.c_str(), r.why.c_str()); return 1; }
        const std::string differs = mv_not_forced(r, s, c);
        printf("FORCE %s %s: timed %s · GPU %.4f ms (%s) · host %.4f ms · th_max %s · digest %s%s%s\n", force.c_str(), differs.empty() ? "ran as forced" : "DID NOT RUN AS FORCED",
               mv_kernels(r).c_str(), r.gpu_ms, r.gpu_level.c_str(), r.host_ms, r.th_max.empty() ? "?" : r.th_max.c_str(), r.digest.c_str(), differs.empty() ? "" : " · ", differs.c_str());
        return differs.empty() ? 0 : 1;
    }

    // the shapes
    std::set<mv_shape> shapes;
    if (!shapes_csv.empty()) {
        for (const std::string & one : split(shapes_csv, ',')) {
            mv_shape s;
            if (!mv_parse_shape(one, &s)) { printf("--shapes: %s is not KIND:WEIGHT:K:M\n", one.c_str()); return 2; }
            if (!ggml_backend_metal_matvec_has_rows(type_by_name(s.weight))) { printf("--shapes: the matvec kernel of %s takes no shape rows\n", s.weight.c_str()); return 2; }
            shapes.insert(s);
        }
    } else if (!model.empty()) {
        ggml_context * meta = nullptr;
        gguf_init_params gp = { true, &meta };
        gguf_context * g = gguf_init_from_file(model.c_str(), gp);
        if (g == nullptr || meta == nullptr) { printf("--model: cannot read %s\n", model.c_str()); return 2; }
        for (ggml_tensor * t = ggml_get_first_tensor(meta); t != nullptr; t = ggml_get_next_tensor(meta, t)) {
            if (ggml_n_dims(t) != 2 || !ggml_backend_metal_matvec_has_rows(t->type)) continue;
            mv_shape s;
            s.weight = ggml_type_name(t->type);
            s.k = t->ne[0];
            s.m = t->ne[1];
            shapes.insert(s);
        }
        gguf_free(g);
        ggml_free(meta);
    } else {
        int32_t kind = 0;
        ggml_type type = GGML_TYPE_COUNT;
        int64_t k = 0, m = 0;
        for (int i = 0; ggml_backend_metal_matvec_shape_row(i, &kind, &type, &k, &m) != 0; ++i) {
            if (k < 1 || m < 1) continue; // an every-shape row names no shape
            mv_shape s;
            s.kind = kind;
            s.weight = ggml_type_name(type);
            s.k = k;
            s.m = m;
            shapes.insert(s);
        }
    }
    if (shapes.empty()) { printf("no shape to measure\n"); return 1; }

    char date[32];
    const time_t now = time(nullptr);
    strftime(date, sizeof(date), "%Y-%m-%d", gmtime(&now));
    printf("matvec shapes: %zu · rounds: %d · rows per simdgroup: %s · prep: %s · columns: %s (forms %s)\n", shapes.size(), rounds, nr0_csv.c_str(), prep_word.c_str(),
           columns_csv.c_str(), forms_csv.c_str());

    std::vector<std::string> rows;
    std::string device, wave, group_bytes;
    bool have_card = false;
    int compared = 0, failed = 0, not_admitted = 0;
    for (const mv_shape & s : shapes) {
        const ggml_type type = type_by_name(s.weight);
        if (!mv_write_weights(weight_path, type, s.k, s.m, s.kind == 1 ? 2 : 1)) { printf("cannot write %s\n", weight_path.c_str()); return 1; }
        std::vector<int> col_counts;
        for (const std::string & v : split(columns_csv, ',')) {
            if (v == "1" || s.kind == 0) col_counts.push_back(atoi(v.c_str())); // a group is one column
        }
        for (const int cols : col_counts) {
            std::vector<mv_cand> cands;
            for (const int prep : { 0, 1 }) {
                if ((prep == 1 && (prep_word == "off" || type != GGML_TYPE_Q4_K)) || (prep == 0 && prep_word == "on" && type == GGML_TYPE_Q4_K)) continue;
                for (const std::string & v : split(nr0_csv, ',')) {
                    for (const std::string & form : cols == 1 ? std::vector<std::string>{ "-" } : split(forms_csv, ',')) {
                        mv_cand c;
                        c.cols = cols;
                        c.prep = prep;
                        c.nr0  = atoi(v.c_str());
                        c.form = form;
                        cands.push_back(c);
                    }
                }
            }
            const std::string env_off = cols == 1 ? "GGML_METAL_KQ_HW_ROWS" : "GGML_METAL_KQ_COLS";
            char what[160];
            snprintf(what, sizeof(what), "%-6s %-5s %6lld x %-6lld cols %d", s.kind == 1 ? "group" : "single", s.weight.c_str(), (long long) s.k, (long long) s.m, cols);
            // what each candidate measured: the rule is applied per prep value once all of them ran
            struct mv_result { mv_cand c; double base_ms = 0, ms = 0, ratio = 1; bool same = false; std::string th_max; };
            std::vector<mv_result> results;
            std::string default_final; // the receipt of the default's launch, "" when its route prints none
            for (const mv_cand & c : cands) {
                const std::string hook = mv_hook(s, c);
                std::vector<double> ratios, t_base, t_cand;
                std::string bad, th_max, default_ran;
                bool refused = false, same = true;
                for (int round = 0; round < rounds && bad.empty() && !refused; ++round) {
                    // the forced run first in even rounds, the default first in odd ones
                    mv_run got, base;
                    if (round % 2 == 0) {
                        got = mv_run_child(self, s, cols, weight_path, trace_path, hook, "");
                        if (got.refused) { refused = true; bad = got.why; break; }
                        base = mv_run_child(self, s, cols, weight_path, trace_path, "", env_off);
                    } else {
                        base = mv_run_child(self, s, cols, weight_path, trace_path, "", env_off);
                        got = mv_run_child(self, s, cols, weight_path, trace_path, hook, "");
                    }
                    if (!base.ok) { bad = "the default run failed: " + base.why; break; }
                    if (!got.ok) { bad = got.refused ? got.why : "the forced run failed: " + got.why; refused = got.refused; break; }
                    if (!have_card) {
                        have_card = true;
                        device = base.device;
                        wave = base.wave;
                        group_bytes = base.group_bytes;
                    }
                    const std::string differs = mv_not_forced(got, s, c);
                    const std::string base_off = mv_base_off(base, s, cols);
                    if (!base_off.empty()) {
                        bad = base_off;
                    } else if (base.device != device || got.device != device || base.wave != wave || got.wave != wave || base.group_bytes != group_bytes || got.group_bytes != group_bytes) {
                        bad = "the card changed during the sweep: a sweep is one card";
                    } else if (!differs.empty()) {
                        bad = differs;
                    } else if (!default_ran.empty() && default_ran != mv_kernels(base)) {
                        bad = "the default ran other kernels in another round (" + default_ran + ", then " + mv_kernels(base) + ")";
                    } else {
                        if (base.digest != got.digest) same = false;
                        t_base.push_back(base.gpu_ms);
                        t_cand.push_back(got.gpu_ms);
                        ratios.push_back(got.gpu_ms / base.gpu_ms);
                        th_max = got.th_max;
                        default_final = base.finals.size() == 1 ? base.finals.begin()->first : "";
                        default_ran = mv_kernels(base); // whatever route the card takes without the rows: its kernels, from the trace
                    }
                }
                char cname[64];
                snprintf(cname, sizeof(cname), "prep %-3s nr0 %d form %-3s", c.prep ? "on" : "off", c.nr0, c.form.c_str());
                if (refused) { printf("%s %s not admitted: %s\n", what, cname, bad.c_str()); ++not_admitted; fflush(stdout); continue; }
                if (!bad.empty()) { printf("%s %s not compared: %s\n", what, cname, bad.c_str()); ++failed; fflush(stdout); continue; }
                ++compared;
                mv_result res;
                res.c = c;
                res.base_ms = median(t_base);
                res.ms = median(t_cand);
                res.ratio = median(ratios);
                res.same = same;
                res.th_max = th_max;
                results.push_back(res);
                printf("%s %s default %9.4f ms · forced %9.4f ms · %+6.1f %% GPU time (gpu_dispatch) · digest %s · th_max %s · default ran %s\n", what, cname, res.base_ms, res.ms, (res.ratio - 1.0) * 100.0,
                       same ? "equal" : "DIFFER", th_max.empty() ? "?" : th_max.c_str(), default_ran.c_str());
                fflush(stdout);
            }
            // The rule, per prep value, the prep on row first (it stands before the prep off row of its shape): the
            // fastest candidate with equal bytes that is at least GAIN faster than the default. Where the card has a
            // row for this shape, column count and prep already, a row is proposed only when it is also at least GAIN
            // faster than that row's own plan measured here; else the card's row stays.
            for (const int prep : { 1, 0 }) {
                const mv_result * best = nullptr;
                int n_prep = 0;
                for (const mv_result & r : results) {
                    if (r.c.prep != prep) continue;
                    ++n_prep;
                    if (r.same && r.ratio <= 1.0 - GAIN && (best == nullptr || r.ms < best->ms)) best = &r;
                }
                if (n_prep == 0 || wave.empty() || group_bytes.empty()) continue;
                const std::string key = std::string("\t") + (s.kind == 1 ? "group" : "single") + "\t" + s.weight + "\t" + std::to_string(s.k) + "\t" + std::to_string(s.m) + "\t" + std::to_string(cols) + "\t" + (prep ? "on" : "off") + "\t";
                // The card's own plan for this shape, column count and prep: what it launches with its rows on and no
                // forced plan (the lookup the card runs, with its checks of the card's facts and its every-shape rows),
                // read from the receipt. It is a row's plan when it is of this column count and prep and is not the default's launch.
                const mv_run own = mv_run_child(self, s, cols, weight_path, trace_path, "", prep == 0 && type == GGML_TYPE_Q4_K ? "GGML_METAL_KQ_PREP" : "");
                if (!own.ok) {
                    printf("NOROW%s-\t-\tthe card's own launch (rows on, no forced plan) could not be read: %s\n", key.c_str(), own.why.c_str());
                    ++failed;
                    continue;
                }
                int own_kind = -1, own_cols = 0, own_nr0 = 0, own_form = 0, own_prep = -1;
                const std::string own_final = own.finals.size() == 1 ? own.finals.begin()->first : "";
                const bool own_read = sscanf(own_final.c_str(), "%d:%d:%d:%d:%d:", &own_kind, &own_cols, &own_nr0, &own_form, &own_prep) == 5;
                const bool has = own_read && own_kind == s.kind && own_cols == cols && own_prep == prep && own_final != default_final;
                const int32_t has_nr0 = own_nr0, has_form = own_form;
                const mv_result * held = nullptr;
                for (const mv_result & r : results) {
                    if (has && r.c.prep == prep && r.c.nr0 == has_nr0 && r.c.form == mv_form_word(cols, has_form) && r.same) held = &r;
                }
                char note[320];
                if (has && held == nullptr) {
                    printf("NOROW%s%d\t%s\tthe card's row was not measured here with equal bytes (it is not among the candidates): nothing is proposed against it\n", key.c_str(), has_nr0, mv_form_word(cols, has_form));
                    continue;
                }
                if (has && (best == nullptr || best == held || best->ms > held->ms * (1.0 - GAIN))) {
                    snprintf(note, sizeof(note), "the card's row stays: %.1f us; the fastest candidate (nr0 %d form %s, %.1f us) is not %.0f %% faster than it", held->ms * 1000.0,
                             best != nullptr ? best->c.nr0 : has_nr0, best != nullptr ? best->c.form.c_str() : mv_form_word(cols, has_form), (best != nullptr ? best->ms : held->ms) * 1000.0, GAIN * 100.0);
                    printf("KEEP%s%d\t%s\t%s\n", key.c_str(), has_nr0, mv_form_word(cols, has_form), note);
                    continue;
                }
                if (best == nullptr) {
                    printf("NOROW%s-\t-\tno candidate is %.0f %% faster than the default with equal bytes\n", key.c_str(), GAIN * 100.0);
                    continue;
                }
                char src[420];
                snprintf(src, sizeof(src), "measured %s by llama-metal-tiled-sweep --matvec, GPU time (gpu_dispatch) %.1f -> %.1f us (%+.1f %%), %d rounds beside the default, th_max %s, bytes same%s",
                         date, best->base_ms * 1000.0, best->ms * 1000.0, (best->ratio - 1.0) * 100.0, rounds, best->th_max.empty() ? "?" : best->th_max.c_str(),
                         has ? (std::string("; against the card's row nr0 ") + std::to_string(has_nr0) + " form " + mv_form_word(cols, has_form) + " at " + std::to_string((int) (held->ms * 1000000.0) / 1000.0).substr(0, 6) + " us").c_str() : "");
                rows.push_back(mv_row_text(card_name(device), wave, group_bytes, s.kind, s.weight, s.k, s.m, cols, prep, best->c.nr0, best->c.form, src));
            }
            fflush(stdout);
            fflush(stdout);
        }
    }
    unlink(weight_path.c_str());
    unlink(trace_path.c_str());

    printf("\ncompared with the default: %d candidates · not admitted by the plan check: %d · not compared: %d\n", compared, not_admitted, failed);
    if (compared == 0) { printf("nothing could be measured\n"); return 1; }
    printf("card: %s (%s)\n", card_name(device).c_str(), device.c_str());
    if (rows.empty()) {
        printf("no rows: no candidate that was compared passes the rule (see the KEEP and NOROW lines)\n");
    } else {
        printf("candidate rows for section [matvec_shapes] of ggml/src/ggml-metal/data/matmul.tsv (cells are separated by one tab; a row of a shape,\n"
               "column count and prep the card has already replaces that row; a prep on row stands before the prep off row of its shape).\n"
               "A row goes into the table once a decode run with it shows the gain in token time:\n");
        for (const std::string & r : rows) printf("%s\n", r.c_str());
    }
    return failed > 0 ? 3 : 0;
}

static void usage(void) {
    printf("usage: llama-metal-tiled-sweep [--weights q4_K,q8_0,...] [--tokens 64,128,512] [--rounds 3]\n"
           "  --weights  weight types to measure (default: every type that has a tiled kernel)\n"
           "  --tokens   column counts of the multiply (prompt tokens in one batch), from 2, ascending; each is judged on its own (default 64,128,512)\n"
           "  --rounds   rounds, each runs the card's own plan and then every compiled plan forced (default 3)\n"
           "       llama-metal-tiled-sweep --matvec ...   the matvec work plan: run --matvec --help\n");
}

int main(int argc, char ** argv) {
    ggml_time_init();
    if (argc == 5 && strcmp(argv[1], "--child") == 0) {
        return child_main(argv[2], argv[3], argv[4]);
    }
    if (argc == 8 && strcmp(argv[1], "--child-mv") == 0) {
        return mv_child_main(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
    }
    if (argc >= 2 && strcmp(argv[1], "--matvec") == 0) {
        return mv_main(argc, argv);
    }
    std::string weights_csv, tokens_csv = "64,128,512";
    int rounds = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--weights" && i + 1 < argc)      weights_csv = argv[++i];
        else if (a == "--tokens" && i + 1 < argc)  tokens_csv = argv[++i];
        else if (a == "--rounds" && i + 1 < argc)  rounds = atoi(argv[++i]);
        else { usage(); return 2; }
    }
    std::vector<int64_t> tokens;
    for (const std::string & t : split(tokens_csv, ',')) {
        char * end = nullptr;
        const long long n = strtoll(t.c_str(), &end, 10);
        // a row is for one column count from 2 (the table's plan_cols)
        if (*end != '\0' || n < 2 || n > 8192 || (!tokens.empty() && n <= tokens.back())) { usage(); return 2; }
        tokens.push_back(n);
    }
    if (tokens.empty() || rounds < 1 || rounds > 99) { usage(); return 2; }

    const int n_plans = ggml_backend_metal_tiled_plan_count();
    if (n_plans < 1) { printf("this build has no compiled tiled plans\n"); return 1; }
    if (n_plans < 2) { printf("this build compiles only the default plan: there is no plan to compare it with\n"); return 1; }
    std::vector<std::string> plans;
    std::vector<std::vector<int32_t>> plan_values;
    std::vector<std::string> plan_suffix; // the end of a plan's entry name
    for (int i = 0; i < n_plans; ++i) {
        plans.push_back(ggml_backend_metal_tiled_plan_id(i));
        std::vector<int32_t> v(11);
        if (ggml_backend_metal_tiled_plan_values(i, v.data(), 11) != 11) { printf("plan %s has no values\n", plans.back().c_str()); return 1; }
        std::string suffix = "_lay";
        for (const int32_t x : v) suffix += "_" + std::to_string(x);
        plan_values.push_back(v);
        plan_suffix.push_back(suffix);
    }

    std::vector<std::string> weights;
    if (weights_csv.empty()) {
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            if (ggml_backend_metal_tiled_has_weight((ggml_type) i) && COLS % ggml_blck_size((ggml_type) i) == 0) weights.push_back(ggml_type_name((ggml_type) i));
        }
    } else {
        for (const std::string & w : split(weights_csv, ',')) {
            const ggml_type t = type_by_name(w);
            if (t == GGML_TYPE_COUNT || !ggml_backend_metal_tiled_has_weight(t) || COLS % ggml_blck_size(t) != 0) {
                printf("--weights: %s is not a weight type with a tiled kernel\n", w.c_str());
                return 2;
            }
            weights.push_back(w);
        }
    }

    char self[4096];
    uint32_t self_size = sizeof(self);
    if (_NSGetExecutablePath(self, &self_size) != 0) { printf("cannot find this program's own path\n"); return 1; }
    const char * tmp = getenv("TMPDIR");
    const std::string weight_path = std::string(tmp != nullptr && tmp[0] != '\0' ? tmp : "/tmp") + "/llama-metal-tiled-sweep-" + std::to_string((long) getpid()) + ".bin";

    char date[32];
    const time_t now = time(nullptr);
    strftime(date, sizeof(date), "%Y-%m-%d", gmtime(&now));

    printf("plans: %d (default %s) · weights: %zu · column counts: %s · rounds: %d · weight %lld x %lld\n", n_plans, plans[0].c_str(), weights.size(),
           tokens_csv.c_str(), rounds, (long long) ROWS, (long long) COLS);
    std::vector<std::string> rows, keeps, norows;
    std::string device, wave, group_bytes; // device: "MTL<index>:<name>"
    bool have_card = false;
    int counts_judged = 0, plans_not_judged = 0, measured_arms = 0;

    for (const std::string & w : weights) {
        const ggml_type type = type_by_name(w);
        const size_t wb = (size_t) ROWS * ggml_row_size(type, COLS);
        {
            std::vector<float> wf((size_t) (ROWS * COLS));
            fill(wf, 1);
            std::vector<uint8_t> wq(wb);
            // some types are quantized with an importance per column: every column gets the same one
            std::vector<float> importance;
            if (ggml_quantize_requires_imatrix(type)) importance.assign((size_t) COLS, 1.0f);
            if (ggml_quantize_chunk(type, wf.data(), wq.data(), 0, ROWS, COLS, importance.empty() ? nullptr : importance.data()) != wb) {
                for (const int64_t n : tokens) norows.push_back("NOROW\t" + w + "\t" + std::to_string(n) + " columns\tggml could not build the weight");
                continue;
            }
            FILE * f = fopen(weight_path.c_str(), "wb");
            if (f == nullptr || fwrite(wq.data(), 1, wb, f) != wb || fclose(f) != 0) { printf("cannot write %s\n", weight_path.c_str()); return 1; }
        }
        // one run at one count, as the judging step reads it
        const auto arm_at = [&](const run & r, int64_t n) {
            tiled_arm a;
            if (!r.ok)                 { a.why = r.why.empty() ? "the run failed" : r.why; return a; }
            if (r.fail.count(n))       { a.why = r.fail.at(n); return a; }
            if (!r.ms.count(n) || !r.hash.count(n)) { a.why = "no result at this count"; return a; }
            if (r.device != device || r.wave != wave || r.group_bytes != group_bytes) { a.why = "the run was on another device or card shape (" + r.device + ")"; return a; }
            a.ok     = true;
            a.ms     = r.ms.at(n);
            a.digest = r.hash.at(n);
            const auto e = r.entries.find(n);
            if (e == r.entries.end() || e->second.empty()) { a.why = "no pipeline in the dispatch receipt"; return a; }
            if (e->second.size() != 1) {
                a.why = "more than one pipeline:";
                for (const std::string & one : e->second) a.why += " " + one;
                return a;
            }
            const std::string & name = *e->second.begin();
            for (size_t p = 0; p < plans.size(); ++p) {
                const std::string & s = plan_suffix[p];
                if (name.rfind("kernel_mul_mat_tiled_", 0) == 0 && name.size() > s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0) a.plan = plans[p];
            }
            if (a.plan.empty()) a.why = "dispatched " + name;
            return a;
        };
        std::map<int64_t, std::vector<tiled_round>> measured;
        for (int round = 0; round < rounds; ++round) {
            // the card's own run first, then every compiled plan forced; each run is a process of its own. by_count[i][n]
            // is the run that holds count n of run i: the batch child, or a child of that count alone when the batch failed.
            std::vector<run> runs;
            std::vector<std::map<int64_t, run>> by_count;
            for (int i = 0; i <= n_plans; ++i) {
                const std::string plan = i == 0 ? "" : plans[(size_t) (i - 1)];
                const run batch = run_child(self, w, weight_path, plan, tokens_csv);
                runs.push_back(batch);
                by_count.push_back({});
                if (!batch.ok && tokens.size() > 1) {
                    printf("%-6s round %d: %s: the run of all counts failed (%s); each count is run alone\n", w.c_str(), round + 1,
                           plan.empty() ? "the card's own run" : plan.c_str(), batch.why.c_str());
                    for (const int64_t n : tokens) {
                        by_count.back()[n] = run_child(self, w, weight_path, plan, std::to_string(n));
                        runs.push_back(by_count.back()[n]);
                    }
                } else {
                    for (const int64_t n : tokens) by_count.back()[n] = batch;
                }
            }
            for (const run & r : runs) {
                if (r.device.empty()) continue;
                if (!have_card) {
                    have_card   = true;
                    device      = r.device;
                    wave        = r.wave;
                    group_bytes = r.group_bytes;
                } else if (r.device != device || r.wave != wave || r.group_bytes != group_bytes) {
                    printf("the card changed during the sweep (%s wave %s group memory %s, then %s wave %s group memory %s): a sweep is one card\n", device.c_str(),
                           wave.c_str(), group_bytes.c_str(), r.device.c_str(), r.wave.c_str(), r.group_bytes.c_str());
                    unlink(weight_path.c_str());
                    return 1;
                }
            }
            for (const int64_t n : tokens) {
                tiled_round one;
                one.own = arm_at(by_count[0].at(n), n);
                for (size_t p = 0; p < plans.size(); ++p) one.forced.push_back(arm_at(by_count[p + 1].at(n), n));
                measured[n].push_back(one);
            }
        }

        for (const int64_t n : tokens) {
            const tiled_verdict v = tiled_judge_count(plans, measured[n], GAIN);
            measured_arms += tiled_measured(measured[n]);
            const std::string at = w + "\t" + std::to_string(n) + " columns\t";
            char text[512];
            if (v.kind != tiled_verdict::NOROW) {
                printf("%-6s %5lld columns: the card's own run: plan %s · %.3f ms\n", w.c_str(), (long long) n, v.own_plan.c_str(), v.own_ms);
                ++counts_judged;
            }
            for (const std::string & line : v.lines) printf("%-6s %5lld columns: %s\n", w.c_str(), (long long) n, line.c_str());
            plans_not_judged += v.not_judged;
            if (v.kind == tiled_verdict::NOROW) {
                norows.push_back("NOROW\t" + at + v.reason);
            } else if (v.kind == tiled_verdict::KEEP) {
                snprintf(text, sizeof(text), "the card's plan %s stays: %.3f ms; the fastest other plan: %s %.3f ms (%+.1f %% time, a row needs %.0f %% less)", v.own_plan.c_str(), v.own_ms,
                         v.plan.c_str(), v.plan_ms, (v.ratio - 1.0) * 100.0, GAIN * 100.0);
                keeps.push_back("KEEP\t" + at + text);
            } else if (wave.empty() || group_bytes.empty()) {
                norows.push_back("NOROW\t" + at + "plan " + v.plan + " wins, but the card's wave width and group memory could not be read");
            } else {
                snprintf(text, sizeof(text), "%lld columns: the card's own plan %s %.3f ms -> %.3f ms (%+.1f %%)", (long long) n, v.own_plan.c_str(), v.own_ms, v.plan_ms, (v.ratio - 1.0) * 100.0);
                const size_t pi = (size_t) (std::find(plans.begin(), plans.end(), v.plan) - plans.begin());
                rows.push_back(tiled_row_text(card_name(device), w, v.plan, n, wave, group_bytes, plan_values[pi].data(),
                                              std::string("measured ") + date + " by llama-metal-tiled-sweep, one random " + w + " weight " + std::to_string(ROWS) + " x " + std::to_string(COLS) +
                                              ", " + std::to_string(rounds) + " rounds, each the card's own run then every compiled plan forced, median multiply time at " + text +
                                              "; digest of the output bytes equal to the card's own run",
                                              "llama-metal-tiled-sweep --weights " + w + " --tokens " + tokens_csv + " --rounds " + std::to_string(rounds)));
            }
            fflush(stdout);
        }
    }
    unlink(weight_path.c_str());

    printf("\njudged: %d weight and column-count pairs · without a verdict: %zu · plans not judged at a count: %d\n", counts_judged, norows.size(), plans_not_judged);
    if (have_card) printf("card: %s (%s)\n", card_name(device).c_str(), device.c_str());
    for (const std::string & k : keeps)  printf("%s\n", k.c_str());
    for (const std::string & k : norows) printf("%s\n", k.c_str());
    if (measured_arms == 0) {
        printf("nothing was measured: no run sent a time and a digest at any column count%s\n", have_card ? "" : " (no run reported a Metal device)");
        return tiled_sweep_exit(0, (int) norows.size(), plans_not_judged);
    }
    if (counts_judged == 0) {
        printf("no rows: the sweep finished and no column count could be judged (the reasons are above)\n");
    } else if (rows.empty()) {
        printf("no rows: at no column count is a plan %.0f %% faster than the plan the card runs there\n", GAIN * 100.0);
    } else {
        printf("rows for section [tiles] of ggml/src/ggml-metal/data/matmul.tsv (cells are separated by one tab; each row is for ONE column\n"
               "count and stands beside the card's other rows; a row is a candidate until prompt speed on a model shows its gain):\n");
        for (const std::string & r : rows) printf("%s\n", r.c_str());
    }
    return tiled_sweep_exit(measured_arms, (int) norows.size(), plans_not_judged);
}
