// llama-metal-tiled-sweep: measures every compiled work plan of the tiled matmul kernel on this machine's Metal card
// and prints the rows to add to section [tiles] of ggml/src/ggml-metal/data/matmul.tsv. No model is needed.
//
// Per weight type: one random weight (fixed seed, ROWS x COLS, quantized once by ggml) and random activations. Each
// plan runs in a child process of this program with GGML_METAL_TEST_TILED_PLAN=<weight>:<plan id> (a process runs one
// plan), which multiplies at each token count and prints the median time and a digest of the output bytes. Each round
// runs the default plan right before every other plan, so drift shows. A plan is compared only when both runs report
// the same device (index and name) and card shape, and the receipt of each shows exactly one route at every token
// count: the tiled kernel in both, or the same other route in both. A row is proposed when the digest of
// the output bytes (FNV-1a, 64 bits) equals the default's at every token count, the plan is at least GAIN faster at
// the largest count and not more than LOSS slower at any other. A child whose receipt does not show the multiply on
// its plan is refused.
//
// Exit code: 0 = every plan was compared (with or without rows), 3 = some plan or weight could not be compared (rows
// are for the rest), 2 = bad arguments, 1 = no Metal device, one compiled plan only, or nothing could be compared.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <mach-o/dyld.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <set>
#include <string>
#include <vector>

static const int64_t ROWS = 4096;   // weight rows
static const int64_t COLS = 2560;   // weight columns
static const int     REPS = 9;      // timed multiplies per token count in a child, after one untimed
static const double  GAIN = 0.03;   // a row needs this much less time at the largest token count
static const double  LOSS = 0.03;   // and may not cost more than this at any other count

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

// ---- child: one plan (from the environment), one weight file, every token count ----

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
        if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("SWEEP_FAIL graph allocation\n"); return 1; }
        std::vector<float> xf((size_t) (n * COLS));
        fill(xf, 2);
        ggml_backend_tensor_set(a, wq.data(), 0, wb);
        ggml_backend_tensor_set(b, xf.data(), 0, xf.size() * sizeof(float));

        std::vector<double> ms;
        for (int r = 0; r <= REPS; ++r) { // r = 0 compiles the kernel and is not timed
            const int64_t t0 = ggml_time_us();
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("SWEEP_FAIL compute\n"); return 1; }
            ggml_backend_synchronize(backend);
            if (r > 0) ms.push_back((double) (ggml_time_us() - t0) / 1000.0);
        }
        const size_t nb = ggml_nbytes(out);
        std::vector<uint8_t> res(nb);
        ggml_backend_tensor_get(out, res.data(), 0, nb);
        printf("SWEEP_RESULT n=%lld ms=%.4f bytes=%zu fnv=%016llx\n", (long long) n, median(ms), nb, (unsigned long long) fnv(res.data(), nb));
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }
    fflush(stdout);
    ggml_backend_free(backend);
    return 0; // the library prints the receipt lines at exit
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
    std::map<int64_t, double> ms;         // per token count
    std::map<int64_t, std::string> hash;
    std::map<int64_t, std::string> route; // the receipt's route of the multiply at this count; absent = no receipt for it
    std::map<int64_t, bool> mixed;        // the receipt shows more than one route at this count (route lists them)
    std::map<int64_t, bool> tiled;        // the multiply at this count ran on the plan's tiled kernel and on nothing else
};

static std::string shell_quote(const std::string & s) {
    std::string q = "'";
    for (char c : s) { if (c == '\'') q += "'\\''"; else q += c; }
    return q + "'";
}

static run run_child(const std::string & self, const std::string & type, const std::string & weight_path, const std::string & plan,
                     const std::string & tokens_csv) {
    run r;
    setenv("GGML_METAL_TEST_TILED_PLAN", (type + ":" + plan).c_str(), 1);
    const std::string cmd = shell_quote(self) + " --child " + shell_quote(type) + " " + shell_quote(weight_path) + " " + shell_quote(tokens_csv) + " 2>&1";
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) { r.why = "could not start the child process"; return r; }
    char buf[4096];
    std::string receipt, last;
    std::map<int64_t, std::set<std::string>> routes;
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
        } else if (line.rfind("TILED_TEST_PLAN ", 0) == 0) {
            receipt = line;
        } else if (line.rfind("TILED_TEST_SHAPE ", 0) == 0) {
            if (cell(line, "weight") == type && atoll(cell(line, "k").c_str()) == COLS && atoll(cell(line, "m").c_str()) == ROWS && !cell(line, "route").empty()) {
                routes[atoll(cell(line, "n").c_str())].insert(cell(line, "route"));
            }
        } else if (line.find("GGML_METAL_TEST_TILED_PLAN") != std::string::npos || line.rfind("SWEEP_FAIL ", 0) == 0) {
            if (r.why.empty()) r.why = line;
        }
        if (!line.empty()) last = line;
    }
    const int status = pclose(p);
    unsetenv("GGML_METAL_TEST_TILED_PLAN");
    for (const auto & at : routes) {
        std::string all;
        for (const std::string & one : at.second) all += (all.empty() ? "" : "+") + one;
        r.route[at.first] = all;
        if (at.second.size() != 1) r.mixed[at.first] = true;
        else if (all == "tiled:" + plan) r.tiled[at.first] = true;
    }
    if (status != 0) {
        if (r.why.empty()) r.why = "the child ended with status " + std::to_string(status) + (last.empty() ? "" : ": " + last);
        return r;
    }
    if (receipt.empty() || cell(receipt, "plan") != plan || atol(cell(receipt, "on_plan").c_str()) < 1) {
        r.why = "the receipt does not show the multiply on plan " + plan + (receipt.empty() ? " (no TILED_TEST_PLAN line)" : ": " + receipt);
        return r;
    }
    r.ok = true;
    return r;
}

static void usage(void) {
    printf("usage: llama-metal-tiled-sweep [--weights q4_K,q8_0,...] [--tokens 64,512] [--rounds 3]\n"
           "  --weights  weight types to measure (default: every type that has a tiled kernel)\n"
           "  --tokens   token counts, ascending; a row is judged at the largest (default 64,512)\n"
           "  --rounds   rounds, each runs the default plan before every other plan (default 3)\n");
}

int main(int argc, char ** argv) {
    ggml_time_init();
    if (argc == 5 && strcmp(argv[1], "--child") == 0) {
        return child_main(argv[2], argv[3], argv[4]);
    }
    std::string weights_csv, tokens_csv = "64,512";
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
        if (*end != '\0' || n < 1 || n > 8192 || (!tokens.empty() && n <= tokens.back())) { usage(); return 2; }
        tokens.push_back(n);
    }
    if (tokens.empty() || rounds < 1 || rounds > 99) { usage(); return 2; }

    const int n_plans = ggml_backend_metal_tiled_plan_count();
    if (n_plans < 1) { printf("this build has no compiled tiled plans\n"); return 1; }
    if (n_plans < 2) { printf("this build compiles only the default plan: there is no plan to compare it with\n"); return 1; }
    std::vector<std::string> plans;
    for (int i = 0; i < n_plans; ++i) plans.push_back(ggml_backend_metal_tiled_plan_id(i));

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

    printf("plans: %d (default %s) · weights: %zu · tokens: %s · rounds: %d · weight %lld x %lld\n", n_plans, plans[0].c_str(), weights.size(),
           tokens_csv.c_str(), rounds, (long long) ROWS, (long long) COLS);
    const int64_t top = tokens.back();
    std::vector<std::string> rows;
    std::string device, wave, group_bytes; // device: "MTL<index>:<name>"
    bool have_card = false;
    int weights_compared = 0, pairs_compared = 0, pairs_failed = 0;

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
                printf("%s: ggml could not build the weight, not compared\n", w.c_str());
                pairs_failed += n_plans - 1;
                continue;
            }
            FILE * f = fopen(weight_path.c_str(), "wb");
            if (f == nullptr || fwrite(wq.data(), 1, wb, f) != wb || fclose(f) != 0) { printf("cannot write %s\n", weight_path.c_str()); return 1; }
        }
        // per plan and token count: the plan's times, and the default's times measured right before each
        std::map<std::string, std::map<int64_t, std::vector<double>>> t_plan, t_base;
        std::map<std::string, bool> same, dead;
        std::map<std::string, std::string> why;
        for (int round = 0; round < rounds; ++round) {
            for (int pi = 1; pi < n_plans; ++pi) {
                const std::string & plan = plans[pi];
                if (dead[plan]) continue;
                // a failure here ends this plan only: a plan that already ran all its rounds keeps its result
                const run base = run_child(self, w, weight_path, plans[0], tokens_csv);
                if (!base.ok || !base.tiled.count(top)) {
                    dead[plan] = true;
                    why[plan] = "the default plan, run before it, did not run the " + std::to_string(top) + "-token multiply on the tiled kernel" + (base.why.empty() ? "" : ": " + base.why);
                    continue;
                }
                if (!have_card) {
                    have_card   = true;
                    device      = base.device;
                    wave        = base.wave;
                    group_bytes = base.group_bytes;
                } else if (base.device != device || base.wave != wave || base.group_bytes != group_bytes) {
                    printf("the card changed during the sweep (%s wave %s group memory %s, then %s wave %s group memory %s): a sweep is one card\n", device.c_str(),
                           wave.c_str(), group_bytes.c_str(), base.device.c_str(), base.wave.c_str(), base.group_bytes.c_str());
                    unlink(weight_path.c_str());
                    return 1;
                }
                const run got = run_child(self, w, weight_path, plan, tokens_csv);
                std::string bad;
                if (!got.ok || !got.tiled.count(top)) {
                    bad = got.why.empty() ? "the multiply did not run on this plan" : got.why;
                } else if (got.device != base.device || got.wave != base.wave || got.group_bytes != base.group_bytes) {
                    bad = "the plan ran on another device or card shape than the default plan (" + got.device + ")";
                }
                for (const int64_t n : tokens) {
                    if (!bad.empty()) break;
                    if (!base.hash.count(n) || !got.hash.count(n)) {
                        bad = "no result at " + std::to_string(n) + " tokens";
                    } else if (!base.route.count(n) || !got.route.count(n)) {
                        bad = "no route receipt at " + std::to_string(n) + " tokens";
                    } else if (base.mixed.count(n) || got.mixed.count(n)) {
                        // two runs that each mix routes can dispatch them in different numbers: one route per run and count, or no comparison
                        bad = "more than one route at " + std::to_string(n) + " tokens (" + base.route.at(n) + ", " + got.route.at(n) + ")";
                    } else if (base.tiled.count(n) != got.tiled.count(n)) {
                        bad = "at " + std::to_string(n) + " tokens only one of the two runs used the tiled kernel alone (" + base.route.at(n) + ", " + got.route.at(n) + ")";
                    } else if (!got.tiled.count(n) && base.route.at(n) != got.route.at(n)) {
                        bad = "at " + std::to_string(n) + " tokens the two runs took different routes (" + base.route.at(n) + ", " + got.route.at(n) + ")";
                    }
                }
                if (!bad.empty()) {
                    dead[plan] = true;
                    why[plan] = bad;
                    continue;
                }
                if (!same.count(plan)) same[plan] = true;
                for (const int64_t n : tokens) {
                    if (base.hash.at(n) != got.hash.at(n)) same[plan] = false;
                    // a count the card does not send to the tiled kernel runs one kernel in both runs: its time says nothing about the plan
                    if (!got.tiled.count(n)) continue;
                    t_base[plan][n].push_back(base.ms.at(n));
                    t_plan[plan][n].push_back(got.ms.at(n));
                }
            }
        }

        std::string best;
        double best_ratio = 1.0;
        std::string best_note;
        int compared = 0;
        for (int pi = 1; pi < n_plans; ++pi) {
            const std::string & plan = plans[pi];
            if (dead[plan]) { printf("%-6s %-18s not compared: %s\n", w.c_str(), plan.c_str(), why[plan].c_str()); ++pairs_failed; continue; }
            ++compared;
            bool wins = same[plan];
            std::string note;
            double ratio_top = 1.0;
            for (const int64_t n : tokens) {
                if (!t_plan[plan].count(n)) { printf("%-6s %-18s %5lld tokens: not on the tiled kernel at this count · digest %s\n", w.c_str(), plan.c_str(), (long long) n, same[plan] ? "equal" : "DIFFER"); continue; }
                const double b = median(t_base[plan][n]), p = median(t_plan[plan][n]);
                std::vector<double> ratios;
                for (size_t i = 0; i < t_plan[plan][n].size(); ++i) ratios.push_back(t_plan[plan][n][i] / t_base[plan][n][i]);
                const double ratio = median(ratios);
                printf("%-6s %-18s %5lld tokens: default %9.3f ms · plan %9.3f ms · %+6.1f %% time · digest %s\n", w.c_str(), plan.c_str(), (long long) n, b, p,
                       (ratio - 1.0) * 100.0, same[plan] ? "equal" : "DIFFER");
                char one[160];
                snprintf(one, sizeof(one), "%s%lld tokens default %.3f ms -> %.3f ms (%+.1f %%)", note.empty() ? "" : ", ", (long long) n, b, p, (ratio - 1.0) * 100.0);
                note += one;
                if (n == top) { ratio_top = ratio; if (ratio > 1.0 - GAIN) wins = false; }
                else if (ratio > 1.0 + LOSS) wins = false;
            }
            if (wins && ratio_top < best_ratio) { best = plan; best_ratio = ratio_top; best_note = note; }
        }
        pairs_compared += compared;
        if (compared == 0) {
            printf("%-6s no plan could be compared with the default plan\n", w.c_str());
            fflush(stdout);
            continue;
        }
        ++weights_compared;
        if (!best.empty() && (wave.empty() || group_bytes.empty())) {
            printf("%-6s plan %s wins, but the card's wave width and group memory could not be read: no row\n", w.c_str(), best.c_str());
            fflush(stdout);
            continue;
        }
        if (best.empty()) {
            printf("%-6s no row: none of the %d plans compared is %.0f %% faster than the default plan (%s) at %lld tokens with an equal digest and no more than %.0f %% slower elsewhere\n",
                   w.c_str(), compared, GAIN * 100.0, plans[0].c_str(), (long long) top, LOSS * 100.0);
        } else {
            std::string values;
            int32_t v[16] = {};
            const int nv = ggml_backend_metal_tiled_plan_values((int) (std::find(plans.begin(), plans.end(), best) - plans.begin()), v, 16);
            for (int i = 0; i < nv; ++i) values += "\t" + std::to_string(v[i]);
            // card, weight, plan, role, no thresholds, the card's shape, the plan's values, no threshold evidence, the measurement, the command
            rows.push_back(card_name(device) + "\t" + w + "\t" + best + "\ttuned\t-\t-\t" + wave + "\t" + group_bytes + values + "\t-\t-\tmeasured " + date + " by llama-metal-tiled-sweep, one random " + w + " weight " + std::to_string(ROWS) + " x " +
                           std::to_string(COLS) + ", " + std::to_string(rounds) + " rounds alternating with the default plan, median multiply time: " + best_note +
                           "; digest of the output bytes equal to the default plan's at every token count\tllama-metal-tiled-sweep --weights " + w + " --tokens " + tokens_csv + " --rounds " + std::to_string(rounds));
        }
        fflush(stdout);
    }
    unlink(weight_path.c_str());

    printf("\ncompared with the default plan (%s): %d plan and weight pairs over %d weights · not compared: %d\n", plans[0].c_str(), pairs_compared, weights_compared, pairs_failed);
    if (pairs_compared == 0) { printf("nothing could be measured\n"); return 1; }
    printf("card: %s (%s)\n", card_name(device).c_str(), device.c_str());
    if (rows.empty()) {
        printf("no rows: no plan that was compared passes the rule\n");
    } else {
        printf("rows for section [tiles] of ggml/src/ggml-metal/data/matmul.tsv (cells are separated by one tab; if the card and weight\n"
               "already have a row there with thresholds, put the plan, the shape, the eleven values and the last two cells into that row):\n");
        for (const std::string & r : rows) printf("%s\n", r.c_str());
    }
    return pairs_failed > 0 ? 3 : 0;
}
