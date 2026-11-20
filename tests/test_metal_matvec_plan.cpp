// Host test of the matvec work plan (ggml/src/ggml-metal/matmul/matvec-plan.cpp). Run: test_metal_matvec_plan <data/matmul.tsv> [overlap | family]
// overlap: the table also holds an every-shape row of the D700 for q6_K at 4 rows, after its rows with a written shape.
// family: the device table gives family gcn1 the Metal architecture name testarch and gcn5 the name vegaarch, and the
// Vega 64's device row names no family (architecture -).
// It reads section [matvec_shapes] of the file itself and asks the rule, built against the generated rows, for every
// row: the row used (its index among the card's rows), rows per simdgroup, form, prep and columns must be the file's.
// Then the cases no table row reaches: a card without rows, the wave facts, the switches, the external rows file, the
// cache rule, dense float geometry, each refusal of the check beside a plan that passes, and the two fallbacks.
#include "ggml.h"
#include "ggml-metal.h"
#include "card-object.h"
#include "matmul/formats.h"
#include "matmul/matvec-plan.h"
#include "matvec-rows.h"
#include "device-rows.h"
#include "device/device-sizes.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int failed = 0;
static int checked = 0;

static void expect(bool ok, const char * what, const std::string & detail = "") {
    checked++;
    if (!ok) {
        failed++;
        printf("FAIL %s%s%s\n", what, detail.empty() ? "" : ": ", detail.c_str());
    }
}

struct file_row {
    std::string card;
    uint32_t wave;
    int kind, type, cols, prep, nr0, form;
    int64_t k, m;
};

static int form_of(const std::string & s) {
    if (s == "-" || s == "a") return 0;
    if (s == "w")   return 1;
    if (s == "ap")  return 2;
    if (s == "wp")  return 3;
    if (s == "ha")  return 4;
    if (s == "hap") return 6;
    return -1;
}

static int type_of(const std::string & s) {
    if (s == "q4_K") return GGML_TYPE_Q4_K;
    if (s == "q5_K") return GGML_TYPE_Q5_K;
    if (s == "q6_K") return GGML_TYPE_Q6_K;
    return -1;
}

static std::vector<file_row> read_shapes(const char * path) {
    std::vector<file_row> rows;
    std::ifstream in(path);
    std::string line, section;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '[') {
            section = line;
            continue;
        }
        if (section != "[matvec_shapes]" || line.empty() || line[0] == '#') {
            continue;
        }
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, '\t')) {
            f.push_back(cell);
        }
        if (f.size() != 12 || f[0] == "card") {
            continue;
        }
        file_row r;
        r.card = f[0];
        r.wave = (uint32_t) std::stoul(f[1]);
        r.kind = f[3] == "group" ? 1 : 0;
        r.type = type_of(f[4]);
        r.k    = f[5] == "any" ? -1 : std::stoll(f[5]);
        r.m    = f[6] == "any" ? -1 : std::stoll(f[6]);
        r.cols = std::stoi(f[7]);
        r.prep = f[8] == "on" ? 1 : 0;
        r.nr0  = std::stoi(f[9]);
        r.form = form_of(f[10]);
        rows.push_back(r);
    }
    return rows;
}

// A card as device/init.m sets it up: its name and facts, then the two loaders of the plan.
static struct ggml_metal_card card_facts(const char * name, uint32_t wave, uint32_t group_bytes, uint32_t l1, uint32_t probed) {
    struct ggml_metal_card p = {};
    p.wave        = wave;
    p.wave_probed = probed;
    p.group_mem   = group_bytes;
    p.l1_per_cu   = l1;
    ggml_metal_matvec_tune_load(&p, name, 1, 1, 1, 1);
    return p;
}

// The family step of device/init.m on one profile, then its loaders again: what a regression that applied a
// recognized family's rows would change is this profile's settings and plans.
static void resolve_family(struct ggml_metal_card * p, const char * name, const char * vendor, const char * metal_name) {
    const struct ggml_metal_family_row * f = ggml_metal_family_row_of(name, vendor, metal_name, p->wave, p->group_mem);
    snprintf(p->family, sizeof(p->family), "%s", f != nullptr ? f->name : "");
    ggml_metal_matvec_tune_load(p, name, 1, 1, 1, 1);
}

// A card of the table with the facts its rows were measured with (the device table's wave and group_bytes), else 64 / 65536.
static struct ggml_metal_card card_of(const char * name, uint32_t wave, uint32_t l1) {
    const struct ggml_metal_device_row * d = ggml_metal_device_row_of(name);
    return card_facts(name, wave, d != nullptr ? (uint32_t) d->group_bytes : 65536, l1, 1);
}

static struct ggml_metal_matvec_in ask(int kind, int type, int64_t k, int64_t m, int cols, int prep_available) {
    struct ggml_metal_matvec_in in = {};
    in.kind           = kind;
    in.weight         = type;
    in.k              = k;
    in.m              = m;
    in.cols           = cols;
    in.prep_available = prep_available;
    in.use_rows       = 1;
    in.model_known    = 1;
    in.act_bytes      = (uint32_t) k * 4;
    return in;
}

static std::string show(const struct ggml_metal_matvec_plan & p) {
    char b[160];
    snprintf(b, sizeof(b), "ok %d nsg %d nr0 %d reduce %d cols %d form %d prep %d row %d set %d default %d", p.ok, p.nsg, p.nr0, p.reduce, p.cols, p.form, p.prep, p.shape_row, p.nr0_set, p.from_default);
    return b;
}

static const char * D700 = "AMD Radeon HD - FirePro D700";
static const char * VEGA = "AMD Radeon Pro Vega 64";
static const char * M5500 = "AMD Radeon Pro 5500M";

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        printf("usage: test_metal_matvec_plan <data/matmul.tsv> [overlap | family]\n");
        return 2;
    }
    const std::string mode = argc == 3 ? argv[2] : "";
    const std::vector<file_row> rows = read_shapes(argv[1]);
    expect(!rows.empty(), "the file has [matvec_shapes] rows");

    // A. every row of the file, with and without prepared headers: the first row of its card that fits, in file order
    int table_cases = 0;
    int table_rows = 0;
    for (size_t i = 0; i < rows.size(); i++) {
        const file_row & r = rows[i];
        if (r.card.rfind("family:", 0) == 0 || r.k < 0) {
            continue; // a family row is never applied (below); an every-shape row has its own cases (mode overlap)
        }
        table_rows++;
        const struct ggml_metal_card card = card_of(r.card.c_str(), r.wave, 16384);
        for (int avail = 0; avail < 2; avail++) {
            int want = -1, at = 0;
            for (size_t j = 0; j < rows.size(); j++) {
                if (rows[j].card != r.card) {
                    continue;
                }
                const file_row & c = rows[j];
                if (want < 0 && c.k >= 0 && c.kind == r.kind && c.type == r.type && c.k == r.k && c.m == r.m && c.cols == r.cols && (c.prep == 0 || avail)) {
                    want = at;
                }
                at++;
            }
            const struct ggml_metal_matvec_in in = ask(r.kind, r.type, r.k, r.m, r.cols, avail);
            const struct ggml_metal_matvec_plan p = ggml_metal_matvec_plan_for(&card, &in);
            char what[200];
            snprintf(what, sizeof(what), "row %zu (%s kind %d type %d %lld x %lld cols %d) headers %d", i, r.card.c_str(), r.kind, r.type, (long long) r.k, (long long) r.m, r.cols, avail);
            if (want < 0) {
                const struct mv_format_row * fmt = ggml_metal_mv_format((enum ggml_type) r.type);
                const bool cache = r.kind == 1; // a group member without a row runs the cache rule's rows
                expect(p.ok == 1 && p.shape_row == -1 && p.cols == 1 && p.form == 0 && p.prep == 0 && (cache ? p.nr0_set == 1 : (p.nr0 == fmt->nr0 && p.nr0_set == 0)), what, "no row fits, wanted the plan without a row, got " + show(p));
            } else {
                // 'want' counts the card's rows in the file; the generated rows of a card stand together in the same order
                const file_row * w = nullptr;
                int n = 0;
                for (size_t j = 0; j < rows.size(); j++) {
                    if (rows[j].card == r.card && n++ == want) {
                        w = &rows[j];
                    }
                }
                expect(p.ok == 1 && p.shape_row == want && p.nr0 == w->nr0 && p.nr0_set == 1 && p.form == w->form && p.prep == w->prep && p.cols == w->cols && p.from_default == 0,
                    what, "wanted row " + std::to_string(want) + " nr0 " + std::to_string(w->nr0) + " form " + std::to_string(w->form) + " prep " + std::to_string(w->prep) + ", got " + show(p));
                const char * why = nullptr;
                expect(ggml_metal_matvec_plan_check(&card, &in, &p, &why) == 1, what, std::string("the check refuses a table row: ") + (why ? why : ""));
            }
            table_cases++;
        }
    }
    printf("table rows: %d card rows with a written shape, %d cases\n", table_rows, table_cases);
    expect(table_rows > 0, "the file has card rows with a written shape");

    const struct ggml_metal_card d700 = card_of(D700, 64, 16384);
    const struct ggml_metal_card vega = card_of(VEGA, 64, 16384);
    const struct ggml_metal_card m5500 = card_of(M5500, 32, 16384);
    expect(d700.n_mv_rows > 0 && vega.n_mv_rows == 0 && m5500.n_mv_rows == 0, "the D700 has shape rows, the Vega 64 and the 5500M none");

    // B. a card without rows runs the format's values; the row sum follows the wave and the card's tail
    {
        struct ggml_metal_matvec_in in = ask(0, GGML_TYPE_Q6_K, 2560, 248320, 1, 0);
        const struct ggml_metal_matvec_plan a = ggml_metal_matvec_plan_for(&d700, &in);
        const struct ggml_metal_matvec_plan b = ggml_metal_matvec_plan_for(&vega, &in);
        const struct ggml_metal_matvec_plan c = ggml_metal_matvec_plan_for(&m5500, &in);
        expect(a.shape_row >= 0 && a.nr0 == 8 && a.reduce == 2, "D700 q6_K 2560 x 248320: its row, tree tail", show(a));
        expect(b.ok == 1 && b.shape_row == -1 && b.nr0 == 2 && b.nsg == 2 && b.nr0_set == 0 && b.reduce == 1, "Vega 64, same multiply: the format's rows, serial tail", show(b));
        expect(c.ok == 1 && c.shape_row == -1 && c.nr0 == 2 && c.reduce == 0, "5500M, same multiply: the format's rows, the wave's own sum", show(c));
        in.kind = 1;
        in.weight = GGML_TYPE_Q4_K;
        const struct ggml_metal_matvec_plan g = ggml_metal_matvec_plan_for(&m5500, &in);
        expect(g.reduce == 1, "a group member at wave 32 sums through shared memory", show(g));
        expect(a.q8_load_pref == 4 && a.q8_act_pref == 16 && a.q8_group == 1 && a.q8_rows8_min == 16384 && a.group_store_first == 1, "the D700's load widths and grouping cells reach the plan", show(a));
        expect(c.q8_load_pref == 1 && c.q8_act_pref == 4 && c.q8_rows8_min == 0 && c.group_store_first == 0, "the 5500M's load widths reach the plan");
    }

    // C. paired halves are a fact of wave 64; D. the column switch
    {
        const struct ggml_metal_matvec_in in = ask(0, GGML_TYPE_Q4_K, 2560, 9216, 2, 1);
        const struct ggml_metal_matvec_plan at64 = ggml_metal_matvec_plan_for(&d700, &in);
        struct ggml_metal_card narrow = d700;
        narrow.wave = 32;
        const struct ggml_metal_matvec_plan at32 = ggml_metal_matvec_plan_for(&narrow, &in);
        expect(at64.cols == 2 && at64.form == 4 && at64.nr0 == 4 && at64.prep == 1, "D700 q4_K 2560 x 9216, 2 columns: the paired-halves row", show(at64));
        expect(at32.cols == 1 && at32.shape_row == -1, "the same rows on a card of wave 32: the paired-halves row is not used", show(at32));
        struct ggml_metal_card off = d700;
        off.kq_cols = 0;
        const struct ggml_metal_matvec_plan no_cols = ggml_metal_matvec_plan_for(&off, &in);
        expect(no_cols.cols == 1 && no_cols.shape_row == -1, "column rows switched off: no column plan", show(no_cols));
        struct ggml_metal_matvec_in none = in;
        none.use_rows = 0;
        const struct ggml_metal_matvec_plan unused = ggml_metal_matvec_plan_for(&d700, &none);
        expect(unused.cols == 1 && unused.shape_row == -1, "use_rows 0: no row is read", show(unused));
    }

    // E. the external rows file wins over the card's one-column row, not over a column row
    {
        struct ggml_metal_matvec_in in = ask(0, GGML_TYPE_Q6_K, 2560, 248320, 1, 0);
        in.rows_file_nr0 = 4;
        const struct ggml_metal_matvec_plan one = ggml_metal_matvec_plan_for(&d700, &in);
        expect(one.nr0 == 4 && one.nr0_set == 1 && one.shape_row == -1, "rows file 4 against the card's row 8: 4", show(one));
        in.cols = 2;
        const struct ggml_metal_matvec_plan two = ggml_metal_matvec_plan_for(&d700, &in);
        expect(two.cols == 2 && two.shape_row >= 0 && two.nr0 == 4 && two.form == 4, "the same with 2 columns: the card's column row", show(two));
    }

    // F. the cache rule: basic quants and group members; G. dense float; H. no kernel
    {
        struct ggml_metal_matvec_in in = ask(0, GGML_TYPE_Q4_0, 1024, 4096, 1, 0);
        expect(ggml_metal_matvec_plan_for(&d700, &in).nr0 == 8, "q4_0, a 4 KB activation in a 16 KB L1: 8 rows");
        in.act_bytes = 2560 * 4;
        expect(ggml_metal_matvec_plan_for(&d700, &in).nr0 == 4, "q4_0, a 10 KB activation: 4 rows");
        in.model_known = 0;
        expect(ggml_metal_matvec_plan_for(&d700, &in).nr0 == 2, "q4_0, no model profile: 2 rows");
        in.model_known = 1;
        struct ggml_metal_card no_l1 = d700;
        no_l1.l1_per_cu = 0;
        expect(ggml_metal_matvec_plan_for(&no_l1, &in).nr0 == 2, "q4_0, no L1 fact: 2 rows");
        // the cache rule's rows only on whole threadgroups: 1028 rows are not a whole number of 8 x 2
        {
            struct ggml_metal_matvec_in part = ask(0, GGML_TYPE_Q4_0, 1024, 1028, 1, 0), full = ask(0, GGML_TYPE_Q4_0, 1024, 1024, 1, 0);
            struct ggml_metal_matvec_in gpart = ask(1, GGML_TYPE_Q6_K, 2560, 1028, 1, 0);
            gpart.model_known = 0;
            struct ggml_metal_card eight = d700;
            eight.mv_rows_unknown = 8;
            expect(ggml_metal_matvec_plan_for(&vega, &full).nr0 == 8 && ggml_metal_matvec_plan_for(&vega, &part).nr0 == 2, "q4_0 1024 x 1028: the cache rule's 8 rows would leave a partial threadgroup, the format's 2", show(ggml_metal_matvec_plan_for(&vega, &part)));
            expect(ggml_metal_matvec_plan_for(&eight, &gpart).nr0 == 2, "a q6_K group member 2560 x 1028 with rows_unknown 8: the format's 2 rows", show(ggml_metal_matvec_plan_for(&eight, &gpart)));
            gpart.m = 1024;
            expect(ggml_metal_matvec_plan_for(&eight, &gpart).nr0 == 8, "the same at 1024 rows: 8 rows");
        }
        struct ggml_metal_matvec_in g = ask(1, GGML_TYPE_Q6_K, 1024, 4096, 1, 0);
        expect(ggml_metal_matvec_plan_for(&vega, &g).nr0 == 8, "a q6_K group member, 4 KB activation: 8 rows");
        const struct ggml_metal_matvec_in q8 = ask(0, GGML_TYPE_Q8_0, 1024, 4096, 1, 0);
        const struct ggml_metal_matvec_plan p8 = ggml_metal_matvec_plan_for(&d700, &q8);
        expect(p8.nsg == 2 && p8.nr0 == 4 && p8.nr0_set == 0 && p8.rows_per_threadgroup == 1, "q8_0: the format's rows, per threadgroup", show(p8));
        const int ks[4] = { 16, 128, 300, 2560 }, nsgs[4] = { 1, 1, 3, 4 }, nr0s[4] = { 32, 2, 2, 2 };
        for (int i = 0; i < 4; i++) {
            const struct ggml_metal_matvec_in f = ask(0, GGML_TYPE_F16, ks[i], 512, 1, 0);
            const struct ggml_metal_matvec_plan pf = ggml_metal_matvec_plan_for(&vega, &f);
            expect(pf.ok == 1 && pf.nsg == nsgs[i] && pf.nr0 == nr0s[i], "f16 geometry from K", "K " + std::to_string(ks[i]) + ": " + show(pf));
        }
        // the tree reduce (the D700's) needs simdgroups x wave to be a power of two: 3 simdgroups sum serially
        const int tks[3] = { 300, 384, 2560 }, tnsg[3] = { 3, 3, 4 }, tred[3] = { 1, 1, 2 };
        for (int i = 0; i < 3; i++) {
            const struct ggml_metal_matvec_in f = ask(0, GGML_TYPE_F16, tks[i], 512, 1, 0);
            const struct ggml_metal_matvec_plan pf = ggml_metal_matvec_plan_for(&d700, &f);
            expect(pf.ok == 1 && pf.nsg == tnsg[i] && pf.reduce == tred[i] && ggml_metal_matvec_plan_check(&d700, &f, &pf, nullptr) == 1, "f16 on the tree card: 3 simdgroups sum serially", "K " + std::to_string(tks[i]) + ": " + show(pf));
        }
        {
            struct ggml_metal_card three = d700;
            three.f_nsg_max = 3;
            const struct ggml_metal_matvec_in f = ask(0, GGML_TYPE_F16, 512, 512, 1, 0);
            const struct ggml_metal_matvec_plan pf = ggml_metal_matvec_plan_for(&three, &f);
            expect(pf.nsg == 3 && pf.reduce == 1, "f16 K 512 with f_nsg_max 3 on the tree card: serial", show(pf));
        }
        // the short entry: one cutoff, the format row's
        {
            const struct mv_format_row * ff = ggml_metal_mv_format(GGML_TYPE_F16);
            expect(ff != nullptr && ff->nr0 == 2 && ff->short_k == 32 && ff->short_rows == 32 && ggml_metal_mv_float_short(ff, 31) && !ggml_metal_mv_float_short(ff, 32) && !ggml_metal_mv_float_short(ggml_metal_mv_format(GGML_TYPE_Q4_K), 8), "the short entry's cutoff is the format row's");
            const struct ggml_metal_matvec_in s31 = ask(0, GGML_TYPE_F32, 31, 512, 1, 0), s32 = ask(0, GGML_TYPE_F32, 32, 512, 1, 0);
            const struct ggml_metal_matvec_plan p31 = ggml_metal_matvec_plan_for(&vega, &s31), p32 = ggml_metal_matvec_plan_for(&vega, &s32);
            expect(p31.nsg == 1 && p31.nr0 == 32 && p32.nsg == 1 && p32.nr0 == 2, "f32 K 31 short, K 32 regular", show(p31) + " | " + show(p32));
        }
        const struct ggml_metal_matvec_in none = ask(0, GGML_TYPE_I32, 1024, 4096, 1, 0);
        expect(ggml_metal_matvec_plan_for(&vega, &none).ok == 0, "a type without a matvec kernel: ok 0");
        expect(ggml_metal_matvec_plan_for(nullptr, &none).ok == 0, "no card: ok 0");
    }

    // I. the check: one plan that passes, then each condition broken alone
    {
        const struct ggml_metal_matvec_in in = ask(0, GGML_TYPE_Q4_K, 2560, 9216, 2, 1);
        const struct ggml_metal_matvec_plan good = ggml_metal_matvec_plan_for(&d700, &in);
        const char * why = "unset";
        expect(ggml_metal_matvec_plan_check(&d700, &in, &good, &why) == 1 && why == nullptr, "control: the D700's 2-column paired-halves plan passes the check", show(good));
        struct refusal { const char * name; struct ggml_metal_matvec_in in; struct ggml_metal_matvec_plan plan; const struct ggml_metal_card * card; const char * word; };
        std::vector<refusal> cases;
        { refusal c = { "q5_K with 2 columns", in, good, &d700, "q4_K or q6_K" }; c.in.weight = GGML_TYPE_Q5_K; c.plan.prep = 0; cases.push_back(c); }
        { refusal c = { "a group member with 2 columns", in, good, &d700, "single matvec" }; c.in.kind = 1; cases.push_back(c); }
        { refusal c = { "paired halves with 1 row", in, good, &d700, "paired halves need 2 or 3 columns" }; c.plan.nr0 = 1; cases.push_back(c); }
        { refusal c = { "paired halves with 4 columns", in, good, &d700, "paired halves need 2 or 3 columns" }; c.plan.cols = 4; cases.push_back(c); }
        { refusal c = { "paired halves at wave 32", in, good, &m5500, "wave width 64" }; cases.push_back(c); }
        { refusal c = { "form 5", in, good, &d700, "form is one of" }; c.plan.form = 5; cases.push_back(c); }
        { refusal c = { "one column with a form", in, good, &d700, "one column has no loop form" }; c.plan.cols = 1; c.plan.form = 2; cases.push_back(c); }
        { refusal c = { "m not whole threadgroups", in, good, &d700, "whole number of nr0 x nsg" }; c.in.m = 9218; cases.push_back(c); }
        { refusal c = { "prep on q6_K", in, good, &d700, "q4_K only" }; c.in.weight = GGML_TYPE_Q6_K; cases.push_back(c); }
        { refusal c = { "a row count for q8_0", in, good, &d700, "does not take a row count" }; c.in.weight = GGML_TYPE_Q8_0; c.plan.cols = 1; c.plan.form = 0; c.plan.prep = 0; cases.push_back(c); }
        { refusal c = { "a row of 3 rows per simdgroup", in, good, &d700, "1, 2, 4 or 8" }; c.plan.nr0 = 3; c.plan.form = 0; cases.push_back(c); }
        { refusal c = { "3 rows with paired halves and no table row", in, good, &d700, "1, 2, 4 or 8" }; c.plan.nr0 = 3; c.plan.shape_row = -1; cases.push_back(c); }
        { refusal c = { "16 rows asked for without a table row", in, good, &d700, "1, 2, 4 or 8" }; c.plan.nr0 = 16; c.plan.shape_row = -1; c.plan.form = 0; cases.push_back(c); }
        { refusal c = { "prep without prepared headers", in, good, &d700, "not available" }; c.in.prep_available = 0; cases.push_back(c); }
        { refusal c = { "a tree over 3 simdgroups", in, good, &d700, "power of two" }; c.plan.nsg = 3; c.plan.cols = 1; c.plan.form = 0; c.plan.shape_row = -1; cases.push_back(c); }
        { refusal c = { "one row per simdgroup for q4_0", in, good, &d700, "2, 4 and 8 rows only" }; c.in.weight = GGML_TYPE_Q4_0; c.plan.nr0 = 1; c.plan.cols = 1; c.plan.form = 0; c.plan.prep = 0; c.plan.shape_row = -1; cases.push_back(c); }
        { refusal c = { "a one-column shape row on an m that leaves a partial threadgroup", in, good, &d700, "with a shape row m is a whole number" }; c.in.m = 9220; c.plan.cols = 1; c.plan.form = 0; c.plan.nr0 = 8; c.plan.shape_row = 0; cases.push_back(c); }
        { refusal c = { "5 columns", in, good, &d700, "columns are 1 to 4" }; c.plan.cols = 5; cases.push_back(c); }
        { refusal c = { "a q5_K group member", in, good, &d700, "no grouped launch" }; c.in.kind = 1; c.in.weight = GGML_TYPE_Q5_K; c.plan.cols = 1; c.plan.form = 0; c.plan.prep = 0; cases.push_back(c); }
        for (const refusal & c : cases) {
            const char * reason = nullptr;
            const int ok = ggml_metal_matvec_plan_check(c.card, &c.in, &c.plan, &reason);
            expect(ok == 0 && reason != nullptr && strstr(reason, c.word) != nullptr, c.name, std::string("wanted a refusal naming '") + c.word + "', got " + (ok ? "pass" : (reason ? reason : "no reason")));
        }
        printf("check: %zu refusals, each beside the passing plan\n", cases.size());
    }

    // J. the fallbacks, with the compiled pipeline's numbers given
    {
        const struct ggml_metal_matvec_in cin = ask(0, GGML_TYPE_Q4_K, 2560, 9216, 2, 1);
        const struct ggml_metal_matvec_in sin = ask(0, GGML_TYPE_Q4_K, 2560, 9216, 1, 1);
        const struct ggml_metal_matvec_plan col = ggml_metal_matvec_plan_for(&d700, &cin);
        struct ggml_metal_matvec_plan one = ggml_metal_matvec_plan_for(&d700, &sin);
        one.nr0  = 8; // a saved one-column plan with its own rows and prep
        one.prep = 1;
        struct ggml_metal_matvec_plan out = {};
        expect(ggml_metal_matvec_plan_fallback(&col, &one, 9216, 4, 2, 64, &out) == GGML_METAL_MATVEC_FALLBACK_NONE && memcmp(&out, &col, sizeof(out)) == 0, "whole threadgroups at wave 64: the column plan stays");
        expect(ggml_metal_matvec_plan_fallback(&col, &one, 9218, 4, 2, 64, &out) == GGML_METAL_MATVEC_FALLBACK_TAIL && memcmp(&out, &one, sizeof(out)) == 0 && out.cols == 1 && out.nr0 == 8 && out.prep == 1,
            "a partial last threadgroup: the saved one-column plan with its rows and prep", show(out));
        expect(ggml_metal_matvec_plan_fallback(&col, &one, 9216, 4, 2, 32, &out) == GGML_METAL_MATVEC_FALLBACK_WIDTH && out.cols == 2 && out.nr0 == 4 && out.prep == 1 && out.form == 0,
            "paired halves on a pipeline compiled to wave 32: the same columns, rows and prep, ordinary form", show(out));
        struct ggml_metal_matvec_plan packed = col;
        packed.form = 6;
        expect(ggml_metal_matvec_plan_fallback(&packed, &one, 9216, 4, 2, 32, &out) == GGML_METAL_MATVEC_FALLBACK_WIDTH && out.form == 2, "the same with packed weights: form ap", show(out));
        expect(ggml_metal_matvec_plan_fallback(&col, &one, 9218, 4, 2, 32, &out) == GGML_METAL_MATVEC_FALLBACK_TAIL && out.cols == 1, "both at once: the tail fallback", show(out));
        expect(ggml_metal_matvec_plan_fallback(&col, &one, 9216, 0, 2, 64, &out) == GGML_METAL_MATVEC_FALLBACK_TAIL, "a pipeline without rows: the tail fallback");
        expect(ggml_metal_matvec_plan_fallback(&one, &one, 9218, 4, 2, 32, &out) == GGML_METAL_MATVEC_FALLBACK_NONE, "a one-column plan has no fallback");
    }

    // K. the matcher shared with the registration of prepared headers
    {
        const struct ggml_device_mv_row row = { D700, 0, GGML_TYPE_Q4_K, 2560, 4096, 8, 1, 1, 0, 64, 32768 };
        expect(ggml_metal_matvec_row_fits(&d700, &row, -1, GGML_TYPE_Q4_K, 2560, 4096, -1) == 1, "the matcher: every kind and column count");
        expect(ggml_metal_matvec_row_fits(&d700, &row, 1, GGML_TYPE_Q4_K, 2560, 4096, 1) == 0, "the matcher: another kind");
        expect(ggml_metal_matvec_row_fits(&d700, &row, 0, GGML_TYPE_Q4_K, 2560, 4096, 2) == 0, "the matcher: another column count");
        expect(ggml_metal_matvec_row_fits(&d700, &row, 0, GGML_TYPE_Q4_K, 2560, 4097, 1) == 0, "the matcher: another shape");
        expect(ggml_metal_matvec_row_fits(&d700, &row, 0, GGML_TYPE_Q6_K, 2560, 4096, 1) == 0, "the matcher: another weight type");
    }

    // L. the settings follow the card's facts: default rows by wave, the card's own row only with the facts written in it
    {
        const struct ggml_metal_matvec_in q6 = ask(0, GGML_TYPE_Q6_K, 2560, 248320, 1, 0);
        const struct ggml_metal_card u32 = card_facts("A card nobody measured", 32, 65536, 16384, 1);
        const struct ggml_metal_card u64 = card_facts("A card nobody measured", 64, 65536, 16384, 1);
        const struct ggml_metal_card u16 = card_facts("A card nobody measured", 16, 65536, 16384, 1);
        const struct ggml_metal_matvec_plan p32 = ggml_metal_matvec_plan_for(&u32, &q6);
        const struct ggml_metal_matvec_plan p64 = ggml_metal_matvec_plan_for(&u64, &q6);
        expect(u32.mv_sum == 0 && p32.reduce == 0 && p32.nr0 == 2 && p32.shape_row == -1, "an unknown card of wave 32: the wave's own sum, the format's rows", show(p32));
        expect(u64.mv_sum == 1 && u64.mv_reduce == 0 && p64.reduce == 1 && p64.nr0 == 2 && p64.shape_row == -1, "an unknown card of wave 64: shared memory, serial tail, the format's rows", show(p64));
        expect(u16.mv_sum == 1 && ggml_metal_matvec_plan_for(&u16, &q6).reduce == 1, "an unknown card of another wave: shared memory");
        expect(u64.q8_load_bytes == 1 && u64.q8_act_load_bytes == 4 && u64.q8_rows8_min == 0 && u64.mv_group == 0 && u64.mv_group_red == 0 && u64.f_scalar_w64 == 0 &&
               u64.l1_share_cut > 0.29f && u64.l1_share_cut < 0.31f && u64.mv_rows_below == 8 && u64.mv_rows_from == 4 && u64.mv_rows_unknown == 2 && u64.f_nsg_max == 4 && u64.f_k_per_sg == 128,
               "an unknown card runs every cell of the default row");
        const struct ggml_metal_card real = card_facts(D700, 64, 32768, 16384, 1);
        const struct ggml_metal_card wave = card_facts(D700, 32, 32768, 16384, 1);
        const struct ggml_metal_card mem  = card_facts(D700, 64, 65536, 16384, 1);
        const struct ggml_metal_card dark = card_facts(D700, 64, 32768, 16384, 0);
        const struct ggml_metal_matvec_plan pr = ggml_metal_matvec_plan_for(&real, &q6);
        expect(real.mv_reduce == 1 && real.q8_load_bytes == 4 && real.kq_prep == 1 && real.kq_cols == 1 && pr.shape_row >= 0 && pr.nr0 == 8, "control: the D700 with its facts runs its rows", show(pr));
        const struct ggml_metal_card * changed[3] = { &wave, &mem, &dark };
        const char * what[3] = { "the D700's name with another wave", "the D700's name with another threadgroup memory", "the D700's name before its wave is probed" };
        for (int i = 0; i < 3; i++) {
            const struct ggml_metal_matvec_plan p = ggml_metal_matvec_plan_for(changed[i], &q6);
            expect(changed[i]->mv_reduce == 0 && changed[i]->q8_load_bytes == 1 && changed[i]->kq_prep == 0 && changed[i]->kq_cols == 0 && p.shape_row == -1 && p.nr0 == 2,
                what[i], "wanted the default rows and no shape row, got mv_reduce " + std::to_string(changed[i]->mv_reduce) + " " + show(p));
        }
        expect(wave.mv_sum == 0 && mem.mv_sum == 1 && dark.mv_sum == 1, "the default row of wave 32 applies by the wave alone, and only once it is probed");
        const struct ggml_device_mv_row row = { D700, 0, GGML_TYPE_Q4_K, 2560, 4096, 8, 1, 1, 0, 64, 32768 };
        expect(ggml_metal_matvec_row_fits(&real, &row, -1, GGML_TYPE_Q4_K, 2560, 4096, -1) == 1 && ggml_metal_matvec_row_fits(&wave, &row, -1, GGML_TYPE_Q4_K, 2560, 4096, -1) == 0,
            "the matcher of the prepared headers: only with the row's facts");
    }

    // M. families: a member is named by its device row or by Metal's architecture name; its family's rows are candidates only
    {
        const struct ggml_metal_family_row * f = ggml_metal_family_row_of(D700, "amd", "", 64, 32768);
        expect(f != nullptr && strcmp(f->name, "gcn1") == 0, "the D700 is a member of gcn1 by its device row");
        expect(ggml_metal_family_row_of(D700, "amd", "", 32, 32768) == nullptr, "not with another wave");
        expect(ggml_metal_family_row_of(D700, "intel", "", 64, 32768) == nullptr && ggml_metal_family_row_of(D700, "", "", 64, 32768) == nullptr && ggml_metal_family_row_of(D700, nullptr, "", 64, 32768) == nullptr, "not with another vendor, or none detected");
        expect(ggml_metal_family_row_of("A card nobody measured", "amd", "", 64, 32768) == nullptr, "a card without a device row and without an architecture name has no family");
        uint32_t n_family = 0, n_card = 0;
        const struct ggml_device_mv_row * fam_rows  = ggml_metal_family_mv_rows_of("gcn1", &n_family);
        const struct ggml_device_mv_row * card_rows = ggml_metal_mv_rows_of(D700, &n_card);
        expect(n_family > 0 && (mode == "overlap" || n_family == n_card), "gcn1 holds the D700's rows as candidates", std::to_string(n_family) + " against " + std::to_string(n_card));
        if (mode.empty()) {
            // a family's rows are copies of its donor's (the one card whose device row names the family), cell by cell
            for (uint32_t i = 0; i < n_family && i < n_card; i++) {
                const struct ggml_device_mv_row & a = fam_rows[i], & b = card_rows[i];
                expect(a.kind == b.kind && a.type == b.type && a.ne00 == b.ne00 && a.ne01 == b.ne01 && a.nr0 == b.nr0 && a.prep == b.prep && a.cols == b.cols && a.form == b.form && a.wave == b.wave && a.group_bytes == b.group_bytes,
                    "a gcn1 shape row is its donor's row", "row " + std::to_string(i));
            }
            int n_families = 0;
            for (int i = 0; ggml_metal_matvec_family_default_rows[i].card != nullptr; i++) {
                const struct ggml_metal_matvec_default_row & a = ggml_metal_matvec_family_default_rows[i];
                const struct ggml_metal_matvec_default_row * b = nullptr;
                // the donor: a member of the family (by its device row) whose settings row equals the family row in every
                // cell; the search goes on past a member that differs in any cell
                const auto same_cells = [](const struct ggml_metal_matvec_default_row & x, const struct ggml_metal_matvec_default_row & y) {
                    return x.wave == y.wave && x.group_bytes == y.group_bytes && x.simd_sum_w64_ok == y.simd_sum_w64_ok && x.mv_reduce == y.mv_reduce && x.f_scalar_w64 == y.f_scalar_w64 && x.mv_group == y.mv_group
                        && x.q8_load_bytes == y.q8_load_bytes && x.q8_rows8_min == y.q8_rows8_min && x.q8_act_load_bytes == y.q8_act_load_bytes && x.mv_group_red == y.mv_group_red && x.mv_sum == y.mv_sum && x.l1_share_cut == y.l1_share_cut
                        && x.rows_below == y.rows_below && x.rows_from == y.rows_from && x.rows_unknown == y.rows_unknown && x.f_nsg_max == y.f_nsg_max && x.f_k_per_sg == y.f_k_per_sg;
                };
                for (int j = 0; j < ggml_metal_device_rows_n && b == nullptr; j++) {
                    const struct ggml_metal_matvec_default_row * c = strcmp(ggml_metal_device_rows[j].arch_family, a.card) == 0 ? ggml_metal_matvec_default_row_of(ggml_metal_device_rows[j].card) : nullptr;
                    if (c != nullptr && same_cells(*c, a)) {
                        b = c;
                    }
                }
                n_families++;
                expect(b != nullptr && a.wave == b->wave && a.group_bytes == b->group_bytes && a.simd_sum_w64_ok == b->simd_sum_w64_ok && a.mv_reduce == b->mv_reduce && a.f_scalar_w64 == b->f_scalar_w64 && a.mv_group == b->mv_group
                    && a.q8_load_bytes == b->q8_load_bytes && a.q8_rows8_min == b->q8_rows8_min && a.q8_act_load_bytes == b->q8_act_load_bytes && a.mv_group_red == b->mv_group_red && a.mv_sum == b->mv_sum && a.l1_share_cut == b->l1_share_cut
                    && a.rows_below == b->rows_below && a.rows_from == b->rows_from && a.rows_unknown == b->rows_unknown && a.f_nsg_max == b->f_nsg_max && a.f_k_per_sg == b->f_k_per_sg,
                    "a family's settings row is its donor's row, cell by cell", a.card);
            }
            expect(n_families == 3, "three family settings rows", std::to_string(n_families));
        }
        expect(ggml_metal_matvec_family_default_row_of("gcn1") != nullptr && ggml_metal_matvec_family_default_row_of("gcn1")->mv_reduce == 1, "and its settings row");
        ggml_metal_mv_rows_of("gcn1", &n_card);
        expect(n_card == 0 && ggml_metal_matvec_default_row_of("gcn1") == nullptr && ggml_metal_matvec_default_row_of("family:gcn1") == nullptr, "a family's rows are in no table a card's name is looked up in");
        if (mode == "family") {
            const char * SIB = "AMD Radeon HD - FirePro D500";
            expect(ggml_metal_family_row_of(SIB, "amd", "testarch", 64, 65536) == nullptr, "the sibling with another threadgroup memory is not a member");
            expect(ggml_metal_family_row_of(SIB, "amd", "otherarch", 64, 32768) == nullptr, "nor with another architecture name");
            expect(ggml_metal_family_row_of(SIB, "intel", "testarch", 64, 32768) == nullptr, "nor with another vendor");
            // one profile through both steps: recognized as gcn1, and still on the default rows
            struct ggml_metal_card sib = card_facts(SIB, 64, 32768, 16384, 1);
            resolve_family(&sib, SIB, "amd", "testarch");
            uint32_t n_candidates = 0;
            ggml_metal_family_mv_rows_of(sib.family, &n_candidates);
            const struct ggml_metal_matvec_in q6 = ask(0, GGML_TYPE_Q6_K, 2560, 248320, 1, 0);
            const struct ggml_metal_matvec_in q4 = ask(0, GGML_TYPE_Q4_K, 2560, 4096, 1, 1);
            const struct ggml_metal_matvec_plan ps = ggml_metal_matvec_plan_for(&sib, &q6);
            const struct ggml_metal_matvec_plan p4 = ggml_metal_matvec_plan_for(&sib, &q4);
            expect(strcmp(sib.family, "gcn1") == 0 && n_candidates > 0, "a sibling without a device row is a member by Metal's architecture name, with candidate rows", sib.family);
            expect(sib.mv_reduce == 0 && sib.q8_load_bytes == 1 && sib.mv_group == 0 && sib.n_mv_rows == 0 && sib.kq_prep == 0 && sib.kq_cols == 0 && ps.shape_row == -1 && ps.nr0 == 2 && ps.reduce == 1 && p4.prep == 0 && p4.nr0 == 2,
                "the recognized sibling runs the default rows, not its family's", show(ps) + " | " + show(p4));
            // a card whose device row names no family is a member by the reported name
            const struct ggml_metal_family_row * v = ggml_metal_family_row_of(VEGA, "amd", "vegaarch", 64, 65536);
            expect(v != nullptr && strcmp(v->name, "gcn5") == 0 && ggml_metal_family_row_of(VEGA, "amd", "", 64, 65536) == nullptr, "a device row without a family: the reported architecture name decides");
        } else {
            expect(ggml_metal_family_row_of("AMD Radeon HD - FirePro D500", "amd", "testarch", 64, 32768) == nullptr, "control: without the family's architecture name in the table the sibling has no family");
            const struct ggml_metal_family_row * v = ggml_metal_family_row_of(VEGA, "amd", "", 64, 65536);
            expect(v != nullptr && strcmp(v->name, "gcn5") == 0, "the Vega 64 is a member of gcn5 by its device row");
        }
    }

    // N. an every-shape row stands for the shapes without a row of their own
    {
        const struct ggml_metal_matvec_in written = ask(0, GGML_TYPE_Q6_K, 2560, 248320, 1, 0);
        const struct ggml_metal_matvec_in other   = ask(0, GGML_TYPE_Q6_K, 1024, 512, 1, 0);
        const struct ggml_metal_matvec_in q4      = ask(0, GGML_TYPE_Q4_K, 1024, 512, 1, 0);
        const struct ggml_metal_matvec_plan pw = ggml_metal_matvec_plan_for(&d700, &written);
        const struct ggml_metal_matvec_plan po = ggml_metal_matvec_plan_for(&d700, &other);
        const struct ggml_metal_matvec_plan p4 = ggml_metal_matvec_plan_for(&d700, &q4);
        expect(pw.shape_row == 0 && pw.nr0 == 8, "q6_K 2560 x 248320 on the D700: its own row", show(pw));
        expect(p4.shape_row == -1 && p4.nr0 == 2, "a q4_K shape without a row: the format's rows", show(p4));
        if (mode == "overlap") {
            expect(po.shape_row > 0 && po.nr0 == 4 && po.nr0_set == 1, "a q6_K shape without a row of its own: the every-shape row", show(po));
            // 1028 rows are not a whole number of 4 x 2: the kernel would read rows 1028 to 1031 of the last threadgroup
            const struct ggml_metal_matvec_in part = ask(0, GGML_TYPE_Q6_K, 2560, 1028, 1, 0);
            const struct ggml_metal_matvec_plan pp = ggml_metal_matvec_plan_for(&d700, &part);
            expect(pp.shape_row == -1 && pp.nr0 == 2 && pp.nr0_set == 0 && pp.from_default == 1, "q6_K 2560 x 1028: the every-shape row would leave a partial threadgroup, the plan without it", show(pp));
            const struct ggml_metal_matvec_in whole = ask(0, GGML_TYPE_Q6_K, 2560, 1024, 1, 0);
            expect(ggml_metal_matvec_plan_for(&d700, &whole).nr0 == 4, "q6_K 2560 x 1024: whole threadgroups, the every-shape row");
            struct ggml_metal_card narrow = d700;
            narrow.wave = 32;
            expect(ggml_metal_matvec_plan_for(&narrow, &other).shape_row == -1, "the every-shape row on a card of another wave: not used");
            struct ggml_metal_matvec_in two = other;
            two.cols = 2;
            expect(ggml_metal_matvec_plan_for(&d700, &two).cols == 1, "the every-shape row is for one column");
        } else {
            expect(po.shape_row == -1 && po.nr0 == 2, "control: without an every-shape row that q6_K shape runs the format's rows", show(po));
        }
    }

    // The test hook's text and the plan it forces (GGML_METAL_TEST_MV_PLAN): what is read, and each refusal.
    {
        struct ggml_metal_matvec_forced f = {};
        char word[16] = "";
        const char * why = nullptr;
        const bool read = ggml_metal_matvec_forced_parse("single:q4_K:9216:2560:4:on:2:ap", &f, word, &why) == 1;
        expect(read && f.kind == 0 && std::string(word) == "q4_K" && f.k == 9216 && f.m == 2560 && f.cols == 4 && f.prep == 1 && f.nr0 == 2 && f.form == 2,
               "hook text single:q4_K:9216:2560:4:on:2:ap is read", why != nullptr ? why : "");
        expect(ggml_metal_matvec_forced_parse("group:q4_K:2560:9216:1:on:8:-", &f, word, &why) == 1 && f.kind == 1 && f.form == 0 && f.nr0 == 8, "hook text of a group is read");
        const char * bad[] = { "", "single:q4_K:2560:4096:1:on:8", "single:q4_K:2560:4096:1:on:8:-:x", "both:q4_K:2560:4096:1:on:8:-", "single::2560:4096:1:on:8:-",
                               "single:q4_K:0:4096:1:on:8:-", "single:q4_K:2560:x:1:on:8:-", "single:q4_K:2560:4096:5:on:8:a", "single:q4_K:2560:4096:1:yes:8:-",
                               "single:q4_K:2560:4096:1:on:3:-", "single:q4_K:2560:4096:1:on:8:zz", "single:q4_K:2560:4096:1:on:8:a", "single:q4_K:2560:4096:2:on:8:-" };
        for (const char * text : bad) {
            why = nullptr;
            expect(ggml_metal_matvec_forced_parse(text, &f, word, &why) == 0 && why != nullptr, "hook text refused", text);
        }

        // a number is checked as written: one that only fits after it is cut to 32 bits is refused
        for (const char * text : { "single:q4_K:2560:4096:4294967298:on:8:a", "single:q4_K:2560:4096:1:on:4294967304:-", "single:q4_K:2560:4096:1:on:99999999999999999999:-" }) {
            why = nullptr;
            expect(ggml_metal_matvec_forced_parse(text, &f, word, &why) == 0 && why != nullptr, "hook text with a number past 32 bits refused", text);
        }
        // the card's row for a shape, column count and prep, as the sweep asks for it
        int32_t r_nr0 = 0, r_form = -1;
        expect(ggml_backend_metal_matvec_card_row(D700, 0, GGML_TYPE_Q4_K, 2560, 4096, 1, 1, &r_nr0, &r_form) == 1 && r_nr0 == 8 && r_form == 0, "the D700's row of q4_K 2560 x 4096 prep on: 8 rows", std::to_string(r_nr0));
        expect(ggml_backend_metal_matvec_card_row(D700, 0, GGML_TYPE_Q4_K, 9216, 2560, 4, 1, &r_nr0, &r_form) == 1 && r_nr0 == 2 && r_form == 2, "the D700's 4-column row of q4_K 9216 x 2560: 2 rows, form ap", std::to_string(r_nr0));
        expect(ggml_backend_metal_matvec_card_row(VEGA, 0, GGML_TYPE_Q4_K, 2560, 4096, 1, 1, &r_nr0, &r_form) == 0 && ggml_backend_metal_matvec_card_row(D700, 1, GGML_TYPE_Q4_K, 2560, 4096, 1, 1, &r_nr0, &r_form) == 0 &&
               ggml_backend_metal_matvec_card_row(D700, 0, GGML_TYPE_Q4_K, 2560, 4096, 1, 0, &r_nr0, &r_form) == 1 && r_nr0 == 8, "a card without the row, another kind: none; prep off is its own row");

        struct ggml_metal_card d = card_of(D700, 64, 16384);
        auto force = [&](const char * text, int prep_available, struct ggml_metal_matvec_plan * plan, const char ** reason) {
            struct ggml_metal_matvec_forced g = {};
            char w[16] = "";
            if (!ggml_metal_matvec_forced_parse(text, &g, w, reason)) {
                return 0;
            }
            g.weight = type_of(w);
            const struct ggml_metal_matvec_in in = ask(g.kind, g.weight, g.k, g.m, g.cols, prep_available);
            return ggml_metal_matvec_forced_plan(&d, &in, &g, plan, reason);
        };
        struct ggml_metal_matvec_plan p = {};
        why = nullptr;
        expect(force("single:q6_K:2560:248320:1:off:4:-", 0, &p, &why) == 1 && p.nr0 == 4 && p.nr0_set == 1 && p.cols == 1 && p.prep == 0 && p.shape_row == -1,
               "forced q6_K 4 rows on a card whose row says 8: the forced plan, no row", show(p));
        expect(force("single:q4_K:2560:4096:1:on:1:-", 1, &p, &why) == 1 && p.nr0 == 1 && p.prep == 1, "forced q4_K 1 row with prepared headers", show(p));
        expect(force("single:q4_K:9216:2560:4:on:2:ap", 1, &p, &why) == 1 && p.cols == 4 && p.form == 2 && p.nr0 == 2 && p.prep == 1, "forced 4 columns, form ap", show(p));
        expect(force("group:q4_K:2560:9216:1:on:8:-", 1, &p, &why) == 1 && p.nr0 == 8 && p.prep == 1 && p.reduce != 0, "forced group plan sums through shared memory", show(p));
        const struct { const char * text; int prep; const char * want; } refused[] = {
            { "single:q5_K:2560:8192:2:off:2:a",  0, "2 to 4 columns need a single matvec of q4_K or q6_K" },
            { "single:q4_K:2560:4096:1:on:8:-",   0, "prepared headers are not available for this weight" },
            { "single:q6_K:2560:4096:1:on:8:-",   1, "prepared headers exist for q4_K only" },
            { "single:q6_K:2560:1028:1:off:4:-",  0, "with a forced row count m is a whole number of nr0 x nsg rows" },
            { "single:q4_K:2560:4096:4:off:2:ha", 0, "paired halves need 2 or 3 columns and 2, 4 or 8 rows" },
            { "group:q5_K:2560:8192:1:off:8:-",   0, "this weight type has no grouped launch" },
            { "group:q4_K:2560:9216:2:off:2:a",   0, "2 to 4 columns need a single matvec of q4_K or q6_K" },
        };
        for (const auto & r : refused) {
            why = nullptr;
            const int ok = force(r.text, r.prep, &p, &why);
            expect(ok == 0 && why != nullptr && std::string(why) == r.want, "forced plan refused with its condition", std::string(r.text) + " -> " + (why != nullptr ? why : "accepted"));
        }
        struct ggml_metal_matvec_forced g = {};
        ggml_metal_matvec_forced_parse("single:q4_K:2560:4096:2:on:4:ha", &g, word, &why);
        g.weight = GGML_TYPE_Q4_K;
        expect(ggml_metal_matvec_forced_is_for(&g, 0, GGML_TYPE_Q4_K, 2560, 4096, 2) == 1, "the hook is for its own multiply");
        expect(ggml_metal_matvec_forced_is_for(&g, 0, GGML_TYPE_Q4_K, 2560, 4096, 1) == 0 && ggml_metal_matvec_forced_is_for(&g, 1, GGML_TYPE_Q4_K, 2560, 4096, 2) == 0 &&
               ggml_metal_matvec_forced_is_for(&g, 0, GGML_TYPE_Q6_K, 2560, 4096, 2) == 0 && ggml_metal_matvec_forced_is_for(&g, 0, GGML_TYPE_Q4_K, 2560, 8192, 2) == 0 &&
               ggml_metal_matvec_forced_is_for(nullptr, 0, GGML_TYPE_Q4_K, 2560, 4096, 2) == 0, "and for no other column count, kind, weight type or shape, nor when unset");
    }

    // A card's sizes come from its exact row of data/device.tsv [devices], through the function device setup calls
    // (device/device-sizes.h). A name one model away has no row: no L1 fact, so the cache rule runs its rows for an
    // unmeasured card even with a model profile and a small activation. The exact name is the control.
    {
        struct ggml_device_profile near = {}, exact = {};
        near.l1_cache_per_cu = 16384; // a value set before the lookup does not survive it
        const bool near_row  = ggml_metal_device_sizes_load(&near, "AMD Radeon Pro Vega 56") != nullptr;
        const bool exact_row = ggml_metal_device_sizes_load(&exact, VEGA) != nullptr;
        expect(!near_row && near.l1_cache_per_cu == 0 && near.compute_units == 0 && near.l2_cache_total == 0 && near.local_bandwidth == 0,
               "a card without a row has no sizes", std::to_string(near.l1_cache_per_cu));
        expect(exact_row && exact.l1_cache_per_cu == 16384 && exact.compute_units == 64, "the Vega 64's row gives its sizes", std::to_string(exact.l1_cache_per_cu));
        const struct ggml_metal_card c_near  = card_facts("AMD Radeon Pro Vega 56", 64, 65536, near.l1_cache_per_cu, 1);
        const struct ggml_metal_card c_exact = card_facts(VEGA, 64, 65536, exact.l1_cache_per_cu, 1);
        const struct ggml_metal_matvec_in in = ask(0, GGML_TYPE_Q4_0, 1024, 4096, 1, 0); // a 4 KB activation, model profile known
        expect(ggml_metal_matvec_plan_for(&c_near, &in).nr0 == 2, "q4_0 on a card without a row: 2 rows", std::to_string(ggml_metal_matvec_plan_for(&c_near, &in).nr0));
        expect(ggml_metal_matvec_plan_for(&c_exact, &in).nr0 == 8, "the same multiply on the Vega 64: 8 rows", std::to_string(ggml_metal_matvec_plan_for(&c_exact, &in).nr0));
    }

    printf("matvec plan: %d checks, %d failed\n", checked, failed);
    return failed == 0 ? 0 : 1;
}
