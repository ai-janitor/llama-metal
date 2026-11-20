// Tiled matmul work plan rule, host only: no GPU, no model.
// usage: test_metal_tiled_plan TUNING_TSV            TUNING_TSV = the matmul.tsv the binary was generated from: for
//                                                    each card and weight type the rule must return the plan that
//                                                    file's tuned rows of [tiles] name (the row for the type, else the any row, else default)
//        test_metal_tiled_plan TUNING_TSV control    the same, built from the check's own control tables (tests/Makefile):
//                                                    the Vega 64's two control rows must each win where they apply, and no
//                                                    part of the test may be skipped
// The parts that need a second compiled plan, a plan with more group memory or a plan with more threads than the
// default find such a row in the compiled table; with none (a table of the default alone) the part is skipped and counted.
#include "matmul/tiled-plan.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int failures = 0;

static void expect(bool cond, const char * what) {
    if (!cond) {
        std::printf("FAIL %s\n", what);
        failures++;
    }
}

struct card_facts {
    const char * name; // the Metal device name ("" = a made-up card of the test)
    uint32_t wave, group_mem;
};

static struct ggml_metal_card object_of(const struct card_facts & c) {
    struct ggml_metal_card card = {};
    card.wave = c.wave;
    card.wave_probed = 1;
    card.group_mem = c.group_mem;
    ggml_metal_tiled_tune_load(&card, c.name);
    return card;
}

static bool same_values(const struct ggml_metal_tiled_plan & lay, const int32_t * want) {
    int32_t v[GGML_METAL_TILED_PLAN_N];
    ggml_metal_tiled_plan_values(&lay, v);
    for (int i = 0; i < GGML_METAL_TILED_PLAN_N; ++i) {
        if (v[i] != want[i]) {
            return false;
        }
    }
    return true;
}

// The cells of the tuned rows of section [tiles] that name a plan (card, weight, plan are the first three cells).
static std::vector<std::vector<std::string>> tuning_rows(const char * path, bool * opened) {
    std::vector<std::vector<std::string>> rows;
    std::ifstream in(path);
    *opened = in.is_open();
    std::string line;
    bool header = true;
    bool tiles = false;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line[0] == '[') {
            tiles = line == "[tiles]";
            header = true;
            continue;
        }
        if (!tiles) {
            continue;
        }
        if (header) {
            header = false;
            continue;
        }
        std::vector<std::string> cells;
        size_t at = 0;
        for (;;) {
            const size_t tab = line.find('\t', at);
            cells.push_back(line.substr(at, tab == std::string::npos ? std::string::npos : tab - at));
            if (tab == std::string::npos) {
                break;
            }
            at = tab + 1;
        }
        if (cells.size() >= 4 && cells[3] == "tuned" && cells[2] != "-") {
            rows.push_back(cells);
        }
    }
    return rows;
}

// The plan id the table names for this card and weight type: its row for the type, else its any row, else the default plan's (row 0).
static std::string named_plan(const std::vector<std::vector<std::string>> & rows, const struct card_facts & c, const char * weight) {
    const std::string card = c.name;
    std::string plan = ggml_metal_tiled_compiled_id(0);
    for (int pass = 0; pass < 2; ++pass) {
        for (const std::vector<std::string> & r : rows) {
            if (r.size() >= 3 && r[0] == card && r[1] == (pass == 0 ? "any" : weight)) {
                plan = r[2];
            }
        }
    }
    return plan;
}

int main(int argc, char ** argv) {
    const bool control = argc == 3 && std::strcmp(argv[2], "control") == 0;
    if (argc != 2 && !control) {
        std::printf("usage: %s TUNING_TSV [control]\n", argv[0]);
        return 2;
    }
    bool opened = false;
    const std::vector<std::vector<std::string>> rows = tuning_rows(argv[1], &opened);
    if (!opened) {
        std::printf("cannot read %s\n", argv[1]);
        return 2;
    }
    int skipped = 0;
    // The Metal device name (the "found device:" log line of each card), wave width and group memory as the device
    // reports them (measured 2026-11-18)
    const struct card_facts cards[] = {
        { "AMD Radeon HD - FirePro D700", 64, 32768 },
        { "AMD Radeon Pro Vega 64",       64, 65536 },
        { "AMD Radeon Pro 5500M",         32, 65536 },
    };
    struct weight_type { int32_t type; const char * name; };
    const struct weight_type types[] = { { GGML_TYPE_Q4_K, "q4_K" }, { GGML_TYPE_Q8_0, "q8_0" } };
    // The plan the kernel carried as literals before the rule = row 0, the default plan.
    const int32_t today[GGML_METAL_TILED_PLAN_N] = { 32, 128, 64, 64, 32, 32, 32, 2, 4, 2, 1 };
    const int n_plans = ggml_metal_tiled_compiled_count();
    expect(n_plans >= 1 && std::memcmp(ggml_metal_tiled_compiled_row(0), today, sizeof(today)) == 0, "row 0 of the compiled table is the default plan");
    expect(ggml_metal_tiled_compiled_find("no_such_plan") == -1, "an id that is not in the table is not found");
    int vega_rows[2] = { -1, -1 };
    for (const struct card_facts & c : cards) {
        const struct ggml_metal_card card = object_of(c);
        for (int ti = 0; ti < 2; ++ti) {
            const int32_t type = types[ti].type;
            const struct ggml_metal_tiled_plan lay = ggml_metal_tiled_plan_for(&card, type, 0);
            const std::string named = named_plan(rows, c, types[ti].name);
            const int want_row = ggml_metal_tiled_compiled_find(named.c_str());
            if (want_row < 0 || lay.row != want_row) {
                std::printf("%s %s: the table names plan %s (row %d), the rule returned row %d\n", c.name, types[ti].name, named.c_str(), want_row, lay.row);
            }
            expect(want_row >= 0 && lay.row == want_row, "plan row: the row the tuning table names for this card and type, else the default");
            expect(want_row >= 0 && same_values(lay, ggml_metal_tiled_compiled_row(want_row)), "plan values are the row's values");
            expect(lay.group_mem_bytes == (lay.bm + lay.bn) * (16 + lay.pad) * 8, "group memory follows the plan");
            // Whether the card's facts admit the named plan is the card's matter (a row may name a plan one card cannot
            // run: the default then runs); the limit checks are on the plan returned.
            if (lay.ok == 1) {
                expect(ggml_metal_tiled_plan_for(&card, type, (uint32_t) lay.threads).ok == 1, "a pipeline limit equal to the plan's threads takes the plan");
                expect(ggml_metal_tiled_plan_for(&card, type, (uint32_t) lay.threads - 1).ok == 0, "a pipeline limit one below the plan's threads refuses it");
            }
            expect(ggml_metal_tiled_plan_to_run(&card, type, 0, -1).ok == 1, "a card of ours gets a plan to run");
            if (std::strcmp(c.name, "AMD Radeon Pro Vega 64") == 0) {
                vega_rows[ti] = lay.row;
            }
        }
    }
    if (control) {
        expect(vega_rows[0] > 0 && vega_rows[1] > 0 && vega_rows[0] != vega_rows[1], "control rows: the row for q4_K and the any row each win where they apply");
    }
    // A tuned plan is the card's only with the shape its row was measured with (control rows: wave 64, 65536 bytes).
    if (control) {
        const struct ggml_metal_card other_wave = object_of({ "AMD Radeon Pro Vega 64", 32, 65536 });
        const struct ggml_metal_card other_mem  = object_of({ "AMD Radeon Pro Vega 64", 64, 32768 });
        struct ggml_metal_card unprobed = {};
        unprobed.wave = 64;
        unprobed.group_mem = 65536;
        ggml_metal_tiled_tune_load(&unprobed, "AMD Radeon Pro Vega 64");
        expect(other_wave.n_tiled_tune == 0, "a card with the row's name and another wave width gets no tuned plan");
        expect(other_mem.n_tiled_tune == 0, "a card with the row's name and another group memory gets no tuned plan");
        expect(unprobed.n_tiled_tune == 0, "a card whose wave width was not probed gets no tuned plan");
        expect(object_of({ "AMD Radeon Pro Vega 64", 64, 65536 }).n_tiled_tune == 2, "the card with the row's shape gets both control rows");
    }
    const struct ggml_metal_card roomy = object_of({ "", 32, 65536 });
    expect(same_values(ggml_metal_tiled_plan_for(&roomy, GGML_TYPE_Q4_K, 0), today), "the default plan is today's layout");
    // Refusals: each fact the rule reads can refuse.
    const struct ggml_metal_card small_mem = object_of({ "", 64, 16384 });
    const struct ggml_metal_card odd_wave  = object_of({ "", 48, 65536 });
    expect(ggml_metal_tiled_plan_for(&small_mem, GGML_TYPE_Q4_K, 0).ok == 0, "group memory below the buffers refuses");
    expect(ggml_metal_tiled_plan_for(&odd_wave, GGML_TYPE_Q4_K, 0).ok == 0, "a group that is not whole waves refuses");
    expect(ggml_metal_tiled_plan_for(nullptr, GGML_TYPE_Q4_K, 0).ok == 0, "no object refuses");
    expect(ggml_metal_tiled_plan_check(&roomy, nullptr, 0).ok == 0, "no values refuse");
    // The checks on values the kernel cannot run: today's layout with one field changed, each must be refused.
    struct refusal { int field; int32_t value; const char * what; };
    const struct refusal refusals[] = {
        { 0, 64,    "warp 64 with the default 32 x 32 warp tile: the warp tiles do not cover the group tile" },
        { 8, 6,     "tm 6: (wm / wmiter) / tm is not a whole number" },
        { 1, 96,    "threads 96: the tiles are not multiples of the fill stride (and the warp count does not cover the tile)" },
        { 2, 60,    "bm 60: not a multiple of the fill stride" },
        { 2, 70000, "bm out of its bound" },
        { 10, 65,   "pad out of its bound" },
        { 4, 16,    "bk other than 32" },
        { 0, 0,     "warp 0" },
    };
    for (const struct refusal & r : refusals) {
        int32_t v[GGML_METAL_TILED_PLAN_N];
        std::memcpy(v, today, sizeof(v));
        v[r.field] = r.value;
        expect(ggml_metal_tiled_plan_check(&roomy, v, 0).ok == 0, r.what);
    }
    const int32_t ones[GGML_METAL_TILED_PLAN_N] = { 32, 128, 64, 64, 32, 1, 1, 2, 4, 2, 1 };
    expect(ggml_metal_tiled_plan_check(&roomy, ones, 0).ok == 0, "wm = wn = 1 (no column pass fits) is refused");
    // Values the kernel can run that differ from the default are accepted.
    const int32_t one_pass[GGML_METAL_TILED_PLAN_N]  = { 32, 128, 64, 64, 32, 32, 32, 1, 8, 2, 1 };
    const int32_t wide_pass[GGML_METAL_TILED_PLAN_N] = { 32, 128, 64, 64, 32, 32, 32, 1, 4, 2, 1 };
    const int32_t padded[GGML_METAL_TILED_PLAN_N]    = { 32, 128, 64, 64, 32, 32, 32, 2, 4, 2, 4 };
    expect(ggml_metal_tiled_plan_check(&roomy, one_pass, 0).ok == 1, "wmiter 1 with tm 8 (8 rows x 4 columns of sums in one pass) can run");
    expect(ggml_metal_tiled_plan_check(&roomy, wide_pass, 0).ok == 1, "wmiter 1 alone (4 rows x 8 columns of sums) can run");
    const struct ggml_metal_tiled_plan pad4 = ggml_metal_tiled_plan_check(&roomy, padded, 0);
    expect(pad4.ok == 1 && pad4.row == -1 && pad4.group_mem_bytes == 128 * 20 * 8, "pad 4 can run; values checked outside the table have no row");
    // Rows of the compiled table the parts below need (-1 = the table has none): any second plan, one whose buffers
    // need more group memory than the default's, one with more threads than the default's.
    const struct ggml_metal_tiled_plan base = ggml_metal_tiled_plan_check(&roomy, ggml_metal_tiled_compiled_row(0), 0);
    const int other = n_plans > 1 ? 1 : -1;
    int more_mem = -1;
    int more_threads = -1;
    for (int r = n_plans - 1; r >= 1; --r) {
        const struct ggml_metal_tiled_plan p = ggml_metal_tiled_plan_check(&roomy, ggml_metal_tiled_compiled_row(r), 0);
        if (p.ok == 1 && p.group_mem_bytes > base.group_mem_bytes) {
            more_mem = r;
        }
        if (p.ok == 1 && p.threads > base.threads) {
            more_threads = r;
        }
    }
    // Selection: an entry for one type leaves the other types on the default; an entry outside the table counts as none.
    struct ggml_metal_card typed = object_of({ "", 32, 65536 });
    if (other > 0) {
        typed.tiled_tune[typed.n_tiled_tune++] = { GGML_TYPE_Q8_0, other };
        typed.tiled_tune[typed.n_tiled_tune++] = { GGML_TYPE_Q5_K, 9999 };
        expect(ggml_metal_tiled_plan_for(&typed, GGML_TYPE_Q8_0, 0).row == other, "an entry for a type applies to that type");
        expect(ggml_metal_tiled_plan_for(&typed, GGML_TYPE_Q4_K, 0).row == 0, "an entry for a type leaves another type on the default");
        expect(ggml_metal_tiled_plan_for(&typed, GGML_TYPE_Q5_K, 0).row == 0, "an entry naming a row outside the table counts as none");
        const struct ggml_metal_tiled_plan kept = ggml_metal_tiled_plan_to_run(&typed, GGML_TYPE_Q8_0, 0, -1);
        expect(kept.ok == 1 && kept.from_default == 0 && kept.row == other, "a selected plan the card can run is kept");
        const struct ggml_metal_tiled_plan forced = ggml_metal_tiled_plan_to_run(&roomy, GGML_TYPE_Q4_K, 0, other);
        expect(forced.ok == 1 && forced.row == other && same_values(forced, ggml_metal_tiled_compiled_row(other)), "a forced row replaces the card's selection");
    } else {
        std::printf("SKIP selection by entry, forced row: the compiled table holds the default alone\n");
        skipped++;
    }
    // The plan to run: on a card whose group memory holds the default's buffers and no more, a selected plan that
    // needs more is replaced by the default, marked.
    if (more_mem > 0) {
        struct ggml_metal_card tight = object_of({ "", 32, (uint32_t) base.group_mem_bytes });
        tight.tiled_tune[tight.n_tiled_tune++] = { -1, more_mem };
        const struct ggml_metal_tiled_plan fell = ggml_metal_tiled_plan_to_run(&tight, GGML_TYPE_Q4_K, 0, -1);
        expect(ggml_metal_tiled_plan_for(&tight, GGML_TYPE_Q4_K, 0).ok == 0, "a plan that needs more group memory than the card has is refused");
        expect(fell.ok == 1 && fell.from_default == 1 && fell.row == 0 && same_values(fell, today), "a selected plan the card cannot run is replaced by the default plan, marked");
        const struct ggml_metal_tiled_plan forced_tight = ggml_metal_tiled_plan_to_run(&tight, GGML_TYPE_Q4_K, 0, more_mem);
        expect(forced_tight.from_default == 1, "a forced row the card cannot run comes back marked as replaced (the dispatch aborts on the mark)");
    } else {
        std::printf("SKIP replacement by the default plan: no compiled plan needs more group memory than the default\n");
        skipped++;
    }
    // The pipeline's thread limit is checked on the plan chosen.
    if (more_threads > 0) {
        struct ggml_metal_card wide = object_of({ "", 32, 65536 });
        wide.tiled_tune[wide.n_tiled_tune++] = { GGML_TYPE_Q8_0, more_threads };
        const uint32_t wide_threads = (uint32_t) ggml_metal_tiled_compiled_row(more_threads)[1];
        expect(ggml_metal_tiled_plan_to_run(&wide, GGML_TYPE_Q8_0, wide_threads, -1).ok == 1, "a limit equal to the selected plan's threads takes it");
        expect(ggml_metal_tiled_plan_to_run(&wide, GGML_TYPE_Q8_0, wide_threads - 1, -1).ok == 0, "a limit one below the selected plan's threads refuses it for that pipeline");
        expect(ggml_metal_tiled_plan_to_run(&wide, GGML_TYPE_Q8_0, (uint32_t) base.threads, 0).ok == 1, "the default plan then runs on its own pipeline at that limit");
    } else {
        std::printf("SKIP thread limit on a selected plan: no compiled plan has more threads than the default\n");
        skipped++;
    }
    expect(ggml_metal_tiled_plan_to_run(&roomy, GGML_TYPE_Q4_K, (uint32_t) base.threads, -1).ok == 1, "a pipeline limit equal to the default plan's threads takes it");
    expect(ggml_metal_tiled_plan_to_run(&roomy, GGML_TYPE_Q4_K, (uint32_t) base.threads - 1, -1).ok == 0, "a pipeline limit one below the default plan's threads refuses it");
    // Every compiled plan is a plan the rule's checks accept on a roomy card.
    for (int r = 0; r < n_plans; ++r) {
        const struct ggml_metal_tiled_plan got = ggml_metal_tiled_plan_check(&roomy, ggml_metal_tiled_compiled_row(r), 0);
        if (got.ok != 1) {
            std::printf("compiled plan %s is refused by the rule's checks\n", ggml_metal_tiled_compiled_id(r));
        }
        expect(got.ok == 1, "a row of the compiled table is a plan the checks accept");
    }
    if (control) {
        expect(skipped == 0, "the control tables leave no part skipped");
    }

    std::printf("tiled plan%s: %d cards, %d compiled plans, %d tuning rows, %d parts skipped, %d failures\n", control ? " (control tables)" : "", (int) (sizeof(cards) / sizeof(cards[0])), n_plans, (int) rows.size(), skipped, failures);
    return failures == 0 ? 0 : 1;
}
