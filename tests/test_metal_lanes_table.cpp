// Lane wiring table checks, host only: no GPU, no model.
// usage: test_metal_lanes_table <path to ggml/src/ggml-metal> [<reason> <test>: the texts the matmul mm-tiled row must hold]
#include "lanes/select.h"
#include "lanes-table.h" // the generated capability names (enum ggml_metal_lane_cap)

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

static void fail(const char * what, const struct ggml_metal_lane_row & row) {
    std::printf("FAIL %s: lane %s family %d route %d weight %d\n", what, row.lane, row.family, row.route, row.weight);
    failures++;
}

// A row that may be picked must leave the values alone, unless it is a route upstream ships that way.
static int check_defaults(const std::vector<ggml_metal_lane_row> & rows, bool print) {
    int bad = 0;
    for (const auto & row : rows) {
        if (!row.on || row.cls == GGML_METAL_LANE_CLASS_EXACT) continue;
        if (row.origin == GGML_METAL_LANE_ORIGIN_UPSTREAM) {
            if (print) std::printf("known: %s on %s narrows by default (upstream)\n", row.lane, row.file);
            continue;
        }
        if (print) fail("default row is not exact", row);
        bad++;
    }
    return bad;
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 4) {
        std::printf("usage: %s <ggml-metal source dir> [<reason> <test>]\n", argv[0]);
        return 2;
    }
    int n = 0, ncap = 0;
    const struct ggml_metal_lane_row * table = ggml_metal_lane_rows_get(&n);
    const struct ggml_metal_lane_cap_row * caps = ggml_metal_lane_cap_rows_get(&ncap);
    std::vector<ggml_metal_lane_row> rows(table, table + n);

    // 1. defaults
    check_defaults(rows, true);

    // 2. every row names a file that exists
    for (const auto & row : rows) {
        const std::string path = std::string(argv[1]) + "/" + row.file;
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) {
            fail("file missing", row);
        } else {
            std::fclose(f);
        }
    }

    // 3. every card with rows, and a card with none, gets an exact or upstream row for every request
    std::vector<uint32_t> masks = { 0u };
    for (int i = 0; i < ncap; ++i) {
        masks.push_back(ggml_metal_lane_ok_mask(caps[i].card));
    }
    if (ggml_metal_lane_ok_mask("Apple M1") != 0 || ggml_metal_lane_ok_mask("") != 0 || ggml_metal_lane_ok_mask(nullptr) != 0) {
        std::printf("FAIL a card with no capability rows has lanes\n");
        failures++;
    }
    // 3a. the shipped capabilities, written here apart from the table: the name each card reports to Metal and how many
    // capabilities are reached and correct on it, and
    // which capabilities those are. A name one character off, in another case or with a space gets none.
    const uint32_t pk = 1u << GGML_METAL_LANE_CAP_PK_MUL_F16;
    static const struct { const char * card; int usable; uint32_t set; } expected[] = {
        { "AMD Radeon Pro 5500M", 1, pk }, { "Intel(R) UHD Graphics 630", 1, pk }, { "AMD Radeon Pro Vega 64", 0, 0 }, { "AMD Radeon HD - FirePro D700", 0, 0 },
        { "AMD Radeon Pro 5500", 0, 0 }, { "amd radeon pro 5500m", 0, 0 }, { "AMD Radeon Pro 5500M ", 0, 0 }, { "Intel(R) UHD Graphics 63", 0, 0 },
    };
    for (const auto & e : expected) {
        const uint32_t mask = ggml_metal_lane_ok_mask(e.card);
        int bits = 0;
        for (int b = 0; b < 32; ++b) bits += (mask >> b) & 1u;
        if (bits != e.usable) {
            std::printf("FAIL card '%s': %d usable capabilities, expected %d\n", e.card, bits, e.usable);
            failures++;
        }
        if (mask != e.set) {
            std::printf("FAIL card '%s': capability set 0x%x, expected 0x%x\n", e.card, mask, e.set);
            failures++;
        }
    }
    // 3b. every shipped capability row, written here apart from the table, in its order: card, capability, in_hardware, reached, correct
    static const struct { const char * card; int capability; int in_hardware, reached, correct; } pinned[] = {
        { "AMD Radeon Pro 5500M", GGML_METAL_LANE_CAP_PK_MUL_F16, 1, 1, 1 },
        { "AMD Radeon Pro 5500M", GGML_METAL_LANE_CAP_DOT2_F32_F16, 1, 0, -1 },
        { "AMD Radeon Pro 5500M", GGML_METAL_LANE_CAP_DOT4_I32_I8, 1, -1, 0 },
        { "AMD Radeon Pro 5500M", GGML_METAL_LANE_CAP_DOT8_I32_I4, 1, -1, -1 },
        { "AMD Radeon Pro 5500M", GGML_METAL_LANE_CAP_MATRIX8X8, 0, 0, 0 },
        { "AMD Radeon Pro Vega 64", GGML_METAL_LANE_CAP_PK_MUL_F16, 1, -1, -1 },
        { "AMD Radeon Pro Vega 64", GGML_METAL_LANE_CAP_DOT2_F32_F16, 0, 0, 0 },
        { "AMD Radeon Pro Vega 64", GGML_METAL_LANE_CAP_DOT4_I32_I8, 0, 0, 0 },
        { "AMD Radeon Pro Vega 64", GGML_METAL_LANE_CAP_DOT8_I32_I4, 0, 0, 0 },
        { "AMD Radeon Pro Vega 64", GGML_METAL_LANE_CAP_MATRIX8X8, 0, 0, 0 },
        { "AMD Radeon HD - FirePro D700", GGML_METAL_LANE_CAP_PK_MUL_F16, 0, -1, -1 },
        { "AMD Radeon HD - FirePro D700", GGML_METAL_LANE_CAP_DOT2_F32_F16, 0, 0, 0 },
        { "AMD Radeon HD - FirePro D700", GGML_METAL_LANE_CAP_DOT4_I32_I8, 0, 0, 0 },
        { "AMD Radeon HD - FirePro D700", GGML_METAL_LANE_CAP_DOT8_I32_I4, 0, 0, 0 },
        { "AMD Radeon HD - FirePro D700", GGML_METAL_LANE_CAP_MATRIX8X8, 0, 0, 0 },
        { "Intel(R) UHD Graphics 630", GGML_METAL_LANE_CAP_PK_MUL_F16, 1, 1, 1 },
        { "Intel(R) UHD Graphics 630", GGML_METAL_LANE_CAP_DOT4_I32_I8, 0, -1, 1 },
    };
    const int npinned = (int) (sizeof(pinned) / sizeof(pinned[0]));
    if (ncap != npinned) {
        std::printf("FAIL pinned rows: the table has %d capability rows, %d are written here\n", ncap, npinned);
        failures++;
    }
    for (int i = 0; i < ncap && i < npinned; ++i) {
        const auto & p = pinned[i];
        if (std::strcmp(caps[i].card, p.card) != 0 || caps[i].capability != p.capability || caps[i].in_hardware != p.in_hardware || caps[i].reached != p.reached || caps[i].correct != p.correct) {
            std::printf("FAIL pinned row %d: '%s' capability %d is %d %d %d, written here: '%s' capability %d is %d %d %d\n", i + 1,
                    caps[i].card, caps[i].capability, caps[i].in_hardware, caps[i].reached, caps[i].correct, p.card, p.capability, p.in_hardware, p.reached, p.correct);
            failures++;
        }
    }
    // 3c. with two more arguments: the reason and test texts of the matmul mm-tiled row, as the table holds them
    if (argc == 4) {
        bool found = false;
        for (const auto & row : rows) {
            if (row.family != GGML_METAL_LANE_FAMILY_MATMUL || row.route != GGML_METAL_LANE_ROUTE_MM_TILED) continue;
            found = true;
            if (std::strcmp(row.reason, argv[2]) != 0 || std::strcmp(row.test, argv[3]) != 0) {
                std::printf("FAIL texts of the mm-tiled row: reason '%s' test '%s'\n", row.reason, row.test);
                failures++;
            }
        }
        if (!found) {
            std::printf("FAIL texts: the table has no matmul mm-tiled row\n");
            failures++;
        }
    }
    // every capability row belongs to one of the four cards named above
    for (int i = 0; i < ncap; ++i) {
        bool known = false;
        for (int e = 0; e < 4; ++e) known = known || std::strcmp(caps[i].card, expected[e].card) == 0;
        if (!known) {
            std::printf("FAIL capability row of card '%s' is not a card of this test\n", caps[i].card);
            failures++;
        }
    }
    const enum ggml_type acts[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 };
    int picks = 0;
    for (uint32_t mask : masks) {
        for (const auto & group : rows) {
            for (int w = 0; w < GGML_TYPE_COUNT; ++w) {
                for (enum ggml_type act : acts) {
                    const struct ggml_metal_lane_row * row = ggml_metal_select_lane_in(
                            rows.data(), n, mask, group.family, group.route, (enum ggml_type) w, act);
                    if (!row) {
                        fail("no row for a request", group);
                        continue;
                    }
                    picks++;
                    const int step = ggml_metal_lane_holds(row->operand, act, row->holds);
                    if (step == GGML_METAL_LANE_HOLDS_NARROWER && row->origin != GGML_METAL_LANE_ORIGIN_UPSTREAM) {
                        fail("selector picked a lane narrower than the tensor", *row);
                    }
                    if (row->cls != GGML_METAL_LANE_CLASS_EXACT && row->origin != GGML_METAL_LANE_ORIGIN_UPSTREAM) {
                        fail("selector picked a lane that is not exact", *row);
                    }
                    if (mask == 0 && row->capability >= 0 && row->origin != GGML_METAL_LANE_ORIGIN_UPSTREAM) {
                        fail("a card with no capability rows got a lane that needs one", *row);
                    }
                }
            }
        }
    }

    // 4. controls on a copy: a narrowing row of ours (an f16 operand for an f32 tensor), on, ahead of the base rows
    std::vector<ggml_metal_lane_row> forced;
    for (const auto & row : rows) {
        if (row.cls == GGML_METAL_LANE_CLASS_EXACT && row.origin == GGML_METAL_LANE_ORIGIN_OURS && row.weight == GGML_TYPE_COUNT) {
            ggml_metal_lane_row bad = row;
            bad.cls = GGML_METAL_LANE_CLASS_NARROWS;
            bad.capability = 0;
            bad.on = 1;
            bad.activation = GGML_TYPE_F32;
            bad.operand = GGML_TYPE_F16;
            bad.holds = GGML_METAL_LANE_HOLDS_NARROWER;
            forced.push_back(bad);
            break;
        }
    }
    forced.insert(forced.end(), rows.begin(), rows.end());
    if (forced.size() != rows.size() + 1) {
        std::printf("FAIL control: the table has no base row of ours to copy\n");
        failures++;
    } else {
        const int at = 0;
        const int nf = (int) forced.size();
        const uint32_t has = 1u << forced[at].capability;
        const int fam = forced[at].family, rt = forced[at].route;
        const enum ggml_type wt = GGML_TYPE_Q4_K;
        // 4a. the defaults check sees it
        if (check_defaults(forced, false) != 1) {
            std::printf("FAIL control: a narrowing default row was not caught\n");
            failures++;
        }
        // 4b. the type comparison drops it for the wider tensor: the pick is still the f32 row
        int step = 0;
        const struct ggml_metal_lane_row * got = ggml_metal_negotiate_lane_in(forced.data(), nf, has, fam, rt, wt, GGML_TYPE_F32, &step);
        if (!got || got == &forced[at] || got->operand != GGML_TYPE_F32 || step != GGML_METAL_LANE_HOLDS_NATIVE) {
            std::printf("FAIL control: a lane narrower than an f32 tensor was not dropped\n");
            failures++;
        }
        // 4c. the same lane wired for an f16 tensor is native and wins over the wider f32 row
        std::vector<ggml_metal_lane_row> typed = forced;
        typed[at].activation = GGML_TYPE_F16;
        typed[at].holds = GGML_METAL_LANE_HOLDS_NATIVE;
        got = ggml_metal_negotiate_lane_in(typed.data(), nf, has, fam, rt, wt, GGML_TYPE_F16, &step);
        if (got != &typed[at] || step != GGML_METAL_LANE_HOLDS_NATIVE) {
            std::printf("FAIL control: a lane native to an f16 tensor was not picked\n");
            failures++;
        }
        // 4d. and not on a card without the capability
        got = ggml_metal_negotiate_lane_in(typed.data(), nf, 0u, fam, rt, wt, GGML_TYPE_F16, &step);
        if (!got || got == &typed[at] || step != GGML_METAL_LANE_HOLDS_WIDER) {
            std::printf("FAIL control: a card without the capability did not land on the wider row\n");
            failures++;
        }
        // 4e. gap: with every wide row of the route gone, the answer is no row, never the narrower lane
        std::vector<ggml_metal_lane_row> gap;
        for (int i = 0; i < nf; ++i) {
            if (i == at || forced[i].family != fam || forced[i].route != rt) gap.push_back(forced[i]);
        }
        if (ggml_metal_negotiate_lane_in(gap.data(), (int) gap.size(), has, fam, rt, wt, GGML_TYPE_F32, nullptr) != nullptr) {
            std::printf("FAIL control: a route with no wide path did not come back as a gap\n");
            failures++;
        }
    }

    std::printf("lanes table: %d rows, %d capability rows, %d picks checked, %d failures\n", n, ncap, picks, failures);
    return failures == 0 ? 0 : 1;
}
