// Host check of a section [launches] (data/device.tsv, or data/matmul.tsv with LAUNCH_MV): the generated ids and rows,
// and the pick function.
// Without LAUNCH_FIXTURE it prints, for every launch of the table it was built from, the numbers a card without a
// row gets at a wave of 32 and of 64 and the numbers of the three cards that have device rows; the caller compares
// the lines with the recorded ones. With LAUNCH_FIXTURE it checks the order of the rows on a table of test rows.
#include <cstdio>
#include <cstring>

#ifdef LAUNCH_MV // data/matmul.tsv [launches]
#include "matmul/mv-launch.h"
#include "mv-launch-rows.h"
#define ggml_metal_launch_names ggml_metal_mv_launch_names
#define ggml_metal_launch_pick  ggml_metal_mv_launch_pick
#define GGML_METAL_LAUNCH_COUNT GGML_METAL_MV_LAUNCH_COUNT
#else            // data/device.tsv [launches]
#include "device/launch-plan.h"
#include "launch-rows.h"
#endif

static int failures = 0;

#ifdef LAUNCH_FIXTURE
static void want(const char * what, struct ggml_metal_launch got, int start, int work_min, int cap, int nsg, int units, int mem_units) {
    if (got.start != start || got.work_min != work_min || got.cap != cap || got.nsg != nsg || got.units != units || got.mem_units != mem_units) {
        printf("FAIL %s: got %d %d %d %d %d %d, wanted %d %d %d %d %d %d\n", what,
               got.start, got.work_min, got.cap, got.nsg, got.units, got.mem_units, start, work_min, cap, nsg, units, mem_units);
        failures++;
    }
}
#endif

int main() {
    int n = 0;
    while (ggml_metal_launch_names[n] != nullptr) {
        n++;
    }
    if (n != GGML_METAL_LAUNCH_COUNT) {
        printf("FAIL %d names for %d ids\n", n, (int) GGML_METAL_LAUNCH_COUNT);
        return 1;
    }
#ifdef LAUNCH_MV
#ifndef EXT_ROWS_W8
#define EXT_ROWS_W8 0 // a K width of 16 on a kernel compiled 8 wide: no launch
#endif
    // rows per threadgroup of the small-batch matvec: width, K width, simdgroups, rows
    static const int ext[][4] = { { 32, 16, 2, 4 }, { 64, 16, 2, 8 }, { 32, 4, 2, 16 }, { 8, 16, 2, EXT_ROWS_W8 }, { 8, 8, 2, 2 }, { 16, 16, 2, 2 }, { 32, 3, 2, 0 }, { 32, 0, 2, 0 } };
    for (const auto & e : ext) {
        const int got = ggml_metal_mv_ext_rows(e[0], e[1], e[2]);
        if (got != e[3]) {
            fprintf(stderr, "FAIL small-batch rows at width %d, K width %d, %d simdgroups: got %d, wanted %d\n", e[0], e[1], e[2], got, e[3]);
            failures++;
        }
    }
#endif
#ifdef LAUNCH_FIXTURE
    want("default row, wave 32",          ggml_metal_launch_pick(GGML_METAL_LAUNCH_T_REDUCE, "another card", 32), 32, 256, 1024, 0, 0, 128);
    want("wave row over the default",     ggml_metal_launch_pick(GGML_METAL_LAUNCH_T_REDUCE, "another card", 64), 64, 256, 1024, 0, 0, 64);
    want("card row over the wave row",    ggml_metal_launch_pick(GGML_METAL_LAUNCH_T_REDUCE, "Test Card A", 64), 64, 512, 1024, 0, 0, 64);
    want("card row at another wave",      ggml_metal_launch_pick(GGML_METAL_LAUNCH_T_REDUCE, "Test Card A", 32), 32, 256, 1024, 0, 0, 128);
    want("no card name",                  ggml_metal_launch_pick(GGML_METAL_LAUNCH_T_REDUCE, nullptr, 64), 64, 256, 1024, 0, 0, 64);
    want("a launch without start",        ggml_metal_launch_pick(GGML_METAL_LAUNCH_T_PACK, "Test Card A", 64), 0, 0, 256, 4, 8, 0);
    want("an id outside the table",       ggml_metal_launch_pick(GGML_METAL_LAUNCH_COUNT, "Test Card A", 64), 0, 0, 0, 0, 0, 0);
    want("a negative id",                 ggml_metal_launch_pick(-1, "Test Card A", 64), 0, 0, 0, 0, 0, 0);
    printf("launch rows (fixture): %d launches, %d failures\n", n, failures);
#else
    static const struct { const char * card; int wave; } cards[] = {
        { "", 32 }, { "", 64 },
        { "AMD Radeon HD - FirePro D700", 64 }, { "AMD Radeon Pro 5500M", 32 }, { "AMD Radeon Pro Vega 64", 64 },
    };
    for (int id = 0; id < n; id++) {
        for (const auto & c : cards) {
            const struct ggml_metal_launch l = ggml_metal_launch_pick(id, c.card, c.wave);
            printf("%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", ggml_metal_launch_names[id], c.card[0] ? c.card : "no row", c.wave,
                   l.start, l.work_min, l.cap, l.nsg, l.units, l.mem_units);
        }
    }
#endif
    return failures == 0 ? 0 : 1;
}
