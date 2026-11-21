// Host check of data/attention.tsv [launch]: fa_launch_pick over the generated rows.
// Without FA_LAUNCH_FIXTURE: the shipped numbers of every route, on a card without a row at a wave of 32 and of 64 and
// with no card at all, equal the numbers written here (the ones the routes held in code). With it: a table of test
// rows, where the row of the wave and the card's row must apply in that order.
#include <cstdio>

#include "flash-attn/launch.h"

static int failures = 0;

static void want(const char * what, struct fa_launch got, int sg, int wide_sg, int wide_head, int wg, int small, int threads) {
    if (got.simdgroups != sg || got.wide_simdgroups != wide_sg || got.wide_head != wide_head || got.workgroups != wg || got.small_queries != small || got.threads != threads) {
        printf("FAIL %s: got %d %d %d %d %d %d, wanted %d %d %d %d %d %d\n", what, got.simdgroups, got.wide_simdgroups, got.wide_head,
               got.workgroups, got.small_queries, got.threads, sg, wide_sg, wide_head, wg, small, threads);
        failures++;
    }
}

int main() {
#ifdef FA_LAUNCH_FIXTURE
    want("vec, wave row",            fa_launch_pick(FA_LAUNCH_VEC, "another card", 64), 2, 0, 0, 32, 20, 0);
    want("vec, card row",            fa_launch_pick(FA_LAUNCH_VEC, "Test Card A", 64), 1, 0, 0, 32, 20, 0);
    want("vec, card at another wave", fa_launch_pick(FA_LAUNCH_VEC, "Test Card A", 32), 4, 0, 0, 32, 20, 0);
    want("vec, no card",             fa_launch_pick(FA_LAUNCH_VEC, nullptr, 0), 4, 0, 0, 32, 20, 0);
    want("scalar, card row",         fa_launch_pick(FA_LAUNCH_SCALAR, "Test Card A", 64), 8, 0, 0, 0, 0, 0);
    printf("attention launch rows (fixture): %d failures\n", failures);
#else
    static const struct { const char * card; int wave; } cards[] = { { nullptr, 0 }, { "a card without a row", 32 }, { "a card without a row", 64 } };
    for (const auto & c : cards) {
        want("scalar",  fa_launch_pick(FA_LAUNCH_SCALAR,  c.card, c.wave), 4, 0, 0, 0, 0, 0);
        want("vec",     fa_launch_pick(FA_LAUNCH_VEC,     c.card, c.wave), 4, 0, 0, 32, 20, 0);
        want("matrix",  fa_launch_pick(FA_LAUNCH_MATRIX,  c.card, c.wave), 4, 8, 512, 0, 0, 0);
        want("prepass", fa_launch_pick(FA_LAUNCH_PREPASS, c.card, c.wave), 0, 0, 0, 0, 0, 32);
    }
    want("a route outside the table", fa_launch_pick(FA_LAUNCH_COUNT, nullptr, 0), 0, 0, 0, 0, 0, 0);
    printf("attention launch rows (shipped): 4 routes, %d failures\n", failures);
#endif
    return failures == 0 ? 0 : 1;
}
