// The launch numbers of a flash attention route on one card: data/attention.tsv [launch], whose header lines say
// what each number is. A number is 0 when the route does not read it.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum fa_launch_route { FA_LAUNCH_SCALAR, FA_LAUNCH_VEC, FA_LAUNCH_MATRIX, FA_LAUNCH_PREPASS, FA_LAUNCH_COUNT };

struct fa_launch {
    int32_t simdgroups;
    int32_t wide_simdgroups;
    int32_t wide_head;
    int32_t workgroups;
    int32_t small_queries;
    int32_t threads;
};

// The numbers of a route on the card named card with a wave of wave threads: the default row, then the row of the
// wave, then the card's own row at that wave. card null and wave 0 give the default row alone: what holds on every card.
struct fa_launch fa_launch_pick(int route, const char * card, int32_t wave);

#ifdef __cplusplus
}
#endif
