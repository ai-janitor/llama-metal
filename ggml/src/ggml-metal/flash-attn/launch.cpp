#include "flash-attn/launch.h"

#include <string.h>

struct fa_launch_row {
    int32_t      route;
    const char * card;
    int32_t      wave;
    int32_t      simdgroups;
    int32_t      wide_simdgroups;
    int32_t      wide_head;
    int32_t      workgroups;
    int32_t      small_queries;
    int32_t      threads;
};

#include "fa-launch-table.h" // generated; this file is its only reader, so a changed number rebuilds this file alone

static void fa_launch_cell(int32_t * out, int32_t cell) {
    if (cell != -1) {
        *out = cell;
    }
}

struct fa_launch fa_launch_pick(int route, const char * card, int32_t wave) {
    struct fa_launch out = { 0, 0, 0, 0, 0, 0 };
    // pass 0: the default row; pass 1: the row of the wave; pass 2: the card's own row at that wave
    for (int pass = 0; pass < 3; pass++) {
        for (const struct fa_launch_row * row = fa_launch_rows; row->route >= 0; row++) {
            if (row->route != route) {
                continue;
            }
            const bool any_card = row->card[0] == 0;
            const bool applies = pass == 0 ? (any_card && row->wave == -1)
                               : pass == 1 ? (any_card && row->wave == wave)
                               : (!any_card && card != nullptr && row->wave == wave && strcmp(row->card, card) == 0);
            if (!applies) {
                continue;
            }
            fa_launch_cell(&out.simdgroups,      row->simdgroups);
            fa_launch_cell(&out.wide_simdgroups, row->wide_simdgroups);
            fa_launch_cell(&out.wide_head,       row->wide_head);
            fa_launch_cell(&out.workgroups,      row->workgroups);
            fa_launch_cell(&out.small_queries,   row->small_queries);
            fa_launch_cell(&out.threads,         row->threads);
        }
    }
    return out;
}
