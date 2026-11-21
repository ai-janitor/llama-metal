#include "device/launch-plan.h"

#include <string.h>

#include "launch-rows.h" // generated; this file is its only reader, so a changed number rebuilds this file alone

static void launch_cell(int32_t * out, int32_t cell, int32_t wave) {
    if (cell == -2) {
        *out = wave;
    } else if (cell != -1) {
        *out = cell;
    }
}

struct ggml_metal_launch ggml_metal_launch_pick(int id, const char * card, int32_t wave) {
    return ggml_metal_launch_pick_rows(ggml_metal_launch_rows, id, card, wave);
}

struct ggml_metal_launch ggml_metal_launch_pick_rows(const struct ggml_metal_launch_row * rows, int id, const char * card, int32_t wave) {
    struct ggml_metal_launch out = { 0, 0, 0, 0, 0, 0 };
    if (id < 0) {
        return out;
    }
    // pass 0: the default row; pass 1: the row of the wave; pass 2: the card's own row at that wave
    for (int pass = 0; pass < 3; pass++) {
        for (const struct ggml_metal_launch_row * row = rows; row->id >= 0; row++) {
            if (row->id != id) {
                continue;
            }
            const bool any_card = row->card[0] == 0;
            const bool applies = pass == 0 ? (any_card && row->wave == -1)
                               : pass == 1 ? (any_card && row->wave == wave)
                               : (!any_card && card != nullptr && row->wave == wave && strcmp(row->card, card) == 0);
            if (!applies) {
                continue;
            }
            launch_cell(&out.start,     row->start,     wave);
            launch_cell(&out.work_min,  row->work_min,  wave);
            launch_cell(&out.cap,       row->cap,       wave);
            launch_cell(&out.nsg,       row->nsg,       wave);
            launch_cell(&out.units,     row->units,     wave);
            launch_cell(&out.mem_units, row->mem_units, wave);
        }
    }
    return out;
}
