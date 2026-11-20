#pragma once

// The flash attention route table. The rows are data/attention.tsv, turned into routes-table.h at configure time.
enum fa_route { FA_ROUTE_INTEL_TILED, FA_ROUTE_SCALAR, FA_ROUTE_MATRIX, FA_ROUTE_VEC };

// Each cell: 1 = must hold, 0 = must not hold, -1 = any. head: the head size the route takes, 0 = any.
// A route function returns 1 after encoding the op; the router returns that value.
struct fa_route_row {
    enum fa_route route;
    int has_matrix; // the card has simdgroup matrices
    int intel;      // the vendor is Intel
    int width32;    // the card's width is 32
    int small;      // small batch
    int kv_same;    // K and V have the same type
    int fp8;        // K is f8_e4m3
    int head;       // head size
};

#include "routes-table.h"

static inline bool fa_route_cell(int cell, bool fact) {
    return cell < 0 || (cell == 1) == fact;
}

static inline bool fa_route_row_matches(const struct fa_route_row * row, bool has_matrix, bool intel, bool width32, bool small, bool kv_same, bool fp8, int head) {
    return (row->head == 0 || row->head == head) && fa_route_cell(row->has_matrix, has_matrix) && fa_route_cell(row->intel, intel) && fa_route_cell(row->width32, width32) &&
           fa_route_cell(row->small, small) && fa_route_cell(row->kv_same, kv_same) && fa_route_cell(row->fp8, fp8);
}
