// Batch route table checks, host only: no GPU, no model.
// usage: test_metal_batch_routes            the decide function against the two predicates it replaced, given the row's cells
//        test_metal_batch_routes control    built with tiled_from 16 on the q4_K rows: fixed cases must land on the expected routes
#include "matmul/batch-route.h"

#include <cstdio>
#include <cstring>

// The two predicates as they stood in dispatch-ext.cpp and dispatch-matmul.cpp before the table. A card with no row
// passes small_min 2, kq_ext_min 4, tiled_from 0, which is those predicates unchanged; a row passes its cells.
static enum mv_batch_route old_route(
        const struct mv_format_row * f, int small_min, int kq_ext_min, int tiled_from, bool matrix, bool vendor_ok, bool f32, bool t0, bool t1,
        int64_t k, int64_t m, int64_t n, bool cols_takes) {
    const enum ggml_type type = f ? f->type : GGML_TYPE_COUNT;
    const bool tiled_can = !t0 && !t1 && !matrix && m >= 64 && f != nullptr && f->tiled && f32 && vendor_ok;
    if (tiled_from > 0 && tiled_can && n >= tiled_from) {
        return MV_BATCH_TILED;
    }
    const bool ext = f32 && (k % 128 == 0) &&
        (
         ((f != nullptr && f->ext == MV_EXT_SMALL) && (n >= small_min && (n <= 8 || (!matrix && type != GGML_TYPE_F16 && type != GGML_TYPE_MXFP4)))) ||
         ((f != nullptr && f->ext == MV_EXT_KQUANT) && (n >= kq_ext_min && (n <= 8 || !matrix)) && (n > 4 || !cols_takes))
        );
    if (ext) {
        return MV_BATCH_EXT;
    }
    const bool tiled = !t0 && !t1 && !matrix && m >= 64 && n > 8 && f != nullptr && f->tiled && f32;
    return tiled && vendor_ok ? MV_BATCH_TILED : MV_BATCH_NONE;
}

static enum mv_batch_route new_route(
        const struct mv_format_row * f, const struct mv_batch_row * row, bool matrix, bool vendor_ok, bool f32, bool t0, bool t1,
        int64_t k, int64_t m, int64_t n, bool cols_takes) {
    struct mv_batch_in in = {};
    in.fmt = f;
    in.ext_from = row ? row->ext_from : 0;
    in.tiled_from = row ? row->tiled_from : 0;
    in.has_matrix_hw = matrix;
    in.tiled_vendor_ok = vendor_ok;
    in.act_f32 = f32;
    in.trans0 = t0;
    in.trans1 = t1;
    in.k = k;
    in.m = m;
    in.n = n;
    in.cols_takes = cols_takes;
    return ggml_metal_batch_route_decide(&in);
}

int main(int argc, char ** argv) {
    const bool control = argc == 2 && std::strcmp(argv[1], "control") == 0;
    if (argc != 1 && !control) {
        std::printf("usage: %s [control]\n", argv[0]);
        return 2;
    }
    int n_rows = 0;
    const struct mv_batch_row * rows = ggml_metal_batch_rows_get(&n_rows);
    const int n_fmt = (int) (sizeof(mv_format_rows) / sizeof(mv_format_rows[0]));
    const int64_t ks[] = { 96, 128, 1024 };
    const int64_t ms[] = { 32, 64, 2560 };

    long cases = 0, diffs = 0, moved = 0;
    // card -1 = a card with no row; then the card of every row
    for (int card = -1; card < n_rows; ++card) {
        for (int fi = -1; fi < n_fmt; ++fi) {
            const struct mv_format_row * f = fi < 0 ? nullptr : &mv_format_rows[fi];
            const struct mv_batch_row * row = (card < 0 || f == nullptr) ? nullptr
                : ggml_metal_batch_row_in(rows, n_rows, rows[card].pci_vendor, rows[card].pci_device, f->type);
            const int kq_ext_min = row && row->ext_from > 0 ? row->ext_from : 4;
            const int small_min  = row && row->ext_from > 0 ? row->ext_from : 2;
            const int tiled_from = row ? row->tiled_from : 0;
            for (int bits = 0; bits < 64; ++bits) {
                const bool matrix = bits & 1, vendor_ok = bits & 2, f32 = bits & 4, t0 = bits & 8, t1 = bits & 16, cols = bits & 32;
                for (int64_t k : ks) {
                    for (int64_t m : ms) {
                        for (int64_t n = 1; n <= 600; ++n) {
                            const bool cols_takes = cols && n <= 4;
                            const enum mv_batch_route was = old_route(f, small_min, kq_ext_min, tiled_from, matrix, vendor_ok, f32, t0, t1, k, m, n, cols_takes);
                            const enum mv_batch_route is  = new_route(f, row, matrix, vendor_ok, f32, t0, t1, k, m, n, cols_takes);
                            cases++;
                            if (row != nullptr && is != old_route(f, 2, 4, 0, matrix, vendor_ok, f32, t0, t1, k, m, n, cols_takes)) {
                                moved++;
                            }
                            if (was != is) {
                                if (diffs < 5 && !control) {
                                    std::printf("FAIL card %d type %d matrix %d vendor %d f32 %d t %d%d k %lld m %lld n %lld cols %d: was %d, is %d\n",
                                        card, f ? (int) f->type : -1, matrix, vendor_ok, f32, t0, t1, (long long) k, (long long) m, (long long) n, cols_takes, was, is);
                                }
                                diffs++;
                            }
                        }
                    }
                }
            }
        }
    }

    if (!control) {
        std::printf("MAIN batch routes: %d rows, %ld cases, %ld moved by a row, %ld failures\n", n_rows, cases, moved, diffs);
        return diffs == 0 ? 0 : 1;
    }

    // control: the q4_K row carries tiled_from 16
    int bad = 0;
    const struct mv_format_row * q4k = ggml_metal_mv_format(GGML_TYPE_Q4_K);
    const struct mv_batch_row * row = nullptr;
    for (int i = 0; i < n_rows; ++i) {
        if (rows[i].type == GGML_TYPE_Q4_K && rows[i].tiled_from == 16) {
            row = &rows[i];
        }
    }
    if (row == nullptr) {
        std::printf("CONTROL tiled-from-16: the table has no q4_K row with tiled_from 16 -- NOT CAUGHT\n");
        return 1;
    }
    bad += new_route(q4k, row, false, true, true, false, false, 2560, 2560, 15, false) != MV_BATCH_EXT;
    bad += new_route(q4k, row, false, true, true, false, false, 2560, 2560, row->ext_from > 2 ? row->ext_from - 1 : 1, false) != MV_BATCH_NONE; // below ext_from: one column at a time
    bad += new_route(q4k, row, false, true, true, false, false, 2560, 2560, 16, false) != MV_BATCH_TILED;
    bad += new_route(q4k, row, false, true, true, false, false, 2560, 32, 16, false) != MV_BATCH_EXT;   // no full tile: stays
    bad += new_route(q4k, row, true, true, true, false, false, 2560, 2560, 16, false) != MV_BATCH_NONE; // matrix hardware: untouched
    bad += new_route(q4k, nullptr, false, true, true, false, false, 2560, 2560, 16, false) != MV_BATCH_EXT; // a card with no row
    bad += new_route(q4k, nullptr, false, true, true, false, false, 2560, 2560, 600, false) != MV_BATCH_EXT;
    std::printf("CONTROL tiled-from-16 (expected to move routes): %ld moved by a row, %d wrong routes%s\n",
        moved, bad, (moved == 0 || bad != 0) ? " -- NOT CAUGHT" : "");
    return (moved > 0 && bad == 0) ? 1 : 3;
}
