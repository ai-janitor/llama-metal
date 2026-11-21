#include "tiled-plan.h"

#include "ggml.h"

#include "tiled-plan-table.h"

#include <cstring>

// tiled_compiled, tiled_compiled_id: the plans the kernel is compiled for = the rows of data/matmul.tsv [tiles]
// (generated header); row 0 is the default plan.
#include "tiled-compiled-table.h"

// The largest sums block and row cache a plan may ask of the kernel (a bound of the rule; the kernel sizes both
// from the plan). matmul/gen-tiles.cmake holds the same numbers.
static const int32_t tiled_sums_max = 16; // float2 sums per thread
static const int32_t tiled_rows_max = 8;  // rows held per thread

// The bounds of each field, in field order: warp, threads, bm, bn, bk, wm, wn, wmiter, tm, tn, pad (bk is 32: the
// block size the weight loaders are written for). The same bounds are in matmul/gen-tiles.cmake.
static const int32_t tiled_min[GGML_METAL_TILED_PLAN_N] = { 1,    1,    1,    1,    32, 1,    1,    1, 2, 1,  0 };
static const int32_t tiled_max[GGML_METAL_TILED_PLAN_N] = { 1024, 1024, 1024, 1024, 32, 1024, 1024, 8, 8, 16, 64 };

int ggml_metal_tiled_compiled_count(void) {
    return (int) (sizeof(tiled_compiled) / sizeof(tiled_compiled[0]));
}

const int32_t * ggml_metal_tiled_compiled_row(int i) {
    return i >= 0 && i < ggml_metal_tiled_compiled_count() ? tiled_compiled[i] : nullptr;
}

const char * ggml_metal_tiled_compiled_id(int i) {
    return i >= 0 && i < ggml_metal_tiled_compiled_count() ? tiled_compiled_id[i] : nullptr;
}

int ggml_metal_tiled_compiled_find(const char * id) {
    for (int i = 0; id != nullptr && i < ggml_metal_tiled_compiled_count(); ++i) {
        if (std::strcmp(tiled_compiled_id[i], id) == 0) {
            return i;
        }
    }
    return -1;
}

void ggml_metal_tiled_tune_load(struct ggml_metal_card * card, const char * device_name) {
    card->n_tiled_tune = 0;
    if (device_name == nullptr || device_name[0] == '\0') {
        return;
    }
    for (const struct ggml_metal_tiled_tune_row * row = tiled_tune_rows; row->plan >= 0; ++row) {
        // A tuned plan is this card's only while the card reports the shape the row was measured with: a name is not
        // one chip, and a width that was never probed is not known.
        const bool shape = card->wave_probed != 0 && card->wave == row->wave && card->group_mem == row->group_bytes;
        if (shape && std::strcmp(row->card, device_name) == 0 && card->n_tiled_tune < GGML_METAL_CARD_TILED_MAX) {
            card->tiled_tune[card->n_tiled_tune++] = { row->weight, row->plan, row->cols };
        }
    }
}

struct ggml_metal_tiled_plan ggml_metal_tiled_plan_check(const struct ggml_metal_card * card, const int32_t v[GGML_METAL_TILED_PLAN_N], uint32_t thread_limit) {
    struct ggml_metal_tiled_plan lay = {};
    lay.row = -1;
    if (card == nullptr || v == nullptr) {
        return lay;
    }
    lay.warp    = v[0];
    lay.threads = v[1];
    lay.bm      = v[2];
    lay.bn      = v[3];
    lay.bk      = v[4];
    lay.wm      = v[5];
    lay.wn      = v[6];
    lay.wmiter  = v[7];
    lay.tm      = v[8];
    lay.tn      = v[9];
    lay.pad     = v[10];
    // Every field inside its bound before any product: the products below stay far inside int32, and every field
    // stays a small positive number. The same bounds are in matmul/gen-tiles.cmake.
    for (int i = 0; i < GGML_METAL_TILED_PLAN_N; ++i) {
        if (v[i] < tiled_min[i] || v[i] > tiled_max[i]) {
            return lay;
        }
    }
    lay.group_mem_bytes = (lay.bm + lay.bn) * (lay.bk / 2 + lay.pad) * 8;

    // The kernel's own arithmetic: warp tiles cover the group tile with one warp each, every thread has whole sums.
    const int32_t per_pass = lay.warp * lay.tm * lay.tn * lay.wmiter;
    const int32_t wniter   = (lay.wm * lay.wn) / per_pass;
    bool shape_ok = lay.tm % 2 == 0 && lay.threads % lay.warp == 0 &&
        lay.bm % lay.wm == 0 && lay.bn % lay.wn == 0 && (lay.bm / lay.wm) * (lay.bn / lay.wn) == lay.threads / lay.warp &&
        (lay.wm * lay.wn) % per_pass == 0 && wniter >= 1 && lay.wm % lay.wmiter == 0 && lay.wn % wniter == 0 &&
        (lay.threads * 2) % lay.bk == 0;
    if (shape_ok) {
        // The kernel divides a thread's index in its warp by (wm / wmiter) / tm: that must be a whole number from 1,
        // and with the column side it must give every thread of the warp its own block of sums.
        const int32_t wsubm = lay.wm / lay.wmiter;
        const int32_t wsubn = lay.wn / wniter;
        const bool thread_map = wsubm % lay.tm == 0 && wsubm / lay.tm >= 1 && wsubn % lay.tn == 0 && wsubn / lay.tn >= 1 &&
            (wsubm / lay.tm) * (wsubn / lay.tn) == lay.warp;
        // The fill writes whole strides of rows (threads * 2 / bk rows at a time): both tiles must be multiples of it.
        const int32_t load_stride = lay.threads * 2 / lay.bk;
        const bool fill_fits = load_stride >= 1 && lay.bm % load_stride == 0 && lay.bn % load_stride == 0;
        shape_ok = thread_map && fill_fits;
    }
    const bool fits_kernel = shape_ok && lay.wmiter * lay.tm * wniter * lay.tn / 2 <= tiled_sums_max && lay.wmiter * lay.tm <= tiled_rows_max;
    // A group is whole waves, its buffers fit the card's group memory, and the built pipeline allows its threads.
    const bool whole_waves = card->wave > 0 && (uint32_t) lay.threads % card->wave == 0;
    const bool fits_mem    = (uint32_t) lay.group_mem_bytes <= card->group_mem;
    const bool fits_limit  = thread_limit == 0 || (uint32_t) lay.threads <= thread_limit;
    lay.ok = fits_kernel && whole_waves && fits_mem && fits_limit ? 1 : 0;
    return lay;
}

// The row of the compiled table the card selects for this weight type and column count. Each entry that fits has a
// level: 3 = for the type at this count, 2 = for the type at every count, 1 = for every type at this count, 0 = for
// every type at every count; the highest level wins, else row 0 (default). An entry for one count fits that count
// only. An entry outside the table (it cannot come from the generated rows) counts as none.
static int32_t tiled_selected_row(const struct ggml_metal_card * card, int32_t weight, int64_t cols) {
    int32_t row = 0;
    int best = -1;
    for (int e = 0; e < card->n_tiled_tune && e < GGML_METAL_CARD_TILED_MAX; ++e) {
        const struct ggml_metal_card_tiled_entry & t = card->tiled_tune[e];
        const bool for_type  = t.weight >= 0;
        const bool for_count = t.cols > 0;
        if ((for_type && t.weight != weight) || (for_count && (int64_t) t.cols != cols) || t.plan < 0 || t.plan >= ggml_metal_tiled_compiled_count()) {
            continue;
        }
        const int level = (for_type ? 2 : 0) + (for_count ? 1 : 0);
        if (level >= best) {
            best = level;
            row = t.plan;
        }
    }
    return row;
}

struct ggml_metal_tiled_plan ggml_metal_tiled_plan_for(const struct ggml_metal_card * card, int32_t weight, int64_t cols, uint32_t thread_limit) {
    if (card == nullptr) {
        struct ggml_metal_tiled_plan none = {};
        none.row = -1;
        return none;
    }
    const int32_t row = tiled_selected_row(card, weight, cols);
    struct ggml_metal_tiled_plan lay = ggml_metal_tiled_plan_check(card, tiled_compiled[row], thread_limit);
    lay.row = row;
    return lay;
}

struct ggml_metal_tiled_plan ggml_metal_tiled_plan_to_run(const struct ggml_metal_card * card, int32_t weight, int64_t cols, uint32_t thread_limit, int32_t forced_row) {
    struct ggml_metal_tiled_plan lay = {};
    lay.row = -1;
    if (card == nullptr) {
        return lay;
    }
    // The plan is chosen without the limit; the limit is then checked on the plan chosen.
    const int32_t row = forced_row >= 0 && forced_row < ggml_metal_tiled_compiled_count() ? forced_row : tiled_selected_row(card, weight, cols);
    lay = ggml_metal_tiled_plan_check(card, tiled_compiled[row], 0);
    lay.row = row;
    if (!lay.ok && row != 0) {
        lay = ggml_metal_tiled_plan_check(card, tiled_compiled[0], 0);
        lay.row = 0;
        lay.from_default = 1;
    }
    if (lay.ok && thread_limit != 0 && (uint32_t) lay.threads > thread_limit) {
        lay.ok = 0;
    }
    return lay;
}

void ggml_metal_tiled_plan_values(const struct ggml_metal_tiled_plan * lay, int32_t out[GGML_METAL_TILED_PLAN_N]) {
    const int32_t v[GGML_METAL_TILED_PLAN_N] = {
        lay->warp, lay->threads, lay->bm, lay->bn, lay->bk, lay->wm, lay->wn, lay->wmiter, lay->tm, lay->tn, lay->pad,
    };
    for (int i = 0; i < GGML_METAL_TILED_PLAN_N; ++i) {
        out[i] = v[i];
    }
}
