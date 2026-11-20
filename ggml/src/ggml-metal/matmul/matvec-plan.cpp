#include "matvec-plan.h"

#include "ggml.h"
#include "ggml-metal.h"
#include "formats.h"

#include "matvec-rows.h"

#include <stdio.h>
#include <string.h>

// The tree reduce halves the threads of a threadgroup step by step: it sums every partial only when their count
// (simdgroups x wave) is a power of two.
static bool mv_tree_fits(int32_t nsg, uint32_t wave) {
    const int64_t n = (int64_t) nsg * (int64_t) wave;
    return n > 0 && (n & (n - 1)) == 0;
}

// The cache rule of the basic quants and the grouped K-quant launch: rows per simdgroup from the share of the
// compute unit's L1 that one activation column takes. The numbers are the card's cells.
static int32_t mv_cache_rows(const struct ggml_metal_card * card, const struct ggml_metal_matvec_in * in) {
    if (!in->model_known || card->l1_per_cu == 0) {
        return (int32_t) card->mv_rows_unknown;
    }
    const float share = (float) in->act_bytes / (float) card->l1_per_cu;
    return (int32_t) (share < card->l1_share_cut ? card->mv_rows_below : card->mv_rows_from);
}

// The card reports the wave and threadgroup memory a row was measured with.
static bool mv_facts_are(const struct ggml_metal_card * card, uint32_t wave, uint32_t group_bytes) {
    return card->wave_probed != 0 && card->wave == wave && card->group_mem == group_bytes;
}

static void mv_apply(struct ggml_metal_card * card, const struct ggml_metal_matvec_default_row * row) {
    if (row == nullptr) {
        return;
    }
    if (row->mv_reduce         >= 0) card->mv_reduce         = (uint32_t) row->mv_reduce;
    if (row->f_scalar_w64      >= 0) card->f_scalar_w64      = (uint32_t) row->f_scalar_w64;
    if (row->mv_group          >= 0) card->mv_group          = (uint32_t) row->mv_group;
    if (row->q8_load_bytes     >= 0) card->q8_load_bytes     = (uint32_t) row->q8_load_bytes;
    if (row->q8_rows8_min      >= 0) card->q8_rows8_min      = (uint32_t) row->q8_rows8_min;
    if (row->q8_act_load_bytes >= 0) card->q8_act_load_bytes = (uint32_t) row->q8_act_load_bytes;
    if (row->mv_group_red      >= 0) card->mv_group_red      = (uint32_t) row->mv_group_red;
    if (row->mv_sum            >= 0) card->mv_sum            = (uint32_t) row->mv_sum;
    if (row->l1_share_cut      >= 0) card->l1_share_cut      = row->l1_share_cut;
    if (row->rows_below        >= 0) card->mv_rows_below     = (uint32_t) row->rows_below;
    if (row->rows_from         >= 0) card->mv_rows_from      = (uint32_t) row->rows_from;
    if (row->rows_unknown      >= 0) card->mv_rows_unknown   = (uint32_t) row->rows_unknown;
    if (row->f_nsg_max         >= 0) card->f_nsg_max         = (uint32_t) row->f_nsg_max;
    if (row->f_k_per_sg        >= 0) card->f_k_per_sg        = (uint32_t) row->f_k_per_sg;
}

static void mv_settings_load(struct ggml_metal_card * card, const char * device_name) {
    card->q8_rows8_min = 0; // never, unless a row says from how many rows
    mv_apply(card, ggml_metal_matvec_any_row_of(0));
    if (card->wave_probed != 0) {
        mv_apply(card, ggml_metal_matvec_any_row_of((int32_t) card->wave));
        const struct ggml_metal_matvec_default_row * own = ggml_metal_matvec_default_row_of(device_name);
        if (own != nullptr && mv_facts_are(card, (uint32_t) own->wave, (uint32_t) own->group_bytes)) {
            mv_apply(card, own);
        }
    }
}

static void mv_rows_load(struct ggml_metal_card * card, const char * device_name, int rows_on, int prep_on, int cols_on) {
    card->mv_rows   = nullptr;
    card->n_mv_rows = 0;
    card->kq_prep   = 0;
    card->kq_cols   = 0;
    if (!rows_on) {
        return;
    }
    card->mv_rows = ggml_metal_mv_rows_of(device_name, &card->n_mv_rows);
    for (uint32_t i = 0; i < card->n_mv_rows; i++) {
        const struct ggml_device_mv_row * row = &card->mv_rows[i];
        if (!mv_facts_are(card, row->wave, row->group_bytes)) {
            continue;
        }
        if (row->prep == 1 && prep_on) {
            card->kq_prep = 1;
        }
        if (row->cols > 1 && cols_on) {
            card->kq_cols = 1;
        }
    }
}

void ggml_metal_matvec_tune_load(struct ggml_metal_card * card, const char * device_name, int rows_on, int prep_on, int cols_on, int group_store_on) {
    if (card == nullptr) {
        return;
    }
    mv_settings_load(card, device_name);
    if (!group_store_on) {
        card->mv_group_red = 0;
    }
    mv_rows_load(card, device_name, rows_on, prep_on, cols_on);
}

int ggml_metal_matvec_row_fits(const struct ggml_metal_card * card, const struct ggml_device_mv_row * row, int32_t kind, int32_t weight, int64_t k, int64_t m, int32_t cols) {
    if (card == nullptr || row == nullptr) {
        return 0;
    }
    if (!mv_facts_are(card, row->wave, row->group_bytes)) {
        return 0;
    }
    const bool shape = (row->ne00 < 0 && row->ne01 < 0) || (row->ne00 == k && row->ne01 == m); // -1 -1 = every shape
    return (kind < 0 || row->kind == kind) && row->type == weight && shape && (cols < 0 || row->cols == cols) ? 1 : 0;
}

int32_t ggml_metal_matvec_shape_row(const struct ggml_metal_card * card, int32_t kind, int32_t weight, int64_t k, int64_t m, int32_t cols, int32_t prep_available) {
    if (card == nullptr || (cols > 1 && card->kq_cols != 1)) {
        return -1;
    }
    for (uint32_t i = 0; i < card->n_mv_rows; i++) {
        const struct ggml_device_mv_row * row = &card->mv_rows[i];
        if ((row->form & 4) != 0 && card->wave != 64) {
            continue; // paired halves are the two halves of a 64-thread wave
        }
        if (ggml_metal_matvec_row_fits(card, row, kind, weight, k, m, cols) && (row->prep != 1 || prep_available)) {
            return (int32_t) i;
        }
    }
    return -1;
}

int ggml_metal_matvec_plan_check(const struct ggml_metal_card * card, const struct ggml_metal_matvec_in * in, const struct ggml_metal_matvec_plan * plan, const char ** why) {
    const char * reason = nullptr;
    const struct mv_format_row * fmt = in != nullptr ? ggml_metal_mv_format((enum ggml_type) in->weight) : nullptr;
    const bool halves = plan != nullptr && (plan->form & 4) != 0;
    if (card == nullptr || in == nullptr || plan == nullptr || !plan->ok) {
        reason = "no plan";
    } else if (plan->cols < 1 || plan->cols > 4) {
        reason = "columns are 1 to 4";
    } else if (plan->form < 0 || plan->form == 5 || plan->form > 6) {
        reason = "form is one of 0, 1, 2, 3, 4, 6";
    } else if (plan->cols == 1 && plan->form != 0) {
        reason = "one column has no loop form";
    } else if (plan->reduce == 2 && !mv_tree_fits(plan->nsg, card->wave)) {
        reason = "the tree reduce needs simdgroups x wave to be a power of two";
    } else if (plan->nr0_set && (fmt == nullptr || !fmt->nr0_override)) {
        reason = "the kernel of this weight type does not take a row count";
    } else if (plan->nr0_set && plan->nr0 != 1 && plan->nr0 != 2 && plan->nr0 != 4 && plan->nr0 != 8) {
        reason = "rows per simdgroup are 1, 2, 4 or 8";
    } else if (plan->nr0_set && plan->nr0 == 1 && !fmt->shape_rows) {
        reason = "the basic quant kernels have instances of 2, 4 and 8 rows only";
    } else if (plan->shape_row >= 0 && plan->cols == 1 && (plan->nr0 * plan->nsg <= 0 || in->m % (plan->nr0 * plan->nsg) != 0)) {
        // the kernels load every weight row of a threadgroup before the output bound
        reason = "with a shape row m is a whole number of nr0 x nsg rows";
    } else if (plan->prep && in->weight != GGML_TYPE_Q4_K) {
        reason = "prepared headers exist for q4_K only";
    } else if (plan->prep && !in->prep_available) {
        reason = "prepared headers are not available for this weight";
    } else if (in->kind == 1 && (fmt == nullptr || !fmt->grouped)) {
        reason = "this weight type has no grouped launch";
    } else if (plan->cols > 1 && (in->kind != 0 || fmt == nullptr || !fmt->shape_rows || in->weight == GGML_TYPE_Q5_K)) {
        reason = "2 to 4 columns need a single matvec of q4_K or q6_K";
    } else if (halves && (plan->cols < 2 || plan->cols > 3 || plan->nr0 < 2)) {
        reason = "paired halves need 2 or 3 columns and 2, 4 or 8 rows";
    } else if (halves && card->wave != 64) {
        reason = "paired halves need wave width 64";
    } else if (plan->cols > 1 && (plan->nr0 * plan->nsg <= 0 || in->m % (plan->nr0 * plan->nsg) != 0)) {
        reason = "with 2 to 4 columns m is a whole number of nr0 x nsg rows";
    }
    if (why != nullptr) {
        *why = reason;
    }
    return reason == nullptr ? 1 : 0;
}

struct ggml_metal_matvec_plan ggml_metal_matvec_plan_for(const struct ggml_metal_card * card, const struct ggml_metal_matvec_in * in) {
    struct ggml_metal_matvec_plan plan = {};
    plan.cols      = 1;
    plan.shape_row = -1;
    if (card == nullptr || in == nullptr) {
        return plan;
    }
    const enum ggml_type type = (enum ggml_type) in->weight;
    const struct mv_format_row * fmt = ggml_metal_mv_format(type);

    // 1. the weight type: simdgroups and rows
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        if (fmt == nullptr || fmt->short_k <= 0 || fmt->nr0 <= 0) {
            return plan; // the type's row gives no dense geometry
        }
        if (ggml_metal_mv_float_short(fmt, in->k)) {
            plan.nsg = 1;
            plan.nr0 = fmt->short_rows;
        } else {
            if (card->f_k_per_sg == 0 || card->f_nsg_max == 0) {
                return plan; // the card's settings were never loaded
            }
            const int32_t by_k = (int32_t) ((in->k + card->f_k_per_sg - 1) / card->f_k_per_sg);
            plan.nsg = by_k < (int32_t) card->f_nsg_max ? by_k : (int32_t) card->f_nsg_max;
            plan.nr0 = fmt->nr0;
        }
        plan.rows_per_threadgroup = 1;
    } else if (fmt != nullptr && fmt->nsg > 0) {
        plan.nsg = fmt->nsg;
        plan.nr0 = fmt->nr0;
        plan.rows_per_threadgroup = fmt->rows_per_threadgroup ? 1 : 0;
    } else {
        return plan;
    }
    plan.ok = 1;

    // 2. the card: how a row sum is reduced, load widths, grouping. A grouped launch always sums through shared memory.
    const bool shared = in->kind == 1 || card->mv_sum == 1;
    plan.reduce            = !shared ? 0 : (card->mv_reduce == 1 ? 2 : 1);
    if (plan.reduce == 2 && !mv_tree_fits(plan.nsg, card->wave)) {
        plan.reduce = 1; // 3 dense simdgroups: the serial sum reads every partial
    }
    plan.q8_load_pref      = (int32_t) card->q8_load_bytes;
    plan.q8_act_pref       = (int32_t) card->q8_act_load_bytes;
    plan.q8_group          = card->mv_group == 1 ? 1 : 0;
    plan.q8_rows8_min      = (int32_t) card->q8_rows8_min;
    plan.group_store_first = card->mv_group_red == 1 ? 1 : 0;

    // 3. rows per simdgroup, for the types whose kernel takes them: the cache rule (basic quants, grouped launch),
    //    or the external rows file, then the card's shape row (K-quants with shape rows).
    const bool shape_type = fmt != nullptr && fmt->shape_rows;
    if (in->kind == 1 || (fmt != nullptr && fmt->nr0_override && !shape_type)) {
        plan.nr0     = mv_cache_rows(card, in);
        plan.nr0_set = 1;
        if (fmt != nullptr && fmt->nr0 > 0 && (plan.nr0 * plan.nsg <= 0 || in->m % (plan.nr0 * plan.nsg) != 0)) {
            // the kernels load every weight row of a threadgroup before the output bound: the cache rule's rows only
            // where m is whole threadgroups of them, else the rows the kernels are compiled for
            plan.nr0 = fmt->nr0;
        }
    }
    if (!shape_type) {
        return plan;
    }
    const bool want_cols = in->kind == 0 && in->cols > 1;
    if (!want_cols && in->kind == 0 && in->rows_file_nr0 > 0) {
        plan.nr0     = in->rows_file_nr0;
        plan.nr0_set = 1;
        return plan;
    }
    const int32_t at = in->use_rows ? ggml_metal_matvec_shape_row(card, in->kind, in->weight, in->k, in->m, want_cols ? in->cols : 1, in->prep_available) : -1;
    if (at < 0) {
        if (in->kind == 0 && in->rows_file_nr0 > 0) {
            plan.nr0     = in->rows_file_nr0;
            plan.nr0_set = 1;
        }
        return plan;
    }
    const struct ggml_device_mv_row * row = &card->mv_rows[at];
    struct ggml_metal_matvec_plan with_row = plan;
    with_row.nr0       = row->nr0;
    with_row.nr0_set   = 1;
    with_row.prep      = row->prep == 1 ? 1 : 0;
    with_row.cols      = row->cols;
    with_row.form      = row->cols > 1 ? row->form : 0;
    with_row.shape_row = at;
    if (!ggml_metal_matvec_plan_check(card, in, &with_row, nullptr)) {
        plan.from_default = 1;
        if (in->kind == 0 && in->rows_file_nr0 > 0) {
            plan.nr0     = in->rows_file_nr0;
            plan.nr0_set = 1;
        }
        return plan;
    }
    return with_row;
}

int ggml_metal_matvec_plan_fallback(const struct ggml_metal_matvec_plan * column, const struct ggml_metal_matvec_plan * single, int64_t m, int32_t pipeline_nr0, int32_t pipeline_nsg, int32_t pipeline_wave, struct ggml_metal_matvec_plan * out) {
    *out = *column;
    if (column->cols <= 1) {
        return GGML_METAL_MATVEC_FALLBACK_NONE;
    }
    const int32_t rows_per_tg = pipeline_nr0 * pipeline_nsg;
    if (rows_per_tg <= 0 || m % rows_per_tg != 0) {
        // the column kernel loads every weight row of a threadgroup before the output bound
        *out = *single;
        return GGML_METAL_MATVEC_FALLBACK_TAIL;
    }
    if ((column->form & 4) != 0 && pipeline_wave != 64) {
        // paired halves split the wave at thread 32
        out->form &= 3;
        return GGML_METAL_MATVEC_FALLBACK_WIDTH;
    }
    return GGML_METAL_MATVEC_FALLBACK_NONE;
}

// One cell of the hook's text: the characters up to the next colon, or the end. Returns the cell's length and moves *at
// past the colon; -1 when the text ended before the cell.
static int mv_forced_cell(const char ** at, char * cell, int cap) {
    const char * p = *at;
    if (p == nullptr) {
        return -1;
    }
    int n = 0;
    while (*p != 0 && *p != ':') {
        if (n + 1 < cap) {
            cell[n] = *p;
        }
        n++;
        p++;
    }
    cell[n < cap ? n : cap - 1] = 0;
    *at = *p == ':' ? p + 1 : nullptr;
    return n;
}

// A whole number of decimal digits in a cell, or -1.
static int64_t mv_forced_number(const char * cell) {
    if (cell[0] == 0) {
        return -1;
    }
    int64_t v = 0;
    for (const char * p = cell; *p != 0; p++) {
        if (*p < '0' || *p > '9' || v > 100000000000ll) {
            return -1;
        }
        v = v * 10 + (*p - '0');
    }
    return v;
}

int ggml_metal_matvec_forced_parse(const char * text, struct ggml_metal_matvec_forced * out, char weight_word[16], const char ** why) {
    static const char * const form_words[] = { "a", "w", "ap", "wp", "ha", "", "hap" };
    const char * reason = nullptr;
    const char * at = text;
    char cell[8][24] = {};
    int n = 0;
    while (n < 8 && mv_forced_cell(&at, cell[n], (int) sizeof(cell[n])) >= 0) {
        n++;
    }
    struct ggml_metal_matvec_forced f = {};
    if (text == nullptr || out == nullptr || weight_word == nullptr) {
        reason = "no value";
    } else if (n != 8 || at != nullptr) {
        reason = "the value is <kind>:<weight>:<k>:<m>:<cols>:<prep>:<nr0>:<form>, eight cells";
    } else {
        f.kind = strcmp(cell[0], "single") == 0 ? 0 : (strcmp(cell[0], "group") == 0 ? 1 : -1);
        f.k    = mv_forced_number(cell[2]);
        f.m    = mv_forced_number(cell[3]);
        const int64_t cols = mv_forced_number(cell[4]), nr0 = mv_forced_number(cell[6]); // checked as written, before they are narrowed
        f.cols = cols >= 1 && cols <= 4 ? (int32_t) cols : 0;
        f.prep = strcmp(cell[5], "on") == 0 ? 1 : (strcmp(cell[5], "off") == 0 ? 0 : -1);
        f.nr0  = nr0 == 1 || nr0 == 2 || nr0 == 4 || nr0 == 8 ? (int32_t) nr0 : 0;
        f.form = -1;
        if (strcmp(cell[7], "-") == 0) {
            f.form = 0;
        }
        for (int i = 0; i < 7 && f.form < 0; i++) {
            if (form_words[i][0] != 0 && strcmp(cell[7], form_words[i]) == 0) {
                f.form = i;
            }
        }
        if (f.kind < 0) {
            reason = "kind is single or group";
        } else if (cell[1][0] == 0 || strlen(cell[1]) > 15) {
            reason = "weight is a weight type name";
        } else if (f.k < 1 || f.m < 1) {
            reason = "k and m are whole numbers above 0";
        } else if (f.cols < 1 || f.cols > 4) {
            reason = "columns are 1 to 4";
        } else if (f.prep < 0) {
            reason = "prep is on or off";
        } else if (f.nr0 != 1 && f.nr0 != 2 && f.nr0 != 4 && f.nr0 != 8) {
            reason = "rows per simdgroup are 1, 2, 4 or 8";
        } else if (f.form < 0) {
            reason = "form is one of -, a, w, ap, wp, ha, hap";
        } else if ((strcmp(cell[7], "-") == 0) != (f.cols == 1)) {
            reason = "form is - with one column and one of a, w, ap, wp, ha, hap with 2 to 4";
        }
    }
    if (reason == nullptr) {
        *out = f;
        snprintf(weight_word, 16, "%s", cell[1]);
    }
    if (why != nullptr) {
        *why = reason;
    }
    return reason == nullptr ? 1 : 0;
}

int ggml_metal_matvec_forced_is_for(const struct ggml_metal_matvec_forced * forced, int32_t kind, int32_t weight, int64_t k, int64_t m, int32_t cols) {
    return forced != nullptr && forced->kind == kind && forced->weight == weight && forced->k == k && forced->m == m && forced->cols == cols ? 1 : 0;
}

int ggml_metal_matvec_forced_plan(const struct ggml_metal_card * card, const struct ggml_metal_matvec_in * in, const struct ggml_metal_matvec_forced * forced, struct ggml_metal_matvec_plan * out, const char ** why) {
    const char * reason = nullptr;
    struct ggml_metal_matvec_plan plan = {};
    if (card == nullptr || in == nullptr || forced == nullptr || out == nullptr) {
        reason = "no plan";
    } else {
        // the rule's plan of the weight type and the card's settings, without the card's rows and the rows file
        struct ggml_metal_matvec_in base = *in;
        base.cols          = 1;
        base.use_rows      = 0;
        base.rows_file_nr0 = 0;
        plan = ggml_metal_matvec_plan_for(card, &base);
        plan.nr0          = forced->nr0;
        plan.nr0_set      = 1;
        plan.cols         = forced->cols;
        plan.form         = forced->form;
        plan.prep         = forced->prep;
        plan.shape_row    = -1;
        plan.from_default = 0;
        if (!ggml_metal_matvec_plan_check(card, in, &plan, &reason)) {
            // reason is set
        } else if (plan.nr0 * plan.nsg <= 0 || in->m % (plan.nr0 * plan.nsg) != 0) {
            // the kernels load every weight row of a threadgroup before the output bound
            reason = "with a forced row count m is a whole number of nr0 x nsg rows";
        }
    }
    if (reason == nullptr) {
        *out = plan;
    }
    if (why != nullptr) {
        *why = reason;
    }
    return reason == nullptr ? 1 : 0;
}

int ggml_backend_metal_matvec_has_rows(enum ggml_type type) {
    const struct mv_format_row * fmt = ggml_metal_mv_format(type);
    return fmt != nullptr && fmt->shape_rows ? 1 : 0;
}

int ggml_backend_metal_matvec_shape_row(int i, int32_t * kind, enum ggml_type * weight, int64_t * k, int64_t * m) {
    if (i < 0 || i >= ggml_metal_hw_mv_rows_n || kind == nullptr || weight == nullptr || k == nullptr || m == nullptr) {
        return 0;
    }
    const struct ggml_device_mv_row * row = &ggml_metal_hw_mv_rows[i];
    *kind   = row->kind;
    *weight = (enum ggml_type) row->type;
    *k      = row->ne00;
    *m      = row->ne01;
    return 1;
}

int ggml_backend_metal_matvec_card_row(const char * card, int32_t kind, enum ggml_type weight, int64_t k, int64_t m, int32_t cols, int32_t prep, int32_t * nr0, int32_t * form) {
    for (int i = 0; card != nullptr && nr0 != nullptr && form != nullptr && i < ggml_metal_hw_mv_rows_n; i++) {
        const struct ggml_device_mv_row * row = &ggml_metal_hw_mv_rows[i];
        if (strcmp(row->card, card) == 0 && row->kind == kind && row->type == (int32_t) weight && row->ne00 == k && row->ne01 == m && row->cols == cols && (row->prep == 1) == (prep == 1)) {
            *nr0  = row->nr0;
            *form = row->form;
            return 1;
        }
    }
    return 0;
}
