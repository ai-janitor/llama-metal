// Work plan of the matvec kernels (matmul/mul-mv.metal and the grouped launch): how much one group takes (simdgroups,
// rows), how a row sum is reduced, columns per launch and their loop form, prepared headers, load widths. One rule
// picks it from the card, the weight type and the multiply's shape; the dispatch files and the pipeline lookup use
// the same struct. The values are cells of data/matmul.tsv: [formats] (per weight type), [matvec_defaults] (per
// card), [matvec_shapes] (per card and shape). A new card or shape is rows there, not code here.
// How a plan reaches the kernel: as function constants that pick an instance compiled into the library already.
#pragma once

#include "card-object.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_matvec_plan {
    // 1 = a matvec kernel exists for the weight type and the fields below are set; 0 = none.
    int32_t ok;
    int32_t nsg;                  // simdgroups per threadgroup
    int32_t nr0;                  // rows per simdgroup, or per threadgroup when rows_per_threadgroup
    int32_t rows_per_threadgroup; // 1 = nr0 counts rows per threadgroup
    int32_t reduce;               // row sum: 0 the wave's own sum, 1 shared memory serial, 2 shared memory tree
    int32_t cols;                 // activation columns in the launch: 1, or 2 to 4
    int32_t form;                 // loop form of a 2 to 4 column launch (bit 0 weights held, bit 1 weights kept packed, bit 2 paired halves); 0 with one column
    int32_t prep;                 // 1 = the entry that reads Q4_K block headers prepared at load
    int32_t q8_load_pref;         // Q8_0 weight load width the card prefers, bytes (1 | 4); the buffers decide (ggml_metal_q8_load_bytes)
    int32_t q8_act_pref;          // Q8_0 activation load width the card prefers, bytes (4 | 16); the buffers decide
    int32_t q8_group;             // 1 = the card runs the Q8_0 matvecs of one activation in one launch
    int32_t q8_rows8_min;         // smallest row count of a single Q8_0 matvec the card runs at 8 whole rows per simdgroup, 0 = never
    int32_t group_store_first;    // 1 = the grouped K-quant launch stores every row before the row sums
    int32_t shape_row;            // index of the card's [matvec_shapes] row used, -1 = none
    int32_t from_default;         // 1 = the shape row that fits failed the check and the plan is the one without it
    int32_t nr0_set;              // 1 = nr0 was given by a row or by the cache rule, 0 = it is the format's
};

// What the rule is asked about one multiply.
struct ggml_metal_matvec_in {
    int32_t  kind;           // 0 single matvec, 1 member of a grouped launch
    int32_t  weight;         // ggml_type of the weights
    int64_t  k;              // the weight's input columns
    int64_t  m;              // the weight's output rows
    int32_t  cols;           // activation columns: 1, or 2 to 4 to ask for a column launch
    int32_t  prep_available; // 1 = prepared block headers exist for this weight (every member, for kind 1)
    int32_t  use_rows;       // 1 = the card's shape rows may apply (kind 1: every member has this type and shape)
    int32_t  model_known;    // 1 = the model profile is populated
    uint32_t act_bytes;      // bytes of one activation column
    int32_t  rows_file_nr0;  // rows per simdgroup from the external rows file for this type and shape, 0 = none
};

// Loads the card's matvec tuning for the card with this Metal device name. Reads card->wave, wave_probed, group_mem.
// 1. The settings, from data/matmul.tsv [matvec_defaults], a later row's written cell replacing the earlier value:
//    the default row (card any, wave any); the default row of the card's wave, once the wave is probed; the row of
//    device_name, while the card reports the wave and threadgroup memory written in it. Family rows are never
//    applied. group_store_on 0 keeps the grouped K-quant launch on the row-at-a-time sums.
// 2. The card's rows of data/matmul.tsv [matvec_shapes] (rows_on 0 = none) and the two switches a row that the
//    card's facts admit turns on: prepared headers (prep_on) and column rows (cols_on).
void ggml_metal_matvec_tune_load(struct ggml_metal_card * card, const char * device_name, int rows_on, int prep_on, int cols_on, int group_store_on);

// One matcher for a [matvec_shapes] row, shared by the rule and by the registration of prepared headers
// (memory/buffers.m): 1 when the card reports the wave and threadgroup memory the row was measured with and the row
// is for this kind, weight type, shape (or every shape) and column count. kind -1 and cols -1 match every kind and
// column count.
int ggml_metal_matvec_row_fits(const struct ggml_metal_card * card, const struct ggml_device_mv_row * row, int32_t kind, int32_t weight, int64_t k, int64_t m, int32_t cols);

// The first shape row of the card that applies (file order), as an index into card->mv_rows, or -1. A row of 2 to 4
// columns applies only where the card's column rows are on; a paired-halves row only at wave width 64; a prep row
// only with prep_available.
int32_t ggml_metal_matvec_shape_row(const struct ggml_metal_card * card, int32_t kind, int32_t weight, int64_t k, int64_t m, int32_t cols, int32_t prep_available);

// The checks: can the kernels run this plan for this multiply on this card? The same conditions matmul/gen-matvec.cmake
// applies to every [matvec_shapes] row. Returns 1, or 0 with *why (when not NULL) naming the condition that failed.
int ggml_metal_matvec_plan_check(const struct ggml_metal_card * card, const struct ggml_metal_matvec_in * in, const struct ggml_metal_matvec_plan * plan, const char ** why);

// The rule: the weight type's values, the card's defaults, then the rows in this order: the external rows file, the
// card's shape row, the format. With in->cols 2 to 4 the plan is a column launch (cols = in->cols) when a column row
// of the card fits, else a one-column plan (cols 1) the caller does not use.
struct ggml_metal_matvec_plan ggml_metal_matvec_plan_for(const struct ggml_metal_card * card, const struct ggml_metal_matvec_in * in);

// A column plan is held to the pipeline as compiled. Returns which fallback applies and writes the plan to run:
//   NONE  the column plan stays;
//   TAIL  m is not a whole number of the compiled pipeline's rows per threadgroup: the saved one-column plan;
//   WIDTH paired halves on a pipeline compiled to a wave other than 64: the same columns, rows and prep, ordinary form.
enum { GGML_METAL_MATVEC_FALLBACK_NONE = 0, GGML_METAL_MATVEC_FALLBACK_TAIL = 1, GGML_METAL_MATVEC_FALLBACK_WIDTH = 2 };
int ggml_metal_matvec_plan_fallback(const struct ggml_metal_matvec_plan * column, const struct ggml_metal_matvec_plan * single, int64_t m, int32_t pipeline_nr0, int32_t pipeline_nsg, int32_t pipeline_wave, struct ggml_metal_matvec_plan * out);

// Test hook GGML_METAL_TEST_MV_PLAN=<kind>:<weight>:<k>:<m>:<cols>:<prep>:<nr0>:<form>, inert unless set: the multiply
// of that kind (single | group), weight type, shape and activation column count runs the plan written in the last
// three cells instead of the rule's. The cells are the words of data/matmul.tsv [matvec_shapes]: prep on | off, nr0
// 1 | 2 | 4 | 8, form - with one column and a | w | ap | wp | ha | hap with 2 to 4. The forced plan goes through
// ggml_metal_matvec_plan_check; the process aborts naming the condition that failed, and also when the value cannot
// be read or the compiled pipeline cannot run the plan as forced. Every other multiply runs the rule's plan.
struct ggml_metal_matvec_forced {
    int32_t kind;   // 0 single, 1 group
    int32_t weight; // ggml_type
    int64_t k;
    int64_t m;
    int32_t cols;
    int32_t prep;
    int32_t nr0;
    int32_t form;
};

// Reads the hook's text. Returns 1 with *out filled except weight, whose word is copied to weight_word (at most 15
// characters; the caller turns it into a ggml_type), or 0 with *why naming what could not be read.
int ggml_metal_matvec_forced_parse(const char * text, struct ggml_metal_matvec_forced * out, char weight_word[16], const char ** why);

// 1 when the forced plan is for this multiply: kind, weight type, shape and activation columns all equal.
int ggml_metal_matvec_forced_is_for(const struct ggml_metal_matvec_forced * forced, int32_t kind, int32_t weight, int64_t k, int64_t m, int32_t cols);

// The plan the hook forces for the multiply `in` (in->cols = its activation columns, in->prep_available = prepared
// headers exist): the rule's plan without the card's rows, with the forced rows, columns, form and prep. Returns 1,
// or 0 with *why naming the failed condition of ggml_metal_matvec_plan_check, or that m is not whole threadgroups.
int ggml_metal_matvec_forced_plan(const struct ggml_metal_card * card, const struct ggml_metal_matvec_in * in, const struct ggml_metal_matvec_forced * forced, struct ggml_metal_matvec_plan * out, const char ** why);

// The hook as this process was started with it, or NULL when it is not set. Aborts when the value cannot be read.
// Defined in matmul/dispatch.cpp.
const struct ggml_metal_matvec_forced * ggml_metal_matvec_forced_env(void);

#ifdef __cplusplus
}
#endif
