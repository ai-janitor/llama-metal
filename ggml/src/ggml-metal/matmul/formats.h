#pragma once

// Per-format facts of the matvec and matmul kernels. The rows are data/matmul.tsv [formats],
// turned into mv-formats-table.h at configure time.
#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>

// Threadgroup memory a format's matvec kernel asks for: nothing, a fixed table, one float per lane,
// or the row sums of every lane (the shared-memory reduce sums nsg waves per row).
enum mv_smem_kind { MV_SMEM_NONE, MV_SMEM_BYTES, MV_SMEM_LANE_FLOATS, MV_SMEM_ROW_SUMS };

// Small-batch kernel: not taken, taken from 2 columns, or taken from the card's column count.
enum mv_ext_kind { MV_EXT_NO, MV_EXT_SMALL, MV_EXT_KQUANT };

// A kernel reads the activation type with this weight type: never, on every route, or on the matrix route only.
enum mv_act_kind { MV_ACT_NO, MV_ACT_YES, MV_ACT_MM };

// Columns (rows for MUL_MAT_ID) from which the matrix kernels take an op on cards with matrix hardware.
#define GGML_METAL_NE11_MM_MIN    8
#define GGML_METAL_NE21_MM_ID_MIN 32

struct mv_format_row {
    enum ggml_type    type;
    int               nsg;                  // simdgroups per threadgroup; 0 = they come from the tensor (the dense float types)
    int               nr0;                  // rows per simdgroup, or per threadgroup when rows_per_threadgroup
    bool              rows_per_threadgroup;
    enum mv_smem_kind smem;
    int               bytes;
    bool              nr0_override;         // the kernel takes its row count from the pipeline (function constant or column form)
    bool              reduce_recompile;     // below width 32 the kernel is compiled again with the shared-memory reduce
    bool              matvec;
    enum mv_ext_kind  ext;
    bool              tiled;
    bool              w64_fallback;
    bool              grouped;
    bool              shape_rows;
    bool              act_bf16;             // a kernel reads a bf16 activation with this weight type
    enum mv_act_kind  act_f16;              // the same for an f16 activation
    bool              ext_wide;             // without matrix hardware the small-batch kernel takes any column count
    int               short_k;              // dense float types: below this many input columns the short entry runs; 0 = the type has none
    int               short_rows;           // rows per threadgroup of the short entry
};

#include "mv-formats-table.h"

// The row of a weight type, or NULL when the type has none.
static inline const struct mv_format_row * ggml_metal_mv_format(enum ggml_type type) {
    for (size_t i = 0; i < sizeof(mv_format_rows)/sizeof(mv_format_rows[0]); ++i) {
        if (mv_format_rows[i].type == type) {
            return &mv_format_rows[i];
        }
    }
    return NULL;
}

// True when a dense float matvec of k input columns runs the short entry (one simdgroup, short_rows rows per
// threadgroup, no threadgroup memory). The rule, the entry lookup and the memory sizing all ask here.
static inline bool ggml_metal_mv_float_short(const struct mv_format_row * row, int64_t k) {
    return row != NULL && row->short_k > 0 && k < row->short_k;
}

// True when the matrix route takes the op. The router and the supports predicate both ask here.
static inline bool ggml_metal_mm_route_takes(bool has_matrix_hw, const struct ggml_tensor * op) {
    if (!has_matrix_hw || op->src[0]->ne[0] < 64) {
        return false;
    }
    if (op->op == GGML_OP_MUL_MAT_ID) {
        return op->src[2]->ne[1] >= GGML_METAL_NE21_MM_ID_MIN;
    }
    return !ggml_is_transposed(op->src[0]) && !ggml_is_transposed(op->src[1]) && op->src[1]->ne[1] > GGML_METAL_NE11_MM_MIN;
}
