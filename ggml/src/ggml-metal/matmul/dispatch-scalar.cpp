// Scalar mul_mv dispatch: the workhorse for single-token generation.
// Extracted from dispatch.cpp — handles all quant types with shmem_reduce
// for non-32-wide GPUs (Intel NW<32, AMD NW=64).
#include "dispatch.h"
#include "matvec-plan.h"
#include "tuning.h"
#include "ggml-metal-probe.h"
#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
GGML_METAL_FAMILY(matmul)
#include "residual-alias-guard.h"

struct nr0_row {
    ggml_type type;
    int64_t ne00;
    int64_t ne01;
    int nr0;
};

static std::vector<nr0_row> load_nr0_rows() {
    std::vector<nr0_row> rows;
    const char * path = std::getenv("GGML_METAL_MV_NR0_ROWS");
    if (path == nullptr) {
        return rows;
    }
    FILE * file = std::fopen(path, "r");
    if (file == nullptr) {
        GGML_LOG_WARN("%s:0: cannot read matvec nr0 rows\n", path);
        return rows;
    }
    const auto parse_int = [](const char * text, int64_t & value) {
        if (*text == '\0') {
            return false;
        }
        errno = 0;
        char * end = nullptr;
        const long long parsed = std::strtoll(text, &end, 10);
        if (errno == ERANGE || *end != '\0') {
            return false;
        }
        value = parsed;
        return true;
    };
    const auto parse_dim = [&parse_int](const char * text, int64_t & value) {
        if (std::strcmp(text, "*") == 0) {
            value = -1;
            return true;
        }
        return parse_int(text, value) && value >= 0;
    };
    char * line = nullptr;
    size_t capacity = 0;
    ssize_t size;
    int line_number = 0;
    while ((size = ::getline(&line, &capacity, file)) >= 0) {
        ++line_number;
        while (size > 0 && (line[size - 1] == '\n' || line[size - 1] == '\r')) {
            line[--size] = '\0';
        }
        const bool complete = std::strlen(line) == (size_t) size;
        if (line_number == 1) {
            if (!complete || std::strcmp(line, "type\tne00\tne01\tnr0") != 0) {
                GGML_LOG_WARN("%s:%d: invalid matvec nr0 rows header\n", path, line_number);
                break;
            }
            continue;
        }
        char * fields[4] = { line, nullptr, nullptr, nullptr };
        bool valid = complete;
        for (int i = 1; i < 4 && valid; ++i) {
            char * tab = std::strchr(fields[i - 1], '\t');
            if (tab == nullptr) {
                valid = false;
            } else {
                *tab = '\0';
                fields[i] = tab + 1;
            }
        }
        nr0_row row = {};
        int64_t nr0 = 0;
        if (valid) {
            valid = std::strchr(fields[3], '\t') == nullptr;
            const struct mv_format_row * fmt = nullptr;
            for (const struct mv_format_row & f : mv_format_rows) {
                if (f.shape_rows && std::strcmp(fields[0], ggml_type_name(f.type)) == 0) {
                    fmt = &f;
                }
            }
            if (fmt != nullptr) {
                row.type = fmt->type;
            } else {
                valid = false;
            }
            valid = valid && parse_dim(fields[1], row.ne00) && parse_dim(fields[2], row.ne01) && parse_int(fields[3], nr0);
            valid = valid && (nr0 == 1 || nr0 == 2 || nr0 == 4 || nr0 == 8);
        }
        if (!valid) {
            GGML_LOG_WARN("%s:%d: ignoring malformed matvec nr0 row\n", path, line_number);
            continue;
        }
        row.nr0 = (int) nr0;
        if (rows.size() < 256) {
            rows.push_back(row);
        }
    }
    if (std::ferror(file) || line_number == 0) {
        GGML_LOG_WARN("%s:%d: cannot read matvec nr0 rows\n", path, line_number + 1);
        rows.clear();
    }
    std::free(line);
    std::fclose(file);
    return rows;
}

static int ggml_metal_mv_nr0_from_rows(ggml_type type, int64_t ne00, int64_t ne01) {
    static const std::vector<nr0_row> rows = load_nr0_rows();
    for (const nr0_row & row : rows) {
        if (row.type == type && (row.ne00 == -1 || row.ne00 == ne00) && (row.ne01 == -1 || row.ne01 == ne01)) {
            return row.nr0;
        }
    }
    return 0;
}

// Runtime opt-in for the scalar Q4_K residual epilogue (FC_MUL_MV_RESIDUAL).
// OFF unless GGML_METAL_Q4_K_RESIDUAL_ENABLE is exactly "1": absent, "0", empty
// or any malformed value keeps the faithful one-node path. It is separate from
// the global fusion policy (ctx->use_fusion), which must ALSO be true.
// Read at every matcher call, never latched per process: a serial activation
// harness sets and restores the variable per row, and a cached value would pin
// a baseline-off row's answer onto every later candidate row.
// This is activation plumbing, not an automatic policy.
static bool ggml_metal_q4_k_residual_requested() {
    const char * value = std::getenv("GGML_METAL_Q4_K_RESIDUAL_ENABLE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// True only when both Metal byte ranges are known, non-overflowing, and do not
// overlap. Fusing an ADD otherwise could replace a source before its final read.
static bool ggml_metal_tensor_ranges_disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    const ggml_metal_buffer_id a_buffer = ggml_metal_get_buffer_id(a);
    const ggml_metal_buffer_id b_buffer = ggml_metal_get_buffer_id(b);
    const size_t a_size = ggml_nbytes(a);
    const size_t b_size = ggml_nbytes(b);

    // A missing Metal object, an empty range, or an overflowing end offset has
    // no safe alias proof, so the optional epilogue must remain disabled.
    if (!a_buffer.metal || !b_buffer.metal || a_size == 0 || b_size == 0 ||
        a_size > std::numeric_limits<size_t>::max() - a_buffer.offs ||
        b_size > std::numeric_limits<size_t>::max() - b_buffer.offs) {
        return false;
    }

    if (a_buffer.metal != b_buffer.metal) {
        return true;
    }

    const size_t a_end = a_buffer.offs + a_size;
    const size_t b_end = b_buffer.offs + b_size;
    return a_end <= b_buffer.offs || b_end <= a_buffer.offs;
}

static bool ggml_metal_q4_k_residual_tensor_ranges_safe(
        const ggml_tensor * op, const ggml_tensor * add, const ggml_tensor * residual) {
    const ggml_metal_buffer_id op_buffer = ggml_metal_get_buffer_id(op);
    const ggml_metal_buffer_id add_buffer = ggml_metal_get_buffer_id(add);
    const ggml_metal_buffer_id residual_buffer = ggml_metal_get_buffer_id(residual);
    const ggml_metal_buffer_id weights_buffer = ggml_metal_get_buffer_id(op->src[0]);
    const ggml_metal_buffer_id activation_buffer = ggml_metal_get_buffer_id(op->src[1]);
    return ggml_metal_q4_k_residual_ranges_safe(
        op_buffer.metal, op_buffer.offs, ggml_nbytes(op),
        add_buffer.metal, add_buffer.offs, ggml_nbytes(add),
        residual_buffer.metal, residual_buffer.offs, ggml_nbytes(residual),
        weights_buffer.metal, weights_buffer.offs, ggml_nbytes(op->src[0]),
        activation_buffer.metal, activation_buffer.offs, ggml_nbytes(op->src[1]));
}

// The opt-in must say why it did not become effective.  This is diagnostic-only:
// it runs only for a requested candidate and does not relax any matcher guard.
static ggml_tensor * ggml_metal_q4_k_residual_reject(
        ggml_metal_op_t ctx, int idx, const char * reason, ggml_tensor * op, ggml_tensor * add) {
    const ggml_metal_buffer_id op_buffer = ggml_metal_get_buffer_id(op);
    const ggml_metal_buffer_id add_buffer = add ? ggml_metal_get_buffer_id(add) : ggml_metal_buffer_id{};
    GGML_LOG_INFO("%s: Q4_K residual epilogue: requested=1 effective=0 reason=%s "
                  "node=%d op=%s next_node=%d next_op=%s "
                  "op_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
                  "add_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
                  "op_buffer=[%p,%zu,%zu] add_buffer=[%p,%zu,%zu]\n",
                  __func__, reason, idx, ggml_op_name(op->op), add ? idx + 1 : -1,
                  add ? ggml_op_name(add->op) : "none",
                  op->ne[0], op->ne[1], op->ne[2], op->ne[3],
                  add ? add->ne[0] : 0, add ? add->ne[1] : 0, add ? add->ne[2] : 0, add ? add->ne[3] : 0,
                  op_buffer.metal, op_buffer.offs, ggml_nbytes(op),
                  add_buffer.metal, add_buffer.offs, add ? ggml_nbytes(add) : 0);
    return nullptr;
}

// Return the ADD's non-matmul source only for the narrow scalar Q4_K residual
// epilogue. Every rejected predicate leaves the legacy single-node dispatch.
static ggml_tensor * ggml_metal_match_q4_k_residual_add(ggml_metal_op_t ctx, int idx, ggml_tensor * op) {
    // The feature needs both its explicit runtime opt-in and the existing
    // global fusion policy; rejection here precedes all matcher side effects.
    if (!ggml_metal_q4_k_residual_requested()) {
        return nullptr;
    }
    if (!ctx->use_fusion) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "fusion_disabled", op, nullptr);
    }
    if (idx + 1 >= ctx->n_nodes()) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "no_next_node", op, nullptr);
    }

    ggml_tensor * add = ctx->node(idx + 1);
    if (op->src[0]->type != GGML_TYPE_Q4_K) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "weight_not_q4_k", op, add);
    }
    if (op->src[1]->type != GGML_TYPE_F32) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "activation_not_f32", op, add);
    }
    if (op->type != GGML_TYPE_F32) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "matmul_output_not_f32", op, add);
    }
    if (!ggml_is_contiguous(op)) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "matmul_not_contiguous", op, add);
    }

    // can_fuse proves the ordered MUL_MAT/ADD pair, one use of the intermediate,
    // no intermediate view/output, and equal shapes before any encoder mutation.
    const ggml_op fusion_ops[] = { GGML_OP_MUL_MAT, GGML_OP_ADD };
    if (!ctx->can_fuse(idx, fusion_ops, 2)) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "ordered_mul_mat_add_rejected", op, add);
    }

    ggml_tensor * residual = add->src[0] == op ? add->src[1] :
                            add->src[1] == op ? add->src[0] : nullptr;

    // The kernel performs elementwise float addition, not broadcasting or a
    // strided view. Its residual address omits the activation column and plane
    // indices, so admit only the one-column, one-plane decode domain.
    if (!residual) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "add_does_not_consume_matmul", op, add);
    }
    if (residual->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "residual_or_add_not_f32", op, add);
    }
    if (!ggml_are_same_shape(op, residual) || !ggml_are_same_shape(op, add)) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "residual_shape_mismatch", op, add);
    }
    if (!ggml_is_contiguous(residual) || !ggml_is_contiguous(add)) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "residual_or_add_not_contiguous", op, add);
    }
    if (op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "not_single_column_decode", op, add);
    }

    // ADD may exactly reuse the MUL_MAT output range: the residual kernel writes
    // that destination from the accumulator and residual, never the old output.
    // Every other input remains strictly disjoint.
    if (!ggml_metal_q4_k_residual_tensor_ranges_safe(op, add, residual)) {
        return ggml_metal_q4_k_residual_reject(ctx, idx, "buffer_ranges_not_disjoint", op, add);
    }

    return residual;
}

// Opt-in Q8_0 decode matvec in the block-owned integer form (kernel_mul_mv_q8_0_f32_lab), GGML_METAL_Q8_LAB=1.
// Lanes per row and simdgroups per row come from GGML_METAL_Q8_LAB_ROWS (ne00, ne01, lanes, simdgroups; -1 = any).
struct q8_lab_row {
    int64_t ne00;
    int64_t ne01;
    int lanes;
    int simdgroups;
};

static bool ggml_metal_q8_lab_requested() {
    static const bool requested = [] {
        const char * value = std::getenv("GGML_METAL_Q8_LAB");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return requested;
}

static std::vector<q8_lab_row> load_q8_lab_rows() {
    std::vector<q8_lab_row> rows;
    const char * path = std::getenv("GGML_METAL_Q8_LAB_ROWS");
    if (path == nullptr || path[0] == '\0') {
        return rows;
    }
    FILE * file = std::fopen(path, "r");
    if (file == nullptr) {
        GGML_LOG_WARN("%s:0: cannot read q8_0 lab rows\n", path);
        return rows;
    }
    char line[256];
    int line_number = 0;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        ++line_number;
        if (line_number == 1) {
            static const char header[] = "ne00\tne01\tlanes\tsimdgroups";
            if (std::strncmp(line, header, sizeof(header) - 1) != 0) {
                GGML_LOG_WARN("%s:1: invalid q8_0 lab rows header\n", path);
                break;
            }
            continue;
        }
        long long ne00 = 0, ne01 = 0;
        int lanes = 0, simdgroups = 0;
        if (std::sscanf(line, "%lld\t%lld\t%d\t%d", &ne00, &ne01, &lanes, &simdgroups) != 4 ||
            (lanes != 4 && lanes != 8) || simdgroups < 1 || simdgroups > 16 || (simdgroups & (simdgroups - 1)) != 0) {
            GGML_LOG_WARN("%s:%d: ignoring malformed q8_0 lab row\n", path, line_number);
            continue;
        }
        rows.push_back({ (int64_t) ne00, (int64_t) ne01, lanes, simdgroups });
    }
    std::fclose(file);
    return rows;
}

static q8_lab_row ggml_metal_q8_lab_geometry(int64_t ne00, int64_t ne01) {
    static const std::vector<q8_lab_row> rows = load_q8_lab_rows();
    for (const q8_lab_row & row : rows) {
        if ((row.ne00 == -1 || row.ne00 == ne00) && (row.ne01 == -1 || row.ne01 == ne01)) {
            return row;
        }
    }
    return { ne00, ne01, 8, 1 };
}

// Returns 1 when the op was encoded, 0 to fall back to the scalar path.
static int dispatch_mul_mat_q8_lab(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return 0;
    }
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1) {
        return 0;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op) || src0->ne[0] % 64 != 0) {
        return 0;
    }
    const ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(src0);
    const ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(src1);
    if (bid_src0.offs % 4 != 0 || bid_src1.offs % 4 != 0) {
        return 0;
    }

    const int32_t ne00 = (int32_t) src0->ne[0];
    const int32_t ne01 = (int32_t) src0->ne[1];
    const q8_lab_row geom = ggml_metal_q8_lab_geometry(ne00, ne01);

    ggml_metal_library_t lib = ctx->lib;
    char name[256];
    snprintf(name, sizeof(name), "kernel_mul_mv_q8_0_f32_lab_nsg=%d_nxpsg=%d", geom.simdgroups, geom.lanes);
    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, (int16_t) geom.simdgroups, FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, (int16_t) geom.lanes,      FC_MUL_MV + 1);
        pipeline = ggml_metal_library_compile_pipeline(lib, "kernel_mul_mv_q8_0_f32_lab", name, cv);
        ggml_metal_cv_free(cv);
    }
    if (!pipeline.pipeline) {
        return 0;
    }

    const int width = ggml_metal_pipeline_thread_execution_width(pipeline);
    if ((width != 32 && width != 64) || width*geom.simdgroups > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        return 0;
    }
    const int rows_per_group = width/geom.lanes;
    const size_t smem = std::max<size_t>(16, (size_t) geom.simdgroups*rows_per_group*sizeof(float));

    ggml_metal_kargs_mul_mv args = {};
    args.ne00 = ne00;
    args.ne01 = ne01;
    args.nb01 = src0->nb[1];
    args.ne10 = (int32_t) src1->ne[0];
    args.ne0  = (int32_t) op->ne[0];
    args.ne1  = (int32_t) op->ne[1];
    args.nr0  = rows_per_group;

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 3);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    ggml_metal_encoder_set_dispatch_shape(enc, ne00, ne01);
    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + rows_per_group - 1)/rows_per_group, 1, 1, width, geom.simdgroups, 1);
    return 1;
}

// A single Q8_0 matvec with at least the hw row's q8_rows8_min rows: 8 whole rows per simdgroup, word loads.
// Taken only where its lane sums pair up exactly as the default kernel's tree does, so the output bytes are the same.
// Returns 1 when encoded, 0 to fall through.
static int dispatch_mul_mat_q8_rows8(ggml_metal_op_t ctx, int idx, const struct ggml_metal_matvec_plan * plan) {
    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
    ggml_tensor * op = ctx->node(idx);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (plan->q8_rows8_min <= 0 || src0->ne[1] < (int64_t) plan->q8_rows8_min) {
        return 0;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return 0;
    }
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1) {
        return 0;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return 0;
    }
    const int width = (int) profile->simd_width;
    const int nsg = plan->nsg, nr0 = ggml_metal_mv_launch_of(profile, GGML_METAL_MV_LAUNCH_Q8_ROWS8).units, lanes_per_block = 4;
    if (nr0 != 4 && nr0 != 8) {
        return 0; // the kernel has instances of 4 and 8 rows only; matmul/mv-launch-rules.cmake takes no other cell
    }
    // same bytes as the default kernel: tree row sum, and one block per lane there (blocks = nsg x width / lanes per block)
    if (width != 64 || plan->reduce != 2 || src0->ne[0] != (int64_t) nsg*(width/lanes_per_block)*32) {
        return 0;
    }
    if (ggml_metal_q8_load_bytes(plan->q8_load_pref, src0) != 4) {
        return 0;
    }
    char name[256];
    snprintf(name, sizeof(name), "kernel_mul_mv_q8_0_f32_rows_nsg=%d_nr0=%d_tail=1_q8load=4", nsg, nr0);
    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(ctx->lib, name);
    if (!pipeline.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, (int16_t) nsg, FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, (int16_t) nr0, FC_MUL_MV + 4);
        ggml_metal_cv_set_int16(cv, (int16_t) 1,   FC_MUL_MV + 7);
        ggml_metal_cv_set_int16(cv, (int16_t) 4,   FC_MUL_MV + 10);
        pipeline = ggml_metal_library_compile_pipeline(ctx->lib, "kernel_mul_mv_q8_0_f32_rows", name, cv);
        ggml_metal_cv_free(cv);
    }
    if (!pipeline.pipeline || ggml_metal_pipeline_thread_execution_width(pipeline) != width ||
        width*nsg > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        return 0;
    }
    const int32_t ne00 = (int32_t) src0->ne[0];
    const int32_t ne01 = (int32_t) src0->ne[1];
    ggml_metal_kargs_mul_mv args = {};
    args.ne00 = ne00;
    args.ne01 = ne01;
    args.nb01 = src0->nb[1];
    args.ne10 = (int32_t) src1->ne[0];
    args.ne0  = (int32_t) op->ne[0];
    args.ne1  = (int32_t) op->ne[1];
    args.nr0  = nr0;

    if (ggml_metal_mv_receipt_requested()) {
        fprintf(stderr, "MV_RECEIPT attempt=none fallback=none final=0:1:%d:0:0:%s\n", nr0, name);
    }
    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src0), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src1), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 3);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, (size_t) nsg*width*nr0*sizeof(float), 0);
    ggml_metal_encoder_set_dispatch_shape(enc, ne00, ne01);
    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nsg*nr0 - 1)/(nsg*nr0), 1, 1, width, nsg, 1);
    return 1;
}

// The 2-4 Q8_0 matvecs that read one activation, packed by the graph optimizer, in one launch (hw row mv_group).
static int dispatch_mul_mat_q8_group(ggml_metal_op_t ctx, int idx, const struct ggml_metal_matvec_plan * plan) {
    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
    ggml_tensor * op = ctx->node(idx);
    if (plan->q8_group != 1 || !ctx->use_fusion || !op->src[1] || !ggml_metal_norm_matvec_mm_ok(op, op->src[1])) {
        return 0;
    }
    const ggml_tensor * act = op->src[1];
    ggml_tensor * mm[4] = { nullptr, nullptr, nullptr, nullptr };
    int nmat = 0;
    // the most matrices of one launch: the card's number, never above the kernel's four slots
    const int group_max = std::min(4, (int) ggml_metal_mv_launch_of(profile, GGML_METAL_MV_LAUNCH_GROUP_Q8).units);
    while (nmat < group_max && idx + nmat < ctx->n_nodes() && ggml_metal_norm_matvec_mm_ok(ctx->node(idx + nmat), act)) {
        mm[nmat] = ctx->node(idx + nmat);
        nmat++;
    }
    if (nmat < 2) {
        return 0;
    }

    const int  width = (int) profile->simd_width;
    const bool sr    = plan->reduce != 0;
    const int  nsg   = plan->nsg;
    const int  nr0   = plan->nr0;
    const int  tail  = plan->reduce == 2 ? 1 : 0;

    char name[256];
    snprintf(name, sizeof(name), "kernel_mul_mvg_q8_0_f32_k=%d_nsg=%d_sr=%d_nr0=%d_tail=%d", nmat, nsg, sr ? 1 : 0, nr0, tail);
    // word loads only when every matrix of the group can take them
    int q8_load = 4;
    for (int k = 0; k < nmat; k++) {
        if (ggml_metal_q8_load_bytes(plan->q8_load_pref, mm[k]->src[0]) != 4) {
            q8_load = 1;
        }
    }
    if (q8_load != 1) {
        const size_t used = strlen(name);
        snprintf(name + used, sizeof(name) - used, "_q8load=%d", q8_load);
    }
    // the matrices of a group share one activation
    const int q8_act = ggml_metal_q8_act_load_bytes(plan->q8_act_pref, mm[0]->src[1]);
    if (q8_act != 4) {
        const size_t used = strlen(name);
        snprintf(name + used, sizeof(name) - used, "_q8act=%d", q8_act);
    }
    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(ctx->lib, name);
    if (!pipeline.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, (int16_t) nsg,  FC_MUL_MV + 0);
        ggml_metal_cv_set_bool (cv, sr,             FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, (int16_t) tail, FC_MUL_MV + 7);
        ggml_metal_cv_set_int16(cv, (int16_t) nmat, FC_MUL_MV_GROUP + 0);
        if (q8_load != 1) {
            ggml_metal_cv_set_int16(cv, (int16_t) q8_load, FC_MUL_MV + 10);
        }
        if (q8_act != 4) {
            ggml_metal_cv_set_int16(cv, (int16_t) q8_act, FC_MUL_MV + 11);
        }
        pipeline = ggml_metal_library_compile_pipeline(ctx->lib, "kernel_mul_mvg_q8_0_f32", name, cv);
        ggml_metal_cv_free(cv);
    }
    if (!pipeline.pipeline || ggml_metal_pipeline_thread_execution_width(pipeline) != width ||
        width*nsg > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        return 0;
    }

    ggml_metal_kargs_norm_mul_mv args = {};
    args.ne00 = (int32_t) act->ne[0];
    int32_t  rows[4]  = { 0, 0, 0, 0 };
    uint64_t nb01[4]  = { 0, 0, 0, 0 };
    int longest = 0, all_rows = 0;
    for (int k = 0; k < nmat; k++) {
        rows[k]  = (int32_t) mm[k]->src[0]->ne[1];
        nb01[k]  = mm[k]->src[0]->nb[1];
        const int groups = (rows[k] + nr0 - 1)/nr0;
        longest  = std::max(longest, groups);
        all_rows += rows[k];
    }
    args.rows0 = rows[0]; args.rows1 = rows[1]; args.rows2 = rows[2]; args.rows3 = rows[3];
    args.nb01_0 = nb01[0]; args.nb01_1 = nb01[1]; args.nb01_2 = nb01[2]; args.nb01_3 = nb01[3];

    // every member is checked for overlap with work still in flight
    for (int i = 1; i < nmat; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i), idx + i)) {
            ggml_metal_op_concurrency_reset(ctx, idx + i);
            break;
        }
    }

    if (ggml_metal_mv_receipt_requested()) {
        fprintf(stderr, "MV_RECEIPT attempt=none fallback=none final=1:1:%d:0:0:%s\n", nr0, name);
    }
    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(act), 1);
    for (int k = 0; k < 4; k++) {
        const ggml_tensor * m = mm[k < nmat ? k : 0];
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(m->src[0]), 4 + k);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(m), 8 + k);
    }
    const int pw = ggml_metal_pipeline_thread_execution_width(pipeline);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, (size_t) (sr ? nsg : 1)*pw*sizeof(float)*nr0, 0);
    ggml_metal_encoder_set_dispatch_shape(enc, args.ne00, all_rows);
    ggml_metal_encoder_dispatch_threadgroups(enc, longest, 1, 1, pw, nsg, 1);
    return nmat;
}

int dispatch_mul_mat_scalar(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
    const struct ggml_model_profile  * model   = ggml_metal_device_get_model_profile(ctx->dev);

    // Note: f32/f16/bf16 scalar kernels have a remaining issue at NW=64 beyond shmem sizing.
    // The q8_0 shmem underallocation fix (nsg * NW * NR0 * sizeof(float)) was necessary but
    // not sufficient — the f16/f32 kernel at NW=64 still produces wrong output.
    // The kernel body itself has an NW=64 bug (likely in the block-stride or remainder logic
    // where NW > NB=32 causes half the threads to skip but the indexing doesn't account for
    // the gap). These skip scalar and fall through to the GCN fallback.
    // Tested 2026-03-28: removing this guard produces divergent output on Qwen3.5-0.8B-Q8_0.
    // Not reproduced later (f16 and bf16 pass on 64-wide cards); f16 and bf16 run scalar where the hw row says f_scalar_w64.
    const bool scalar_f_broken = (profile->simd_width == 64 &&
        (op->src[0]->type == GGML_TYPE_F32 ||
         ((op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) && ggml_metal_device_get_card(ctx->dev)->f_scalar_w64 != 1)));

    if (scalar_f_broken) {
        return 0;
    }

    // Q8_0: the grouped launch and the 8-row launch take their grouping, row sum, simdgroups and load widths from
    // the matvec work plan of this multiply.
    if (op->src[0]->type == GGML_TYPE_Q8_0) {
        struct ggml_metal_matvec_in q8_in = {};
        q8_in.weight      = (int32_t) GGML_TYPE_Q8_0;
        q8_in.k           = op->src[0]->ne[0];
        q8_in.m           = op->src[0]->ne[1];
        q8_in.cols        = 1;
        q8_in.use_rows    = 1;
        q8_in.model_known = model != nullptr && model->populated ? 1 : 0;
        q8_in.act_bytes   = (uint32_t) op->src[0]->ne[0] * (uint32_t) sizeof(float);
        const struct ggml_metal_matvec_plan q8_plan = ggml_metal_matvec_plan_for(ggml_metal_device_get_card(ctx->dev), &q8_in);
        if (q8_plan.ok && q8_plan.q8_group == 1) {
            const int n = dispatch_mul_mat_q8_group(ctx, idx, &q8_plan);
            if (n > 0) {
                return n;
            }
        }
        if (q8_plan.ok && dispatch_mul_mat_q8_rows8(ctx, idx, &q8_plan) == 1) {
            return 1;
        }
    }
    if (ggml_metal_q8_lab_requested() && op->src[0]->type == GGML_TYPE_Q8_0) {
        const int n = dispatch_mul_mat_q8_lab(ctx, idx);
        if (n > 0) {
            return n;
        }
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // The matvec work plan of this multiply (matmul/matvec-plan.h): the rule reads the weight type's values, the card's
    // defaults and rows. Rows per simdgroup of a K-quant: the external rows file first, then the card's shape row.
    const ggml_type type = op->src[0]->type;
    const struct mv_format_row * type_row = ggml_metal_mv_format(type);
    const bool kquant_rows = type_row != nullptr && type_row->shape_rows;

    const bool src1_trans_mv = ggml_is_transposed(op->src[1]);
    ggml_tensor * residual = ggml_metal_match_q4_k_residual_add(ctx, idx, op);
    const bool use_residual = residual != nullptr;

    // A Q4_K weight with headers prepared at load (one plane, one f32 column, no residual epilogue) may take a prep
    // row: its entry point reads them at 5.
    const bool kq_prep_test = kquant_rows && ggml_metal_kq_prep_test_requested();
    ggml_metal_buffer_id bid_kq_prep = { nullptr, 0 };
    if (kquant_rows && type == GGML_TYPE_Q4_K && !use_residual && !src1_trans_mv && op->src[1]->type == GGML_TYPE_F32 &&
        ne02 == 1 && ne03 == 1 && ne11 == 1 && ne12 == 1 && ne13 == 1 && (ggml_metal_device_get_card(ctx->dev)->kq_prep == 1 || kq_prep_test)) {
        bid_kq_prep = ggml_metal_get_kq_prep_id(op->src[0]);
    }

    struct ggml_metal_matvec_in in = {};
    in.kind           = 0;
    in.weight         = (int32_t) type;
    in.k              = ne00;
    in.m              = ne01;
    in.cols           = 1;
    in.prep_available = bid_kq_prep.metal != nullptr ? 1 : 0;
    in.use_rows       = 1;
    in.model_known    = model != nullptr && model->populated ? 1 : 0;
    in.act_bytes      = (uint32_t) ne00 * (uint32_t) sizeof(float);
    in.rows_file_nr0  = kquant_rows ? ggml_metal_mv_nr0_from_rows(type, ne00, ne01) : 0;
    struct ggml_metal_matvec_plan plan = ggml_metal_matvec_plan_for(ggml_metal_device_get_card(ctx->dev), &in);
    if (kq_prep_test && bid_kq_prep.metal != nullptr) {
        plan.prep = 1;
    }

    // The test hook GGML_METAL_TEST_MV_PLAN (matmul/matvec-plan.h) names this multiply: its plan replaces the rule's.
    const struct ggml_metal_matvec_forced * forced = ggml_metal_matvec_forced_env();
    const bool forced_here = ggml_metal_matvec_forced_is_for(forced, 0, (int32_t) type, ne00, ne01, ne11) == 1;
    if (forced_here) {
        if (use_residual || src1_trans_mv || op->src[1]->type != GGML_TYPE_F32 || ne02 != 1 || ne03 != 1 || ne12 != 1 || ne13 != 1) {
            GGML_ABORT("GGML_METAL_TEST_MV_PLAN: the multiply is not one plane of f32 activation columns without a fused add");
        }
        bid_kq_prep = forced->prep == 1 ? ggml_metal_get_kq_prep_id(op->src[0]) : ggml_metal_buffer_id{ nullptr, 0 };
        struct ggml_metal_matvec_in forced_in = in;
        forced_in.cols           = ne11;
        forced_in.prep_available = bid_kq_prep.metal != nullptr ? 1 : 0;
        const char * why = nullptr;
        if (!ggml_metal_matvec_forced_plan(ggml_metal_device_get_card(ctx->dev), &forced_in, forced, &plan, &why)) {
            GGML_ABORT("GGML_METAL_TEST_MV_PLAN: %s", why);
        }
    }

    // A column row of this card takes a 2-4 column decode batch of this weight in one launch (grid 1 in y). The
    // one-column plan is kept: a column plan the compiled pipeline cannot run falls back to it.
    const struct ggml_metal_matvec_plan plan_single = plan;
    const ggml_metal_buffer_id bid_kq_prep_single = bid_kq_prep;
    if (!forced_here && kquant_rows && type != GGML_TYPE_Q5_K && ne11 >= 2 && ne11 <= 4 && !use_residual && !src1_trans_mv && op->src[1]->type == GGML_TYPE_F32 &&
        ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1) {
        int test_nr0 = 0;
        int test_form = 0;
        const bool cols_test = ggml_metal_kq_cols_test(&test_nr0, &test_form);
        ggml_metal_buffer_id bid_cols = { nullptr, 0 };
        if (type == GGML_TYPE_Q4_K && (ggml_metal_device_get_card(ctx->dev)->kq_prep == 1 || ggml_metal_kq_prep_test_requested())) {
            bid_cols = ggml_metal_get_kq_prep_id(op->src[0]);
        }
        struct ggml_metal_matvec_in cols_in = in;
        cols_in.cols           = ne11;
        cols_in.prep_available = bid_cols.metal != nullptr ? 1 : 0;
        const struct ggml_metal_matvec_plan cols_plan = ggml_metal_matvec_plan_for(ggml_metal_device_get_card(ctx->dev), &cols_in);
        if (cols_plan.cols > 1) {
            plan = cols_plan;
            bid_kq_prep = bid_cols;
        } else if (cols_test) {
            // the test switch asks for a column launch without a row: its values are not checked here, the pipeline as compiled decides below
            plan.cols      = ne11;
            plan.form      = test_form;
            plan.nr0       = test_nr0;
            plan.nr0_set   = 1;
            plan.prep      = bid_cols.metal != nullptr ? 1 : 0;
            plan.shape_row = -1;
            bid_kq_prep = bid_cols;
        }
        if (plan.cols > 1 && (ne11 == 4 || profile->simd_width != 64)) {
            plan.form &= 3; // paired halves take 2 or 3 columns and are the two halves of a 64-thread wave
        }
    }
    char pipeline_name[256];
    char * name_out = ggml_metal_mv_receipt_requested() ? pipeline_name : nullptr;
    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op, &plan, src1_trans_mv, use_residual, name_out);

    // The column kernel is held to the pipeline as compiled, whatever the row or the test switch asked for.
    const struct ggml_metal_matvec_plan plan_attempt = plan;
    int fallback = GGML_METAL_MATVEC_FALLBACK_NONE;
    int fallback_rows = 0;
    int fallback_wave = 0;
    if (pipeline.pipeline && plan.cols > 1) {
        fallback_rows = pipeline.nr0 * pipeline.nsg;
        fallback_wave = ggml_metal_pipeline_thread_execution_width(pipeline);
        struct ggml_metal_matvec_plan next = plan;
        fallback = ggml_metal_matvec_plan_fallback(&plan, &plan_single, ne01, pipeline.nr0, pipeline.nsg, fallback_wave, &next);
        if (fallback != GGML_METAL_MATVEC_FALLBACK_NONE) {
            plan = next;
            if (fallback == GGML_METAL_MATVEC_FALLBACK_TAIL) {
                bid_kq_prep = bid_kq_prep_single; // a partial last threadgroup takes the per-column launch
            }
            pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op, &plan, src1_trans_mv, use_residual, name_out);
        }
    }
    const bool kq_prep = plan.prep == 1;
    const int ncols = plan.cols > 1 ? plan.cols : 0;

    if (forced_here && (!pipeline.pipeline || fallback != GGML_METAL_MATVEC_FALLBACK_NONE || pipeline.nr0 != forced->nr0)) {
        GGML_ABORT("GGML_METAL_TEST_MV_PLAN: the compiled pipeline cannot run the forced plan (%s, rows per simdgroup %d)",
            !pipeline.pipeline ? "no pipeline" : fallback == GGML_METAL_MATVEC_FALLBACK_TAIL ? "m is not whole threadgroups of it" :
            fallback == GGML_METAL_MATVEC_FALLBACK_WIDTH ? "its wave is not 64" : "it runs another row count", pipeline.pipeline ? pipeline.nr0 : 0);
    }
    if (!pipeline.pipeline) {
        return 0;
    }

    // The launch geometry comes from the pipeline as compiled for the plan.
    const int nr0 = pipeline.nr0;
    const int nr1 = pipeline.nr1;
    const int nsg = pipeline.nsg;
    const int mv_simd_w = ggml_metal_pipeline_thread_execution_width(pipeline);
    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.tg_x_offset    =*/ 0,
        /*.tg_y_offset    =*/ 0,
        /*.n_tg_x         =*/ 0,  // set below for persistent dispatch
        /*.compute_units   =*/ (int32_t)profile->compute_units,  // for CU-local row clustering (0 = disabled)
        /*.grid_x          =*/ 0,  // set below after grid computation
    };

    // the receipt of a launch that is encoded: the pipeline named is the one bound below
    if (ggml_metal_mv_receipt_requested()) {
        char attempt[64] = "none";
        if (plan_attempt.cols > 1) {
            snprintf(attempt, sizeof(attempt), "%d:%d:%d:%d", plan_attempt.cols, plan_attempt.nr0, plan_attempt.form, plan_attempt.prep);
        }
        char why[96] = "none";
        if (fallback == GGML_METAL_MATVEC_FALLBACK_TAIL) {
            snprintf(why, sizeof(why), "tail: m %d is not a whole number of %d", ne01, fallback_rows);
        } else if (fallback == GGML_METAL_MATVEC_FALLBACK_WIDTH) {
            snprintf(why, sizeof(why), "width: compiled wave %d", fallback_wave);
        }
        fprintf(stderr, "MV_RECEIPT attempt=%s fallback=%s final=0:%d:%d:%d:%d:%s\n", attempt, why,
            plan.cols, pipeline.nr0, plan.form, plan.prep, pipeline_name);
    }
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    // FC606 writes the matched ADD result directly and reads the residual at 4;
    // the false path retains the original output binding and no extra buffer.
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(use_residual ? ctx->node(idx + 1) : op), 3);
    if (use_residual) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(residual), 4);
    }
    if (kq_prep) {
        ggml_metal_encoder_set_buffer(enc, bid_kq_prep, 5);
    }
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    // shape attribution for profile=3: bucket dispatches by ne00 x ne01
    ggml_metal_encoder_set_dispatch_shape(enc, ne00, ne01);

    // Grid computation: how many threadgroups along each dimension.
    // grid_x: depends on type — dense/q8_0 types use nr0 rows per TG,
    //         while other quant types use nr0*nsg (more rows per TG).
    // grid_y: src1 rows (batch dimension), 1 for matvec.
    // grid_z: attention heads × batch (3rd and 4th tensor dimensions).
    const int32_t grid_x = plan.rows_per_threadgroup
                           ? ((ne01 + nr0 - 1) / nr0)
                           : ((ne01 + nr0*nsg - 1) / (nr0*nsg));
    const int32_t grid_y = ncols > 0 ? 1 : ((ne11 + nr1 - 1) / nr1);
    const int32_t grid_z = ne12 * ne13;

    // Pass grid_x to kernel for CU-local row clustering remapping.
    args.grid_x = grid_x;

    const int32_t total_tg = grid_x * grid_y * grid_z;
    const int32_t chunk = (int32_t)profile->max_threadgroups_per_dispatch / ggml_metal_mv_launch_of(profile, GGML_METAL_MV_LAUNCH_SPLIT_MV).units;

    // Persistent threadgroup dispatch: DISABLED — regresses all model sizes.
    //
    // Tested 2026-03-28: reducing grid_x to CU count (32 on D700) causes:
    //   0.6B: 39.49 → 29.49 (-25%)
    //   4B:   10.56 →  6.32 (-40%)
    //   9B:    7.17 →  4.53 (-37%)
    //
    // Root cause: fewer TGs = fewer wavefronts in flight = less memory
    // latency hiding. With 32 TGs on 32 CUs, each CU uses 1 of 12
    // wavefront slots. Sequential slice processing within a single TG
    // cannot compensate for the massive loss of memory-level parallelism.
    // L2 locality from sequential slices is irrelevant when the weight
    // matrix (>2 MB even for 0.6B) doesn't fit in L2 (768 KB on D700).
    //
    // The persistent loop infrastructure remains in all k-quant and basic
    // quant kernels (n_tg_x == 0 → loop runs once, zero overhead). It may
    // be useful on GPUs with fewer CUs or larger L2 caches where the
    // wavefront/locality tradeoff is different.
    //
    // Types with the persistent loop: q2_K..q6_K, q4_0..q5_1.
    // IQ types lack it — shmem lookup tables conflict with reduction buffer.
    int32_t eff_grid_x = grid_x;

    // With persistent TGs, the effective total TG count is reduced.
    // Timeout chunking still uses original grid_x for its threshold check,
    // but dispatches use eff_grid_x (which may be smaller).
    const int32_t eff_total_tg = eff_grid_x * grid_y * grid_z;

    if (chunk > 0 && eff_total_tg > chunk) {
        const int32_t chunk_x = chunk / (grid_y * grid_z);
        if (chunk_x > 0) {
            for (int32_t x_off = 0; x_off < eff_grid_x; x_off += chunk_x) {
                const int32_t x_count = ((x_off + chunk_x) < eff_grid_x) ? chunk_x : (eff_grid_x - x_off);
                args.tg_x_offset = x_off;
                ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                ggml_metal_encoder_dispatch_threadgroups(enc, x_count, grid_y, grid_z, mv_simd_w, nsg, 1);
            }
        } else {
            const int32_t chunk_y = chunk / grid_z;
            const int32_t eff_chunk_y = (chunk_y > 0) ? chunk_y : 1;
            for (int32_t x_off = 0; x_off < eff_grid_x; x_off++) {
                for (int32_t y_off = 0; y_off < grid_y; y_off += eff_chunk_y) {
                    const int32_t y_count = ((y_off + eff_chunk_y) < grid_y) ? eff_chunk_y : (grid_y - y_off);
                    args.tg_x_offset = x_off;
                    args.tg_y_offset = y_off;
                    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                    ggml_metal_encoder_dispatch_threadgroups(enc, 1, y_count, grid_z, mv_simd_w, nsg, 1);
                }
            }
        }
    } else {
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, eff_grid_x, grid_y, grid_z, mv_simd_w, nsg, 1);
    }
    // Consume the ADD only after its specialized pipeline was acquired and every
    // dispatch chunk has been submitted; otherwise the legacy path consumes one.
    if (use_residual) {
        // Activation trace after every ACTUAL fused dispatch (pipeline acquired, all
        // chunks submitted), so each activation row carries its own effective evidence.
        // requested comes from the env opt-in, effective from the route taken; there is
        // no process-wide "already traced" state that could silence a later row.
        GGML_LOG_INFO("%s: Q4_K residual epilogue: requested=1 effective=1 nodes_consumed=2 (ne00=%d ne01=%d)\n",
                      __func__, (int) ne00, (int) ne01);
        return 2;
    }
    return 1;
}

// Fused rms_norm -> MUL by the norm weight -> 1-4 Q8_0 matvecs reading it (GGML_METAL_NORM_MATVEC=1).
// The graph optimizer packs the matvecs right after the norm; the normed vector is never written.
int ggml_metal_op_norm_matvec(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    if (!ctx->use_fusion || idx + 2 >= ctx->n_nodes()) {
        return 0;
    }
    ggml_tensor * norm = ctx->node(idx);
    ggml_tensor * mul  = ctx->node(idx + 1);
    const ggml_tensor * x  = norm->src[0];
    const ggml_tensor * nw = mul->src[1];
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || mul->src[0] != norm || !nw ||
        x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ggml_nrows(x) != 1 || x->ne[0] % 64 != 0 ||
        nw->type != GGML_TYPE_F32 || !ggml_is_contiguous(nw) || !ggml_are_same_shape(nw, x) ||
        mul->type != GGML_TYPE_F32 || (mul->flags & GGML_TENSOR_FLAG_OUTPUT) || (norm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    ggml_tensor * mm[4] = { nullptr, nullptr, nullptr, nullptr };
    int nmat = 0;
    while (nmat < 4 && idx + 2 + nmat < ctx->n_nodes() && ggml_metal_norm_matvec_mm_ok(ctx->node(idx + 2 + nmat), mul)) {
        mm[nmat] = ctx->node(idx + 2 + nmat);
        nmat++;
    }
    if (nmat == 0) {
        return 0;
    }
    // the norm and its weighted output are not written: nothing else may read them
    int uses_norm = 0, uses_mul = 0, x_index = -1;
    for (int i = 0; i < ctx->graph_n_nodes(); i++) {
        const ggml_tensor * t = ctx->graph_node(i);
        x_index = t == x ? i : x_index;
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            uses_norm += t->src[s] == norm;
            uses_mul  += t->src[s] == mul;
        }
    }
    if (uses_norm != 1 || uses_mul != nmat) {
        return 0;
    }

    // the residual add that produced x stored its mean of squares in its slot (ggml_metal_op_add_scale)
    const bool has_mean = ggml_metal_norm_matvec_add_scale() && x->op == GGML_OP_ADD && ggml_metal_add_scale_applies(x) &&
                          x_index >= 0 && x_index < GGML_METAL_NORM_MEANS_SLOTS;
    const int lanes = 8;

    ggml_metal_library_t lib = ctx->lib;
    char name[256];
    snprintf(name, sizeof(name), "kernel_norm_mul_mv_q8_0_f32_lab_k=%d_mean=%d_nxpsg=%d", nmat, has_mean ? 1 : 0, lanes);
    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, (int16_t) nmat,  FC_NORM_MUL_MV + 0);
        ggml_metal_cv_set_bool (cv, has_mean,        FC_NORM_MUL_MV + 1);
        ggml_metal_cv_set_int16(cv, (int16_t) lanes, FC_NORM_MUL_MV + 2);
        pipeline = ggml_metal_library_compile_pipeline(lib, "kernel_norm_mul_mv_q8_0_f32_lab", name, cv);
        ggml_metal_cv_free(cv);
    }
    if (!pipeline.pipeline) {
        return 0;
    }
    const int width = ggml_metal_pipeline_thread_execution_width(pipeline);
    if (width != 32 && width != 64) {
        return 0;
    }
    const int rows_per_group = width/lanes;

    ggml_metal_kargs_norm_mul_mv args = {};
    args.ne00 = (int32_t) x->ne[0];
    memcpy(&args.eps, norm->op_params, sizeof(float));
    int32_t  rows[4]  = { 0, 0, 0, 0 };
    int32_t  start[4] = { 0, 0, 0, 0 };
    uint64_t nb01[4]  = { 0, 0, 0, 0 };
    int total = 0;
    for (int k = 0; k < nmat; k++) {
        rows[k]  = (int32_t) mm[k]->src[0]->ne[1];
        nb01[k]  = mm[k]->src[0]->nb[1];
        start[k] = total;
        total += (rows[k] + rows_per_group - 1)/rows_per_group;
    }
    args.rows0 = rows[0]; args.rows1 = rows[1]; args.rows2 = rows[2]; args.rows3 = rows[3];
    args.nb01_0 = nb01[0]; args.nb01_1 = nb01[1]; args.nb01_2 = nb01[2]; args.nb01_3 = nb01[3];
    args.start1 = start[1]; args.start2 = start[2]; args.start3 = start[3];

    // every member is checked for overlap with work still in flight
    for (int i = 1; i < 2 + nmat; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i), idx + i)) {
            ggml_metal_op_concurrency_reset(ctx, idx + i);
            break;
        }
    }

    ggml_metal_buffer_id bid_x = ggml_metal_get_buffer_id(x);
    const ggml_metal_buffer_id bid_mean = has_mean ? ggml_metal_device_get_norm_means(ctx->dev, x_index) : bid_x;

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_x, 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(nw), 2);
    ggml_metal_encoder_set_buffer  (enc, bid_mean, 3);
    for (int k = 0; k < 4; k++) {
        const ggml_tensor * m = mm[k < nmat ? k : 0];
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(m->src[0]), 4 + k);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(m), 8 + k);
    }
    ggml_metal_encoder_set_dispatch_shape(enc, args.ne00, total*rows_per_group);
    ggml_metal_encoder_dispatch_threadgroups(enc, total, 1, 1, width, 1, 1);
    ggml_metal_tiled_test_route(ctx, idx, 2 + nmat, "norm-matvec");
    return 2 + nmat;
}

// Residual add that also stores the mean of squares of its output in its slot of the device-owned buffer.
int ggml_metal_op_add_scale(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);
    const int slot = ctx->graph_index(idx);
    if (slot >= GGML_METAL_NORM_MEANS_SLOTS) {
        return 0;
    }
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, "kernel_add_scale_f32");
    if (!pipeline.pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, "kernel_add_scale_f32", "kernel_add_scale_f32", nullptr);
    }
    GGML_ASSERT(pipeline.pipeline);
    const int width = ggml_metal_pipeline_thread_execution_width(pipeline);
    const int n4 = (int) (op->ne[0]/4);
    int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ((n4 + width - 1)/width)*width);
    nth = std::max(nth, width);

    ggml_metal_kargs_add_scale args = { (int32_t) op->ne[0] };
    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_device_get_norm_means(ctx->dev, slot), 4);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, 32*sizeof(float), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);
    return 1;
}
