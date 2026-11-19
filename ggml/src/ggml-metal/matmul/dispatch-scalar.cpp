// Scalar mul_mv dispatch: the workhorse for single-token generation.
// Extracted from dispatch.cpp — handles all quant types with shmem_reduce
// for non-32-wide GPUs (Intel NW<32, AMD NW=64).
#include "dispatch.h"
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
static int dispatch_mul_mat_q8_rows8(ggml_metal_op_t ctx, int idx) {
    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
    ggml_tensor * op = ctx->node(idx);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (profile->q8_rows8_min == 0 || src0->ne[1] < (int64_t) profile->q8_rows8_min) {
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
    const int nsg = N_SG_Q8_0, nr0 = 8, lanes_per_block = 4;
    // same bytes as the default kernel: tree row sum, and one block per lane there (blocks = nsg x width / lanes per block)
    if (width != 64 || profile->mv_reduce != 1 || src0->ne[0] != (int64_t) nsg*(width/lanes_per_block)*32) {
        return 0;
    }
    if (ggml_metal_q8_load_bytes(profile, src0) != 4) {
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
static int dispatch_mul_mat_q8_group(ggml_metal_op_t ctx, int idx) {
    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
    ggml_tensor * op = ctx->node(idx);
    if (profile->mv_group != 1 || !ctx->use_fusion || !op->src[1] || !ggml_metal_norm_matvec_mm_ok(op, op->src[1])) {
        return 0;
    }
    const ggml_tensor * act = op->src[1];
    ggml_tensor * mm[4] = { nullptr, nullptr, nullptr, nullptr };
    int nmat = 0;
    while (nmat < 4 && idx + nmat < ctx->n_nodes() && ggml_metal_norm_matvec_mm_ok(ctx->node(idx + nmat), act)) {
        mm[nmat] = ctx->node(idx + nmat);
        nmat++;
    }
    if (nmat < 2) {
        return 0;
    }

    const int  width = (int) profile->simd_width;
    const bool sr    = width != 32;
    const int  nsg   = N_SG_Q8_0;
    const int  nr0   = N_R0_Q8_0;
    const int  tail  = (sr && profile->mv_reduce == 1) ? 1 : 0;

    char name[256];
    snprintf(name, sizeof(name), "kernel_mul_mvg_q8_0_f32_k=%d_nsg=%d_sr=%d_nr0=%d_tail=%d", nmat, nsg, sr ? 1 : 0, nr0, tail);
    // word loads only when every matrix of the group can take them
    int q8_load = 4;
    for (int k = 0; k < nmat; k++) {
        if (ggml_metal_q8_load_bytes(profile, mm[k]->src[0]) != 4) {
            q8_load = 1;
        }
    }
    if (q8_load != 1) {
        const size_t used = strlen(name);
        snprintf(name + used, sizeof(name) - used, "_q8load=%d", q8_load);
    }
    // the matrices of a group share one activation
    const int q8_act = ggml_metal_q8_act_load_bytes(profile, mm[0]->src[1]);
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
         ((op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) && profile->f_scalar_w64 != 1)));

    if (scalar_f_broken) {
        return 0;
    }

    if (profile->mv_group == 1 && op->src[0]->type == GGML_TYPE_Q8_0) {
        const int n = dispatch_mul_mat_q8_group(ctx, idx);
        if (n > 0) {
            return n;
        }
    }

    if (op->src[0]->type == GGML_TYPE_Q8_0 && dispatch_mul_mat_q8_rows8(ctx, idx) == 1) {
        return 1;
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

    // Compute optimal NR0 from model + hardware profile.
    // For basic quant types (q4_0..q5_1), this drives the function constant FC_mul_mv_nr0,
    // giving the kernel a dynamic NR0 at compile time (8 for small models, 4 for large).
    // For k-quant types, the function constant is set but ignored — they still use the
    // template parameter (NR0 override requires changing all k-quant entry points too).
    const ggml_type type = op->src[0]->type;
    const struct mv_format_row * type_row = ggml_metal_mv_format(type);
    const bool kquant_rows = type_row != nullptr && type_row->shape_rows;
    int nr0_dynamic = kquant_rows ? ggml_metal_mv_nr0_from_rows(type, ne00, ne01) : ggml_metal_compute_nr0_basic(profile, model, ne00);

    const bool use_shmem_reduce = (profile->simd_width != 32);
    const bool src1_trans_mv = ggml_is_transposed(op->src[1]);
    ggml_tensor * residual = ggml_metal_match_q4_k_residual_add(ctx, idx, op);
    const bool use_residual = residual != nullptr;

    // Per-shape row of this card (hw mv rows); a row of the rows file above wins. A Q4_K weight with headers prepared
    // at load (one plane, one f32 column, no residual epilogue) may take a prep row: its entry point reads them at 5.
    bool kq_prep = false;
    ggml_metal_buffer_id bid_kq_prep = { nullptr, 0 };
    if (kquant_rows) {
        const bool kq_prep_test = ggml_metal_kq_prep_test_requested();
        if (type == GGML_TYPE_Q4_K && !use_residual && !src1_trans_mv && op->src[1]->type == GGML_TYPE_F32 &&
            ne02 == 1 && ne03 == 1 && ne11 == 1 && ne12 == 1 && ne13 == 1 && (profile->kq_prep == 1 || kq_prep_test)) {
            bid_kq_prep = ggml_metal_get_kq_prep_id(op->src[0]);
        }
        if (nr0_dynamic == 0) {
            const struct ggml_device_mv_row * row = ggml_metal_mv_hw_row(profile, 0, type, ne00, ne01, 1, bid_kq_prep.metal != nullptr);
            if (row != nullptr) {
                nr0_dynamic = row->nr0;
                kq_prep = row->prep == 1;
            }
        }
        if (kq_prep_test && bid_kq_prep.metal != nullptr) {
            kq_prep = true;
        }
    }
    // A cols row of this card takes a 2-4 column decode batch of this weight in one launch (grid 1 in y).
    int ncols = 0;
    int col_form = 0;
    if (kquant_rows && type != GGML_TYPE_Q5_K && ne11 >= 2 && ne11 <= 4 && !use_residual && !src1_trans_mv && op->src[1]->type == GGML_TYPE_F32 &&
        ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1) {
        int test_nr0 = 0;
        int test_form = 0;
        const bool cols_test = ggml_metal_kq_cols_test(&test_nr0, &test_form);
        ggml_metal_buffer_id bid_cols = { nullptr, 0 };
        if (type == GGML_TYPE_Q4_K && (profile->kq_prep == 1 || ggml_metal_kq_prep_test_requested())) {
            bid_cols = ggml_metal_get_kq_prep_id(op->src[0]);
        }
        const struct ggml_device_mv_row * cols_row = ggml_metal_mv_hw_row(profile, 0, type, ne00, ne01, ne11, bid_cols.metal != nullptr);
        if (cols_row != nullptr || cols_test) {
            ncols = ne11;
            col_form = cols_row != nullptr ? cols_row->form : test_form;
            if (ne11 == 4) {
                col_form &= 3; // paired halves take 2 or 3 columns
            }
            nr0_dynamic = cols_row != nullptr ? cols_row->nr0 : test_nr0;
            kq_prep = cols_row != nullptr ? cols_row->prep == 1 : bid_cols.metal != nullptr;
            bid_kq_prep = bid_cols;
        }
    }
    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op, use_shmem_reduce, src1_trans_mv, nr0_dynamic, use_residual, kq_prep, ncols, col_form);

    if (!pipeline.pipeline) {
        return 0;
    }

    // NR0 comes from the pipeline (which used our nr0_dynamic override for basic types,
    // or the hardcoded N_R0_* default for other types).
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
    const int32_t grid_x = ggml_metal_mv_rows_per_threadgroup(op->src[0]->type)
                           ? ((ne01 + nr0 - 1) / nr0)
                           : ((ne01 + nr0*nsg - 1) / (nr0*nsg));
    const int32_t grid_y = ncols > 0 ? 1 : ((ne11 + nr1 - 1) / nr1);
    const int32_t grid_z = ne12 * ne13;

    // Pass grid_x to kernel for CU-local row clustering remapping.
    args.grid_x = grid_x;

    const int32_t total_tg = grid_x * grid_y * grid_z;
    const int32_t chunk = (int32_t)profile->max_threadgroups_per_dispatch / 16;

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
