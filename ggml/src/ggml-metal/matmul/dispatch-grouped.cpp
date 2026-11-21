// Grouped q4_K/q6_K fusion dispatch — fuses 2-8 consecutive matvec ops into
// a single Metal dispatch with more threadgroups to improve GPU occupancy.
// Extracted from dispatch.cpp to keep each dispatch strategy in its own file.
#include "dispatch.h"
#include "matvec-plan.h"
#include "tuning.h"
#include "../lanes/select.h"
#include "ggml-metal-probe.h"
#include <cstdlib>
#include <cstring>
GGML_METAL_FAMILY(matmul)

static bool read_grouped_env() {
    const char * value = std::getenv("GGML_METAL_MV_GROUPED");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

// GGML_METAL_MV_GROUPED=0 disables grouped matvec dispatch.
static bool grouped_enabled() {
    static const bool on = read_grouped_env();
    return on;
}

int dispatch_mul_mat_grouped(ggml_metal_op_t ctx, int idx) {
    if (!grouped_enabled()) { return 0; }
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
    const struct ggml_metal_card * card = ggml_metal_device_get_card(ctx->dev);
    const struct ggml_model_profile  * model   = ggml_metal_device_get_model_profile(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // =========================================================================
    // Grouped mul_mv dispatch — fuse consecutive matmuls into one dispatch.
    // =========================================================================
    // On both GCN (D700, NW=64) and RDNA (5500M, NW=32), individual matmul
    // dispatches have too few threadgroups to fully occupy the GPU. This scans
    // forward through the compute graph and fuses up to 8 consecutive q4_K/q6_K
    // matvec operations into a single dispatch with more threadgroups.
    //
    // Requirements for fusion:
    //   - Same ne00 (K dimension / inner dimension must match)
    //   - Matvec only (ne11 == 1, i.e., single token generation)
    //   - Same Metal buffer for src0 and dst (offsets within the buffer differ)
    //   - q4_K or q6_K quant type (both use THREADS_PER_BLOCK=16)
    //   - All types must share the same N_R0_KQUANT for correct grid computation
    //
    // Returns the number of ops consumed (>1 if fusion happened).
    const struct mv_format_row * fmt = ggml_metal_mv_format(op->src[0]->type);
    if (ne11 != 1 ||
        fmt == nullptr || !fmt->grouped ||
        op->src[1]->type != GGML_TYPE_F32) {
        return 0;
    }

    // Group descriptor — MUST match the Metal kernel's mv_group struct layout
    // exactly (same field order, sizes, alignment). Any mismatch causes
    // silent data corruption in the kernel.
    struct mv_group_host {
        uint64_t src0_off;    // byte offset of this group's weights from base
        uint64_t dst_off;     // byte offset of this group's output from base
        int32_t  ne01;        // number of output rows for this group
        int32_t  nb01;        // byte stride between weight rows
        uint64_t nb02;        // byte stride for 3rd dim (attention heads)
        uint64_t nb03;        // byte stride for 4th dim (batch)
        int32_t  tg_start;    // first threadgroup index for this group
        int32_t  quant_type;  // 0=q4_K, 1=q6_K (matches MV_GROUP_TYPE_* in kernel)
    };

    // Kernel arguments — MUST match ggml_metal_kargs_mul_mv_grouped in the
    // Metal kernel exactly. Contains shared parameters + per-group descriptors.
    struct kargs_grouped {
        int32_t  ne00;        // K dimension (shared across all groups)
        int32_t  ne02;        // number of attention heads (src0)
        int32_t  ne10;        // src1 row length (== ne00)
        int32_t  ne11;        // src1 rows (1 for matvec)
        int32_t  ne12;        // src1 heads (broadcast dimension)
        uint64_t nb10;        // src1 byte strides
        uint64_t nb11;
        uint64_t nb12;
        uint64_t nb13;
        int32_t  ne0;         // output row length
        int32_t  ne1;         // output rows
        int16_t  r2;          // GQA broadcast ratio (ne12/ne02)
        int16_t  r3;          // batch broadcast ratio
        int32_t  n_groups;    // number of fused groups (2..8)
        int32_t  compute_units;   // CU count from hardware profile
        int32_t  l1_cache_per_cu; // L1 cache bytes per CU from hardware profile
        mv_group_host groups[8];  // per-group descriptors (MAX_MV_GROUPS=8)
    };

    // Always use shmem reduction for the grouped kernel — it's the safe
    // path on all devices. GCN needs it (simd_sum broken at NW=64),
    // RDNA doesn't need it but it works fine and keeps the kernel uniform.
    // The plan of a group member (matmul/matvec-plan.h): shared-memory sums, rows per simdgroup from the cache rule;
    // the card's group row may replace them once the members are known (below).
    struct ggml_metal_matvec_in in = {};
    in.kind        = 1;
    in.weight      = (int32_t) op->src[0]->type;
    in.k           = ne00;
    in.m           = ne01;
    in.cols        = 1;
    in.model_known = model != nullptr && model->populated ? 1 : 0;
    in.act_bytes   = (uint32_t) ne00 * (uint32_t) sizeof(float);
    const struct ggml_metal_matvec_plan plan = ggml_metal_matvec_plan_for(card, &in);
    // The test hook GGML_METAL_TEST_MV_PLAN (matmul/matvec-plan.h): a plan forced on the single matvec of this weight
    // keeps it out of a grouped launch; a plan forced on the group replaces the rows and prep chosen below.
    const struct ggml_metal_matvec_forced * forced = ggml_metal_matvec_forced_env();
    if (ggml_metal_matvec_forced_is_for(forced, 0, in.weight, ne00, ne01, ne11)) {
        return 0;
    }
    const bool forced_here = ggml_metal_matvec_forced_is_for(forced, 1, in.weight, ne00, ne01, ne11) == 1;
    // Grouped kernels retain their existing output path; FC606 is scalar Q4_K-only. This pipeline gives the
    // simdgroup count and the wave width: it is the format's own one-column entry, at the format's row count.
    struct ggml_metal_matvec_plan width_plan = plan;
    width_plan.nr0     = fmt->nr0;
    width_plan.nr0_set = 0;
    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op, &width_plan, false, false, nullptr);

    if (!pipeline.pipeline) {
        return 0;
    }

    const int nsg = pipeline.nsg;  // simdgroups per threadgroup (N_SG_Q4_K)
    // The grouped kernel is compiled with the plan's rows per simdgroup as a function constant.
    int nr0 = forced_here ? forced->nr0 : plan.nr0;
    const int mv_simd_w = ggml_metal_pipeline_thread_execution_width(pipeline);  // 64 on GCN

    // Base buffer IDs — all groups must be in the same Metal buffer.
    // We store byte offsets relative to these base pointers.
    struct ggml_metal_buffer_id bid_src0_base = ggml_metal_get_buffer_id(op->src[0]);
    struct ggml_metal_buffer_id bid_dst_base  = ggml_metal_get_buffer_id(op);

    // Scan forward through the compute graph, collecting compatible ops.
    kargs_grouped gargs = {};
    int n_groups = 0;
    int total_tg = 0;     // cumulative threadgroup count across all groups
    int n_consumed = 0;   // how many ops from the graph we consumed

    // Per-shape group row of this card: members of the row's type and shape run at its rows per simdgroup
    // (a prep row needs every member's prepared headers in one run); the scan repeats with that value.
    bool kq_prep = false;
    ggml_metal_buffer_id bid_kq_prep = { nullptr, 0 };
    const ggml_tensor * member_src0[8] = {};
    // the most members of one launch: the card's number, never above the 8 descriptors the kernel holds
    const int group_max = std::min(8, (int) ggml_metal_mv_launch_of(profile, GGML_METAL_MV_LAUNCH_GROUP_KQ).units);
    for (int pass = 0; pass < 2; pass++) {
    gargs = {};
    n_groups = 0;
    total_tg = 0;
    n_consumed = 0;
    bid_src0_base = ggml_metal_get_buffer_id(op->src[0]);
    bid_dst_base  = ggml_metal_get_buffer_id(op);

    bool whole = true; // every member's rows are whole threadgroups at nr0
    for (int g = 0; g < group_max && (idx + g) < ctx->n_nodes(); g++) {
        ggml_tensor * gop = ctx->node(idx + g);
        if (gop->op != GGML_OP_MUL_MAT) break;

        // Only fuse q4_K and q6_K — they share THREADS_PER_BLOCK=16
        // and N_R0_KQUANT, so threadgroup geometry is identical.
        const struct mv_format_row * gfmt = ggml_metal_mv_format(gop->src[0]->type);
        const bool is_q4k = (gop->src[0]->type == GGML_TYPE_Q4_K);
        if (gfmt == nullptr || !gfmt->grouped) break;
        if (gop->src[0]->ne[0] != ne00) break;  // K dimension must match
        if (gop->src[1]->ne[1] != 1) break;     // must be matvec (batch=1)
        if (g > 0 && !(gop->flags & GGML_TENSOR_FLAG_COMPUTE)) break;
        // a matvec the test hook forces as a single launch is no member: the scan ends before it
        if (g > 0 && ggml_metal_matvec_forced_is_for(forced, 0, (int32_t) gop->src[0]->type, gop->src[0]->ne[0], gop->src[0]->ne[1], (int32_t) gop->src[1]->ne[1])) break;

        // One activation buffer is bound for the entire group. Exact tensor
        // identity also prevents a later group's activation consuming an earlier
        // group's output: every group must read the same pre-existing input.
        if (gop->src[1] != op->src[1]) break;

        // All groups must share the same underlying Metal buffer
        // (we pass one buffer binding and use byte offsets).
        struct ggml_metal_buffer_id bid_src0_g = ggml_metal_get_buffer_id(gop->src[0]);
        struct ggml_metal_buffer_id bid_dst_g  = ggml_metal_get_buffer_id(gop);
        if (bid_src0_g.metal != bid_src0_base.metal) break;
        if (bid_dst_g.metal != bid_dst_base.metal) break;

        // Compute how many threadgroups this group needs.
        // Each TG covers nr0 * nsg rows = N_R0_KQUANT * N_SG_Q4_K rows.
        int gne01 = (int32_t)gop->src[0]->ne[1];
        int tg_count = (gne01 + nr0 * nsg - 1) / (nr0 * nsg);
        whole = whole && gne01 % (nr0 * nsg) == 0;

        // Collect absolute offsets first: graph order need not match allocation
        // order. A later member can precede the first in either Metal buffer.
        gargs.groups[n_groups].src0_off    = bid_src0_g.offs;
        gargs.groups[n_groups].dst_off     = bid_dst_g.offs;
        // Metal buffer bindings must precede every accessed member. Subtracting
        // a later binding underflows uint64 offsets and can discard GPU stores.
        if (bid_src0_g.offs < bid_src0_base.offs) bid_src0_base.offs = bid_src0_g.offs;
        if (bid_dst_g.offs  < bid_dst_base.offs)  bid_dst_base.offs  = bid_dst_g.offs;
        gargs.groups[n_groups].ne01        = gne01;
        gargs.groups[n_groups].nb01        = (int32_t)gop->src[0]->nb[1];
        gargs.groups[n_groups].nb02        = gop->src[0]->nb[2];
        gargs.groups[n_groups].nb03        = gop->src[0]->nb[3];
        gargs.groups[n_groups].tg_start    = total_tg;
        gargs.groups[n_groups].quant_type  = is_q4k ? 0 : 1;
        member_src0[n_groups] = gop->src[0];

        total_tg += tg_count;
        n_groups++;
        n_consumed++;
    }

    if (forced_here && n_groups >= 2 && !whole) {
        GGML_ABORT("GGML_METAL_TEST_MV_PLAN: with a forced row count m is a whole number of nr0 x nsg rows");
    }
    if (pass == 0 && !forced_here && !whole && nr0 != fmt->nr0) {
        // the kernel loads every weight row of a threadgroup before the output bound: a member that would leave a
        // partial threadgroup at these rows sends the launch back to the rows the kernel is compiled for
        nr0 = fmt->nr0;
        continue;
    }
    if (pass == 0 && n_groups >= 2 && (card->n_mv_rows > 0 || forced_here) && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1) {
        // same type and shape on every member; headers prepared for all, in one run of one buffer
        bool same = true;
        bool prep_ok = (card->kq_prep == 1 || (forced_here && forced->prep == 1)) && op->src[0]->type == GGML_TYPE_Q4_K;
        int first = 0; // the member with the lowest weight offset: the bindings start there
        ggml_metal_buffer_id member_prep[8] = {};
        for (int group = 0; group < n_groups; ++group) {
            same = same && member_src0[group]->type == op->src[0]->type && member_src0[group]->ne[1] == op->src[0]->ne[1];
            if (gargs.groups[group].src0_off < gargs.groups[first].src0_off) {
                first = group;
            }
            member_prep[group] = prep_ok ? ggml_metal_get_kq_prep_id(member_src0[group]) : ggml_metal_buffer_id{ nullptr, 0 };
            prep_ok = prep_ok && member_prep[group].metal != nullptr && member_prep[group].metal == member_prep[0].metal;
        }
        for (int group = 0; group < n_groups && prep_ok; ++group) {
            const uint64_t woff = gargs.groups[group].src0_off - gargs.groups[first].src0_off;
            prep_ok = woff % 144 == 0 && member_prep[group].offs >= member_prep[first].offs &&
                      member_prep[group].offs - member_prep[first].offs == woff / 144 * 32;
        }
        struct ggml_metal_matvec_in row_in = in;
        row_in.use_rows       = same ? 1 : 0;
        row_in.prep_available = prep_ok ? 1 : 0;
        if (forced_here) {
            struct ggml_metal_matvec_plan forced_plan = {};
            const char * why = same ? nullptr : "the members of the launch differ in weight type or shape";
            if (!same || !ggml_metal_matvec_forced_plan(card, &row_in, forced, &forced_plan, &why)) {
                GGML_ABORT("GGML_METAL_TEST_MV_PLAN: %s", why);
            }
            kq_prep = forced_plan.prep == 1;
            if (kq_prep) {
                bid_kq_prep = member_prep[first];
            }
            break;
        }
        const struct ggml_metal_matvec_plan row_plan = ggml_metal_matvec_plan_for(card, &row_in);
        if (row_plan.shape_row >= 0) {
            kq_prep = row_plan.prep == 1;
            if (kq_prep) {
                bid_kq_prep = member_prep[first];
            }
            if (row_plan.nr0 != nr0) {
                nr0 = row_plan.nr0;
                continue;
            }
        }
    }
    break;
    }

    if (forced_here && n_groups >= 2 && !(ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1)) {
        GGML_ABORT("GGML_METAL_TEST_MV_PLAN: the grouped launch is not one plane of one f32 activation column");
    }
    // Only use grouped dispatch if we found 2+ ops to fuse.
    // With 1 op, the overhead of the group lookup isn't worth it.
    if (n_groups < 2) {
        return 0;
    }

    // Rebase every descriptor against the minimum binding, keeping all accesses
    // nonnegative even when weights or outputs were allocated out of graph order.
    for (int group = 0; group < n_groups; ++group) {
        gargs.groups[group].src0_off -= bid_src0_base.offs;
        gargs.groups[group].dst_off  -= bid_dst_base.offs;
    }

    // Fill in the shared parameters (same for all groups).
    gargs.ne00 = ne00;
    gargs.ne02 = ne02;
    gargs.ne10 = ne10;
    gargs.ne11 = ne11;
    gargs.ne12 = ne12;
    gargs.nb10 = nb10;
    gargs.nb11 = nb11;
    gargs.nb12 = nb12;
    gargs.nb13 = nb13;
    gargs.ne0  = ne0;
    gargs.ne1  = ne1;
    gargs.r2   = r2;
    gargs.r3   = r3;
    gargs.n_groups = n_groups;
    gargs.compute_units   = (int32_t)profile->compute_units;
    gargs.l1_cache_per_cu = (int32_t)profile->l1_cache_per_cu;

    // The grouped kernel is compiled with NR0 as a function constant
    // (FC_grouped_nr0), allowing different NR0 per model. Pipeline is
    // cached by name (includes NR0), so each variant compiles only once.
    char gpipeline_name[128];
    const int16_t selected_lane = ggml_metal_select_lane_id(
        profile, GGML_METAL_LANE_FAMILY_MATMUL, GGML_METAL_LANE_ROUTE_MV_GROUPED, op->src[0]->type, op->src[1]->type);
    // The card's hw row picks the reduce: every row stored before the row sums, or one row at a time.
    const bool g_red = plan.group_store_first == 1;
    const char * gbase = kq_prep ? (g_red ? "kernel_mul_mv_grouped_q4k_prep_red_f32" : "kernel_mul_mv_grouped_q4k_prep_f32")
                                 : (g_red ? "kernel_mul_mv_grouped_kquant_red_f32" : "kernel_mul_mv_grouped_kquant_f32");
    snprintf(gpipeline_name, sizeof(gpipeline_name),
        "%s_nr0=%d_lane=%d", gbase, nr0, selected_lane);

    auto gpipeline = ggml_metal_library_get_pipeline(lib, gpipeline_name);
    if (!gpipeline.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, (int16_t)nr0, FC_MUL_MV_GROUPED + 0);
        gpipeline = ggml_metal_library_compile_pipeline(lib,
            gbase, gpipeline_name, cv);
        ggml_metal_cv_free(cv);
    }

    if (!gpipeline.pipeline) {
        return 0;
    }
    if (ggml_metal_mv_receipt_requested()) {
        fprintf(stderr, "MV_RECEIPT attempt=none fallback=none final=1:1:%d:0:%d:%s\n", nr0, kq_prep ? 1 : 0, gpipeline_name);
    }

    ggml_metal_encoder_set_pipeline(enc, gpipeline);
    ggml_metal_encoder_set_bytes   (enc, &gargs, sizeof(gargs), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0_base, 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, bid_dst_base, 3);
    if (kq_prep) {
        ggml_metal_encoder_set_buffer(enc, bid_kq_prep, 5);
    }
    // Shared memory for cross-lane reduction: NSG x NW x NR0 x sizeof(float)
    // Use nr0 (from tuning) not pipeline.smem (which used hardcoded NR0)
    const size_t grouped_smem = (size_t)nsg * mv_simd_w * nr0 * sizeof(float);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, grouped_smem, 0);
    // Dispatch all fused groups as one grid: total_tg threadgroups.
    // Each TG has mv_simd_w x nsg threads (= 64 x 2 = 128 on GCN).
    ggml_metal_encoder_dispatch_threadgroups(enc, total_tg, 1, ne12 * ne13, mv_simd_w, nsg, 1);

    return n_consumed;  // tell caller we consumed multiple ops
}
