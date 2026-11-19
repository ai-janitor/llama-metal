#include "../ops/ggml-metal-ops-internal.h"
#include "act-copy.h"
#include "lanes/select.h"

#include <algorithm>
#include <mutex>
#include <set>
#include <vector>

struct ggml_metal_act_copy {
    enum ggml_type                   precision = GGML_TYPE_COUNT;
    ggml_metal_buffer_t              buf       = nullptr;
    size_t                           size      = 0;
    // a grown buffer is kept until the context ends: a graph still running may read it
    std::vector<ggml_metal_buffer_t> retired;
    std::mutex                       mu;
    std::set<uint64_t>               logged;
};

// The ops that may ask for a copy, with the operand they copy. A family that joins adds its line.
static const struct { enum ggml_op op; int family; int route; int src; } act_copy_users[] = {
    { GGML_OP_MUL_MAT, GGML_METAL_LANE_FAMILY_MATMUL, GGML_METAL_LANE_ROUTE_MM_TILED, 1 },
};

struct ggml_metal_act_copy * ggml_metal_act_copy_init(void) {
    return new ggml_metal_act_copy();
}

void ggml_metal_act_copy_free(struct ggml_metal_act_copy * st) {
    if (!st) return;
    if (st->buf) ggml_metal_buffer_free(st->buf);
    for (ggml_metal_buffer_t b : st->retired) ggml_metal_buffer_free(b);
    delete st;
}

void ggml_metal_act_copy_set_precision(struct ggml_metal_act_copy * st, enum ggml_type type) {
    if (st) st->precision = type;
}

void ggml_metal_act_copy_reserve(struct ggml_metal_act_copy * st, ggml_metal_device_t dev, struct ggml_cgraph * gf) {
    if (!st || (st->precision != GGML_TYPE_BF16 && st->precision != GGML_TYPE_F16)) return;
    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(dev);
    int n = 0;
    const struct ggml_metal_act_copy_row * rows = ggml_metal_act_copy_rows_get(&n);
    const size_t elem = ggml_type_size(st->precision);
    size_t need = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const struct ggml_tensor * node = ggml_graph_node(gf, i);
        for (const auto & user : act_copy_users) {
            if (node->op != user.op || node->src[user.src] == nullptr) continue;
            const struct ggml_tensor * src = node->src[user.src];
            const uint64_t bytes = (uint64_t) elem * (uint64_t) ggml_nelements(src);
            int why = 0;
            // the largest re-read count: the op's own count is checked when it is encoded
            ggml_metal_act_copy_decide_in(rows, n, profile->pci_vendor, profile->pci_device, user.family, user.route,
                    st->precision, src->type, INT64_MAX, bytes, &why);
            if (why == GGML_METAL_ACT_COPY_APPLIES) need = std::max(need, (size_t) bytes);
        }
    }
    if (need > st->size) {
        if (st->buf) st->retired.push_back(st->buf);
        st->buf  = ggml_metal_buffer_init(dev, need, false);
        st->size = st->buf ? need : 0;
    }
}

struct ggml_metal_act_plan ggml_metal_act_copy_plan(ggml_metal_op_t ctx, int idx, int family, int route, const struct ggml_tensor * src, int64_t rereads) {
    struct ggml_metal_act_plan plan = {};
    plan.type = GGML_TYPE_COUNT;
    plan.why  = GGML_METAL_ACT_COPY_NO_PRECISION;
    struct ggml_metal_act_copy * st = ctx->act_copy;
    if (!st) return plan;

    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
    const size_t elem = st->precision == GGML_TYPE_COUNT ? 0 : ggml_type_size(st->precision);
    const uint64_t bytes = (uint64_t) elem * (uint64_t) ggml_nelements(src);
    int n = 0;
    const struct ggml_metal_act_copy_row * rows = ggml_metal_act_copy_rows_get(&n);
    plan.row = ggml_metal_act_copy_decide_in(rows, n, profile->pci_vendor, profile->pci_device, family, route,
            st->precision, src->type, rereads, bytes, &plan.why);
    if (plan.why != GGML_METAL_ACT_COPY_APPLIES || st->buf == nullptr || bytes > st->size) {
        return plan;
    }

    int n_lanes = 0;
    const struct ggml_metal_lane_row * lanes = ggml_metal_lane_rows_get(&n_lanes);
    plan.lane = ggml_metal_negotiate_lane_in(lanes, n_lanes, profile->lane_ok, family, route,
            ctx->node(idx)->src[0]->type, st->precision, &plan.lane_step);

    plan.applies = 1;
    plan.type    = st->precision;
    plan.bid     = ggml_metal_buffer_base_id(st->buf);
    plan.nb[0]   = elem;
    plan.nb[1]   = plan.nb[0] * (uint64_t) src->ne[0];
    plan.nb[2]   = plan.nb[1] * (uint64_t) src->ne[1];
    plan.nb[3]   = plan.nb[2] * (uint64_t) src->ne[2];

    // one line per family, route and weight type: the two rows are the record of the choice
    const uint64_t key = ((uint64_t) family << 40) | ((uint64_t) route << 32) | (uint64_t) ctx->node(idx)->src[0]->type;
    bool first = false;
    {
        std::lock_guard<std::mutex> lock(st->mu);
        first = st->logged.insert(key).second;
    }
    if (first) {
        static const char * const steps[] = { "?", "native", "exact-other", "wider", "narrower" };
        GGML_LOG_INFO("act-copy: family %d route %d weight %s on %04x:%04x: copy %s (%s) · lane %s step %s (%s)\n",
                family, route, ggml_type_name(ctx->node(idx)->src[0]->type), profile->pci_vendor, profile->pci_device,
                ggml_type_name(plan.type), plan.row->reason,
                plan.lane ? plan.lane->lane : "none", steps[plan.lane ? plan.lane_step : 0], plan.lane ? plan.lane->reason : "no lane holds the type");
    }
    return plan;
}

void ggml_metal_act_copy_encode(ggml_metal_op_t ctx, int idx, const struct ggml_metal_act_plan * plan, const struct ggml_tensor * src) {
    struct ggml_metal_act_slot & slot = ctx->act_slot;
    const struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);

    bool held = slot.valid && slot.metal == bid_src.metal && slot.offs == bid_src.offs && slot.type == (int) plan->type;
    for (int i = 0; held && i < 4; ++i) {
        held = slot.ne[i] == src->ne[i] && slot.nb[i] == src->nb[i];
    }
    if (held) {
        return;
    }

    // ops encoded so far may still read the previous copy
    ggml_metal_op_concurrency_reset(ctx, idx);

    auto pipeline = ggml_metal_library_get_pipeline_cpy(ctx->lib, src->type, plan->type);

    const int32_t ne00 = (int32_t) src->ne[0];
    const int32_t ne01 = (int32_t) src->ne[1];
    const int32_t ne02 = (int32_t) src->ne[2];
    const int32_t ne03 = (int32_t) src->ne[3];

    // threadgroup shape as ggml_metal_op_cpy picks it for two plain element types
    const int64_t nk0 = ne00;
    int nth = std::min<int>(nk0*ne01, 256);
    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;
        if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }
    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ src->nb[0],
        /*.nb01 =*/ src->nb[1],
        /*.nb02 =*/ src->nb[2],
        /*.nb03 =*/ src->nb[3],
        /*.ne0  =*/ ne00,
        /*.ne1  =*/ ne01,
        /*.ne2  =*/ ne02,
        /*.ne3  =*/ ne03,
        /*.nb0  =*/ plan->nb[0],
        /*.nb1  =*/ plan->nb[1],
        /*.nb2  =*/ plan->nb[2],
        /*.nb3  =*/ plan->nb[3],
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes   (ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx->enc, bid_src,   1);
    ggml_metal_encoder_set_buffer  (ctx->enc, plan->bid, 2);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    // the op's kernel reads the copy only after it is written
    ggml_metal_op_concurrency_reset(ctx, idx);

    slot.valid = true;
    slot.metal = bid_src.metal;
    slot.offs  = bid_src.offs;
    slot.span  = ggml_nbytes(src);
    slot.type  = (int) plan->type;
    for (int i = 0; i < 4; ++i) {
        slot.ne[i] = src->ne[i];
        slot.nb[i] = src->nb[i];
    }
}

void ggml_metal_act_copy_note_write(ggml_metal_op_t ctx, const struct ggml_tensor * node) {
    struct ggml_metal_act_slot & slot = ctx->act_slot;
    if (!slot.valid) {
        return;
    }
    const struct ggml_metal_buffer_id bid = ggml_metal_get_buffer_id(node);
    if (bid.metal != slot.metal) {
        return;
    }
    const size_t n = ggml_nbytes(node);
    if (bid.offs < slot.offs + slot.span && slot.offs < bid.offs + n) {
        slot.valid = false;
    }
}
