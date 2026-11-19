// Encode the validated small fusion patterns with the reference norm geometry.
#include "../ops/ggml-metal-ops-internal.h"
#include <cstdio>
#include <cstring>
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(norm) static ggml_metal_pipeline_with_params small_pipeline(ggml_metal_library_t lib,
        const char * base, bool c4) {
    char name[128];
    std::snprintf(name, sizeof(name), "%s%s", base, c4 ? "_4" : "");
    auto result = ggml_metal_library_get_pipeline(lib, name);
    if (!result.pipeline) {
        result = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }
    result.c4 = c4;
    return result;
}

static void small_check_concurrency(ggml_metal_op_t ctx, int idx, int count) {
    for (int i = 1; i < count; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i), idx + i)) {
            ggml_metal_op_concurrency_reset(ctx, idx + i);
            break;
        }
    }
}

static ggml_metal_kargs_norm small_norm_args(const ggml_tensor * norm,
        const ggml_tensor * out, bool c4, float multiplier) {
    const ggml_tensor * src = norm->src[0];
    ggml_metal_kargs_norm result = {};
    result.ne00 = src->ne[0];
    result.ne00_t = c4 ? src->ne[0] / 4 : src->ne[0];
    result.nb1 = out->nb[1];
    result.nb2 = out->nb[2];
    result.nb3 = out->nb[3];
    std::memcpy(&result.eps, norm->op_params, sizeof(float));
    result.nef1[0] = src->ne[1];
    result.nef2[0] = src->ne[2];
    result.nef3[0] = src->ne[3];
    result.nbf1[0] = src->nb[1];
    result.nbf2[0] = src->nb[2];
    result.nbf3[0] = src->nb[3];
    result.scale = multiplier;
    return result;
}

static int small_reference_threads(ggml_metal_op_t ctx,
        const ggml_metal_pipeline_with_params & pipeline, int width) {
    int threads = ggml_metal_device_get_profile(ctx->dev)->simd_width;
    const int maximum = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    while (threads < width && threads < maximum) {
        threads *= 2;
    }
    return std::min(width, std::min(threads, maximum));
}

static int small_encode_gate(ggml_metal_op_t ctx, int idx, int count) {
    const ggml_tensor * add = ctx->node(idx);
    const ggml_tensor * mul = ctx->node(idx + 2);
    // one element per thread: the exact exp and log of a float4 run component after component
    auto pipeline = small_pipeline(ctx->lib, "kernel_small_gate_g_f32", false);
    GGML_ASSERT(pipeline.pipeline);
    ggml_metal_kargs_small_gate args = {};
    args.ne0 = add->ne[0];
    args.ne1 = add->ne[1];
    args.ne2 = add->ne[2];
    args.ne3 = add->ne[3];
    args.nb01 = add->src[0]->nb[1];
    args.nb02 = add->src[0]->nb[2];
    args.nb03 = add->src[0]->nb[3];
    args.nb1 = mul->nb[1];
    args.nb2 = mul->nb[2];
    args.nb3 = mul->nb[3];
    const int threads = std::min(args.ne0,
            std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)));
    small_check_concurrency(ctx, idx, count);
    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(add->src[0]), 1);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(add->src[1]), 2);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(mul->src[1]), 3);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(mul), 4);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc,
            args.ne1 * args.ne2 * args.ne3, 1, 1, threads, 1, 1);
    return count;
}

// The split_l2 (node index in ctx) whose two norms read the q and k heads of this conv output as consecutive
// contiguous rows from offset 0, when the small fusion would take it as one launch; -1 otherwise.
int ggml_metal_op_conv_l2_target(ggml_metal_op_t ctx, int from, const ggml_tensor * conv_out) { GGML_METAL_ZONE_BEGIN(encode);
    for (int j = from; j + 3 < ctx->n_nodes(); ++j) {
        const ggml_tensor * norm0 = ctx->node(j);
        if (norm0->op != GGML_OP_RMS_NORM) {
            continue;
        }
        const ggml_tensor * t = norm0->src[0];
        while (t != nullptr && t != conv_out && t->view_src != nullptr) t = t->view_src;
        if (t != conv_out) {
            continue;
        }
        int count = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(j, GGML_METAL_FUSION_FULL, &count);
        if (fusion == nullptr || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_SPLIT_L2 || count != 4) {
            return -1;
        }
        const ggml_tensor * norm1 = ctx->node(j + 2);
        const ggml_tensor * q = norm0->src[0];
        const ggml_tensor * k = norm1->src[0];
        const int head_dim = (int) q->ne[0];
        const size_t row = (size_t) head_dim*sizeof(float);
        const ggml_tensor * tk = k;
        while (tk != nullptr && tk != conv_out && tk->view_src != nullptr) tk = tk->view_src;
        if (tk != conv_out || k->ne[0] != head_dim || head_dim % 4 != 0 || q->view_offs != 0 ||
            q->nb[0] != sizeof(float) || k->nb[0] != sizeof(float) || q->nb[1] != row || k->nb[1] != row ||
            q->ne[2]*q->ne[3] != 1 || k->ne[2]*k->ne[3] != 1 || conv_out->ne[1] != 1 ||
            k->view_offs != (size_t) ggml_nrows(q)*row || conv_out->ne[0] % head_dim != 0 ||
            ggml_get_op_params_f32(ctx->node(j + 1), 1) != 0.0f || ggml_get_op_params_f32(ctx->node(j + 3), 1) != 0.0f) {
            return -1;
        }
        // the same thread rule small_encode_norm uses
        const int width = head_dim/4;
        if (small_reference_threads(ctx, ggml_metal_library_get_pipeline_norm_scale(ctx->lib, norm0), width) != width ||
            small_reference_threads(ctx, ggml_metal_library_get_pipeline_norm_scale(ctx->lib, norm1), width) != width) {
            return -1;
        }
        return j;
    }
    return -1;
}

static int small_encode_norm(ggml_metal_op_t ctx, int idx, int count, bool gated) {
    const ggml_tensor * norm0 = ctx->node(idx);
    const ggml_tensor * norm1 = gated ? norm0 : ctx->node(idx + 2);
    const ggml_tensor * out0 = ctx->node(idx + (gated ? 3 : 1));
    const ggml_tensor * out1 = gated ? out0 : ctx->node(idx + 3);
    // The conv_state launch that produced q and k already wrote both norms (hw row conv_l2_fold).
    if (!gated && std::find(ctx->split_l2_folded.begin(), ctx->split_l2_folded.end(), norm0) != ctx->split_l2_folded.end()) {
        return count;
    }
    const bool c4 = norm0->ne[0] % 4 == 0;
    const int width = c4 ? norm0->ne[0] / 4 : norm0->ne[0];
    auto reference0 = gated ? ggml_metal_library_get_pipeline_norm(ctx->lib, norm0, 2)
                            : ggml_metal_library_get_pipeline_norm_scale(ctx->lib, norm0);
    const int threads = small_reference_threads(ctx, reference0, width);
    if (threads != width) {
        return 1;
    }
    if (!gated) {
        auto reference1 = ggml_metal_library_get_pipeline_norm_scale(ctx->lib, norm1);
        if (small_reference_threads(ctx, reference1, width) != threads) {
            return 1;
        }
    }
    auto pipeline = small_pipeline(ctx->lib,
            gated ? "kernel_small_gated_rms_norm_f32" : "kernel_small_split_l2_f32", c4);
    GGML_ASSERT(pipeline.pipeline);
    if (threads > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        return 1;
    }
    pipeline.smem = reference0.smem;
    ggml_metal_kargs_small_norm args = {};
    args.norm[0] = small_norm_args(norm0, out0, c4,
            gated ? 1.0f : ggml_get_op_params_f32(out0, 0));
    args.norm[1] = small_norm_args(norm1, out1, c4,
            gated ? 1.0f : ggml_get_op_params_f32(out1, 0));
    for (int j = 0; j < 3; ++j) {
        args.ne[0][j] = norm0->ne[j + 1];
        args.ne[1][j] = norm1->ne[j + 1];
    }
    const ggml_tensor * gate = gated ? ctx->node(idx + 2)->src[0] : norm0->src[0];
    args.nbg1 = gate->nb[1];
    args.nbg2 = gate->nb[2];
    args.nbg3 = gate->nb[3];
    const ggml_tensor * second = gated ? ctx->node(idx + 1)->src[1] : norm1->src[0];
    small_check_concurrency(ctx, idx, count);
    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(norm0->src[0]), 1);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(second), 2);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(gate), 3);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(out0), 4);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(out1), 5);
    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, pipeline.smem, 0);
    const int rows = ggml_nrows(norm0) + (gated ? 0 : ggml_nrows(norm1));
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, rows, 1, 1, threads, 1, 1);
    return count;
}

// Gate copy, sigmoid, attention output copy and multiply as one launch; 1 when the four nodes do not line up.
int ggml_metal_op_attn_gate(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    if (idx + 3 >= ctx->n_nodes() || ctx->node(idx)->op != GGML_OP_CONT) {
        return 1;
    }
    const ggml_tensor * mul = ctx->node(idx + 3);
    const ggml_tensor * cg = nullptr, * sig = nullptr, * out = nullptr;
    if (!ggml_metal_attn_gate_match(mul, &cg, &sig, &out) ||
        ctx->node(idx) != cg || ctx->node(idx + 1) != sig || ctx->node(idx + 2) != out) {
        return 1;
    }
    int uses_cg = 0, uses_sig = 0, uses_out = 0;
    for (int i = 0; i < ctx->graph_n_nodes(); i++) {
        const ggml_tensor * t = ctx->graph_node(i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            uses_cg  += t->src[s] == cg;
            uses_sig += t->src[s] == sig;
            uses_out += t->src[s] == out;
        }
    }
    if (uses_cg != 1 || uses_sig != 1 || uses_out != 1) {
        return 1;
    }
    const ggml_tensor * gv = cg->src[0];
    const ggml_tensor * ov = out->src[0];
    const int64_t rows = gv->ne[1];
    const size_t row_bytes = (size_t) gv->ne[0]*sizeof(float);
    // every element sits in one threadgroup: 1 element per thread, else 2; a larger row stays unfused
    const int elements = (int) (gv->ne[0]/4*rows);
    ggml_metal_pipeline_with_params pipeline = {};
    int threads = 0;
    for (int e = 1; e <= 2 && threads == 0; ++e) {
        char name[64];
        std::snprintf(name, sizeof(name), "kernel_small_attn_gate_f32_4_e=%d", e);
        pipeline = ggml_metal_library_get_pipeline(ctx->lib, name);
        if (!pipeline.pipeline) {
            ggml_metal_cv_t cv = ggml_metal_cv_init();
            ggml_metal_cv_set_int16(cv, (int16_t) e, FC_SMALL_ATTN_GATE + 0);
            pipeline = ggml_metal_library_compile_pipeline(ctx->lib, "kernel_small_attn_gate_f32_4", name, cv);
            ggml_metal_cv_free(cv);
        }
        if (!pipeline.pipeline) {
            return 1;
        }
        const int width = ggml_metal_pipeline_thread_execution_width(pipeline);
        if (elements % e == 0 && (elements/e) % width == 0 &&
            elements/e <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            threads = elements/e;
        }
    }
    if (threads == 0) {
        return 1;
    }
    ggml_metal_kargs_small_gate args = {};
    args.ne0  = (int32_t) (gv->ne[0]/4);
    args.ne1  = (int32_t) rows;
    args.nb01 = gv->nb[1];
    args.nb1  = row_bytes;
    small_check_concurrency(ctx, idx, 4);
    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(gv), 1);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(ov), 2);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(mul), 3);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, 1, 1, 1, threads, 1, 1);
    return 4;
}

// Residual add, the rms norm of its sum and the weight multiply as one launch; 1 when the nodes do not line up.
int ggml_metal_op_add_norm(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    if (idx + 2 >= ctx->n_nodes()) {
        return 1;
    }
    const ggml_tensor * add  = ctx->node(idx);
    const ggml_tensor * norm = ctx->node(idx + 1);
    const ggml_tensor * mul  = ctx->node(idx + 2);
    if (!ggml_metal_add_scale_applies(add) || norm->op != GGML_OP_RMS_NORM || norm->src[0] != add ||
        mul->op != GGML_OP_MUL || mul->src[0] != norm || mul->type != GGML_TYPE_F32) {
        return 1;
    }
    const ggml_tensor * w = mul->src[1];
    if (w == nullptr || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || ggml_nelements(w) != add->ne[0] ||
        !ggml_are_same_shape(mul, add)) {
        return 1;
    }
    int uses_norm = 0;
    for (int i = 0; i < ctx->graph_n_nodes(); i++) {
        const ggml_tensor * t = ctx->graph_node(i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            uses_norm += t->src[s] == norm;
        }
    }
    if (uses_norm != 1) {
        return 1;
    }
    const int width = (int) (add->ne[0]/4);
    auto reference = ggml_metal_library_get_pipeline_norm(ctx->lib, norm, 2);
    const int threads = small_reference_threads(ctx, reference, width);
    if (threads != width) {
        return 1;
    }
    auto pipeline = small_pipeline(ctx->lib, "kernel_small_add_norm_f32", true);
    if (!pipeline.pipeline || threads > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        return 1;
    }
    pipeline.smem = reference.smem;
    ggml_metal_kargs_norm args = small_norm_args(norm, mul, true, 1.0f);
    small_check_concurrency(ctx, idx, 3);
    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(add->src[0]), 1);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(add->src[1]), 2);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(w), 3);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(add), 4);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(mul), 5);
    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, 1, 1, 1, threads, 1, 1);
    return 3;
}

int ggml_metal_op_fuse_small(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    const ggml_op op = ctx->node(idx)->op;
    if (op != GGML_OP_ADD && op != GGML_OP_RMS_NORM) {
        return 1;
    }
    int count = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &count);
    if (!fusion) {
        return 1;
    }
    switch (ggml_metal_fusion_get_id(fusion)) {
        case GGML_METAL_FUSION_GATE_G:         return small_encode_gate(ctx, idx, count);
        case GGML_METAL_FUSION_GATED_RMS_NORM: return small_encode_norm(ctx, idx, count, true);
        case GGML_METAL_FUSION_SPLIT_L2:       return small_encode_norm(ctx, idx, count, false);
        default: return 1;
    }
}
