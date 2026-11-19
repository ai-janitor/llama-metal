#include "../ops/ggml-metal-ops-internal.h"
#include "ggml-metal-probe.h"
#include <cstring>
GGML_METAL_FAMILY(ssm)
int ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    int n_fuse = 1;
    bool use_silu = false;
    if (ctx->use_fusion) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SSM_CONV_SILU) {
            n_fuse = n;
            use_silu = true;
        }
    }
    const ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(
        n_fuse > 1 ? ctx->node(idx + n_fuse - 1) : op);
    if (n_fuse > 1 && !ggml_metal_op_concurrency_check(ctx, ctx->node(idx + 1), idx + 1)) {
        ggml_metal_op_concurrency_reset(ctx, idx + 1);
    }

    const bool has_state_dst = (op->src[2] != nullptr);

    ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.has_state_dst =*/ has_state_dst ? 1 : 0,
    };

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    const bool use_batched = (ne1 > 1);

    if (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        int BATCH_SIZE;
        if      (ne1 > 128) BATCH_SIZE = 256;
        else if (ne1 > 64 ) BATCH_SIZE = 128;
        else if (ne1 > 32 ) BATCH_SIZE = 64;
        else if (ne1 > 16 ) BATCH_SIZE = 32;
        else if (ne1 > 8  ) BATCH_SIZE = 16;
        else if (ne1 > 4  ) BATCH_SIZE = 8;
        else                BATCH_SIZE = 2;

        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE, use_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,                                  3);
        if (has_state_dst) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 4);
        } else {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);
        }

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        const int n_token_batches = (ne1 + BATCH_SIZE - 1) / BATCH_SIZE;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv(lib, op, use_silu);

        const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
        // Pack multiple channels per TG to fill the native wavefront.
        // On D700/GCN (NW=64): 64 channels per TG instead of 1 → no idle wavefront lanes.
        // On NW=32: 32 channels per TG. Grid adjusts via ceil division.
        const int tg_width = (int)profile->simd_width;

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,                                  3);
        if (has_state_dst) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 4);
        } else {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);
        }

        const int grid_x = (ne01 + tg_width - 1) / tg_width;  // ceil(channels / tg_width)
        ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, ne1, ne02, tg_width, 1, 1);
    }

    return n_fuse;
}

// CONCAT + CPY + SSM_CONV + silu of one decode token in one launch (GGML_METAL_FUSION_CONV_STATE);
// returns 0 when the table does not match, so the four nodes encode as before.
int ggml_metal_op_conv_state(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    if (!ctx->use_fusion) {
        return 0;
    }
    int n = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
    if (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_CONV_STATE || n != 4) {
        return 0;
    }

    ggml_tensor * cat  = ctx->node(idx);
    ggml_tensor * cpy  = ctx->node(idx + 1);
    ggml_tensor * conv = ctx->node(idx + 2);
    ggml_tensor * un   = ctx->node(idx + 3);

    // every member is checked for overlap with work still in flight
    for (int i = 1; i < n; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i), idx + i)) {
            ggml_metal_op_concurrency_reset(ctx, idx + i);
            break;
        }
    }

    const int nc = (int) cat->ne[0];

    ggml_metal_kargs_conv_state args = {
        /*.nc       =*/ nc,
        /*.channels =*/ (int32_t) cat->ne[1],
        /*.nbs1     =*/ cat->src[0]->nb[1],
        /*.nbx1     =*/ cat->src[1]->nb[1],
        /*.nbw1     =*/ conv->src[1]->nb[1],
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_state(ctx->lib, nc);

    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);
    const int tg_width = (int) profile->simd_width;

    // hw row conv_l2_fold: this launch also writes the split_l2 norms of the q and k heads it produces;
    // the conv output itself is still written (its v part has another reader)
    if (profile->conv_l2_fold == 1 && ggml_metal_fuse_small_enabled() && nc % 4 == 0) {
        const int l2 = ggml_metal_op_conv_l2_target(ctx, idx + 4, un);
        if (l2 >= 0) {
            const ggml_tensor * norm0 = ctx->node(l2);
            const ggml_tensor * out0  = ctx->node(l2 + 1);
            const ggml_tensor * norm1 = ctx->node(l2 + 2);
            const ggml_tensor * out1  = ctx->node(l2 + 3);
            const int head_dim = (int) norm0->ne[0];
            ggml_metal_kargs_conv_state_l2 a2 = {};
            a2.cs = args;
            a2.head_dim = head_dim;
            a2.rows[0] = (int32_t) ggml_nrows(norm0);
            a2.rows[1] = (int32_t) ggml_nrows(norm1);
            a2.ne00[0] = (int32_t) norm0->src[0]->ne[0];
            a2.ne00[1] = (int32_t) norm1->src[0]->ne[0];
            std::memcpy(&a2.eps[0], norm0->op_params, sizeof(float));
            std::memcpy(&a2.eps[1], norm1->op_params, sizeof(float));
            a2.scale[0] = ggml_get_op_params_f32(out0, 0);
            a2.scale[1] = ggml_get_op_params_f32(out1, 0);
            a2.nb1[0] = out0->nb[1];
            a2.nb1[1] = out1->nb[1];
            auto p2 = ggml_metal_library_get_pipeline_conv_state_l2(ctx->lib);
            if (p2.pipeline && head_dim <= ggml_metal_pipeline_max_theads_per_threadgroup(p2)) {
                ggml_metal_encoder_t enc = ctx->enc;
                ggml_metal_encoder_set_pipeline(enc, p2);
                ggml_metal_encoder_set_bytes   (enc, &a2, sizeof(a2), 0);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(cat->src[0]),  1);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(cat->src[1]),  2);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(conv->src[1]), 3);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(un),           4);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(cpy),          5);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(out0),         6);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(out1),         7);
                ggml_metal_encoder_set_threadgroup_memory_size(enc, (size_t) (head_dim + head_dim/4 + 4)*sizeof(float), 0);
                ggml_metal_encoder_dispatch_threadgroups(enc, args.channels/head_dim, 1, 1, head_dim, 1, 1);
                ctx->split_l2_folded.push_back(norm0);
                return n;
            }
        }
    }

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(cat->src[0]),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(cat->src[1]),  2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(conv->src[1]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(un),           4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(cpy),          5);
    ggml_metal_encoder_dispatch_threadgroups(enc, (args.channels + tg_width - 1)/tg_width, 1, 1, tg_width, 1, 1);

    return n;
}

int ggml_metal_op_ssm_scan(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const ggml_tensor * src3 = op->src[3];
    const ggml_tensor * src4 = op->src[4];
    const ggml_tensor * src5 = op->src[5];
    const ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb00         =*/ nb00,
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb10         =*/ nb10,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.ns12         =*/ nb12/nb10,
        /*.nb13         =*/ nb13,
        /*.nb20         =*/ nb20,
        /*.nb21         =*/ nb21,
        /*.ns21         =*/ nb21/nb20,
        /*.nb22         =*/ nb22,
        /*.ne30         =*/ ne30,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.ns42         =*/ nb42/nb40,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.ns52         =*/ nb52/nb50,
        /*.nb53         =*/ nb53,
        /*.nb0          =*/ nb0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_ssm_scan(lib, op);

    GGML_ASSERT(d_state <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 7);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         8);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, d_inner, n_head, n_seqs, d_state, 1, 1);

    return 1;
}

int ggml_metal_op_rwkv(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    auto pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &B, sizeof(B), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &T, sizeof(T), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &C, sizeof(C), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &H, sizeof(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

int ggml_metal_op_gated_linear_attn(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    // k, v, q, g: {head_size, n_head, n_tokens}; state: {head_size*head_size*n_head, n_seqs}
    const int32_t head_size = op->src[0]->ne[0];
    const int32_t n_head    = op->src[0]->ne[1];
    const int32_t n_tokens  = op->src[0]->ne[2];
    const int32_t n_seqs    = op->src[4]->ne[1];

    ggml_metal_kargs_gla args = {
        /*.head_size    =*/ head_size,
        /*.n_head       =*/ n_head,
        /*.n_seq_tokens =*/ n_tokens/n_seqs,
        /*.n_seqs       =*/ n_seqs,
        /*.scale        =*/ ggml_get_op_params_f32(op, 0),
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_GATED_LINEAR_ATTN);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         6);

    ggml_metal_encoder_dispatch_threadgroups(enc, n_seqs*n_head, 1, 1, head_size, 1, 1);

    return 1;
}

int ggml_metal_op_solve_tri(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_solve_tri args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    const int sw = ggml_metal_pipeline_thread_execution_width(pipeline);
    ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + nsg - 1)/nsg, ne02, ne03, sw, nsg, 1);

    return 1;
}

// Reference: ggml-org/llama.cpp 4e5116f47, ggml-metal-ops.cpp:2062 (MIT).
// INTENT/PDL: bind q/k/v and shared geometry; FU-MU-14 writes snapshots directly into the cache when matched.
int ggml_metal_op_gated_delta_net(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net(lib, op);

    // when fused with the trailing cache cpy, the snapshots are written straight into the
    // recurrent cache and the cpy is skipped (see GGML_METAL_FUSION_GDN_CACHE)
    ggml_metal_buffer_id bid_out = ggml_metal_get_buffer_id(op);
    uint64_t nb_out = 0;
    int n_fuse = 1;

    if (ctx->use_fusion) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_GDN_CACHE) {
            const ggml_tensor * dst_cache = ctx->node(idx + 1)->src[1]; // cache view

            bid_out = ggml_metal_get_buffer_id(dst_cache);
            nb_out  = dst_cache->nb[2]/sizeof(float);
            n_fuse = 2;


            if (ctx->debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: GATED_DELTA_NET + CPY\n", __func__);
            }
        }
    }

    int ida = 0;

    ggml_metal_kargs_gated_delta_net args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20,
        /*.ne21 =*/ ne21,
        /*.ne22 =*/ ne22,
        /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20,
        /*.nb21 =*/ nb21,
        /*.nb22 =*/ nb22,
        /*.nb23 =*/ nb23,
        /*.ns02 =*/ (int32_t) (nb02/sizeof(float)),
        /*.ns12 =*/ (int32_t) (nb12/sizeof(float)),
        /*.ns22 =*/ (int32_t) (nb22/sizeof(float)),
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.nb_out =*/ nb_out,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++); // args
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++); // state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst (attn)
    ggml_metal_encoder_set_buffer  (enc, bid_out,                              ida++); // state_out

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/nsg, op->src[2]->ne[1], op->src[2]->ne[3], pipeline.nr1, nsg, 1);

    return n_fuse;
}
