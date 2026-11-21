#include "../ops/ggml-metal-ops-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(misc) int ggml_metal_op_argmax(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_argmax args = {
        /*.ne00 = */ ne00,
        /*.nb01 = */ nb01,
    };

    auto pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    const struct ggml_metal_launch launch = ggml_metal_op_launch(ctx, GGML_METAL_LAUNCH_ARGMAX);
    int nth = launch.start;
    while (nth < ne00 && nth*ne01*ne02*ne03 < launch.work_min) {
        nth *= 2;
    }

    // a single aligned f32 row is read as 16-byte values by one threadgroup
    if (op->src[0]->type == GGML_TYPE_F32 && nrows == 1 && ne00 % 4 == 0 && ggml_metal_get_buffer_id(op->src[0]).offs % 16 == 0) {
        pipeline = ggml_metal_library_get_pipeline_argmax_w4(lib);
        nth = std::min(launch.cap, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_adamw(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_adamw args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_sgd(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_sgd args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_count_equal(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    {
        ggml_metal_kargs_memset args = { /*.val =*/ 0 };

        auto pipeline = ggml_metal_library_get_pipeline_memset(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx, idx);

    {
        ggml_metal_kargs_count_equal args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
        };

        auto pipeline = ggml_metal_library_get_pipeline_count_equal(lib, op);

        const size_t smem = pipeline.smem;

        int nth = ggml_metal_op_launch(ctx, GGML_METAL_LAUNCH_COUNT_EQUAL).start*pipeline.nsg;

        if (nth > (int) ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}

int ggml_metal_op_top_k_mask(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * mask   = op->src[0];
    const ggml_tensor * top_k  = op->src[1];
    const ggml_tensor * score  = op->src[2];
    const ggml_tensor * blocks = op->src[3];
    const ggml_tensor * extra  = op->src[4];

    GGML_ASSERT(op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->type == op->type);
    GGML_ASSERT(top_k->type == GGML_TYPE_I32);
    GGML_ASSERT(!score  || score->type  == GGML_TYPE_F32);
    GGML_ASSERT(!blocks || blocks->type == GGML_TYPE_I32);
    GGML_ASSERT(!extra  || extra->type  == GGML_TYPE_I32);

    const int32_t n_kv    = mask->ne[0];
    const int32_t n_top   = (int32_t) top_k->ne[0];
    const int32_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    const int32_t n_blk   = blocks ? blocks->ne[0] : 0;
    const int32_t n_extra = extra  ? extra->ne[0]  : 0;

    const ggml_metal_buffer_id bid = ggml_metal_get_buffer_id(op);

    // the vector fill stores 4 elements at a time
    const int32_t ts      = ggml_type_size(op->type);
    const int64_t n_elems = ggml_nelements(op);

    const bool has_score  = score  != NULL;
    const bool has_blocks = blocks != NULL;
    const bool has_extra  = extra  != NULL;

    const bool vec = n_elems % 4 == 0 && bid.offs % (4*ts) == 0;

    ggml_metal_kargs_top_k_mask args = {
        /*.n_kv    =*/ n_kv,
        /*.m1      =*/ (int32_t) mask->ne[1],
        /*.m2      =*/ (int32_t) mask->ne[2],
        /*.m3      =*/ (int32_t) mask->ne[3],
        /*.n_top   =*/ n_top,
        /*.n_pick  =*/ n_pick,
        /*.n_blk   =*/ n_blk,
        /*.n_extra =*/ n_extra,
        /*.n_elems =*/ n_elems,
        /*.nb1     =*/ op->nb[1],
        /*.nb2     =*/ op->nb[2],
        /*.nb3     =*/ op->nb[3],
        /*.nbm1    =*/ mask->nb[1],
        /*.nbm2    =*/ mask->nb[2],
        /*.nbm3    =*/ mask->nb[3],
        /*.nbk0    =*/ top_k->nb[0],
        /*.nbk1    =*/ top_k->nb[1],
        /*.nbk2    =*/ top_k->nb[2],
        /*.nbk3    =*/ top_k->nb[3],
        /*.nbs0    =*/ score  ? score->nb[0]  : 0,
        /*.nbs1    =*/ score  ? score->nb[1]  : 0,
        /*.nbs2    =*/ score  ? score->nb[2]  : 0,
        /*.nbs3    =*/ score  ? score->nb[3]  : 0,
        /*.nbb0    =*/ blocks ? blocks->nb[0] : 0,
        /*.nbb1    =*/ blocks ? blocks->nb[1] : 0,
        /*.nbe0    =*/ extra  ? extra->nb[0]  : 0,
        /*.nbe1    =*/ extra  ? extra->nb[1]  : 0,
        /*.nbe2    =*/ extra  ? extra->nb[2]  : 0,
        /*.nbe3    =*/ extra  ? extra->nb[3]  : 0,
    };

    // the fill covers the whole dst, the scatter overwrites the kept cells with the mask values
    {
        auto pipeline_fill = ggml_metal_library_get_pipeline_top_k_mask(lib, op, true, false, false, false, vec);

        const int64_t n_fill  = n_elems/(vec ? 4 : 1);
        const int32_t nth_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_fill);

        int nth = ggml_metal_op_launch(ctx, GGML_METAL_LAUNCH_TOP_K_MASK).start;

        while (nth < nth_max && nth < n_fill) {
            nth *= 2;
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline_fill);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid, 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, (n_fill + nth - 1)/nth, 1, 1, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx, idx);
    }

    {
        auto pipeline = ggml_metal_library_get_pipeline_top_k_mask(lib, op, false, has_score, has_blocks, has_extra, false);

        const int32_t n_max   = n_top + n_extra;
        const int32_t nth_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);

        int nth = ggml_metal_op_launch(ctx, GGML_METAL_LAUNCH_TOP_K_MASK).start;

        while (nth < nth_max && nth < n_max) {
            nth *= 2;
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(mask),  1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(top_k), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(score  ? score  : mask), 3);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(blocks ? blocks : mask), 4);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(extra  ? extra  : mask), 5);
        ggml_metal_encoder_set_buffer(enc, bid, 6);

        ggml_metal_encoder_dispatch_threadgroups(enc, op->ne[1], op->ne[2], op->ne[3], nth, 1, 1);
    }

    return 1;
}

int ggml_metal_op_dsv4_hc(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;
    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc(ctx->lib, op);

    // one simdgroup per tile: the kernels index by the threadgroup's width
    const int nw = ggml_metal_pipeline_thread_execution_width(pipeline);

    ggml_metal_encoder_set_pipeline(enc, pipeline);

    switch (op->op) {
        case GGML_OP_DSV4_HC_COMB:
            {
                const ggml_tensor * mixes = op->src[0];
                const ggml_tensor * scale = op->src[1];
                const ggml_tensor * base  = op->src[2];

                GGML_ASSERT(mixes->type == GGML_TYPE_F32);
                GGML_ASSERT(scale->type == GGML_TYPE_F32);
                GGML_ASSERT(base->type  == GGML_TYPE_F32);
                GGML_ASSERT(op->type    == GGML_TYPE_F32);
                GGML_ASSERT(mixes->ne[0] == 24);
                GGML_ASSERT(op->ne[0] == 4 && op->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_comb args = {
                    /*.n_tokens =*/ (int32_t) mixes->ne[1],
                    /*.n_iter   =*/ ggml_get_op_params_i32(op, 1),
                    /*.nb_m0    =*/ mixes->nb[0],
                    /*.nb_m1    =*/ mixes->nb[1],
                    /*.nb_s0    =*/ scale->nb[0],
                    /*.nb_b0    =*/ base->nb[0],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                    /*.eps      =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(mixes), 1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(scale), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(base),  3);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),    4);

                // One SIMDgroup owns one 4x4 Sinkhorn matrix. Packing up to four
                // independent tokens per threadgroup keeps both decode and prompt
                // dispatches compact without any threadgroup-memory synchronization.
                const int nsg = std::min(ggml_metal_op_launch(ctx, GGML_METAL_LAUNCH_DSV4_HC).nsg, args.n_tokens);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (args.n_tokens + nsg - 1)/nsg, 1, 1, nw, nsg, 1);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                const ggml_tensor * x       = op->src[0];
                const ggml_tensor * weights = op->src[1];

                GGML_ASSERT(x->type       == GGML_TYPE_F32);
                GGML_ASSERT(weights->type == GGML_TYPE_F32);
                GGML_ASSERT(op->type      == GGML_TYPE_F32);

                ggml_metal_kargs_dsv4_hc_pre args = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[2],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_x2    =*/ x->nb[2],
                    /*.nb_w0    =*/ weights->nb[0],
                    /*.nb_w1    =*/ weights->nb[1],
                    /*.nb_w2    =*/ weights->nb[2],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.scale    =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),       1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),      3);

                const int n_tiles = (args.n_embd + nw - 1)/nw;
                const int nsg = std::min(ggml_metal_op_launch(ctx, GGML_METAL_LAUNCH_DSV4_HC).nsg, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, nw, nsg, 1);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                const ggml_tensor * x        = op->src[0];
                const ggml_tensor * residual = op->src[1];
                const ggml_tensor * post     = op->src[2];
                const ggml_tensor * comb     = op->src[3];

                GGML_ASSERT(x->type        == GGML_TYPE_F32);
                GGML_ASSERT(residual->type == GGML_TYPE_F32);
                GGML_ASSERT(post->type     == GGML_TYPE_F32);
                GGML_ASSERT(op->type       == GGML_TYPE_F32);
                GGML_ASSERT(residual->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_post args = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[1],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_r0    =*/ residual->nb[0],
                    /*.nb_r1    =*/ residual->nb[1],
                    /*.nb_r2    =*/ residual->nb[2],
                    /*.nb_p0    =*/ post->nb[0],
                    /*.nb_p1    =*/ post->nb[1],
                    /*.nb_c0    =*/ comb ? comb->nb[0] : 0,
                    /*.nb_c1    =*/ comb ? comb->nb[1] : 0,
                    /*.nb_c2    =*/ comb ? comb->nb[2] : 0,
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),        1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(residual), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(post),     3);
                if (comb) {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(comb), 4);
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   5);
                } else {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   4);
                }

                const int n_tiles = (args.n_embd + nw - 1)/nw;
                const int nsg = std::min(ggml_metal_op_launch(ctx, GGML_METAL_LAUNCH_DSV4_HC).nsg, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, nw, nsg, 1);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    return 1;
}

int ggml_metal_op_out_prod(ggml_metal_op_t ctx, int idx) { GGML_METAL_ZONE_BEGIN(encode);
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    ggml_metal_kargs_out_prod args = {
        /*.ne0  =*/ (int32_t) op->ne[0],
        /*.ne1  =*/ (int32_t) op->ne[1],
        /*.ne2  =*/ (int32_t) op->ne[2],
        /*.ne01 =*/ (int32_t) src0->ne[1],
        /*.dps2 =*/ (int32_t) (op->ne[2] / src0->ne[2]),
        /*.dps3 =*/ (int32_t) (op->ne[3] / src0->ne[3]),
        /*.nb00 =*/ src0->nb[0],
        /*.nb01 =*/ src0->nb[1],
        /*.nb02 =*/ src0->nb[2],
        /*.nb03 =*/ src0->nb[3],
        /*.nb10 =*/ src1->nb[0],
        /*.nb11 =*/ src1->nb[1],
        /*.nb12 =*/ src1->nb[2],
        /*.nb13 =*/ src1->nb[3],
        /*.nb0  =*/ op->nb[0],
        /*.nb1  =*/ op->nb[1],
        /*.nb2  =*/ op->nb[2],
        /*.nb3  =*/ op->nb[3],
    };

    auto pipeline = ggml_metal_library_get_pipeline_out_prod(ctx->lib);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src0), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src1), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (op->ne[0] + 31)/32, op->ne[1], op->ne[2]*op->ne[3], 32, 1, 1);

    return 1;
}
