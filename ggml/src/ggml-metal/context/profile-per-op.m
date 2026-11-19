#import "context/ggml-metal-context-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(context) enum ggml_status ggml_metal_graph_compute_per_op(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    @autoreleasepool {
        {
            ctx->profile_n--;
            id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

            // per-op-type accumulators indexed by ggml_op enum
            double   op_time_ms[GGML_OP_COUNT];  // total GPU time per op type
            uint32_t op_count  [GGML_OP_COUNT];   // number of dispatches per op type
            memset(op_time_ms, 0, sizeof(op_time_ms));
            memset(op_count,   0, sizeof(op_count));

            double total_gpu_ms = 0;
            int    n_profiled   = 0;

            for (int i = 0; i < gf->n_nodes; ++i) {
                struct ggml_tensor * node = gf->nodes[i];

                // skip noop ops (same set as ggml_metal_op_encode_impl)
                if (node->op == GGML_OP_NONE      ||
                    node->op == GGML_OP_RESHAPE    ||
                    node->op == GGML_OP_VIEW       ||
                    node->op == GGML_OP_TRANSPOSE  ||
                    node->op == GGML_OP_PERMUTE) {
                    continue;
                }

                // skip nodes that don't need compute
                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // create a fresh command buffer for this single op
                id<MTLCommandBuffer> cmd_buf = ggml_metal_new_command_buffer(queue, true);

                // encode this single node: idx_start=i, idx_end=i+1, no fusion, no concurrency
                ggml_metal_op_t ctx_op = ggml_metal_op_init(
                    ctx->dev,
                    cmd_buf,
                    gf,
                    i, i + 1,
                    /*use_fusion=*/false,
                    /*use_concurrency=*/false,
                    /*use_capture=*/false,
                    ctx->debug_graph,
                    ctx->debug_fusion,
                    /*profiler=*/NULL,
                    /*profiler_cb_idx=*/-1
#ifdef GGML_METAL_TEST_OBSERVER
                    , ctx->observer, ctx->observer_user_data
#endif
                    );
                ggml_metal_op_set_act_copy(ctx_op, ctx->act_copy);

                for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
                    const int res = ggml_metal_op_encode(ctx_op, idx);
                    if (res == 0) {
                        break;
                    }
                    idx += res - 1;
                }

                ggml_metal_op_free(ctx_op);

                // commit and wait synchronously to get accurate per-op GPU timing
                [cmd_buf commit];
                [cmd_buf waitUntilCompleted];

                // check for errors
                if ([cmd_buf status] != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_ERROR("%s: profile: command buffer failed for node %d (%s) with status %d\n",
                        __func__, i, ggml_op_name(node->op), (int)[cmd_buf status]);
                    if ([cmd_buf status] == MTLCommandBufferStatusError) {
                        GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }
                    return ggml_metal_graph_failed(ctx, cmd_buf, i);
                }

                // read GPU timestamps and accumulate per-op-type timing
                double start = [cmd_buf GPUStartTime];
                double end   = [cmd_buf GPUEndTime];
                double dt_ms = (end - start) * 1000.0;

                op_time_ms[node->op] += dt_ms;
                op_count  [node->op] += 1;
                total_gpu_ms += dt_ms;
                n_profiled++;
            }

            // print per-op-type summary, sorted by total time descending
            if (n_profiled > 0) {
                // build sorted index of active op types
                int active_ops[GGML_OP_COUNT];
                int n_active = 0;
                for (int i = 0; i < GGML_OP_COUNT; ++i) {
                    if (op_count[i] > 0) {
                        active_ops[n_active++] = i;
                    }
                }

                // simple insertion sort by total time descending
                for (int i = 1; i < n_active; ++i) {
                    int key = active_ops[i];
                    int j = i - 1;
                    while (j >= 0 && op_time_ms[active_ops[j]] < op_time_ms[key]) {
                        active_ops[j + 1] = active_ops[j];
                        j--;
                    }
                    active_ops[j + 1] = key;
                }

                fprintf(stderr, "[METAL_PROFILE] per-op breakdown (%.2f ms total, %d nodes):\n",
                    total_gpu_ms, n_profiled);
                for (int i = 0; i < n_active; ++i) {
                    int op = active_ops[i];
                    double pct = (total_gpu_ms > 0) ? (op_time_ms[op] / total_gpu_ms * 100.0) : 0;
                    double avg = op_time_ms[op] / op_count[op];
                    fprintf(stderr, "  %-24s %8.2f ms (%5.1f%%) x%-5u avg=%.3f ms\n",
                        ggml_op_name((enum ggml_op) op),
                        op_time_ms[op], pct, op_count[op], avg);
                }
            }

            // all ops completed synchronously — nothing for synchronize() to wait on
            ctx->cmd_buf_last = nil;

            return GGML_STATUS_SUCCESS;
        }
    }
}
