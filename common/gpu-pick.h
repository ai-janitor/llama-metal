#pragma once

#include "llama.h"

#include "common.h"

// greedy token pick on the GPU: one argmax node in the graph, one token read back

// returns a greedy backend sampler to attach to the context, or nullptr when the pick is off
struct llama_sampler * common_gpu_pick_init(const common_params & params, const struct common_sampler * smpl);

// the token picked on the GPU for output idx, or LLAMA_TOKEN_NULL
llama_token common_gpu_pick_token(struct llama_context * ctx, int idx);
