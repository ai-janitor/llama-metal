#include "ggml.h"
#include "../models.h"

// Unfused gated delta-net decode path.
// Elementwise fallback for multi-sequence decode (n_seqs > 1).
// Inputs must be preprocessed: q/k l2-normed, q scaled, beta sigmoided, state reshaped to 4D.

std::pair<ggml_tensor *, ggml_tensor *> llm_build_qwen35::build_delta_net_unfused(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * beta,
        ggml_tensor * state,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_UNUSED(S_k);
    GGML_UNUSED(n_tokens);

    // Reshape gate and beta for elementwise ops
    ggml_tensor * g_t    = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, g), 1, 1, H_k, n_seqs);
    ggml_tensor * beta_t = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, beta), 1, 1, H_k, n_seqs);

    // Decay state by gate
    g_t = ggml_exp(ctx0, g_t);
    state = ggml_mul(ctx0, state, g_t);

    // Compute memory retrieval: kv_mem = sum_j(state[i,j] * k[j]) for each head
    ggml_tensor * k_t_unsqueezed = ggml_reshape_4d(ctx0, k, 1, S_v, H_v, n_seqs);
    ggml_tensor * kv_mem         = ggml_mul(ctx0, state, k_t_unsqueezed);
    kv_mem = ggml_transpose(ctx0, ggml_sum_rows(ctx0, ggml_transpose(ctx0, kv_mem)));

    // Delta rule: v_diff = v - kv_mem, delta = v_diff * beta
    ggml_tensor * v_t    = ggml_reshape_4d(ctx0, v, S_v, 1, H_v, n_seqs);
    ggml_tensor * v_diff = ggml_sub(ctx0, v_t, kv_mem);
    ggml_tensor * delta  = ggml_mul(ctx0, v_diff, beta_t);

    // State update: state += outer(k, delta)
    ggml_tensor * k_t_delta = ggml_mul(ctx0, ggml_repeat_4d(ctx0, k_t_unsqueezed, S_v, S_v, H_v, n_seqs), delta);
    state = ggml_add(ctx0, state, k_t_delta);

    // Output: query the updated state
    ggml_tensor * q_t_unsqueezed = ggml_reshape_4d(ctx0, q, 1, S_v, H_v, n_seqs);
    ggml_tensor * state_q        = ggml_mul(ctx0, state, q_t_unsqueezed);
    ggml_tensor * core_attn_out  = ggml_transpose(ctx0, ggml_sum_rows(ctx0, ggml_transpose(ctx0, state_q)));
    ggml_tensor * new_state      = state;

    cb(core_attn_out, "output_tokens", il);
    cb(new_state, "new_state", il);

    return {core_attn_out, new_state};
}
