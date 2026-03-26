#include "ggml.h"
#include "../models.h"

// Fused gated delta-net decode path.
// Uses the ggml_gated_delta_net Metal kernel for single-sequence decode (n_seqs == 1).
// Inputs must be preprocessed: q/k l2-normed, q scaled, beta sigmoided, state reshaped to 4D.
//
// When state_dst is non-null, the kernel writes the new state directly to
// the cache buffer (eliminating a ggml_cpy). The returned new_state tensor
// IS state_dst in that case.

std::pair<ggml_tensor *, ggml_tensor *> llm_build_qwen35::build_delta_net_fused(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * beta,
        ggml_tensor * state,
        ggml_tensor * state_dst,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_seqs == 1);

    // g and beta are contiguous (from ggml_mul and ggml_sigmoid respectively).
    // Reshape is sufficient — no need to materialize with cont.
    ggml_tensor * g_t    = ggml_reshape_4d(ctx0, g, 1, 1, H_k, n_seqs);
    ggml_tensor * beta_t = ggml_reshape_4d(ctx0, beta, 1, 1, H_k, n_seqs);

    ggml_tensor * q3 = ggml_reshape_3d(ctx0, q, S_k, H_k, n_tokens);
    ggml_tensor * k3 = ggml_reshape_3d(ctx0, k, S_k, H_k, n_tokens);
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, v, S_v, H_v, n_tokens);

    ggml_tensor * result;
    ggml_tensor * core_attn_out;
    ggml_tensor * new_state;

    if (state_dst) {
        // Direct cache write: kernel writes state to state_dst, dst is output-only.
        result = ggml_gated_delta_net_ext(ctx0, k3, v3, q3, g_t, beta_t, state, state_dst, 1.0f);
        cb(result, "delta_net_result", il);

        const int64_t n_embd = S_v * H_v;

        // result is [S*H, n_tokens] — only attention output, no state portion.
        core_attn_out = ggml_reshape_4d(ctx0, result, S_v, 1, H_v, n_seqs);

        // State was written directly to state_dst by the kernel.
        new_state = state_dst;
    } else {
        // Legacy packed output: result = [S*H, n_tokens + S*n_seqs]
        result = ggml_gated_delta_net(ctx0, k3, v3, q3, g_t, beta_t, state, 1.0f);
        cb(result, "delta_net_result", il);

        const int64_t n_embd = S_v * H_v;

        core_attn_out = ggml_view_1d(ctx0, result, n_embd * n_tokens, 0);
        core_attn_out = ggml_reshape_4d(ctx0, core_attn_out, S_v, 1, H_v, n_seqs);

        new_state = ggml_view_1d(ctx0, result,
            n_embd * S_v * n_seqs,
            n_embd * n_tokens * sizeof(float));
        new_state = ggml_reshape_4d(ctx0, new_state, S_v, S_v, H_v, n_seqs);
    }

    cb(core_attn_out, "output_tokens", il);
    cb(new_state, "new_state", il);

    return {core_attn_out, new_state};
}
