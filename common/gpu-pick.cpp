#include "gpu-pick.h"

#include "ggml.h"
#include "log.h"
#include "sampling.h"

static const char * gpu_pick_name(const struct llama_sampler * /*smpl*/) {
    return "gpu-pick";
}

// CPU path: the first maximum
static void gpu_pick_apply(struct llama_sampler * /*smpl*/, llama_token_data_array * cur_p) {
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
            cur_p->selected = i;
        }
    }
}

static bool gpu_pick_backend_init(struct llama_sampler * /*smpl*/, ggml_backend_buffer_type_t /*buft*/, uint32_t /*n_outputs_max_per_seq*/) {
    return true;
}

static void gpu_pick_backend_apply(struct llama_sampler * /*smpl*/, struct ggml_context * ctx, struct ggml_cgraph * /*gf*/, struct llama_sampler_data * data) {
    data->sampled = ggml_argmax(ctx, ggml_reshape_1d(ctx, data->logits, ggml_nelements(data->logits)));
    data->logits  = nullptr; // only the token goes back to the host
}

static struct llama_sampler_i gpu_pick_i = {
    /* .name              = */ gpu_pick_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ gpu_pick_apply,
    /* .reset             = */ nullptr,
    /* .clone             = */ nullptr,
    /* .free              = */ nullptr,
    /* .backend_init      = */ gpu_pick_backend_init,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ gpu_pick_backend_apply,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

struct llama_sampler * common_gpu_pick_init(const common_params & params, const struct common_sampler * smpl) {
    if (!params.sampling.gpu_pick) {
        return nullptr;
    }

    if (params.sampling.backend_sampling || !common_sampler_is_greedy_only(smpl)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOG_WRN("gpu-pick: off (sampler is not greedy)\n");
        }
        return nullptr;
    }

    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init(&gpu_pick_i, nullptr));
    return chain;
}

llama_token common_gpu_pick_token(struct llama_context * ctx, int idx) {
    return llama_get_sampled_token_ith(ctx, idx);
}
