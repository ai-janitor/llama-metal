#include "ggml.h"
#include "llama.h"
#include "completion-status.hpp"
#include "sampling-config.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Print the bounded-completion interface because the driver deliberately avoids
// the common CLI layer: it owns the observer reset and receipt boundaries.
static void print_usage(const char * program) {
    // A real newline ends the usage line; the receipt ABI uses the same style.
    std::fprintf(stderr, "usage: %s -m MODEL -o RECEIPT [-n TOKENS] [-ngl LAYERS] [--backend-sampling|--no-backend-sampling --samplers temperature --temp 0 --seed 42] [PROMPT]\n", program);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string receipt_path;
    std::string prompt = "Hello my name is";
    int n_gpu_layers = 99;
    int n_predict = 32;
    // Requested sampling policy; empty means the retained greedy default.
    phase_timing_sampling_config sampling_config;

    int arg_index = 1;
    for (; arg_index < argc; ++arg_index) {
        const std::string option = argv[arg_index];
        if (option == "--backend-sampling" || option == "--no-backend-sampling" ||
            option == "--samplers" || option == "--temp" || option == "--seed") {
            // Sampling flags are rejected here, before any model initialization.
            const bool needs_value = option == "--samplers" || option == "--temp" || option == "--seed";
            const char * value = nullptr;
            if (needs_value) {
                if (arg_index + 1 == argc) {
                    std::string diagnostic;
                    phase_timing_apply_sampling_option(sampling_config, option, nullptr, diagnostic);
                    std::fprintf(stderr, "%s\n", diagnostic.c_str());
                    print_usage(argv[0]);
                    return 1;
                }
                value = argv[++arg_index];
            }
            std::string diagnostic;
            if (phase_timing_apply_sampling_option(sampling_config, option, value, diagnostic) != phase_timing_sampling_status::ok) {
                std::fprintf(stderr, "%s\n", diagnostic.c_str());
                print_usage(argv[0]);
                return 1;
            }
            continue;
        }
        if (option.rfind("-", 0) == 0 && option != "-m" && option != "-o" && option != "-n" && option != "-ngl") {
            // An unknown flag is never silently treated as prompt text.
            std::fprintf(stderr, "unknown option: %s\n", argv[arg_index]);
            print_usage(argv[0]);
            return 1;
        }
        if (std::strcmp(argv[arg_index], "-m") == 0 || std::strcmp(argv[arg_index], "-o") == 0 || std::strcmp(argv[arg_index], "-n") == 0 || std::strcmp(argv[arg_index], "-ngl") == 0) {
            if (arg_index + 1 == argc) {
                print_usage(argv[0]);
                return 1;
            }

            const char * value = argv[++arg_index];
            if (std::strcmp(argv[arg_index - 1], "-m") == 0) {
                model_path = value;
            } else if (std::strcmp(argv[arg_index - 1], "-o") == 0) {
                receipt_path = value;
            } else {
                try {
                    const int parsed_value = std::stoi(value);
                    if (std::strcmp(argv[arg_index - 1], "-n") == 0) {
                        n_predict = parsed_value;
                    } else {
                        n_gpu_layers = parsed_value;
                    }
                } catch (...) {
                    print_usage(argv[0]);
                    return 1;
                }
            }
        } else {
            break;
        }
    }

    {
        // A partial explicit policy rejects before the model is touched.
        std::string diagnostic;
        if (phase_timing_validate_sampling_policy(sampling_config, diagnostic) != phase_timing_sampling_status::ok) {
            std::fprintf(stderr, "%s\n", diagnostic.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || receipt_path.empty() || n_predict < 0) {
        print_usage(argv[0]);
        return 1;
    }

    // Everything after the options is the production completion prompt.
    if (arg_index < argc) {
        prompt = argv[arg_index++];
        for (; arg_index < argc; ++arg_index) {
            prompt += " ";
            prompt += argv[arg_index];
        }
    }

    // Backends must be registered before the model chooses its GPU placement.
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Build the explicit chain before the context so the backend arm can
    // register seq 0 at context creation; the default arm stays greedy.
    // Both matched explicit arms use the production temp/dist samplers, and
    // only registration differs between them.
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (sampling_config.explicit_policy()) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(sampling_config.temp_value));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(sampling_config.seed_value));
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    }
    // The seq config must outlive the context; it is freed after it below.
    llama_sampler_seq_config seq_config { 0, sampler };
    const phase_timing_sampler_plan sampler_plan = phase_timing_make_sampler_plan(sampling_config);

    llama_context_params context_params = llama_context_default_params();
    // One prompt batch plus bounded generated tokens gives decode its complete range.
    context_params.n_ctx = n_prompt + n_predict;
    context_params.n_batch = n_prompt;
    if (sampler_plan.register_seq0) {
        context_params.samplers = &seq_config;
        context_params.n_samplers = 1;
    }
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_sampler_free(sampler);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Establish the prompt before the selected decode-only receipt window.
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(context, batch) != 0) {
        std::fprintf(stderr, "prompt decode failed\n");
        // Context frees before the chain it may have registered.
        llama_free(context);
        llama_sampler_free(sampler);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Sampling consumes the prompt decode's logits before the selected window begins.
    llama_token next_token = llama_sampler_sample(sampler, context, -1);
    bool reached_end_of_generation = llama_vocab_is_eog(vocab, next_token);

    // Reset only after prompt work so the receipt contains decode-token work alone.
    const bool write_receipt = ggml_l0_enabled();
    if (write_receipt) {
        ggml_l0_reset();
    }
    const double receipt_start = write_receipt ? ggml_l0_now() : 0;

    int n_generated = 0;
    int exit_code = 0;
    while (n_generated < n_predict && !reached_end_of_generation) {
        batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(context, batch) != 0) {
            std::fprintf(stderr, "decode failed\n");
            exit_code = 1;
            break;
        }

        // Sampling remains the final consumer for each measured decode.
        next_token = llama_sampler_sample(sampler, context, -1);
        reached_end_of_generation = llama_vocab_is_eog(vocab, next_token);
        ++n_generated;
    }
    exit_code = phase_timing_completion_exit_code(n_predict, n_generated, exit_code);
    const double receipt_end = write_receipt ? ggml_l0_now() : 0;

    // No synchronization is added here: the final sampling call is the consumer boundary.
    if (write_receipt) {
        FILE * const receipt = std::fopen(receipt_path.c_str(), "w");
        if (receipt == nullptr) {
            std::fprintf(stderr, "failed to open receipt: %s\n", receipt_path.c_str());
            exit_code = 1;
        } else {
            // Stable receipt identity: this invocation has one selected decode epoch.
            std::fprintf(receipt, "SAMPLE command=decode rep=0 trial=0 start=%.9f end=%.9f\n", receipt_start, receipt_end);
            ggml_l0_dump(receipt);
            const int refused_full_n = n_generated < n_predict ? 1 : 0;
            std::fprintf(receipt, "COMPLETION requested=%d actual=%d refused-full-N=%d eos=%d\n",
                n_predict, n_generated, refused_full_n, reached_end_of_generation ? 1 : 0);
            std::fclose(receipt);
            // Route is read back from the literal records just written, so a
            // flag alone never claims backend execution and a zero-payload
            // sample record falls back instead of passing.
            std::vector<std::string> receipt_lines;
            if (FILE * const readback = std::fopen(receipt_path.c_str(), "r")) {
                char buffer[4096];
                while (std::fgets(buffer, sizeof(buffer), readback) != nullptr) {
                    receipt_lines.push_back(buffer);
                }
                std::fclose(readback);
            }
            const phase_timing_route_counts route_counts = phase_timing_count_route_records(receipt_lines);
            const phase_timing_route route = phase_timing_classify_route(route_counts, n_generated, sampling_config);
            if (FILE * const append = std::fopen(receipt_path.c_str(), "a")) {
                std::fprintf(append, "SAMPLING requested=%s policy=%s temp=%.1f seed=%u route=%s raw=%d sample-positive=%d sample-zero=%d decoded=%d\n",
                    sampling_config.backend_sampling ? "backend" : "cpu",
                    sampling_config.explicit_policy() ? sampling_config.samplers_value.c_str() : "greedy",
                    sampling_config.explicit_policy() ? static_cast<double>(sampling_config.temp_value) : 0.0,
                    sampling_config.explicit_policy() ? sampling_config.seed_value : 0,
                    phase_timing_route_name(route), route_counts.raw_logits_copy,
                    route_counts.sample_token_copy_positive, route_counts.sample_token_copy_zero, n_generated);
                std::fclose(append);
            }
        }
    }

    std::fprintf(stderr, "generated %d tokens\n", n_generated);
    // Context frees before the chain it may have registered.
    llama_free(context);
    llama_sampler_free(sampler);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
