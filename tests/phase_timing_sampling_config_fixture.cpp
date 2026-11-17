#include "sampling-config.hpp"

#include <cstdio>
#include <string>
#include <vector>

// Non-elidable check: assert() would vanish under NDEBUG, so failures are
// counted and returned as the process result in every build mode.
static int failures = 0;
static void check(bool condition, const char * name) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s\n", name);
        ++failures;
    }
}

static size_t lifecycle_index(const std::vector<std::string> & lifecycle, const std::string & event) {
    for (size_t i = 0; i < lifecycle.size(); ++i) {
        if (lifecycle[i] == event) {
            return i;
        }
    }
    return lifecycle.size();
}

int main() {
    std::string diagnostic;

    // Valid exact public flags map to the one bounded explicit policy.
    {
        phase_timing_sampling_config config;
        check(phase_timing_apply_sampling_option(config, "--no-backend-sampling", nullptr, diagnostic) == phase_timing_sampling_status::ok, "valid-cpu-flag");
        check(phase_timing_apply_sampling_option(config, "--samplers", "temperature", diagnostic) == phase_timing_sampling_status::ok, "valid-samplers");
        check(phase_timing_apply_sampling_option(config, "--temp", "0", diagnostic) == phase_timing_sampling_status::ok, "valid-temp");
        check(phase_timing_apply_sampling_option(config, "--seed", "42", diagnostic) == phase_timing_sampling_status::ok, "valid-seed");
        check(phase_timing_validate_sampling_policy(config, diagnostic) == phase_timing_sampling_status::ok, "valid-policy");
        check(config.explicit_policy() && !config.backend_sampling, "valid-cpu-policy");
        check(!phase_timing_make_sampler_plan(config).register_seq0, "cpu-no-registration");
    }
    {
        phase_timing_sampling_config config;
        check(phase_timing_apply_sampling_option(config, "--backend-sampling", nullptr, diagnostic) == phase_timing_sampling_status::ok, "valid-backend-flag");
        check(phase_timing_apply_sampling_option(config, "--samplers", "temperature", diagnostic) == phase_timing_sampling_status::ok, "backend-samplers");
        check(phase_timing_apply_sampling_option(config, "--temp", "0", diagnostic) == phase_timing_sampling_status::ok, "backend-temp");
        check(phase_timing_apply_sampling_option(config, "--seed", "42", diagnostic) == phase_timing_sampling_status::ok, "backend-seed");
        const auto plan = phase_timing_make_sampler_plan(config);
        check(plan.register_seq0, "backend-registration-only-enabled");
        // Cleanup order is observed, not assumed: context before chain.
        check(lifecycle_index(plan.lifecycle, "sampler-chain:create") < lifecycle_index(plan.lifecycle, "context:create"), "chain-before-context");
        check(lifecycle_index(plan.lifecycle, "context:register-seq0") < lifecycle_index(plan.lifecycle, "context:free"), "register-before-free");
        check(lifecycle_index(plan.lifecycle, "context:free") < lifecycle_index(plan.lifecycle, "sampler-chain:free"), "context-freed-before-chain");
    }

    // Default with no policy flags remains greedy and unregistered.
    {
        const phase_timing_sampling_config config;
        check(!config.explicit_policy(), "default-greedy");
        check(phase_timing_validate_sampling_policy(config, diagnostic) == phase_timing_sampling_status::ok, "default-valid");
        check(!phase_timing_make_sampler_plan(config).register_seq0, "default-no-registration");
    }

    // Unsupported values are a validation result, distinct from a parse pass.
    {
        phase_timing_sampling_config config;
        check(phase_timing_apply_sampling_option(config, "--samplers", "top-k", diagnostic) == phase_timing_sampling_status::unsupported_value, "unsupported-samplers-not-accepted");
        check(config.samplers_seen && config.samplers_value == "top-k", "unsupported-value-retained");
        check(phase_timing_apply_sampling_option(config, "--temp", "0.5", diagnostic) == phase_timing_sampling_status::unsupported_value, "unsupported-temp");
        check(phase_timing_apply_sampling_option(config, "--seed", "7", diagnostic) == phase_timing_sampling_status::unsupported_value, "unsupported-seed");
    }

    // Missing values and unknown flags reject before any model work.
    {
        phase_timing_sampling_config config;
        check(phase_timing_apply_sampling_option(config, "--samplers", nullptr, diagnostic) == phase_timing_sampling_status::missing_value, "missing-samplers-value");
        check(phase_timing_apply_sampling_option(config, "--temp", nullptr, diagnostic) == phase_timing_sampling_status::missing_value, "missing-temp-value");
        check(phase_timing_apply_sampling_option(config, "--seed", nullptr, diagnostic) == phase_timing_sampling_status::missing_value, "missing-seed-value");
        check(phase_timing_apply_sampling_option(config, "--bogus", nullptr, diagnostic) == phase_timing_sampling_status::unknown_option, "unknown-option");
    }

    // A partial explicit policy is incomplete, never silently completed.
    {
        phase_timing_sampling_config config;
        check(phase_timing_apply_sampling_option(config, "--temp", "0", diagnostic) == phase_timing_sampling_status::ok, "partial-temp-parses");
        check(phase_timing_validate_sampling_policy(config, diagnostic) == phase_timing_sampling_status::incomplete_policy, "partial-policy-incomplete");
    }

    // Literal route records use a real tab. Positive payload proves a token;
    // a zero-payload sample event is fallback, never backend success.
    {
        phase_timing_sampling_config backend;
        std::string ignored;
        phase_timing_apply_sampling_option(backend, "--backend-sampling", nullptr, ignored);
        phase_timing_apply_sampling_option(backend, "--samplers", "temperature", ignored);
        phase_timing_apply_sampling_option(backend, "--temp", "0", ignored);
        phase_timing_apply_sampling_option(backend, "--seed", "42", ignored);

        const std::vector<std::string> backend_records = {
            "L0\tevent\tsample_token_copy\t1.000000000\t1.100000000\t1\n",
        };
        const auto backend_counts = phase_timing_count_route_records(backend_records);
        check(backend_counts.sample_token_copy_positive == 1 && backend_counts.raw_logits_copy == 0, "backend-positive-counts");
        check(phase_timing_classify_route(backend_counts, 1, backend) == phase_timing_route::backend_sampled, "backend-positive-route");

        const std::vector<std::string> zero_records = {
            "L0\tevent\tsample_token_copy\t1.000000000\t1.100000000\t0",
        };
        const auto zero_counts = phase_timing_count_route_records(zero_records);
        check(zero_counts.sample_token_copy_zero == 1 && zero_counts.sample_token_copy_positive == 0, "zero-payload-counts");
        check(phase_timing_classify_route(zero_counts, 1, backend) == phase_timing_route::fallback, "zero-payload-not-backend");

        phase_timing_sampling_config cpu = backend;
        cpu.backend_sampling = false;
        const std::vector<std::string> cpu_records = {
            "L0\tevent\traw_logits_copy\t1.000000000\t1.100000000\t4096",
        };
        check(phase_timing_classify_route(phase_timing_count_route_records(cpu_records), 1, cpu) == phase_timing_route::cpu_policy, "cpu-route");
        check(phase_timing_classify_route(phase_timing_count_route_records(cpu_records), 0, cpu) == phase_timing_route::no_decode, "no-decode-route");
        check(phase_timing_classify_route(phase_timing_count_route_records(backend_records), 1, cpu) == phase_timing_route::fallback, "cpu-fallback-route");

        // A zero-payload sample record next to an otherwise clean route
        // disqualifies it: the window did not run the policy cleanly.
        const std::vector<std::string> backend_mixed_records = {
            "L0\tevent\tsample_token_copy\t1.000000000\t1.100000000\t1\n",
            "L0\tevent\tsample_token_copy\t1.200000000\t1.300000000\t0\n",
        };
        const auto backend_mixed_counts = phase_timing_count_route_records(backend_mixed_records);
        check(backend_mixed_counts.sample_token_copy_positive == 1 && backend_mixed_counts.sample_token_copy_zero == 1 &&
              backend_mixed_counts.raw_logits_copy == 0, "backend-mixed-zero-counts");
        check(phase_timing_classify_route(backend_mixed_counts, 1, backend) == phase_timing_route::fallback, "backend-mixed-zero-not-backend");

        const std::vector<std::string> cpu_mixed_records = {
            "L0\tevent\traw_logits_copy\t1.000000000\t1.100000000\t4096\n",
            "L0\tevent\tsample_token_copy\t1.200000000\t1.300000000\t0\n",
        };
        const auto cpu_mixed_counts = phase_timing_count_route_records(cpu_mixed_records);
        check(cpu_mixed_counts.raw_logits_copy == 1 && cpu_mixed_counts.sample_token_copy_zero == 1 &&
              cpu_mixed_counts.sample_token_copy_positive == 0, "cpu-mixed-zero-counts");
        check(phase_timing_classify_route(cpu_mixed_counts, 1, cpu) == phase_timing_route::fallback, "cpu-mixed-zero-not-cpu-policy");
    }

    if (failures != 0) {
        std::fprintf(stderr, "phase_timing_sampling_config_fixture: %d failures\n", failures);
        return 1;
    }
    std::fprintf(stdout, "phase_timing_sampling_config_fixture: ok\n");
    return 0;
}
