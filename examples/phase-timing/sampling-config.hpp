#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Pure sampling-policy seam for the bounded phase-timing driver.
// This header has no llama or Metal dependency so a CPU fixture can prove the
// public flag mapping, the registration decision, and the receipt route without
// loading a model. The driver includes the same header, so the fixture tests
// the production seam rather than a copy of it.

// Result of parsing or validating one sampling option. Unsupported and
// incomplete are distinct: a value can parse cleanly yet be outside the one
// policy this bounded driver supports (temperature 0, seed 42).
enum class phase_timing_sampling_status {
    ok,
    missing_value,
    unknown_option,
    unsupported_value,
    incomplete_policy,
};

// Requested sampling policy collected from the driver command line.
// A raw field is kept for every spelled value so validation, not parsing, owns
// the supported-policy decision and diagnostics can name the rejected value.
struct phase_timing_sampling_config {
    // True only for --backend-sampling; --no-backend-sampling clears it.
    bool backend_sampling = false;
    // Whether --samplers was spelled, and its literal value.
    bool samplers_seen = false;
    std::string samplers_value;
    // Whether --temp was spelled, and its parsed value.
    bool temp_seen = false;
    float temp_value = 0.0f;
    // Whether --seed was spelled, and its parsed value.
    bool seed_seen = false;
    uint32_t seed_value = 0;

    // Any sampling-policy flag makes the policy explicit; none means greedy.
    bool explicit_policy() const {
        return samplers_seen || temp_seen || seed_seen;
    }

    // The bounded explicit policy is a triple, never a partial chain.
    bool complete_policy() const {
        return samplers_seen && temp_seen && seed_seen;
    }
};

// Parse one option. `value` is nullptr when the option takes no value or when
// a required value is absent at end of argv. On unsupported_value the literal
// value stays in `config`, so the caller can report parse-vs-validation
// accurately instead of pretending the flag was accepted.
inline phase_timing_sampling_status phase_timing_apply_sampling_option(
        phase_timing_sampling_config & config,
        const std::string & option,
        const char * value,
        std::string & diagnostic) {
    diagnostic.clear();
    if (option == "--backend-sampling") {
        config.backend_sampling = true;
        return phase_timing_sampling_status::ok;
    }
    if (option == "--no-backend-sampling") {
        config.backend_sampling = false;
        return phase_timing_sampling_status::ok;
    }
    if (option == "--samplers" || option == "--temp" || option == "--seed") {
        if (value == nullptr) {
            diagnostic = option + " requires a value";
            return phase_timing_sampling_status::missing_value;
        }
        if (option == "--samplers") {
            config.samplers_seen = true;
            config.samplers_value = value;
            if (config.samplers_value != "temperature") {
                diagnostic = "unsupported --samplers value: " + config.samplers_value;
                return phase_timing_sampling_status::unsupported_value;
            }
            return phase_timing_sampling_status::ok;
        }
        if (option == "--temp") {
            // Parse the whole token; a trailing suffix is not a temperature.
            const std::string text = value;
            size_t consumed = 0;
            float parsed = 0.0f;
            try {
                parsed = std::stof(text, &consumed);
            } catch (...) {
                diagnostic = "unsupported --temp value: " + text;
                return phase_timing_sampling_status::unsupported_value;
            }
            if (consumed != text.size()) {
                diagnostic = "unsupported --temp value: " + text;
                return phase_timing_sampling_status::unsupported_value;
            }
            config.temp_seen = true;
            config.temp_value = parsed;
            if (parsed != 0.0f) {
                diagnostic = "unsupported --temp value: " + text;
                return phase_timing_sampling_status::unsupported_value;
            }
            return phase_timing_sampling_status::ok;
        }
        // --seed: digits only, so a sign or suffix cannot wrap into a seed.
        const std::string text = value;
        uint64_t parsed = 0;
        bool digits = !text.empty();
        for (const char ch : text) {
            if (ch < '0' || ch > '9') {
                digits = false;
                break;
            }
            parsed = parsed * 10 + static_cast<uint64_t>(ch - '0');
            if (parsed > 0xffffffffULL) {
                digits = false;
                break;
            }
        }
        if (!digits) {
            diagnostic = "unsupported --seed value: " + text;
            return phase_timing_sampling_status::unsupported_value;
        }
        config.seed_seen = true;
        config.seed_value = static_cast<uint32_t>(parsed);
        if (config.seed_value != 42) {
            diagnostic = "unsupported --seed value: " + text;
            return phase_timing_sampling_status::unsupported_value;
        }
        return phase_timing_sampling_status::ok;
    }
    diagnostic = "unknown option: " + option;
    return phase_timing_sampling_status::unknown_option;
}

// Validate the collected policy before any model initialization.
// No explicit policy is the retained greedy default and is always valid.
inline phase_timing_sampling_status phase_timing_validate_sampling_policy(
        const phase_timing_sampling_config & config,
        std::string & diagnostic) {
    diagnostic.clear();
    if (!config.explicit_policy()) {
        return phase_timing_sampling_status::ok;
    }
    if (!config.complete_policy()) {
        diagnostic = "explicit sampling policy requires --samplers temperature --temp 0 --seed 42";
        return phase_timing_sampling_status::incomplete_policy;
    }
    if (config.samplers_value != "temperature" || config.temp_value != 0.0f || config.seed_value != 42) {
        diagnostic = "unsupported sampling policy: --samplers " + config.samplers_value;
        return phase_timing_sampling_status::unsupported_value;
    }
    return phase_timing_sampling_status::ok;
}

// Registration and cleanup plan shared by the driver and the fixture.
// Registration is enabled only by the backend arm of a valid explicit policy;
// both matched arms otherwise build the same chain in the same order.
struct phase_timing_sampler_plan {
    // True when seq 0 must be bound in llama_context_params before creation.
    bool register_seq0 = false;
    // Ordered lifecycle events the driver follows for an explicit policy.
    std::vector<std::string> lifecycle;
};

inline phase_timing_sampler_plan phase_timing_make_sampler_plan(
        const phase_timing_sampling_config & config) {
    phase_timing_sampler_plan plan;
    std::string ignored;
    plan.register_seq0 = config.backend_sampling && config.explicit_policy() &&
        phase_timing_validate_sampling_policy(config, ignored) == phase_timing_sampling_status::ok;
    // The chain is created before the context so a backend registration can
    // point at it, and the context is freed before the chain on every path.
    plan.lifecycle.push_back("sampler-chain:create");
    plan.lifecycle.push_back("context:create");
    if (plan.register_seq0) {
        plan.lifecycle.push_back("context:register-seq0");
    }
    plan.lifecycle.push_back("context:free");
    plan.lifecycle.push_back("sampler-chain:free");
    return plan;
}

// Copy-producer counts read from literal L0 receipt records.
// The producer payload for sample_token_copy is t_sampled.size(): a zero
// payload is an event with no sampled token and must never prove a backend
// route. Raw records carry a byte count and are counted by name.
struct phase_timing_route_counts {
    int raw_logits_copy = 0;
    int sample_token_copy_positive = 0;
    int sample_token_copy_zero = 0;
};

// Count literal tab-delimited records exactly as ggml_l0_dump writes them:
// "L0", "event", name, start, end, payload separated by a real tab (0x09).
inline phase_timing_route_counts phase_timing_count_route_records(const std::vector<std::string> & lines) {
    phase_timing_route_counts counts;
    for (const std::string & raw_line : lines) {
        // Strip only the record terminator; tabs inside the record stay literal.
        std::string line = raw_line;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        // Split on the real tab character, never on a backslash escape.
        std::vector<std::string> fields;
        size_t begin = 0;
        while (true) {
            const size_t tab = line.find('\t', begin);
            if (tab == std::string::npos) {
                fields.push_back(line.substr(begin));
                break;
            }
            fields.push_back(line.substr(begin, tab - begin));
            begin = tab + 1;
        }
        if (fields.size() < 6 || fields[0] != "L0" || fields[1] != "event") {
            continue;
        }
        if (fields[2] == "raw_logits_copy") {
            counts.raw_logits_copy++;
        } else if (fields[2] == "sample_token_copy") {
            // The payload decides token presence; the event name alone does not.
            bool digits = !fields[5].empty();
            uint64_t payload = 0;
            for (const char ch : fields[5]) {
                if (ch < '0' || ch > '9') {
                    digits = false;
                    break;
                }
                payload = payload * 10 + static_cast<uint64_t>(ch - '0');
            }
            if (!digits) {
                continue;
            }
            if (payload > 0) {
                counts.sample_token_copy_positive++;
            } else {
                counts.sample_token_copy_zero++;
            }
        }
    }
    return counts;
}

// Route proven by observed copy producers, never by the requested flag alone.
enum class phase_timing_route {
    no_decode,
    cpu_policy,
    backend_sampled,
    fallback,
};

inline const char * phase_timing_route_name(phase_timing_route route) {
    switch (route) {
        case phase_timing_route::no_decode: return "no-decode";
        case phase_timing_route::cpu_policy: return "cpu-policy";
        case phase_timing_route::backend_sampled: return "backend-sampled";
        case phase_timing_route::fallback: return "fallback";
    }
    return "fallback";
}

// Classify one receipt window. Per decoded window the CPU policy expects one
// raw copy and no positive sample copy; the backend policy expects the exact
// opposite. Anything else, including a zero-payload sample event, is fallback.
inline phase_timing_route phase_timing_classify_route(
        const phase_timing_route_counts & counts,
        int decoded_windows,
        const phase_timing_sampling_config & config) {
    if (decoded_windows <= 0) {
        return phase_timing_route::no_decode;
    }
    // Invariant: a zero-payload sample record means a window did not run the
    // policy cleanly, so it disqualifies both routes even beside a clean copy.
    if (counts.sample_token_copy_zero != 0) {
        return phase_timing_route::fallback;
    }
    if (config.explicit_policy() && config.backend_sampling) {
        const bool backend_proven = counts.raw_logits_copy == 0 &&
            counts.sample_token_copy_positive == decoded_windows;
        return backend_proven ? phase_timing_route::backend_sampled : phase_timing_route::fallback;
    }
    const bool cpu_proven = counts.raw_logits_copy == decoded_windows &&
        counts.sample_token_copy_positive == 0;
    return cpu_proven ? phase_timing_route::cpu_policy : phase_timing_route::fallback;
}
