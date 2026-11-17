#pragma once

// Keep the bounded-completion process result independent of receipt formatting.
inline int phase_timing_completion_exit_code(int requested_tokens, int completed_tokens, int existing_exit_code) {
    if (existing_exit_code != 0) {
        return existing_exit_code;
    }

    return completed_tokens < requested_tokens ? 1 : 0;
}
