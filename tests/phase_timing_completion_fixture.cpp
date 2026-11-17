#include "completion-status.hpp"

#include <cassert>

int main() {
    // Full completion succeeds; early EOS refuses a full-N claim.
    assert(phase_timing_completion_exit_code(4, 4, 0) == 0);
    assert(phase_timing_completion_exit_code(4, 1, 0) != 0);
    // A decode failure remains nonzero rather than being overwritten.
    assert(phase_timing_completion_exit_code(4, 4, 7) == 7);
    return 0;
}
