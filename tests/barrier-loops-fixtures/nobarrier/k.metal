// self-test case of tests/check_barrier_loops.sh (not built)
kernel void k(device float * x) {
    for (int i = 0; i < 4; ++i) {
        if (x[i] < 0.0f) {
            continue;
        }
        x[i] += 1.0f;
    }
}
