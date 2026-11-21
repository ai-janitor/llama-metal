// self-test case of tests/check_barrier_loops.sh (not built)
kernel void k(device float * x) {
    x[0] = 1.0f;
}
