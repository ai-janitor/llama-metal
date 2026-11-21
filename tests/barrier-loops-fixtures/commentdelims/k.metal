// self-test case of tests/check_barrier_loops.sh (not built)
kernel void k(uint x [[thread_index_in_threadgroup]], device float * y) {
    // a note that mentions /* in passing
    if (x == 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // and a later note that mentions */
    y[0] = 1.0f;
}
