// self-test case of tests/check_barrier_loops.sh (not built)
kernel void k(uint x [[thread_index_in_threadgroup]], device float * y) {
    if (x == 0) {
        y[0] = 1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
