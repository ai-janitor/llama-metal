// self-test case of tests/check_barrier_loops.sh (not built)
kernel void k(uint a [[thread_index_in_threadgroup]], constant int & n, device float * y) {
    if (n > 4) {
        y[a] = 1.0f;
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
