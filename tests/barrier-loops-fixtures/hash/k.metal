// self-test case of tests/check_barrier_loops.sh (not built)
kernel void k(uint sgitg [[simdgroup_index_in_threadgroup]], device float * x) {
    for (int i = 0; i < 4; ++i) {
        if (x[i] < 0.0f) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        x[i] += 1.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        x[i] *= 2.0f;
    }
}
