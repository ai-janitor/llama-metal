// self-test case of tests/check_barrier_loops.sh (not built)
kernel void k(uint sgitg [[simdgroup_index_in_threadgroup]], device float * x) {
    for (uint i = sgitg; i < 8; i += 4)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    x[0] = 1.0f;
}
