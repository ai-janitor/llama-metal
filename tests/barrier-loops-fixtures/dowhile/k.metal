// self-test case of tests/check_barrier_loops.sh (not built)
kernel void k(constant int & n, device float * x) {
    int i = 0;
    do {
        x[i] += 1.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (x[i] > 2.0f) {
            i++;
            continue;
        }
        i++;
    } while (i < n);
}
