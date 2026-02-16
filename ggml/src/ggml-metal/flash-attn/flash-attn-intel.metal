// Flash Attention TILED kernel for Intel Gen9 iGPUs
// Tiles K/V into shared memory for cooperative reuse across threads,
// eliminating the per-thread global memory reads that bottleneck the scalar kernel.
//
// Thread decomposition (flat tid, no simdgroup indexing):
//   TG_SIZE = 128 threads
//   D_SPLIT = 16 — threads collaborating on one QK dot product
//   COL_GROUPS = TG_SIZE / D_SPLIT = 8 — KV positions processed in parallel
//   Bc = 32 — KV tile size loaded cooperatively into shared memory
//   Br = 1 — single query per threadgroup
//
// Thread identity:
//   d_tid   = tid % D_SPLIT     // head dimension slice (0..15)
//   col_tid = tid / D_SPLIT     // KV column group (0..7)
//   Each thread handles Bc/COL_GROUPS = 4 KV positions per tile
//
// Shared memory layout (f16 for K/V tiles):
//   Q:       DK floats
//   K tile:  Bc * DK halfs
//   V tile:  Bc * DV halfs
//   Scores:  Bc floats (attention scores for current tile)
//   Reduce:  TG_SIZE floats (scratch for cross-thread reductions)

// Dequant helpers — copied from scalar kernel to keep files self-contained
template<typename block_t>
inline float tiled_load_element_q8_0(device const char * base, uint64_t row_offset, int elem_idx) {
    device const block_q8_0 * block = (device const block_q8_0 *)(base + row_offset);
    const int block_idx = elem_idx / 32;
    const int offset = elem_idx % 32;
    return float(block[block_idx].qs[offset]) * float(block[block_idx].d);
}

template<typename block_t>
inline float tiled_load_element_q4_0(device const char * base, uint64_t row_offset, int elem_idx) {
    device const block_q4_0 * block = (device const block_q4_0 *)(base + row_offset);
    const int block_idx = elem_idx / 32;
    const int offset = elem_idx % 32;
    device const uint8_t * qs = block[block_idx].qs;
    const int byte_idx = offset / 2;
    const int nibble_shift = (offset % 2) * 4;
    const uint8_t nibble = (qs[byte_idx] >> nibble_shift) & 0x0F;
    const float d = float(block[block_idx].d);
    return float(nibble) * d - 8.0f * d;
}

template<typename block_t>
inline float tiled_load_element_f16(device const char * base, uint64_t row_offset, int elem_idx) {
    device const half * ptr = (device const half *)(base + row_offset);
    return float(ptr[elem_idx]);
}

// Shared memory reduction: sum val across D_SPLIT threads within the same column group
// d_tid is the thread's index within its D_SPLIT group (0..D_SPLIT-1)
// col_tid identifies which column group this thread belongs to
// scratch is TG_SIZE floats of shared memory
inline float shmem_reduce_sum(float val, int d_tid, int col_tid, int D_SPLIT,
                              threadgroup float * scratch, uint tid) {
    scratch[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Only thread 0 of each D_SPLIT group accumulates
    float result = 0.0f;
    if (d_tid == 0) {
        const int base_idx = col_tid * D_SPLIT;
        for (int i = 0; i < D_SPLIT; i++) {
            result += scratch[base_idx + i];
        }
    }
    // Broadcast result back through shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d_tid == 0) {
        scratch[col_tid] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    result = scratch[col_tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

template<int DK, int DV, typename kblock_t, typename vblock_t>
inline void flash_attn_ext_tiled_impl(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  char * shmem_raw [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort    tid      [[thread_index_in_threadgroup]])
{
    // Constants
    constexpr int TG_SIZE    = 128;
    constexpr int D_SPLIT    = 16;
    constexpr int COL_GROUPS = TG_SIZE / D_SPLIT;  // 8
    constexpr int Bc         = 32;
    constexpr int COLS_PER_THREAD = Bc / COL_GROUPS;  // 4

    // Thread identity from flat tid
    const int d_tid   = tid % D_SPLIT;     // 0..15: which head-dim slice
    const int col_tid = tid / D_SPLIT;     // 0..7: which column group

    // Query/batch indices
    const int iq1 = tgpig.x + tg_offset[0];
    const int iq2 = tgpig.y;
    const int iq3 = tgpig.z;

    if (iq1 >= args.ne01) return;

    const int KV = args.ne11;

    // GQA: map Q head -> K/V head
    const int ik2 = iq2 / (args.ne02 / args.ne_12_2);

    // Q pointer (f32)
    device const float * q_ptr = (device const float *)(q + iq1 * args.nb01 + iq2 * args.nb02 + iq3 * args.nb03);

    // K/V base pointers
    device const char * k_head = k + ik2 * args.nb12 + iq3 * args.nb13;
    device const char * v_head = v + ik2 * args.nb22 + iq3 * args.nb23;

    // Mask pointer (f16)
    const bool has_mask = args.ne31 > 0;
    device const half * mask_row = nullptr;
    if (has_mask) {
        mask_row = (device const half *)(mask
            + iq1 * args.nb31
            + (iq2 % args.ne32) * args.nb32
            + (iq3 % args.ne33) * args.nb33);
    }

    // ALiBi slope
    float slope = 1.0f;
    if (args.max_bias > 0.0f) {
        const short h = iq2;
        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;
        slope = pow(base, float(exph));
    }

    const bool has_softcap = args.logit_softcap != 0.0f;

    // --- Shared memory layout ---
    // Q: DK floats
    // K tile: Bc * DK halfs
    // V tile: Bc * DV halfs
    // Scores: Bc floats
    // Reduce: TG_SIZE floats
    threadgroup float * sh_q       = (threadgroup float *)shmem_raw;
    threadgroup half  * sh_k       = (threadgroup half  *)(sh_q + DK);
    threadgroup half  * sh_v       = (threadgroup half  *)(sh_k + Bc * DK);
    threadgroup float * sh_scores  = (threadgroup float *)(sh_v + Bc * DV);
    threadgroup float * sh_reduce  = sh_scores + Bc;

    // --- Load Q into shared memory (cooperative, all 128 threads) ---
    // Each thread loads ceil(DK/128) elements, pre-scaled
    for (int i = tid; i < DK; i += TG_SIZE) {
        sh_q[i] = q_ptr[i] * args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Per-thread accumulators ---
    // Each thread accumulates DV/D_SPLIT elements of the output
    constexpr int O_PER_THREAD = (DV + D_SPLIT - 1) / D_SPLIT;
    float o[O_PER_THREAD];
    for (int j = 0; j < O_PER_THREAD; j++) {
        o[j] = 0.0f;
    }
    float max_score = -__FLT_MAX__ / 2.0f;
    float sum_exp   = 0.0f;

    // --- Main KV tile loop ---
    for (int tile_start = 0; tile_start < KV; tile_start += Bc) {
        const int tile_end = min(tile_start + Bc, KV);
        const int tile_size = tile_end - tile_start;

        // --- Cooperative K load: all 128 threads load Bc*DK elements into shared memory ---
        const int k_total = tile_size * DK;
        for (int i = tid; i < k_total; i += TG_SIZE) {
            const int kv_pos = i / DK;
            const int d_idx  = i % DK;
            const int global_kv = tile_start + kv_pos;
            const uint64_t k_row_offset = (uint64_t)global_kv * args.nb11;

            float val;
            if (is_same<kblock_t, block_q8_0>::value) {
                val = tiled_load_element_q8_0<kblock_t>(k_head, k_row_offset, d_idx);
            } else if (is_same<kblock_t, block_q4_0>::value) {
                val = tiled_load_element_q4_0<kblock_t>(k_head, k_row_offset, d_idx);
            } else {
                val = tiled_load_element_f16<kblock_t>(k_head, k_row_offset, d_idx);
            }
            sh_k[kv_pos * DK + d_idx] = half(val);
        }
        // Zero-pad K tile for positions beyond KV
        for (int i = tid; i < Bc * DK; i += TG_SIZE) {
            const int kv_pos = i / DK;
            if (kv_pos >= tile_size) {
                sh_k[i] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- QK dot products ---
        // Each thread in col_group computes partial dot for its COLS_PER_THREAD KV positions
        // over D_SPLIT head elements, then reduce across D_SPLIT threads
        for (int c = 0; c < COLS_PER_THREAD; c++) {
            const int kv_local = col_tid * COLS_PER_THREAD + c;  // 0..31
            const int kv_global = tile_start + kv_local;

            float partial_dot = 0.0f;
            if (kv_local < tile_size) {
                // Each d_tid thread handles elements [d_tid*DK/D_SPLIT .. (d_tid+1)*DK/D_SPLIT)
                const int d_start = d_tid * (DK / D_SPLIT);
                const int d_end   = d_start + (DK / D_SPLIT);
                for (int d = d_start; d < d_end; d++) {
                    partial_dot += sh_q[d] * float(sh_k[kv_local * DK + d]);
                }
            }

            // Reduce partial dots across D_SPLIT threads to get full score
            float score = shmem_reduce_sum(partial_dot, d_tid, col_tid, D_SPLIT, sh_reduce, tid);

            // Softcap
            if (has_softcap) {
                score = args.logit_softcap * precise::tanh(score);
            }

            // Apply mask
            if (has_mask && kv_global < KV) {
                score += slope * float(mask_row[kv_global]);
            }

            // Store score for this KV position (only need one thread per column group)
            if (d_tid == 0 && kv_local < tile_size) {
                sh_scores[kv_local] = score;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Cooperative V load ---
        const int v_total = tile_size * DV;
        for (int i = tid; i < v_total; i += TG_SIZE) {
            const int kv_pos = i / DV;
            const int d_idx  = i % DV;
            const int global_kv = tile_start + kv_pos;
            const uint64_t v_row_offset = (uint64_t)global_kv * args.nb21;

            float val;
            if (is_same<vblock_t, block_q8_0>::value) {
                val = tiled_load_element_q8_0<vblock_t>(v_head, v_row_offset, d_idx);
            } else if (is_same<vblock_t, block_q4_0>::value) {
                val = tiled_load_element_q4_0<vblock_t>(v_head, v_row_offset, d_idx);
            } else {
                val = tiled_load_element_f16<vblock_t>(v_head, v_row_offset, d_idx);
            }
            sh_v[kv_pos * DV + d_idx] = half(val);
        }
        // Zero-pad V tile
        for (int i = tid; i < Bc * DV; i += TG_SIZE) {
            const int kv_pos = i / DV;
            if (kv_pos >= tile_size) {
                sh_v[i] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Online softmax + V accumulation ---
        // Each thread processes its COLS_PER_THREAD KV positions
        for (int c = 0; c < COLS_PER_THREAD; c++) {
            const int kv_local = col_tid * COLS_PER_THREAD + c;
            if (kv_local >= tile_size) continue;

            const float score = sh_scores[kv_local];

            // Online softmax update
            float new_max   = max(max_score, score);
            float factor    = fast::exp(max_score - new_max);
            float exp_score = fast::exp(score - new_max);

            max_score = new_max;
            sum_exp   = sum_exp * factor + exp_score;

            // Rescale existing accumulators and add weighted V
            // Each thread handles DV/D_SPLIT elements of V
            const int v_start = d_tid * (DV / D_SPLIT);
            const int v_end   = v_start + (DV / D_SPLIT);
            for (int vi = 0; vi < O_PER_THREAD; vi++) {
                const int v_idx = v_start + vi;
                if (v_idx < v_end && v_idx < DV) {
                    o[vi] = o[vi] * factor + exp_score * float(sh_v[kv_local * DV + v_idx]);
                }
            }
        }
    }

    // --- Cross-thread reduction ---
    // Each thread has partial o[] for its D_SPLIT slice and its COL_GROUPS subset of KV positions
    // Need to reduce across col_groups (sum the partial softmax results with rescaling)

    // Step 1: Reduce max_score across all 8 col_groups via shared memory
    // Use sh_reduce for this
    sh_reduce[tid] = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each d_tid group needs to find global max across all col_tids sharing same d_tid
    float global_max = -__FLT_MAX__ / 2.0f;
    for (int g = 0; g < COL_GROUPS; g++) {
        global_max = max(global_max, sh_reduce[g * D_SPLIT + d_tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Rescale sum_exp and o[] to global_max, then reduce across col_groups
    float rescale = fast::exp(max_score - global_max);
    sum_exp *= rescale;
    for (int vi = 0; vi < O_PER_THREAD; vi++) {
        o[vi] *= rescale;
    }

    // Reduce sum_exp across col_groups
    sh_reduce[tid] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum_exp = 0.0f;
    for (int g = 0; g < COL_GROUPS; g++) {
        global_sum_exp += sh_reduce[g * D_SPLIT + d_tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce o[] across col_groups via shared memory
    // Each thread writes its O_PER_THREAD values, then col_tid==0 sums
    const int v_start = d_tid * (DV / D_SPLIT);
    for (int vi = 0; vi < O_PER_THREAD; vi++) {
        const int v_idx = v_start + vi;
        if (v_idx >= DV) break;

        sh_reduce[tid] = o[vi];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col_tid == 0) {
            float total = 0.0f;
            for (int g = 0; g < COL_GROUPS; g++) {
                total += sh_reduce[g * D_SPLIT + d_tid];
            }
            o[vi] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Attention sinks ---
    const bool has_sinks = args.ns10 > 0;
    if (has_sinks && col_tid == 0) {
        // Thread d_tid==0 loads the sink value, others get -INF
        const float s = d_tid == 0 ? ((device const float *) sinks)[iq2] : -__FLT_MAX__ / 2.0f;

        // Find max across d_tid threads via shared memory
        sh_reduce[d_tid] = max(global_max, s);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sink_val = sh_reduce[0];
        for (int i = 1; i < D_SPLIT; i++) {
            sink_val = max(sink_val, sh_reduce[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float ms = fast::exp(global_max - sink_val);

        // Sum exp(s - sink_val) across threads
        sh_reduce[d_tid] = fast::exp(s - sink_val);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vs_sum = 0.0f;
        for (int i = 0; i < D_SPLIT; i++) {
            vs_sum += sh_reduce[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        global_sum_exp = global_sum_exp * ms + vs_sum;
        for (int vi = 0; vi < O_PER_THREAD; vi++) {
            o[vi] *= ms;
        }
    }

    // --- Final normalization and write ---
    if (col_tid == 0) {
        for (int vi = 0; vi < O_PER_THREAD; vi++) {
            o[vi] = (global_sum_exp > 0.0f) ? (o[vi] / global_sum_exp) : 0.0f;
        }

        device float * dst_ptr = (device float *)dst;
        const int dst_base = iq3 * args.ne2 * args.ne1 * DV
                           + iq1 * args.ne1 * DV
                           + iq2 * DV;
        for (int vi = 0; vi < O_PER_THREAD; vi++) {
            const int v_idx = v_start + vi;
            if (v_idx < DV) {
                dst_ptr[dst_base + v_idx] = o[vi];
            }
        }
    }
}

// --- Kernel wrappers ---
// F16 K/V variants

#define TILED_KERNEL_WRAPPER(host, dk, dv, ktype, vtype) \
[[host_name(host)]] \
kernel void kernel_flash_attn_ext_tiled_##dk##_##dv##_##ktype( \
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]], \
    device const char * q     [[buffer(1)]], \
    device const char * k     [[buffer(2)]], \
    device const char * v     [[buffer(3)]], \
    device const char * mask  [[buffer(4)]], \
    device       char * dst   [[buffer(5)]], \
    device const char * sinks [[buffer(6)]], \
    constant uint32_t * tg_offset [[buffer(7)]], \
    threadgroup  char * shmem [[threadgroup(0)]], \
    uint3   tgpig    [[threadgroup_position_in_grid]], \
    ushort    tid      [[thread_index_in_threadgroup]]) \
{ \
    flash_attn_ext_tiled_impl<dk, dv, ktype, vtype>( \
        args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tid); \
}

// F16 dk64/dv64
TILED_KERNEL_WRAPPER("kernel_flash_attn_ext_tiled_dk64_dv64", 64, 64, half2, half2)

// Q8_0 dk64/dv64
TILED_KERNEL_WRAPPER("kernel_flash_attn_ext_tiled_q8_0_dk64_dv64", 64, 64, block_q8_0, block_q8_0)

// Q4_0 dk64/dv64
TILED_KERNEL_WRAPPER("kernel_flash_attn_ext_tiled_q4_0_dk64_dv64", 64, 64, block_q4_0, block_q4_0)

#undef TILED_KERNEL_WRAPPER
