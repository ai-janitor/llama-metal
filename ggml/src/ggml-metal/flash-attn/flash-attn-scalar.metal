// Flash Attention SCALAR fallback for GPUs without simdgroup matrix multiply
// Algorithm from Apple MLX sdpa_vector.h, adapted for ggml tensor layout
// Runs on any Metal GPU with simdgroup_reduction (AMD, older Apple, etc.)
//
// Thread decomposition: BN simdgroups × BD threads
//   - Each simdgroup processes one KV position at a time, stepping by BN
//   - Each thread handles DK/BD elements of Q/K and DV/BD elements of V
//   - QK dot product reduced via simd_sum within each simdgroup
//   - Cross-simdgroup reduction via shared memory after all KV processed
//
// BN is DYNAMIC at runtime: BN = threadgroup_size / BD
// BD = [[threads_per_simdgroup]] (runtime: 32 on AMD, 8-32 on Intel Gen9)
// Dispatch uses BN=4 with BD from pipeline's threadExecutionWidth to set threadgroup size,
// but Intel Gen9 may run with smaller BD at runtime → more actual simdgroups.
// Min supported SIMD width: 8 (Intel Gen9 worst case)

// Helper: load single element from quantized K/V cache
template<typename block_t>
inline float load_element_q8_0(device const char * base, uint64_t row_offset, int elem_idx) {
    // Q8_0: 32-element blocks
    device const block_q8_0 * block = (device const block_q8_0 *)(base + row_offset);
    const int block_idx = elem_idx / 32;
    const int offset = elem_idx % 32;
    return float(block[block_idx].qs[offset]) * float(block[block_idx].d);
}

template<typename block_t>
inline float load_element_q4_0(device const char * base, uint64_t row_offset, int elem_idx) {
    // Q4_0: 32-element blocks (16 bytes packed)
    device const block_q4_0 * block = (device const block_q4_0 *)(base + row_offset);
    const int block_idx = elem_idx / 32;
    const int offset = elem_idx % 32;

    // Extract 4-bit nibble from packed storage
    device const uint8_t * qs = block[block_idx].qs;
    const int byte_idx = offset / 2;
    const int nibble_shift = (offset % 2) * 4;
    const uint8_t nibble = (qs[byte_idx] >> nibble_shift) & 0x0F;

    // Dequantize: value = nibble * scale - 8 * scale
    const float d = float(block[block_idx].d);
    return float(nibble) * d - 8.0f * d;
}

template<typename block_t>
inline float load_element_f16(device const char * base, uint64_t row_offset, int elem_idx) {
    device const half * ptr = (device const half *)(base + row_offset);
    return float(ptr[elem_idx]);
}

template<int DK, int DV, typename kblock_t, typename vblock_t>
inline void flash_attn_ext_scalar_impl(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    const int BD = (int)simd_size;  // runtime: 32 on AMD, 8-32 on Intel Gen9
    const int BN = (int)(tptg.x) / BD;  // dynamic: more simdgroups when BD is smaller

    // Compile-time max allocation for smallest supported SIMD width (8)
    // Intel Gen9 can run with BD=8 despite pipeline reporting 32
    constexpr int max_qk_per_thread = (DK + 7) / 8;
    constexpr int max_v_per_thread  = (DV + 7) / 8;

    // Runtime: actual elements per thread
    const int qk_per_thread = (DK + BD - 1) / BD;
    const int v_per_thread  = (DV + BD - 1) / BD;

    const int iq1 = tgpig.x + tg_offset[0];  // query position (with chunk offset)
    const int iq2 = tgpig.y;  // Q head index
    const int iq3 = tgpig.z;  // batch index

    if (iq1 >= args.ne01) return;

    const int KV = args.ne11;

    // GQA: map Q head → K/V head
    const int ik2 = iq2 / (args.ne02 / args.ne_12_2);

    // Q pointer (f32, contiguous within row)
    device const float * q_ptr = (device const float *)(q + iq1 * args.nb01 + iq2 * args.nb02 + iq3 * args.nb03);

    // K/V base pointers — offset by KV head and batch
    device const char * k_head = k + ik2 * args.nb12 + iq3 * args.nb13;
    device const char * v_head = v + ik2 * args.nb22 + iq3 * args.nb23;

    // Mask pointer (f16) — offset by query position, head, batch
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

    // Load Q into registers (pre-scaled; args.scale already divided by softcap if needed)
    // Guard against out-of-bounds when DK is not a multiple of BD
    float q_reg[max_qk_per_thread];
    for (int j = 0; j < qk_per_thread; j++) {
        const int q_idx = simd_lid * qk_per_thread + j;
        q_reg[j] = (q_idx < DK) ? q_ptr[q_idx] * args.scale : 0.0f;
    }

    // Per-thread accumulators
    float o[max_v_per_thread];
    for (int j = 0; j < v_per_thread; j++) {
        o[j] = 0;
    }
    float max_score = -__FLT_MAX__ / 2.0f;
    float sum_exp   = 0;

    // Main KV loop: each simdgroup steps through KV by BN
    for (int i = simd_gid; i < KV; i += BN) {
        const uint64_t k_row_offset = (uint64_t)i * args.nb11;

        // QK dot product: partial per thread, then simd_sum
        // Load K elements based on type (F16, Q8_0, or Q4_0)
        // Guard against out-of-bounds when DK is not a multiple of BD
        float score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            const int k_idx = simd_lid * qk_per_thread + j;
            if (k_idx >= DK) break;
            float k_val;

            // Type dispatch: load K element based on kblock_t
            if (is_same<kblock_t, block_q8_0>::value) {
                k_val = load_element_q8_0<kblock_t>(k_head, k_row_offset, k_idx);
            } else if (is_same<kblock_t, block_q4_0>::value) {
                k_val = load_element_q4_0<kblock_t>(k_head, k_row_offset, k_idx);
            } else {
                // F16 path (half2 = placeholder for F16)
                k_val = load_element_f16<kblock_t>(k_head, k_row_offset, k_idx);
            }

            score += q_reg[j] * k_val;
        }
        score = simd_sum(score);

        // Logit softcap: tanh(score/softcap) * softcap (score already divided by softcap via prescale)
        if (has_softcap) {
            score = args.logit_softcap * precise::tanh(score);
        }

        // Apply additive mask with ALiBi slope. -INF entries naturally → exp(-INF) = 0
        if (has_mask) {
            score += slope * float(mask_row[i]);
        }

        // Online softmax update (all lanes have same score after simd_sum)
        float new_max   = max(max_score, score);
        float factor    = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp   = sum_exp * factor + exp_score;

        // Fused V accumulation: O = O * factor + exp_score * V
        // Load V elements based on type (F16, Q8_0, or Q4_0)
        // Guard against out-of-bounds when DV is not a multiple of BD
        const uint64_t v_row_offset = (uint64_t)i * args.nb21;
        for (int j = 0; j < v_per_thread; j++) {
            const int v_idx = simd_lid * v_per_thread + j;
            if (v_idx >= DV) break;
            float v_val;

            // Type dispatch: load V element based on vblock_t
            if (is_same<vblock_t, block_q8_0>::value) {
                v_val = load_element_q8_0<vblock_t>(v_head, v_row_offset, v_idx);
            } else if (is_same<vblock_t, block_q4_0>::value) {
                v_val = load_element_q4_0<vblock_t>(v_head, v_row_offset, v_idx);
            } else {
                // F16 path
                v_val = load_element_f16<vblock_t>(v_head, v_row_offset, v_idx);
            }

            o[j] = o[j] * factor + exp_score * v_val;
        }
    }

    // --- Cross-simdgroup reduction ---
    // Shared memory layout: [BN max] [BN sum] [BN*BD output scratch]
    threadgroup float * sg_max = shmem;
    threadgroup float * sg_sum = shmem + BN;
    threadgroup float * sg_out = shmem + 2 * BN;

    if (simd_lid == 0) {
        sg_max[simd_gid] = max_score;
        sg_sum[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global max across all simdgroups
    float global_max = sg_max[0];
    for (int g = 1; g < BN; g++) {
        global_max = max(global_max, sg_max[g]);
    }

    // Global sum_exp, rescaled to global max
    float global_sum_exp = 0;
    for (int g = 0; g < BN; g++) {
        global_sum_exp += sg_sum[g] * fast::exp(sg_max[g] - global_max);
    }

    // Rescale this simdgroup's partial output to global max
    float rescale = fast::exp(max_score - global_max);
    for (int j = 0; j < v_per_thread; j++) {
        o[j] *= rescale;
    }

    // Sum output across simdgroups via shared memory
    // Guard against out-of-bounds when DV is not a multiple of BD
    for (int j = 0; j < v_per_thread; j++) {
        const int v_idx = simd_lid * v_per_thread + j;
        sg_out[simd_gid * BD + simd_lid] = (v_idx < DV) ? o[j] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {
            float total = 0;
            for (int g = 0; g < BN; g++) {
                total += sg_out[g * BD + simd_lid];
            }
            o[j] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Attention sinks: fold pre-computed per-head sink value into softmax state
    const bool has_sinks = args.ns10 > 0;
    if (has_sinks && simd_gid == 0) {
        const float s = simd_lid == 0 ? ((device const float *) sinks)[iq2] : -__FLT_MAX__ / 2.0f;
        const float sink_val = simd_max(max(global_max, s));

        const float ms = fast::exp(global_max - sink_val);
        const float vs = fast::exp(s - sink_val);

        global_sum_exp = global_sum_exp * ms + simd_sum(vs);

        for (int j = 0; j < v_per_thread; j++) {
            o[j] *= ms;
        }
    }

    // Final normalization
    if (simd_gid == 0) {
        for (int j = 0; j < v_per_thread; j++) {
            o[j] = (global_sum_exp > 0) ? (o[j] / global_sum_exp) : 0;
        }
    }

    // Write output (only simdgroup 0)
    // dst layout: [DV, ne1(heads), ne2(N), ne3(batch)] contiguous f32
    // Guard against out-of-bounds when DV is not a multiple of BD
    if (simd_gid == 0) {
        device float * dst_ptr = (device float *)dst;
        const int dst_base = iq3 * args.ne2 * args.ne1 * DV
                           + iq1 * args.ne1 * DV
                           + iq2 * DV;
        for (int j = 0; j < v_per_thread; j++) {
            const int v_idx = simd_lid * v_per_thread + j;
            if (v_idx < DV) {
                dst_ptr[dst_base + v_idx] = o[j];
            }
        }
    }
}

// Kernel wrappers for all head sizes
// F16 K/V variants (use half2 as placeholder type for F16 path)

[[host_name("kernel_flash_attn_ext_scalar_dk32_dv32")]]
kernel void kernel_flash_attn_ext_scalar_dk32_dv32(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<32, 32, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_dk40_dv40")]]
kernel void kernel_flash_attn_ext_scalar_dk40_dv40(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<40, 40, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_dk48_dv48")]]
kernel void kernel_flash_attn_ext_scalar_dk48_dv48(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<48, 48, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_dk64_dv64")]]
kernel void kernel_flash_attn_ext_scalar_dk64_dv64(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<64, 64, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_dk128_dv128")]]
kernel void kernel_flash_attn_ext_scalar_dk128_dv128(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<128, 128, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_dk256_dv256")]]
kernel void kernel_flash_attn_ext_scalar_dk256_dv256(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<256, 256, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

// Q8_0 K/V variants
[[host_name("kernel_flash_attn_ext_scalar_q8_0_dk64_dv64")]]
kernel void kernel_flash_attn_ext_scalar_q8_0_dk64_dv64(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<64, 64, block_q8_0, block_q8_0>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_q8_0_dk128_dv128")]]
kernel void kernel_flash_attn_ext_scalar_q8_0_dk128_dv128(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<128, 128, block_q8_0, block_q8_0>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_q8_0_dk256_dv256")]]
kernel void kernel_flash_attn_ext_scalar_q8_0_dk256_dv256(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<256, 256, block_q8_0, block_q8_0>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

// Q4_0 K/V variants
[[host_name("kernel_flash_attn_ext_scalar_q4_0_dk64_dv64")]]
kernel void kernel_flash_attn_ext_scalar_q4_0_dk64_dv64(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<64, 64, block_q4_0, block_q4_0>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_q4_0_dk128_dv128")]]
kernel void kernel_flash_attn_ext_scalar_q4_0_dk128_dv128(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<128, 128, block_q4_0, block_q4_0>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_q4_0_dk256_dv256")]]
kernel void kernel_flash_attn_ext_scalar_q4_0_dk256_dv256(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<256, 256, block_q4_0, block_q4_0>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

// Additional head sizes for broader model support
[[host_name("kernel_flash_attn_ext_scalar_dk72_dv72")]]
kernel void kernel_flash_attn_ext_scalar_dk72_dv72(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<72, 72, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_dk80_dv80")]]
kernel void kernel_flash_attn_ext_scalar_dk80_dv80(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<80, 80, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_dk96_dv96")]]
kernel void kernel_flash_attn_ext_scalar_dk96_dv96(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<96, 96, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_dk112_dv112")]]
kernel void kernel_flash_attn_ext_scalar_dk112_dv112(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<112, 112, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_dk192_dv192")]]
kernel void kernel_flash_attn_ext_scalar_dk192_dv192(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<192, 192, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}

[[host_name("kernel_flash_attn_ext_scalar_dk576_dv576")]]
kernel void kernel_flash_attn_ext_scalar_dk576_dv576(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    constant uint32_t * tg_offset [[buffer(7)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    ushort3 tptg     [[threads_per_threadgroup]],
    ushort  simd_gid [[simdgroup_index_in_threadgroup]],
    ushort  simd_lid [[thread_index_in_simdgroup]],
    ushort  simd_size [[threads_per_simdgroup]])
{
    flash_attn_ext_scalar_impl<576, 576, half2, half2>(args, q, k, v, mask, dst, sinks, tg_offset, shmem, tgpig, tptg, simd_gid, simd_lid, simd_size);
}
