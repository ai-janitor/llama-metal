// INTENT: distribute IQ blocks across the actual SIMD width.
// PDL: advance each lane by NW blocks, matching its NW-wide reduction.
// =============================================================================
// Scalar matmul kernels for k-quant types (q2_K through q6_K, iq2_xxs through iq4_xs).
// =============================================================================
//
// These kernels compute matrix-vector products for GGML's "k-quant" formats.
// k-quants use multi-level scale packing: each superblock (QK_K=256 elements)
// has a superblock-level scale (d/dmin) plus per-group sub-scales packed in
// a compact format that requires bitmask extraction.
//
// All kernels follow the same pattern:
//   1. Thread-to-data mapping: NW threads are split into groups (THREADS_PER_BLOCK),
//      each group collaborates on one superblock.
//   2. Activation reuse: activation elements are loaded once per superblock and
//      reused across nr0 weight rows (controlled by N_R0_* constants).
//   3. Cross-lane reduction: partial sums from all threads are combined via
//      simd_sum (NW=32) or threadgroup-memory reduction (shmem_reduce for NW!=32).
//
// Key variables used across all kernels:
//   NSG          — simdgroups per threadgroup (function constant FC_mul_mv_nsg)
//   NW           — SIMD width (function constant N_SIMDWIDTH, 32 or 64)
//   eff_tiisg    — effective lane index (remapped when using shmem_reduce)
//   eff_sgitg    — effective simdgroup index
//   first_row    — first output row assigned to this simdgroup
//   nr0          — how many consecutive rows each simdgroup processes
//   nb           — number of superblocks per row (ne00 / QK_K)
//
#include "dequant.metal"
#include "block-dot.metal"

// =============================================================================
// q2_K: 2-bit quantization with per-group scales and min values.
// =============================================================================
// block_q2_K layout (84 bytes per superblock of 256 elements):
//   - scales: uint8[16]  — packed 4-bit scales (low nibble) and mins (high nibble)
//   - qs:     uint8[64]  — 2-bit quants packed (4 per byte)
//   - dm:     half2       — superblock scale (d) and min scale (dmin)
//
template<int nr0, typename args_t, short ARITHMETIC_LANE = block_dot_lane_fp32>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;    // simdgroups per threadgroup
    const short NW = N_SIMDWIDTH;       // native SIMD width (32 or 64)
    // When using shmem_reduce (NW != 32), remap thread indices from flat tidx
    // to ensure consistent NW-wide reduction. When using simd_sum, use hardware indices.
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;  // superblocks per row

    // Grid position: slice-invariant coordinates (constant across persistent loop iterations)
    const int r1 = tgpig.y + args.tg_y_offset;  // src1 row index (0 for matvec)
    const int im = tgpig.z;                       // combined 3rd+4th dim (attention heads × batch)

    // Broadcast indices for grouped-query attention (GQA).
    const uint i12 = im%args.ne12;  // head index in src1
    const uint i13 = im/args.ne12;  // batch index

    // Activation offset (constant — doesn't depend on which weight row slice)
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    // -------------------------------------------------------------------------
    // Thread-to-data mapping for q2_K (8 threads per superblock of QK_K=256).
    // -------------------------------------------------------------------------
    // q2_K stores 2-bit weights: 4 values per byte, 256 values per superblock.
    // With 8 threads per superblock, each thread processes 32 elements.
    //
    //   ix (0..NW/8-1) — which superblock this thread starts on (stride across blocks)
    //   it (0..7)       — thread position within the 8-thread block
    const short ix = eff_tiisg/8;   // superblock index (strides by NW/8 concurrent blocks)
    const block_dot_q2_k_fragment owned = block_dot_q2_k_legacy::owned_fragment(eff_tiisg%8);

    // Persistent threadgroup loop setup.
    // total_slices: how many (NSG * nr0)-row chunks the output matrix has.
    // grid_stride: when n_tg_x > 0 (persistent mode), each TG loops over slices
    //   at stride n_tg_x. When n_tg_x == 0 (normal mode), grid_stride == total_slices
    //   so the loop runs exactly once — identical to non-persistent behavior.
    const int rows_per_slice = NSG * nr0;
    const int total_slices = (args.ne01 + rows_per_slice - 1) / rows_per_slice;
    const int grid_stride = args.n_tg_x > 0 ? args.n_tg_x : total_slices;

    // -------------------------------------------------------------------------
    // CU-local row clustering: TESTED 2026-03-28, no improvement (-0.9%).
    // -------------------------------------------------------------------------
    // Hypothesis: remap TG→slice so TGs on the same CU access contiguous rows,
    // reducing DRAM page misses. Tested with round-robin CU model (TG i → CU i%32).
    // Result: neutral. GCN likely uses fill-first scheduling (not round-robin),
    // so consecutive TGs already go to the same CU. Remapping scattered what was
    // already well-ordered. Also tested chunked dispatch (64 TGs/chunk) — -31.5%
    // regression, confirming occupancy/latency hiding is critical. The bandwidth
    // gap is NOT fixable by row remapping alone.
    //
    // Infrastructure kept: compute_units and grid_x fields in args, remapping code
    // below (disabled via the false && guard). May help on GPUs with different
    // scheduling models or fewer CUs.
    const int base_tg = (int)tgpig.x + args.tg_x_offset;
    int effective_tg = base_tg;
    if (false && args.compute_units > 0 && args.grid_x > args.compute_units) {
        const int n_cus = args.compute_units;
        const int tgs_per_cu = (args.grid_x + n_cus - 1) / n_cus;
        const int cu_id     = base_tg % n_cus;
        const int local_idx = base_tg / n_cus;
        const int remapped  = cu_id * tgs_per_cu + local_idx;
        effective_tg = (remapped < args.grid_x) ? remapped : base_tg;
    }

    // -------------------------------------------------------------------------
    // Persistent threadgroup loop: each TG processes multiple row slices.
    // -------------------------------------------------------------------------
    // Instead of one TG per slice (many TGs, scattered DRAM reads), we launch
    // fewer TGs (ideally = CU count) and each loops over slices at stride grid_x.
    // Consecutive slices processed by the same CU keep weight data in L1/L2.
    for (int slice = effective_tg; slice < total_slices; slice += grid_stride) {

    // first_row: first output row this simdgroup is responsible for in THIS slice.
    // Each threadgroup covers NSG simdgroups × nr0 rows per slice.
    const int first_row = (slice * NSG + eff_sgitg) * nr0;

    // Weight offset (changes per slice — different weight rows each iteration)
    // i12/r2 and i13/r3 handle GQA broadcast (multiple src1 heads share one src0 head).
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float sumf[nr0]={0.f};  // partial dot-product accumulators, one per output row (reset each slice)

    // Main loop: iterate over superblocks, striding by NW/8 concurrent blocks.
    // Each thread processes one superblock per iteration, with NW/8 blocks in flight.
    for (int ib = ix; ib < nb; ib += 4) {
        const block_dot_q2_k_activation prepared = block_dot_q2_k_legacy::prepare_activation<ARITHMETIC_LANE>(owned, y + ib*QK_K);

        // Inner row loop: process nr0 consecutive weight rows using the SAME
        // prepared activation data. This is the activation reuse optimization.
        for (short row = 0; row < nr0; row++) {
            // Advance to next weight row (nb01 bytes between rows)
            // qs += args.nb01/2;   // uint16_t pointer: nb01 bytes / 2
            // sc += args.nb01;     // uint8_t pointer: nb01 bytes
            // dh += args.nb01/2;   // half pointer: nb01 bytes / 2
            // The typed row-byte address replaces those historical pointer advances.
            device const char * row_bytes = (device const char *)(x + ib) + row*args.nb01;
            block_dot_q2_k_legacy::accumulate<ARITHMETIC_LANE>((device const block_q2_K *)row_bytes, owned, prepared, sumf[row]);
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }

    } // end persistent threadgroup loop (slice loop)
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// P11.63: literal packed-fp16 V3 entry (q2_K); the host does not route to it. Faithful default above stays fp32.
[[host_name("kernel_mul_mv_q2_K_f32_v3")]]
kernel void kernel_mul_mv_q2_K_f32_v3(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &, block_dot_lane_packed_fp16>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t, short ARITHMETIC_LANE = block_dot_lane_fp32>
void kernel_mul_mv_q3_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    // Grid position: slice-invariant coordinates (constant across persistent loop iterations)
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    // -------------------------------------------------------------------------
    // Work distribution for q3_K: 8 threads per superblock of QK_K=256 elements.
    // -------------------------------------------------------------------------
    constexpr short THREADS_PER_BLOCK = block_dot_q3_k_legacy::threads_per_block;
    const short CONCURRENT_BLOCKS = NW / THREADS_PER_BLOCK;
    const short block_id      = eff_tiisg / THREADS_PER_BLOCK;
    const short lane_in_block = eff_tiisg % THREADS_PER_BLOCK;

    // Persistent threadgroup loop setup.
    // When n_tg_x == 0 (normal mode), grid_stride == total_slices → loop runs once.
    const int rows_per_slice = NSG * nr0;
    const int total_slices = (args.ne01 + rows_per_slice - 1) / rows_per_slice;
    const int grid_stride = args.n_tg_x > 0 ? args.n_tg_x : total_slices;

    // -------------------------------------------------------------------------
    // Persistent threadgroup loop: each TG processes multiple row slices.
    // -------------------------------------------------------------------------
    for (int slice = (int)tgpig.x + args.tg_x_offset; slice < total_slices; slice += grid_stride) {

    const int first_row = (slice * NSG + eff_sgitg) * nr0;
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = {0.f};  // partial dot-product accumulators, one per output row (reset each slice)

    const block_dot_q3_k_fragment owned = block_dot_q3_k_legacy::owned_fragment(lane_in_block);
    // Two persistent accumulators span every block for this output-row slice.
    float sumf1[nr0] = {0.f};
    float sumf2[nr0] = {0.f};
    for (int i = block_id; i < nb; i += CONCURRENT_BLOCKS) {
        const block_dot_q3_k_activation prepared = block_dot_q3_k_legacy::prepare_activation<ARITHMETIC_LANE>(owned, yy + i*QK_K);
        for (short row = 0; row < nr0; ++row) {
            // nb01 is a byte stride between rows; cast before applying it.
            device const block_q3_K * block = (device const block_q3_K *)((device const char *)(x + i) + row*args.nb01);
            block_dot_q3_k_legacy::accumulate<ARITHMETIC_LANE>(block, owned, prepared, sumf1[row], sumf2[row]);
        }
    }
    // Normalize after the complete block loop; per-block normalization changes rounding.
    for (int row = 0; row < nr0; ++row) {
        sumf[row] += (sumf1[row] + 0.25f * sumf2[row]) / (1 << owned.quant_shift);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < nr0; ++row) {
            sumf[row] = simd_sum(sumf[row]);
        }
        if (tiisg == 0) {
            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                dst_f32[first_row + row] = sumf[row];
            }
        }
    }

    } // end persistent threadgroup loop (slice loop)
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// P11.63: literal packed-fp16 V3 entry (q3_K); the host does not route to it. Faithful default above stays fp32.
[[host_name("kernel_mul_mv_q3_K_f32_v3")]]
kernel void kernel_mul_mv_q3_K_f32_v3(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &, block_dot_lane_packed_fp16>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

constant bool FC_mul_mv_residual [[function_constant(FC_MUL_MV_RESIDUAL)]];

// Residual-capable core. When FC_mul_mv_residual is true the epilogue stores
// (row result + residual[row]) into dst; the residual address omits the column
// and plane index, so the host only enables it for one-column, one-plane decode.
// Specializations that never set FC606 (e.g. kernel_mul_mv_id_q4_K_f32) take the
// plain-store path via is_function_constant_defined.
// PREP: the block headers come from entries unpacked once at load (prep, 32 bytes per block, block ordinal =
// row * blocks per row + block); false never reads prep.
template<int nr0, typename args_t, bool PREP = false>
void kernel_mul_mv_q4_K_f32_residual_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const float * residual,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    // Work distribution: 16 threads per superblock (QK_K=256 elements).
    // With NW threads in the wavefront, we process NW/16 superblocks concurrently.
    constexpr short THREADS_PER_BLOCK = block_dot_q4_k_swar::threads_per_block;
    const short CONCURRENT_BLOCKS = NW / THREADS_PER_BLOCK;
    const short block_id      = eff_tiisg / THREADS_PER_BLOCK;
    const short lane_in_block = eff_tiisg % THREADS_PER_BLOCK;

    const int nb = args.ne00/QK_K;

    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    // Byte offset into activation (src1) — same for all row slices.
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    // Total number of row-slices along the X dimension.
    // Each slice covers NSG * nr0 rows (e.g., 2*4 = 8 rows).
    const int rows_per_slice = NSG * nr0;
    const int total_slices = (args.ne01 + rows_per_slice - 1) / rows_per_slice;

    // Grid stride for the persistent threadgroup loop.
    // n_tg_x = number of threadgroups launched in X dimension (set by dispatch).
    // When n_tg_x == total_slices, each TG processes one slice (non-persistent, same as before).
    // When n_tg_x < total_slices, each TG loops over multiple slices at stride n_tg_x.
    const int grid_stride = args.n_tg_x > 0 ? args.n_tg_x : total_slices;

    // -------------------------------------------------------------------------
    // Persistent threadgroup loop: each TG processes multiple row slices.
    // -------------------------------------------------------------------------
    // Instead of one TG per slice (many TGs, scattered DRAM reads), we launch
    // fewer TGs (ideally = CU count) and each loops over slices at stride grid_x.
    // Consecutive slices processed by the same CU keep weight data in L1/L2.
    for (int slice = (int)tgpig.x + args.tg_x_offset; slice < total_slices; slice += grid_stride) {

    const int first_row = (slice * NSG + eff_sgitg) * nr0;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float sumf[nr0]={0.f};

    // Bitmasks for unpacking 6-bit scales from the 12-byte scales array.
    // q4_K stores 8 × 6-bit scale pairs in a non-trivial packed layout:
    //   - kmask1 (0x3f3f): extracts lower 6 bits of each byte (scale value)
    //   - kmask2 (0x0f0f): extracts lower 4 bits of each byte (low part of min scale)
    //   - kmask3 (0xc0c0): extracts upper 2 bits of each byte (high part of min scale)
    // Historical variable names and operations below now live in block-dot-q4-k.metal;
    // traversal and byte-row stride remain here.
    // sc8 declared locally within the row loop to avoid potential
    // thread-private aliasing issues on Intel Metal at NW<32
    // -------------------------------------------------------------------------
    // Thread-to-data mapping within 16-thread block (matches Vulkan mul_mat_vec_q4_k).
    // -------------------------------------------------------------------------
    //   ix (= block_id)  — which superblock this thread group processes
    //   itid (0..15)     — thread position within the 16-thread block
    //   il   (0..3)      — element group index
    //   ir   (0..3)      — position within group
    //   v_im (0 or 1)    — which 128-element half of the 256-element superblock
    //   v_in (0 or 1)    — interleaved position within the half
    //   l0   (0..28, step 4) — starting element offset
    // element group (0..3)
    // position within group (0..3)
    // 0 = lower 128 elements, 1 = upper 128
    // interleave: even/odd groups
    // element offset: 0,4,8,...,28
    // byte offset into qs[] for this thread's quants
    // float offset into activation for this thread's data
    // The exact activation base is 64*scale_pair+chunk, not a 128-element half offset.
    //
    // Each thread reads 4 quants from qs[] and 4 floats from activation,
    // covering 16 elements total (4 sub-groups of 4). The 16 threads together
    // cover all 256 elements of the superblock.
    const block_dot_q4_k_fragment owned = block_dot_q4_k_swar::owned_fragment(lane_in_block);
    // Main loop: iterate over superblocks, striding by CONCURRENT_BLOCKS.
    for (int ib = block_id; ib < nb; ib += CONCURRENT_BLOCKS) {
        // 4 vec4 loads (16 floats per thread, half of old 32)
        const block_dot_q4_k_activation prepared = block_dot_q4_k_swar::prepare_activation<block_dot_lane_fp32>(owned, y + ib*QK_K);
        for (short row = 0; row < nr0; row++) {
            // Inner row loop: process nr0 rows reusing same activation loads.
            // Unpack 6-bit scales from the packed 12-byte scales array.
            // sc16[0..1] = quantization scales (lower 6 bits of each byte)
            // sc16[2..3] = min-value scales (upper 2 bits from sc[0,2] + lower 4 bits from sc[4])
            // After unpacking, sc8[0..7] gives individual 8-bit scale values:
            //   sc8[0,1,4,5] = quantization scales for 4 sub-groups
            //   sc8[2,3,6,7] = min-value scales for 4 sub-groups
            // lower 6 bits of scales[0..1]
            // min scales (low)
            // min scales (high)
            // Declare sc8 locally after sc16 is populated to ensure
            // the compiler always reads the freshly-computed values.
            // Load 4 bytes of packed 4-bit quants and split into low/high nibbles.
            // qs0 = quants at q_offset, qs64 = quants at q_offset+64 (upper half)
            // +64 bytes (upper 128-element half)
            // low nibbles (values 0..15)
            // high nibbles
            // Dot products: activation × dequantized weights for 4 sub-groups.
            // sx/sy handle lower 128 elements, sz/sw handle upper 128.
            // lower half, low nibbles
            // lower half, high nibbles
            // upper half, low nibbles
            // upper half, high nibbles
            // smin: min-value correction term.
            // q4_K dequant: value = d * (quant * scale) - dmin * min_scale * sum(activation)
            // Each activation group is weighted by its min-value scale (sc8[2,3,6,7]).
            // Exact operation: d*quant_scale*quant - dmin*min_scale.
            // Final accumulation: d * weighted_quant_sums - dmin * smin
            // dm.x = d (superblock scale), dm.y = dmin (superblock min scale)
            // Advance to next weight row (nb01 bytes between rows)
            // uint32_t pointer
            // uint16_t pointer
            // half pointer
            // Preserve the original byte row stride before restoring the Q4 block type.
            device const block_q4_K * block = (device const block_q4_K *) ((device const char *) (x + ib) + row*args.nb01);
            if (PREP) {
                device const uint * header = (device const uint *) (prep + (uint64_t) ((first_row + row) * nb + ib) * 32);
                block_dot_q4_k_swar::accumulate_prepared<block_dot_lane_fp32>(block, owned, prepared, sumf[row], header);
            } else {
                block_dot_q4_k_swar::accumulate<block_dot_lane_fp32>(block, owned, prepared, sumf[row]);
            }
        }
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;
    const bool add_residual = is_function_constant_defined(FC_mul_mv_residual) && FC_mul_mv_residual;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // All rows written, one barrier, then lane r sums row r in the original order.
        for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int row = ltid; row < nr0; row += NW) {
            if (first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = add_residual ? tot + residual[first_row + row] : tot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (int row = 0; row < nr0; ++row) {
            sumf[row] = simd_sum(sumf[row]);
        }
        if (tiisg == 0) {
            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                dst_f32[first_row + row] = add_residual ? sumf[row] + residual[first_row + row] : sumf[row];
            }
        }
    }

    } // end persistent threadgroup loop (slice loop)
}

// Original nine-argument impl signature, kept for callers that have no residual
// operand (kernel_mul_mv_id_q4_K_f32 binds it through mmv_fn). The residual
// pointer is never read there: FC606 is undefined or false for those pipelines.
template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    kernel_mul_mv_q4_K_f32_residual_impl<nr0, args_t>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const float * residual [[buffer(4)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    if (FC_mul_mv_nr0 == 1) {
        kernel_mul_mv_q4_K_f32_residual_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, residual, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        kernel_mul_mv_q4_K_f32_residual_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, residual, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 8) {
        kernel_mul_mv_q4_K_f32_residual_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, residual, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q4_K_f32_residual_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, residual, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

// Q4_K matvec of a weight whose block headers were unpacked once at load (hw mv rows with prep): the headers at
// buffer 5; one plane, one column, no residual.
[[host_name("kernel_mul_mv_q4_K_prep_f32")]]
kernel void kernel_mul_mv_q4_K_prep_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep [[buffer(5)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    if (FC_mul_mv_nr0 == 1) {
        kernel_mul_mv_q4_K_f32_residual_impl<1, constant ggml_metal_kargs_mul_mv &, true>(args, src0, src1, dst, (device const float *) dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        kernel_mul_mv_q4_K_f32_residual_impl<4, constant ggml_metal_kargs_mul_mv &, true>(args, src0, src1, dst, (device const float *) dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 8) {
        kernel_mul_mv_q4_K_f32_residual_impl<8, constant ggml_metal_kargs_mul_mv &, true>(args, src0, src1, dst, (device const float *) dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q4_K_f32_residual_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &, true>(args, src0, src1, dst, (device const float *) dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

template<int nr0, typename args_t, short ARITHMETIC_LANE = block_dot_lane_fp32>
void kernel_mul_mv_q5_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    // Grid position: slice-invariant coordinates (constant across persistent loop iterations)
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    // Work distribution: 8 threads per superblock (QK_K=256 elements).
    // q5_K uses fewer threads than q4_K because each thread handles more data.
    constexpr short THREADS_PER_BLOCK = block_dot_q5_k_scalar_swar::threads_per_block;
    const short CONCURRENT_BLOCKS = NW / THREADS_PER_BLOCK;
    const short block_id      = eff_tiisg / THREADS_PER_BLOCK;
    const short lane_in_block = eff_tiisg % THREADS_PER_BLOCK;

    // Persistent threadgroup loop setup.
    // When n_tg_x == 0 (normal mode), grid_stride == total_slices → loop runs once.
    const int rows_per_slice = NSG * nr0;
    const int total_slices = (args.ne01 + rows_per_slice - 1) / rows_per_slice;
    const int grid_stride = args.n_tg_x > 0 ? args.n_tg_x : total_slices;

    // -------------------------------------------------------------------------
    // Persistent threadgroup loop: each TG processes multiple row slices.
    // -------------------------------------------------------------------------
    for (int slice = (int)tgpig.x + args.tg_x_offset; slice < total_slices; slice += grid_stride) {

    const int first_row = (slice * NSG + eff_sgitg) * nr0;
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0]={0.f};  // reset each slice

    // "fragment" is an MSL stage keyword, so the owned fragment uses another local name.
    const block_dot_q5_k_fragment owned = block_dot_q5_k_scalar_swar::owned_fragment(lane_in_block);
    for (int i = block_id; i < nb; i += CONCURRENT_BLOCKS) {
        const block_dot_q5_k_activation prepared = block_dot_q5_k_scalar_swar::prepare_activation<ARITHMETIC_LANE>(owned, yy + i * QK_K);
        // Inner row loop: process nr0 rows reusing same activation data.
        for (short row = 0; row < nr0; ++row) {
            device const block_q5_K * block = (device const block_q5_K *) ((device const char *) (x + i) + row * args.nb01);
            block_dot_q5_k_scalar_swar::accumulate<ARITHMETIC_LANE>(block, owned, prepared, sumf[row]);
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // All rows written, one barrier, then lane r sums row r in the original order.
        for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int row = ltid; row < nr0; row += NW) {
            if (first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (int row = 0; row < nr0; ++row) {
            const float tot = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = tot;
            }
        }
    }

    } // end persistent threadgroup loop (slice loop)
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    if (FC_mul_mv_nr0 == 1) {
        kernel_mul_mv_q5_K_f32_impl<1, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        kernel_mul_mv_q5_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 8) {
        kernel_mul_mv_q5_K_f32_impl<8, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

// The host does not route to this literal V3 entry yet; keeping it separate
// preserves the faithful default while forcing compilation of the Q5_K true-half path.
[[host_name("kernel_mul_mv_q5_K_f32_v3")]]
kernel void kernel_mul_mv_q5_K_f32_v3(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &, block_dot_lane_packed_fp16>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t, short ARITHMETIC_LANE = block_dot_lane_fp32>
void kernel_mul_mv_q6_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    // Grid position: slice-invariant coordinates (constant across persistent loop iterations)
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    // Native NW-wide work distribution: 16 threads collaborate on each superblock.
    constexpr short THREADS_PER_BLOCK = block_dot_q6_k_scalar_swar::threads_per_block;
    const short CONCURRENT_BLOCKS = NW / THREADS_PER_BLOCK;
    const short block_id      = eff_tiisg / THREADS_PER_BLOCK;
    const short lane_in_block = eff_tiisg % THREADS_PER_BLOCK;

    // Persistent threadgroup loop setup.
    // When n_tg_x == 0 (normal mode), grid_stride == total_slices → loop runs once.
    const int rows_per_slice = NSG * nr0;
    const int total_slices = (args.ne01 + rows_per_slice - 1) / rows_per_slice;
    const int grid_stride = args.n_tg_x > 0 ? args.n_tg_x : total_slices;

    // -------------------------------------------------------------------------
    // Persistent threadgroup loop: each TG processes multiple row slices.
    // -------------------------------------------------------------------------
    for (int slice = (int)tgpig.x + args.tg_x_offset; slice < total_slices; slice += grid_stride) {

    const int first_row = (slice * NSG + eff_sgitg) * nr0;
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = { 0.f };  // reset each slice

    // "fragment" is an MSL stage keyword, so the owned fragment uses another local name.
    const block_dot_q6_k_fragment owned = block_dot_q6_k_scalar_swar::owned_fragment(lane_in_block);
    for (int i = block_id; i < nb; i += CONCURRENT_BLOCKS) {
        const block_dot_q6_k_activation prepared = block_dot_q6_k_scalar_swar::prepare_activation<ARITHMETIC_LANE>(owned, yy + i * QK_K);
        // Inner row loop: process nr0 rows reusing same activation data.
        for (short row = 0; row < nr0; ++row) {
            device const block_q6_K * block = (device const block_q6_K *) ((device const char *) (x + i) + row * args.nb01);
            block_dot_q6_k_scalar_swar::accumulate<ARITHMETIC_LANE>(block, owned, prepared, sumf[row]);
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // All rows written, one barrier, then lane r sums row r in the original order.
        for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int row = ltid; row < nr0; row += NW) {
            if (first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        // simd_sum path — works at NW=8 and NW=32. Broken at NW=16 on Intel.
        for (int row = 0; row < nr0; ++row) {
            sumf[row] = simd_sum(sumf[row]);
        }
        if (tiisg == 0) {
            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                dst_f32[first_row + row] = sumf[row];
            }
        }
    }

    } // end persistent threadgroup loop (slice loop)
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    if (FC_mul_mv_nr0 == 1) {
        kernel_mul_mv_q6_K_f32_impl<1, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        kernel_mul_mv_q6_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 8) {
        kernel_mul_mv_q6_K_f32_impl<8, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &, block_dot_lane_fp32>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

// P11.63: literal packed-fp16 V3 entry (q6_K); the host does not route to it. Faithful default above stays fp32.
[[host_name("kernel_mul_mv_q6_K_f32_v3")]]
kernel void kernel_mul_mv_q6_K_f32_v3(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &, block_dot_lane_packed_fp16>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// ======================= "True" 2-bit


// Columns per launch of the *_cols entry points (2, 3 or 4) and their loop form (bit 0 = weights held, bit 1 = packed,
// bit 2 = paired halves at 2 or 3 columns).
constant short FC_mul_mv_ncols_raw [[function_constant(FC_MUL_MV + 12)]];
constant short FC_mul_mv_ncols = is_function_constant_defined(FC_mul_mv_ncols_raw) ? FC_mul_mv_ncols_raw : 2;
constant short FC_mul_mv_col_form_raw [[function_constant(FC_MUL_MV + 13)]];
constant short FC_mul_mv_col_form = is_function_constant_defined(FC_mul_mv_col_form_raw) ? FC_mul_mv_col_form_raw : 0;

// Q4_K matvec over NC activation columns in one launch: a lane loads each block's weights once for all columns.
// HOLD_W reloads the activations per row instead of keeping them; PACK keeps the weights packed and unpacks per column.
template<int nr0, int NC, typename args_t, bool PREP, bool HOLD_W, bool PACK>
void kernel_mul_mv_q4_K_f32_cols_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    constexpr short THREADS_PER_BLOCK = block_dot_q4_k_swar::threads_per_block;
    const short CONCURRENT_BLOCKS = NW / THREADS_PER_BLOCK;
    const short block_id      = eff_tiisg / THREADS_PER_BLOCK;
    const short lane_in_block = eff_tiisg % THREADS_PER_BLOCK;

    const int nb = args.ne00/QK_K;
    const int r1 = (tgpig.y + args.tg_y_offset) * NC;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    const int first_row = (((int) tgpig.x + args.tg_x_offset) * NSG + eff_sgitg) * nr0;
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);

    float sumf[nr0*NC] = {0.f};

    const block_dot_q4_k_fragment owned = block_dot_q4_k_swar::owned_fragment(lane_in_block);
    for (int ib = block_id; ib < nb; ib += CONCURRENT_BLOCKS) {
        block_dot_q4_k_activation held[NC];
        if (!HOLD_W) {
            for (short c = 0; c < NC; ++c) {
                held[c] = block_dot_q4_k_swar::prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*QK_K);
            }
        }
        for (short row = 0; row < nr0; ++row) {
            device const block_q4_K * block = (device const block_q4_K *) ((device const char *) (x + ib) + row*args.nb01);
            device const uint * header = (device const uint *) (prep + (uint64_t) ((first_row + row) * nb + ib) * 32);
            if (PACK) {
                block_dot_q4_k_swar::raw w;
                if (PREP) {
                    w = block_dot_q4_k_swar::load_raw_prepared(block, owned, header);
                } else {
                    w = block_dot_q4_k_swar::load_raw(block, owned);
                }
                for (short c = 0; c < NC; ++c) {
                    if (HOLD_W) {
                        const block_dot_q4_k_activation act = block_dot_q4_k_swar::prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*QK_K);
                        block_dot_q4_k_swar::accumulate_raw(w, act, sumf[row*NC + c]);
                    } else {
                        block_dot_q4_k_swar::accumulate_raw(w, held[c], sumf[row*NC + c]);
                    }
                }
            } else {
                block_dot_q4_k_swar::weights w;
                if (PREP) {
                    w = block_dot_q4_k_swar::load_weights_prepared(block, owned, header);
                } else {
                    w = block_dot_q4_k_swar::load_weights(block, owned);
                }
                for (short c = 0; c < NC; ++c) {
                    if (HOLD_W) {
                        const block_dot_q4_k_activation act = block_dot_q4_k_swar::prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*QK_K);
                        block_dot_q4_k_swar::accumulate_weights(w, act, sumf[row*NC + c]);
                    } else {
                        block_dot_q4_k_swar::accumulate_weights(w, held[c], sumf[row*NC + c]);
                    }
                }
            }
        }
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0 * NC;
        for (int k = 0; k < nr0*NC; ++k) {
            buf[NW * k + ltid] = sumf[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Lane k sums accumulator k (row k / NC, column k % NC) in lane order, as the one-column kernel does.
        for (int k = ltid; k < nr0*NC; k += NW) {
            const int row = k / NC;
            const int c = k % NC;
            if (first_row + row < args.ne0 && r1 + c < args.ne11) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * k + i]; }
                dst_f32[(int64_t) c*args.ne0 + first_row + row] = tot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (int k = 0; k < nr0*NC; ++k) {
            sumf[k] = simd_sum(sumf[k]);
        }
        if (tiisg == 0) {
            for (int k = 0; k < nr0*NC; ++k) {
                if (first_row + k / NC < args.ne0 && r1 + k % NC < args.ne11) {
                    dst_f32[(int64_t) (k % NC)*args.ne0 + first_row + k / NC] = sumf[k];
                }
            }
        }
    }
}

template<int NC, typename args_t, bool PREP, bool HOLD_W, bool PACK>
void kernel_mul_mv_q4_K_f32_cols_rows(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if (FC_mul_mv_nr0 == 1) {
        kernel_mul_mv_q4_K_f32_cols_impl<1, NC, args_t, PREP, HOLD_W, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 2) {
        kernel_mul_mv_q4_K_f32_cols_impl<2, NC, args_t, PREP, HOLD_W, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        kernel_mul_mv_q4_K_f32_cols_impl<4, NC, args_t, PREP, HOLD_W, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q4_K_f32_cols_impl<8, NC, args_t, PREP, HOLD_W, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

template<int NC, typename args_t, bool PREP>
void kernel_mul_mv_q4_K_f32_cols_form(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if (FC_mul_mv_col_form == 0) {
        kernel_mul_mv_q4_K_f32_cols_rows<NC, args_t, PREP, false, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_col_form == 1) {
        kernel_mul_mv_q4_K_f32_cols_rows<NC, args_t, PREP, true, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_col_form == 2) {
        kernel_mul_mv_q4_K_f32_cols_rows<NC, args_t, PREP, false, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q4_K_f32_cols_rows<NC, args_t, PREP, true, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

// One pass of a half wave: block ib of RH rows for NC columns, activations loaded once per pass.
template<int RH, int NC, typename args_t, bool PREP, bool PACK>
static inline void kernel_mul_mv_q4_K_f32_half_pass(
        args_t args,
        device const block_q4_K * x,
        device const char * src1,
        device const char * prep,
        uint64_t offset1,
        int first_row,
        int nb,
        thread const block_dot_q4_k_fragment & owned,
        int ib,
        short h,
        thread float * s) {
    block_dot_q4_k_activation held[NC];
    for (short c = 0; c < NC; ++c) {
        held[c] = block_dot_q4_k_swar::prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*QK_K);
    }
    for (short rr = 0; rr < RH; ++rr) {
        const int row = RH*h + rr;
        device const block_q4_K * block = (device const block_q4_K *) ((device const char *) (x + ib) + row*args.nb01);
        if (PACK) {
            block_dot_q4_k_swar::raw w;
            if (PREP) {
                w = block_dot_q4_k_swar::load_raw_prepared(block, owned, (device const uint *) (prep + (uint64_t) ((first_row + row) * nb + ib) * 32));
            } else {
                w = block_dot_q4_k_swar::load_raw(block, owned);
            }
            for (short c = 0; c < NC; ++c) {
                block_dot_q4_k_swar::accumulate_raw(w, held[c], s[rr*NC + c]);
            }
        } else {
            block_dot_q4_k_swar::weights w;
            if (PREP) {
                w = block_dot_q4_k_swar::load_weights_prepared(block, owned, (device const uint *) (prep + (uint64_t) ((first_row + row) * nb + ib) * 32));
            } else {
                w = block_dot_q4_k_swar::load_weights(block, owned);
            }
            for (short c = 0; c < NC; ++c) {
                block_dot_q4_k_swar::accumulate_weights(w, held[c], s[rr*NC + c]);
            }
        }
    }
}

// Q4_K matvec with paired halves (64-lane waves, shared-memory reduce): each half of a wave works RH rows and
// two blocks per pass; a lane's two banks land in the one-column kernel's lane slots, so the row sum order is unchanged.
template<int RH, int NC, typename args_t, bool PREP, bool PACK>
void kernel_mul_mv_q4_K_f32_half_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    constexpr int nr0 = 2*RH;
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort ltid = tidx % NW;
    const ushort eff_sgitg = tidx / NW;
    const short h = ltid / 32;
    const short t = ltid % 32;
    const short b = t / 16;

    const int nb = args.ne00/QK_K;
    const int r1 = (tgpig.y + args.tg_y_offset) * NC;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    const int first_row = (((int) tgpig.x + args.tg_x_offset) * NSG + eff_sgitg) * nr0;
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);

    const block_dot_q4_k_fragment owned = block_dot_q4_k_swar::owned_fragment(t % 16);
    float bank0[RH*NC] = {0.f};
    float bank1[RH*NC] = {0.f};
    for (int ib = b; ib < nb; ib += 4) {
        kernel_mul_mv_q4_K_f32_half_pass<RH, NC, args_t, PREP, PACK>(args, x, src1, prep, offset1, first_row, nb, owned, ib, h, bank0);
        if (ib + 2 < nb) {
            kernel_mul_mv_q4_K_f32_half_pass<RH, NC, args_t, PREP, PACK>(args, x, src1, prep, offset1, first_row, nb, owned, ib + 2, h, bank1);
        }
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0 * NC;
    for (short rr = 0; rr < RH; ++rr) {
        for (short c = 0; c < NC; ++c) {
            const int k = (RH*h + rr)*NC + c;
            buf[NW * k + t] = bank0[rr*NC + c];
            buf[NW * k + t + 32] = bank1[rr*NC + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = ltid; k < nr0*NC; k += NW) {
        const int row = k / NC;
        const int c = k % NC;
        if (first_row + row < args.ne0 && r1 + c < args.ne11) {
            float tot = 0.0f;
            for (short i = 0; i < NW; i++) { tot += buf[NW * k + i]; }
            dst_f32[(int64_t) c*args.ne0 + first_row + row] = tot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template<int NC, typename args_t, bool PREP, bool PACK>
void kernel_mul_mv_q4_K_f32_half_rows(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if (FC_mul_mv_nr0 == 2) {
        kernel_mul_mv_q4_K_f32_half_impl<1, NC, args_t, PREP, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        kernel_mul_mv_q4_K_f32_half_impl<2, NC, args_t, PREP, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q4_K_f32_half_impl<4, NC, args_t, PREP, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

template<typename args_t, bool PREP>
void kernel_mul_mv_q4_K_f32_half_cols(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const bool pack = (FC_mul_mv_col_form & 2) != 0;
    if (FC_mul_mv_ncols == 2) {
        if (pack) { kernel_mul_mv_q4_K_f32_half_rows<2, args_t, PREP, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); } else { kernel_mul_mv_q4_K_f32_half_rows<2, args_t, PREP, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); }
    } else {
        if (pack) { kernel_mul_mv_q4_K_f32_half_rows<3, args_t, PREP, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); } else { kernel_mul_mv_q4_K_f32_half_rows<3, args_t, PREP, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); }
    }
}

[[host_name("kernel_mul_mv_q4_K_cols_f32")]]
kernel void kernel_mul_mv_q4_K_cols_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    if ((FC_mul_mv_col_form & 4) != 0) {
        kernel_mul_mv_q4_K_f32_half_cols<constant ggml_metal_kargs_mul_mv &, false>(args, src0, src1, dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
        return;
    }
    if (FC_mul_mv_ncols == 2) {
        kernel_mul_mv_q4_K_f32_cols_form<2, constant ggml_metal_kargs_mul_mv &, false>(args, src0, src1, dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_ncols == 3) {
        kernel_mul_mv_q4_K_f32_cols_form<3, constant ggml_metal_kargs_mul_mv &, false>(args, src0, src1, dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q4_K_f32_cols_form<4, constant ggml_metal_kargs_mul_mv &, false>(args, src0, src1, dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

[[host_name("kernel_mul_mv_q4_K_prep_cols_f32")]]
kernel void kernel_mul_mv_q4_K_prep_cols_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep [[buffer(5)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    if ((FC_mul_mv_col_form & 4) != 0) {
        kernel_mul_mv_q4_K_f32_half_cols<constant ggml_metal_kargs_mul_mv &, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
        return;
    }
    if (FC_mul_mv_ncols == 2) {
        kernel_mul_mv_q4_K_f32_cols_form<2, constant ggml_metal_kargs_mul_mv &, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_ncols == 3) {
        kernel_mul_mv_q4_K_f32_cols_form<3, constant ggml_metal_kargs_mul_mv &, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q4_K_f32_cols_form<4, constant ggml_metal_kargs_mul_mv &, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

// Q6_K matvec over NC activation columns in one launch: a lane loads each block's weights once for all columns.
// HOLD_W reloads the activations per row instead of keeping them; PACK keeps the weights packed and unpacks per column.
template<int nr0, int NC, typename args_t, bool HOLD_W, bool PACK>
void kernel_mul_mv_q6_K_f32_cols_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    constexpr short THREADS_PER_BLOCK = block_dot_q6_k_scalar_swar::threads_per_block;
    const short CONCURRENT_BLOCKS = NW / THREADS_PER_BLOCK;
    const short block_id      = eff_tiisg / THREADS_PER_BLOCK;
    const short lane_in_block = eff_tiisg % THREADS_PER_BLOCK;

    const int nb = args.ne00/QK_K;
    const int r1 = (tgpig.y + args.tg_y_offset) * NC;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    const int first_row = (((int) tgpig.x + args.tg_x_offset) * NSG + eff_sgitg) * nr0;
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);

    float sumf[nr0*NC] = {0.f};

    const block_dot_q6_k_fragment owned = block_dot_q6_k_scalar_swar::owned_fragment(lane_in_block);
    for (int ib = block_id; ib < nb; ib += CONCURRENT_BLOCKS) {
        block_dot_q6_k_activation held[NC];
        if (!HOLD_W) {
            for (short c = 0; c < NC; ++c) {
                held[c] = block_dot_q6_k_scalar_swar::prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*QK_K);
            }
        }
        for (short row = 0; row < nr0; ++row) {
            device const block_q6_K * block = (device const block_q6_K *) ((device const char *) (x + ib) + row*args.nb01);
            if (PACK) {
                const block_dot_q6_k_scalar_swar::raw w = block_dot_q6_k_scalar_swar::load_raw(block, owned);
                for (short c = 0; c < NC; ++c) {
                    if (HOLD_W) {
                        const block_dot_q6_k_activation act = block_dot_q6_k_scalar_swar::prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*QK_K);
                        block_dot_q6_k_scalar_swar::accumulate_raw(w, act, sumf[row*NC + c]);
                    } else {
                        block_dot_q6_k_scalar_swar::accumulate_raw(w, held[c], sumf[row*NC + c]);
                    }
                }
            } else {
                const block_dot_q6_k_scalar_swar::weights w = block_dot_q6_k_scalar_swar::load_weights(block, owned);
                for (short c = 0; c < NC; ++c) {
                    if (HOLD_W) {
                        const block_dot_q6_k_activation act = block_dot_q6_k_scalar_swar::prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*QK_K);
                        block_dot_q6_k_scalar_swar::accumulate_weights(w, act, sumf[row*NC + c]);
                    } else {
                        block_dot_q6_k_scalar_swar::accumulate_weights(w, held[c], sumf[row*NC + c]);
                    }
                }
            }
        }
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0 * NC;
        for (int k = 0; k < nr0*NC; ++k) {
            buf[NW * k + ltid] = sumf[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Lane k sums accumulator k (row k / NC, column k % NC) in lane order, as the one-column kernel does.
        for (int k = ltid; k < nr0*NC; k += NW) {
            const int row = k / NC;
            const int c = k % NC;
            if (first_row + row < args.ne0 && r1 + c < args.ne11) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * k + i]; }
                dst_f32[(int64_t) c*args.ne0 + first_row + row] = tot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (int k = 0; k < nr0*NC; ++k) {
            sumf[k] = simd_sum(sumf[k]);
        }
        if (tiisg == 0) {
            for (int k = 0; k < nr0*NC; ++k) {
                if (first_row + k / NC < args.ne0 && r1 + k % NC < args.ne11) {
                    dst_f32[(int64_t) (k % NC)*args.ne0 + first_row + k / NC] = sumf[k];
                }
            }
        }
    }
}

template<int NC, typename args_t, bool HOLD_W, bool PACK>
void kernel_mul_mv_q6_K_f32_cols_rows(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if (FC_mul_mv_nr0 == 1) {
        kernel_mul_mv_q6_K_f32_cols_impl<1, NC, args_t, HOLD_W, PACK>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 2) {
        kernel_mul_mv_q6_K_f32_cols_impl<2, NC, args_t, HOLD_W, PACK>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        kernel_mul_mv_q6_K_f32_cols_impl<4, NC, args_t, HOLD_W, PACK>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q6_K_f32_cols_impl<8, NC, args_t, HOLD_W, PACK>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

template<int NC, typename args_t>
void kernel_mul_mv_q6_K_f32_cols_form(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if (FC_mul_mv_col_form == 0) {
        kernel_mul_mv_q6_K_f32_cols_rows<NC, args_t, false, false>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_col_form == 1) {
        kernel_mul_mv_q6_K_f32_cols_rows<NC, args_t, true, false>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_col_form == 2) {
        kernel_mul_mv_q6_K_f32_cols_rows<NC, args_t, false, true>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q6_K_f32_cols_rows<NC, args_t, true, true>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

// One pass of a half wave: block ib of RH rows for NC columns, activations loaded once per pass.
template<int RH, int NC, typename args_t, bool PACK>
static inline void kernel_mul_mv_q6_K_f32_half_pass(
        args_t args,
        device const block_q6_K * x,
        device const char * src1,
        device const char * prep,
        uint64_t offset1,
        int first_row,
        int nb,
        thread const block_dot_q6_k_fragment & owned,
        int ib,
        short h,
        thread float * s) {
    block_dot_q6_k_activation held[NC];
    for (short c = 0; c < NC; ++c) {
        held[c] = block_dot_q6_k_scalar_swar::prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*QK_K);
    }
    for (short rr = 0; rr < RH; ++rr) {
        const int row = RH*h + rr;
        device const block_q6_K * block = (device const block_q6_K *) ((device const char *) (x + ib) + row*args.nb01);
        if (PACK) {
            const block_dot_q6_k_scalar_swar::raw w = block_dot_q6_k_scalar_swar::load_raw(block, owned);
            for (short c = 0; c < NC; ++c) {
                block_dot_q6_k_scalar_swar::accumulate_raw(w, held[c], s[rr*NC + c]);
            }
        } else {
            const block_dot_q6_k_scalar_swar::weights w = block_dot_q6_k_scalar_swar::load_weights(block, owned);
            for (short c = 0; c < NC; ++c) {
                block_dot_q6_k_scalar_swar::accumulate_weights(w, held[c], s[rr*NC + c]);
            }
        }
    }
}

// Q6_K matvec with paired halves (64-lane waves, shared-memory reduce): each half of a wave works RH rows and
// two blocks per pass; a lane's two banks land in the one-column kernel's lane slots, so the row sum order is unchanged.
template<int RH, int NC, typename args_t, bool PACK>
void kernel_mul_mv_q6_K_f32_half_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    constexpr int nr0 = 2*RH;
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort ltid = tidx % NW;
    const ushort eff_sgitg = tidx / NW;
    const short h = ltid / 32;
    const short t = ltid % 32;
    const short b = t / 16;

    const int nb = args.ne00/QK_K;
    const int r1 = (tgpig.y + args.tg_y_offset) * NC;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    const int first_row = (((int) tgpig.x + args.tg_x_offset) * NSG + eff_sgitg) * nr0;
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);

    const block_dot_q6_k_fragment owned = block_dot_q6_k_scalar_swar::owned_fragment(t % 16);
    float bank0[RH*NC] = {0.f};
    float bank1[RH*NC] = {0.f};
    for (int ib = b; ib < nb; ib += 4) {
        kernel_mul_mv_q6_K_f32_half_pass<RH, NC, args_t, PACK>(args, x, src1, prep, offset1, first_row, nb, owned, ib, h, bank0);
        if (ib + 2 < nb) {
            kernel_mul_mv_q6_K_f32_half_pass<RH, NC, args_t, PACK>(args, x, src1, prep, offset1, first_row, nb, owned, ib + 2, h, bank1);
        }
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0 * NC;
    for (short rr = 0; rr < RH; ++rr) {
        for (short c = 0; c < NC; ++c) {
            const int k = (RH*h + rr)*NC + c;
            buf[NW * k + t] = bank0[rr*NC + c];
            buf[NW * k + t + 32] = bank1[rr*NC + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = ltid; k < nr0*NC; k += NW) {
        const int row = k / NC;
        const int c = k % NC;
        if (first_row + row < args.ne0 && r1 + c < args.ne11) {
            float tot = 0.0f;
            for (short i = 0; i < NW; i++) { tot += buf[NW * k + i]; }
            dst_f32[(int64_t) c*args.ne0 + first_row + row] = tot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template<int NC, typename args_t, bool PACK>
void kernel_mul_mv_q6_K_f32_half_rows(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if (FC_mul_mv_nr0 == 2) {
        kernel_mul_mv_q6_K_f32_half_impl<1, NC, args_t, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        kernel_mul_mv_q6_K_f32_half_impl<2, NC, args_t, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q6_K_f32_half_impl<4, NC, args_t, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

template<typename args_t>
void kernel_mul_mv_q6_K_f32_half_cols(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const bool pack = (FC_mul_mv_col_form & 2) != 0;
    if (FC_mul_mv_ncols == 2) {
        if (pack) { kernel_mul_mv_q6_K_f32_half_rows<2, args_t, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); } else { kernel_mul_mv_q6_K_f32_half_rows<2, args_t, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); }
    } else {
        if (pack) { kernel_mul_mv_q6_K_f32_half_rows<3, args_t, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); } else { kernel_mul_mv_q6_K_f32_half_rows<3, args_t, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); }
    }
}

[[host_name("kernel_mul_mv_q6_K_cols_f32")]]
kernel void kernel_mul_mv_q6_K_cols_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    if ((FC_mul_mv_col_form & 4) != 0) {
        kernel_mul_mv_q6_K_f32_half_cols<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
        return;
    }
    if (FC_mul_mv_ncols == 2) {
        kernel_mul_mv_q6_K_f32_cols_form<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_ncols == 3) {
        kernel_mul_mv_q6_K_f32_cols_form<3, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        kernel_mul_mv_q6_K_f32_cols_form<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    // Load shared-memory lookup tables for iq2_xxs dequantization.
    // The tables need 256 uint64_t entries (svalues) and 128 uint8_t entries (ssigns).
    // On GPUs with NW < 32 (e.g., Intel iGPU with NW=16), the total thread count per
    // threadgroup may differ from the assumed 64 (2 simdgroups * 32 threads). We use
    // tidx (flat thread index within the threadgroup) to ensure all entries are loaded
    // regardless of NW. Total threads = NSG_dispatch * 32 (dispatch always uses 32
    // threads per simdgroup slot), so tidx covers 0..(NSG_dispatch*32-1).
    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        const int total_threads = NW * NSG;
        const int nval = 4;
        for (int pos = (NW*eff_sgitg + eff_tiisg)*nval; pos < 256; pos += total_threads*nval) {
            for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        }
        const int nval2 = 2;
        for (int pos = (NW*eff_sgitg + eff_tiisg)*nval2; pos < 128; pos += total_threads*nval2) {
            for (int i = 0; i < nval2; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = eff_tiisg;

    // Inner loop fix for variable SIMD width (NW).
    //
    // The original loop used a stride of 32, assuming NW=32 (one thread per ib32
    // position per stride). On Intel iGPU with NW=16, only positions 0..15 are
    // visited per stride, leaving positions 16..31 unprocessed. For k=1024,
    // nb32 = 32, so half the data is skipped, causing ~50% error.
    //
    // Fix: stride by NW instead of 32. Each thread handles ib32 positions
    // ix, ix+NW, ix+2*NW, ... covering all nb32 positions. The y pointer is
    // re-derived per ib32 position (y + 32*ib32) instead of using incremental
    // advancement, which was tied to the old stride of 32.
    for (int ib32 = ix; ib32 < nb32; ib32 += NW) {

        // Re-derive y pointer for this ib32 position.
        // Each ib32 corresponds to a 32-float chunk of the input vector.
        device const float * y4 = y + 32 * ib32;

        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot * 0.25f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = sum_all * 0.25f;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {
    kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xs * x = (device const block_iq2_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 512);
    {
        const int total_threads = NW * NSG;
        const int nval = 8;
        for (int pos = (NW*eff_sgitg + eff_tiisg)*nval; pos < 512; pos += total_threads*nval) {
            for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xs_grid[pos + i];
        }
        const int nval2 = 2;
        for (int pos = (NW*eff_sgitg + eff_tiisg)*nval2; pos < 128; pos += total_threads*nval2) {
            for (int i = 0; i < nval2; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += NW) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (short l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (short l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += args.nb01/2;
            q2 += args.nb01/2;
            sc += args.nb01;
        }

        y4 += 32 * NW;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot * 0.25f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = sum_all * 0.25f;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<N_R0_IQ2_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        const int total_threads = NW * NSG;
        const int nval = 4;
        for (int pos = (NW*eff_sgitg + eff_tiisg)*nval; pos < 256; pos += total_threads*nval) {
            for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        }
        const int nval2 = 2;
        for (int pos = (NW*eff_sgitg + eff_tiisg)*nval2; pos < 128; pos += total_threads*nval2) {
            for (int i = 0; i < nval2; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += NW) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += args.nb01/2;
            q3  += args.nb01;
            gas += args.nb01/2;
        }

        y4 += 32 * NW;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot * 0.5f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = sum_all * 0.5f;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq3_s * x = (device const block_iq3_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        const int total_threads = NW * NSG;
        const int nval = 8;
        for (int pos = (NW*eff_sgitg + eff_tiisg)*nval; pos < 512; pos += total_threads*nval) {
            for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += NW) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * NW;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<N_R0_IQ3_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_s * x = (device const block_iq2_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    //threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
    //{
    //    int nval = 32;
    //    int pos  = (32*sgitg + tiisg)*nval;
    //    for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2s_grid[pos + i];
    //    threadgroup_barrier(mem_flags::mem_threadgroup);
    //}

    const short ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += NW) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >>  4));

            float2 sum = {0};
            for (short l = 0; l < 2; ++l) {
                //const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                //const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                for (short j = 0; j < 8; ++j) {
                    sum[0] += yl[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * NW;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot * 0.25f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = sum_all * 0.25f;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<N_R0_IQ2_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_s * x = (device const block_iq1_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    const short ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += NW) {
        float sumy = 0;
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint16_t * qh = xr->qh + ib;
        device const half     * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));

            float sum = 0;
            for (short j = 0; j < 4; ++j) {
                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01/2;
        }

        y4 += 32 * NW;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<N_R0_IQ1_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_m_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_m * x = (device const block_iq1_m *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    const short ix = eff_tiisg;

    device const float * y4 = y + 32 * ix;

    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += NW) {
        float4 sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint8_t  * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (short row = 0; row < nr0; row++) {
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f};
            for (short j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

            sc += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01;
        }

        y4 += 32 * NW;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * nr0;
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // Loop unconditionally — uniform barrier count across simdgroups (Intel NW<32 fix)
            for (int row = 0; row < nr0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int NR0, typename args_t>
void kernel_mul_mv_iq4_nl_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * NR0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK4_NL;
    const int ns01 = args.nb01/args.nb00;

    // Native NW-wide work distribution: 2 threads per quant block.
    constexpr short THREADS_PER_BLOCK = 2;
    const short CONCURRENT_BLOCKS = NW / THREADS_PER_BLOCK;
    const short block_id      = eff_tiisg / THREADS_PER_BLOCK;
    const short lane_in_block = eff_tiisg % THREADS_PER_BLOCK;

    shmem_f32[eff_tiisg] = kvalues_iq4nl_f[eff_tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf[NR0]={0.f};

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    {
        const short ix = block_id;
        const short it = lane_in_block;

        float4 yl[4];

        device const float * yb = y + ix*QK4_NL + it*8;

        // [TAG_MUL_MV_WEIRD]
        for (int ib = ix; ib < nb && ib < ns01; ib += CONCURRENT_BLOCKS) {
            device const float4 * y4 = (device const float4 *)yb;
            yl[0] = y4[0];
            yl[1] = y4[4];
            yl[2] = y4[1];
            yl[3] = y4[5];

            for (short row = 0; row < NR0; row++) {
                device const block_iq4_nl & xb = x[row*ns01 + ib];
                device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

                float4 acc1 = {0.f}, acc2 = {0.f};

                aux32[0] = q4[0] | (q4[1] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
                qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
                acc1 += yl[0] * qf1;
                acc2 += yl[1] * qf2;

                aux32[0] = q4[2] | (q4[3] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
                qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
                acc1 += yl[2] * qf1;
                acc2 += yl[3] * qf2;

                acc1 += acc2;

                sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
            }

            yb += CONCURRENT_BLOCKS * QK4_NL;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0;
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int NR0, typename args_t>
void kernel_mul_mv_iq4_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + eff_sgitg) * NR0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK_K;
    const int ns01 = args.nb01/args.nb00;

    // Native NW-wide work distribution: 16 threads per superblock.
    constexpr short THREADS_PER_BLOCK = 16;
    const short CONCURRENT_BLOCKS = NW / THREADS_PER_BLOCK;
    const short block_id      = eff_tiisg / THREADS_PER_BLOCK;
    const short lane_in_block = eff_tiisg % THREADS_PER_BLOCK;

    shmem_f32[eff_tiisg] = kvalues_iq4nl_f[eff_tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf[NR0]={0.f};

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    {
        const short ix = block_id;
        const short it = lane_in_block;       // 0..15
        const short ib = it/2;
        const short il = it%2;

        float4 yl[4];

        device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    // [TAG_MUL_MV_WEIRD]
    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += CONCURRENT_BLOCKS) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; ++row) {
            device const block_iq4_xs & xb = x[row*ns01 + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += CONCURRENT_BLOCKS * QK_K;
    }
    } // end block scope

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0;
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}

template<int NR0, typename args_t>
void kernel_mul_mv_mxfp4_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x + args.tg_x_offset;
    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + eff_sgitg) * NR0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_mxfp4 * x = (device const block_mxfp4 *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    const int nb   = args.ne00/QK_MXFP4;
    const int ns01 = args.nb01/args.nb00; // this can be larger than nb for permuted src0 tensors

    const short n_pairs = NW/2;    // number of (ix,it) pairs: 16 for NW=32, 8 for NW=16
    const short ix = eff_tiisg/2;  // 0...n_pairs-1
    const short it = eff_tiisg%2;  // 0 or 1

    shmem_f32[eff_tiisg] = kvalues_mxfp4_f[eff_tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK_MXFP4 + it*8;

    // note: just the check `ib < nb` is enough, but adding the redundant `&& ib < ns01` check makes the kernel a bit faster
    //       no idea why that is - needs some deeper investigation [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += n_pairs) {
        device const float4 * y4 = (device const float4 *) yb;

        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_mxfp4 & xb = x[row*ns01 + ib];
            device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 8*it);

            float4 acc1 = yl[0]*float4(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
            float4 acc2 = yl[1]*float4(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
            float4 acc3 = yl[2]*float4(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
            float4 acc4 = yl[3]*float4(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            sumf[row] += e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }

        yb += n_pairs * QK_MXFP4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0;
        // Loop always runs NR0 iterations so all threads hit the same number of
        // barriers — required for deadlock-free operation when ne0 is not a
        // multiple of NSG*NR0 (some simdgroups may have out-of-bounds first_row).
        for (int row = 0; row < NR0; ++row) {
            buf[NW * row + ltid] = sumf[row];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ltid == 0 && first_row + row < args.ne0) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                dst_f32[first_row + row] = tot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_mul_mv_mxfp4_f32")]]
kernel void kernel_mul_mv_mxfp4_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tidx [[thread_index_in_threadgroup]]) {

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, tidx);
}
