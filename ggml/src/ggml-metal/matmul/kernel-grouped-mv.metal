// =============================================================================
// Grouped matvec — multiple weight matrices processed in a single GPU dispatch.
// =============================================================================
//
// Problem: On GCN (D700), each individual matmul dispatch has too few
// threadgroups to fully occupy the GPU. The command processor takes time
// to start each dispatch, and small dispatches finish before the memory
// pipeline warms up.
//
// Solution: Fuse N consecutive matmul operations into ONE dispatch with
// N× the threadgroups. Each threadgroup looks up which weight matrix it
// belongs to via a group table, then runs the appropriate dequant kernel.
//
// Supports mixed quant types (e.g. q4_K + q6_K) in the same dispatch —
// each threadgroup branches once on grp.quant_type, then runs type-specific
// code with zero divergence within the threadgroup.
//
// IMPORTANT: All fused types MUST use the same NR0 (rows per simdgroup)
// because the dispatch code computes threadgroup counts using a single nr0.
// This is enforced by N_R0_KQUANT in ggml-metal-impl.h.

#include "dequant.metal"
#include "block-dot.metal"

// Function constant for NR0 (rows per simdgroup) — set at pipeline compile time
// based on the model profile's L1 cache pressure ratio.
// Small models (L1 pressure < 0.3): NR0=8 → more activation reuse per L1 load.
// Large models (L1 pressure > 0.7): NR0=4 → avoid L1 thrashing.
constant short FC_grouped_nr0 [[function_constant(FC_MUL_MV_GROUPED + 0)]];

// Maximum number of weight matrices that can be fused into one dispatch.
// Limited by the constant buffer size for the group table.
#define MAX_MV_GROUPS 8

// Quant type identifiers — used to branch in the kernel.
// Must match the values set in dispatch.cpp (MV_GROUP_TYPE_Q4_K=0, Q6_K=1).
#define MV_GROUP_TYPE_Q4_K 0
#define MV_GROUP_TYPE_Q6_K 1

// Per-group descriptor: identifies one weight matrix within the fused dispatch.
// Layout must match mv_group_host in dispatch.cpp exactly (same field order/sizes).
struct mv_group {
    uint64_t src0_off;   // byte offset of this group's weights from the base buffer
    uint64_t dst_off;    // byte offset of this group's output from the base buffer
    int32_t  ne01;       // number of output rows in this group's weight matrix
    int32_t  nb01;       // byte stride between rows of this group's weights
    uint64_t nb02;       // byte stride for 3rd dimension (multi-head attention)
    uint64_t nb03;       // byte stride for 4th dimension (batch)
    int32_t  tg_start;   // first threadgroup index belonging to this group
    int32_t  quant_type;  // MV_GROUP_TYPE_Q4_K or MV_GROUP_TYPE_Q6_K
};

// Kernel arguments — shared across all groups in the dispatch.
// src1 (activation/input vector) is the same for all groups.
struct ggml_metal_kargs_mul_mv_grouped {
    int32_t  ne00;       // K dimension (number of elements per row, shared across groups)
    int32_t  ne02;       // number of attention heads (src0 3rd dimension)
    int32_t  ne10;       // src1 row length (should == ne00)
    int32_t  ne11;       // number of src1 rows (1 for matvec)
    int32_t  ne12;       // src1 3rd dimension (broadcast dimension)
    uint64_t nb10;       // src1 byte stride between elements
    uint64_t nb11;       // src1 byte stride between rows
    uint64_t nb12;       // src1 byte stride for 3rd dimension
    uint64_t nb13;       // src1 byte stride for 4th dimension
    int32_t  ne0;        // output row length
    int32_t  ne1;        // output number of rows
    int16_t  r2;         // broadcast ratio for 3rd dim: ne12/ne02 (for GQA)
    int16_t  r3;         // broadcast ratio for 4th dim: ne13/ne03
    int32_t  n_groups;   // how many groups are fused in this dispatch

    // Hardware profile fields — passed from host so the kernel can self-tune.
    // compute_units: number of CUs on this GPU (32 on D700, 22 on 5500M).
    //   Used to size the persistent threadgroup loop stride.
    // l1_cache_per_cu: L1 data cache bytes per CU (16384 on D700, 16384 on 5500M).
    //   Used to determine optimal NR0 based on activation size vs cache pressure.
    int32_t  compute_units;
    int32_t  l1_cache_per_cu;

    mv_group groups[MAX_MV_GROUPS];
};

// Mixed-type grouped matvec: q4_K and q6_K in one dispatch.
// Both types use THREADS_PER_BLOCK=16, so threadgroup geometry is identical.
// PREP: every member is Q4_K and its block headers come from entries unpacked once at load (prep, 32 bytes per
// block; a member's first entry is its weight offset / 144 from the binding); false never reads prep.
// RED: every row stored first, then lane k sums row k (same lane order per row).
template<bool PREP, bool RED>
void kernel_mul_mv_grouped_kquant_f32_impl(
        constant ggml_metal_kargs_mul_mv_grouped & args,
        device const char * src0,   // weight data (base pointer, groups offset from here)
        device const char * src1,   // input activation vector (shared across all groups)
        device       char * dst,    // output buffer (base pointer, groups offset from here)
        device const char * prep,
        threadgroup  char * shmem,  // shared memory for cross-lane reduction
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {

    // -------------------------------------------------------------------------
    // Hardware constants — adapted to any SIMD width via function constant.
    // Works on GCN (NW=64, D700) and RDNA (NW=32, 5500M).
    // Always uses shmem reduction (uniform path, avoids simd_sum issues on GCN).
    // -------------------------------------------------------------------------
    const short NW = N_SIMDWIDTH;   // native SIMD width from function constant (32 or 64)
    const short NSG = 2;            // simdgroups per threadgroup (= N_SG_Q4_K)
    // NR0_MAX: compile-time maximum for array sizing (Metal doesn't support VLAs).
    // NR0: effective rows per simdgroup, set via function constant from model profile.
    // The kernel allocates sumf[NR0_MAX] but only uses sumf[0..NR0-1].
    constexpr short NR0_MAX = 8;
    const short NR0 = min((short)FC_grouped_nr0, NR0_MAX);

    // Effective lane/simdgroup indices for shmem reduction.
    // Remap from flat tidx to ensure consistent NW-wide reduction regardless of
    // the hardware's actual simdgroup assignment. Required on GCN (simd_sum broken
    // at NW=64) and harmless on RDNA (NW=32, where tidx%NW == tiisg anyway).
    const ushort eff_tiisg = tidx % NW;  // effective lane index within wavefront
    const ushort eff_sgitg = tidx / NW;  // effective simdgroup index (0 or 1)

    // -------------------------------------------------------------------------
    // Group lookup: determine which weight matrix this threadgroup processes.
    // -------------------------------------------------------------------------
    // The dispatch launches total_tg threadgroups across ALL fused matmuls.
    // Each group has a tg_start marking where its threadgroups begin.
    // We find which group owns this threadgroup by scanning the start indices.
    const int global_tg = tgpig.x;
    int group_idx = 0;
    for (int g = 1; g < args.n_groups; g++) {
        if (global_tg >= args.groups[g].tg_start) group_idx = g;
    }
    constant const mv_group & grp = args.groups[group_idx];
    const int raw_local_tg = global_tg - grp.tg_start;  // threadgroup index within this group

    int local_tg = raw_local_tg;

    // -------------------------------------------------------------------------
    // Work distribution within a threadgroup.
    // -------------------------------------------------------------------------
    // Both q4_K and q6_K use 16 threads per superblock (THREADS_PER_BLOCK=16).
    // A superblock is QK_K=256 elements of the weight matrix.
    // CONCURRENT_BLOCKS = NW/16: how many superblocks processed in parallel.
    //   NW=64 (GCN/D700):  4 concurrent superblocks
    //   NW=32 (RDNA/5500M): 2 concurrent superblocks
    constexpr short THREADS_PER_BLOCK = 16;
    const short CONCURRENT_BLOCKS = NW / THREADS_PER_BLOCK;
    const short block_id      = eff_tiisg / THREADS_PER_BLOCK;  // which superblock (0..3)
    const short lane_in_block = eff_tiisg % THREADS_PER_BLOCK;  // position within superblock (0..15)

    // nb = number of superblocks per row (each superblock covers QK_K=256 elements)
    const int nb = args.ne00 / QK_K;

    const int r1 = tgpig.y;   // src1 row index (always 0 for matvec, >0 for small batch)
    const int im = tgpig.z;   // combined index for 3rd+4th dimensions (attention heads × batch)

    // first_row: the first output row this simdgroup is responsible for.
    // Each threadgroup covers NSG simdgroups × NR0 rows = NSG*NR0 rows total.
    const int first_row = (local_tg * NSG + eff_sgitg) * NR0;

    // Broadcast indices for grouped-query attention (GQA).
    // r2/r3 are the ratios: multiple src1 heads may share one src0 head.
    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    // Byte offsets into the weight (src0) and activation (src1) buffers.
    // grp.src0_off is the offset from the base buffer to this group's weights.
    const uint64_t offset0 = grp.src0_off + first_row * grp.nb01 + (i12 / args.r2) * grp.nb02 + (i13 / args.r3) * grp.nb03;
    const uint64_t offset1 = r1 * args.nb11 + (i12) * args.nb12 + (i13) * args.nb13;

    // y = input activation vector (same for all weight rows)
    device const float * y = (device const float *)(src1 + offset1);

    // Accumulator: one partial sum per output row this simdgroup handles.
    // The row loop reuses the activation loads across NR0 rows (activation reuse).
    float sumf[NR0_MAX] = {0.f};  // sized at compile-time max, only NR0 entries used

    // =========================================================================
    // Branch on quant type — divergence is per-threadgroup, not per-thread,
    // so there is no SIMD divergence penalty.
    // =========================================================================
    if (grp.quant_type == MV_GROUP_TYPE_Q4_K) {
        // =====================================================================
        // q4_K dequantization
        // =====================================================================
        // block_q4_K layout (144 bytes per superblock of 256 elements):
        //   - dm:     half2 (4B) — scale (d) and min scale (dmin)
        //   - scales: uint8[12]  — 6-bit quantized per-group scales and mins
        //   - qs:     uint8[128] — 4-bit quantized weights (2 elements per byte)
        device const block_q4_K * x = (device const block_q4_K *)(src0 + offset0);

        // Bitmasks for unpacking the 6-bit scales from the 12-byte scales array.
        // The scales are packed across 6 uint16_t values in a non-trivial layout
        // inherited from the GGML quantization format.
        // Historical variable names and operations below now live in block-dot-q4-k.metal;
        // traversal and byte-row stride remain here.
        // Thread-to-data mapping within a 16-thread block.
        // Each thread processes 16 elements (4 floats × 4 groups) from the 256-element superblock.
        //
        //   itid (0..15) → which of the 16 threads in this block
        //   il (0..3)    → which of 4 element groups (each group = 4 elements)
        //   ir (0..3)    → position within element group
        //   v_im (0..1)  → which 128-element half of the superblock
        //   v_in (0..1)  → interleaved position within the half
        //   l0           → starting element offset within the half (0,4,8,...,28)
        // 0 = lower 128 elements, 1 = upper 128
        // interleave: even/odd groups
        // byte offset into qs[] for this thread's quants
        // float offset into activation for this thread's data
        // The exact activation base is 64*scale_pair+chunk, not a 128-element half offset.
        const block_dot_q4_k_fragment owned = block_dot_q4_k_swar::owned_fragment(lane_in_block);
        // Main loop: iterate over superblocks, striding by CONCURRENT_BLOCKS
        // so that 4 blocks process 4 different superblocks in parallel.
        for (int ib = block_id; ib < nb; ib += CONCURRENT_BLOCKS) {
            // Load 4 float4 vectors from the activation (input) vector.
            // These are REUSED across NR0 weight rows — this is the key optimization.
            // by10/by132 cover the lower 128-element half, by20/by232 the upper half.
            const block_dot_q4_k_activation prepared = block_dot_q4_k_swar::prepare_activation<block_dot_lane_fp32>(owned, y + ib*QK_K);
            for (short row = 0; row < NR0; row++) {
                // Preserve the original byte row stride before restoring the Q4 block type.
                device const block_q4_K * block = (device const block_q4_K *) ((device const char *) (x + ib) + row*grp.nb01);
                if (PREP) {
                    device const uint * header = (device const uint *) (prep + (grp.src0_off / 144 + (uint64_t) ((first_row + row) * nb + ib)) * 32);
                    block_dot_q4_k_swar::accumulate_prepared<block_dot_lane_fp32>(block, owned, prepared, sumf[row], header);
                } else {
                    block_dot_q4_k_swar::accumulate<block_dot_lane_fp32>(block, owned, prepared, sumf[row]);
                }
            }
        }
    } else {
        // =====================================================================
        // q6_K dequantization
        // =====================================================================
        // block_q6_K layout (210 bytes per superblock of 256 elements):
        //   - ql:     uint8[128] — lower 4 bits of 6-bit weights
        //   - qh:     uint8[64]  — upper 2 bits of 6-bit weights
        //   - scales: int8[16]   — per-group scales
        //   - d:      half       — super-block scale
        device const block_q6_K * x = (device const block_q6_K *)(src0 + offset0);

        const block_dot_q6_k_fragment owned = block_dot_q6_k_grouped_legacy::owned_fragment(lane_in_block);

        for (int i = block_id; i < nb; i += CONCURRENT_BLOCKS) {
            const block_dot_q6_k_grouped_activation prepared = block_dot_q6_k_grouped_legacy::prepare_activation<block_dot_lane_fp32>(owned, y + i * QK_K);

            // Inner row loop — same activation reuse pattern as q4_K.
            for (short row = 0; row < NR0; ++row) {
                device const block_q6_K * block = (device const block_q6_K *) ((device const char *) (x + i) + row * grp.nb01);
                block_dot_q6_k_grouped_legacy::accumulate<block_dot_lane_fp32>(block, owned, prepared, sumf[row]);
            }
        }
    }

    // =========================================================================
    // Cross-lane reduction using threadgroup (shared) memory.
    // =========================================================================
    // Each thread has computed a partial dot product for its slice of the row.
    // We need to sum across all NW threads in the wavefront to get the final
    // result. On GCN, simd_sum doesn't work at NW=64, so we use shared memory.
    //
    // Buffer layout: [simdgroup_0: NW×NR0 floats][simdgroup_1: NW×NR0 floats]
    // Total shared memory: NSG × NW × NR0 × sizeof(float) bytes.
    device float * dst_f32 = (device float *)(dst + grp.dst_off) + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    // Wave32 supports the native reduction already used by scalar matvecs.
    // The function-constant width selects this path uniformly for the whole TG.
    constexpr short NATIVE_REDUCTION_SIMD_WIDTH = 32;
    if (NW == NATIVE_REDUCTION_SIMD_WIDTH) {
        // Each simdgroup owns different output rows; do not add their sums.
        // Every lane participates even when the final row is outside the matrix.
        for (int row = 0; row < NR0; ++row) {
            const float total = simd_sum(sumf[row]);
            // The native lane leader writes this simdgroup's valid row only.
            if (tiisg == 0 && first_row + row < grp.ne01) {
                dst_f32[first_row + row] = total;
            }
        }
        // No cross-simdgroup scratch or barriers are needed for disjoint rows.
        return;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort ltid = tidx % NW;
    threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0;
    if (RED) {
        for (int row = 0; row < NR0; ++row) {
            buf[NW * row + ltid] = sumf[row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ltid < NR0 && first_row + ltid < grp.ne01) {
            float tot = 0.0f;
            for (short i = 0; i < NW; i++) { tot += buf[NW * ltid + i]; }
            dst_f32[first_row + ltid] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return;
    }
    for (int row = 0; row < NR0; ++row) {
        // All threads write their partial sum for this row.
        buf[NW * row + ltid] = sumf[row];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 accumulates the full sum and writes the final result.
        // Guard: first_row + row < ne01 prevents out-of-bounds writes when
        // the number of rows isn't evenly divisible by NR0*NSG.
        if (ltid == 0 && first_row + row < grp.ne01) {
            float tot = 0.0f;
            for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
            dst_f32[first_row + row] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_mul_mv_grouped_kquant_f32(
        constant ggml_metal_kargs_mul_mv_grouped & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tidx  [[thread_index_in_threadgroup]]) {
    kernel_mul_mv_grouped_kquant_f32_impl<false, false>(args, src0, src1, dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// The grouped launch whose members are all Q4_K weights with prepared block headers (hw mv rows, kind group, prep):
// the headers at buffer 5.
kernel void kernel_mul_mv_grouped_q4k_prep_f32(
        constant ggml_metal_kargs_mul_mv_grouped & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        device const char * prep [[buffer(5)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tidx  [[thread_index_in_threadgroup]]) {
    kernel_mul_mv_grouped_kquant_f32_impl<true, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
}

// The grouped launches with every row stored before the row sums.
kernel void kernel_mul_mv_grouped_kquant_red_f32(
        constant ggml_metal_kargs_mul_mv_grouped & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tidx  [[thread_index_in_threadgroup]]) {
    kernel_mul_mv_grouped_kquant_f32_impl<false, true>(args, src0, src1, dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

kernel void kernel_mul_mv_grouped_q4k_prep_red_f32(
        constant ggml_metal_kargs_mul_mv_grouped & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst  [[buffer(3)]],
        device const char * prep [[buffer(5)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tidx  [[thread_index_in_threadgroup]]) {
    kernel_mul_mv_grouped_kquant_f32_impl<true, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
}
