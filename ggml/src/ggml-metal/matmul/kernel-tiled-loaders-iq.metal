// Tiled matmul A-matrix loader structs for IQ quantization types (QK_K=256)
//
// IQ types use importance quantization with grid-based lookup tables.
// Each QK_K=256 block spans 8 K-tiles (256/32=8).
// Loaders track block_idx and sub_block (0..7) within the current QK_K block.

// ============================================================================
// iq4_xs — 4-bit importance quantization with per-sub-block scales
// ============================================================================

inline void load_a_iq4_xs_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;
    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_iq4_xs * block_ptr = (device const block_iq4_xs *)(src0_row + loadc * nb01) + block_idx;

        const int ls = ((block_ptr->scales_l[sub_block/2] >> 4*(sub_block%2)) & 0xf) |
                       (((block_ptr->scales_h >> 2*sub_block) & 3) << 4);
        const float d = (float)block_ptr->d * (ls - 32);

        device const uint8_t * qs = block_ptr->qs + 16 * sub_block;

        if (loadr < 8) {
            const uint8_t byte0 = qs[2*loadr];
            const uint8_t byte1 = qs[2*loadr + 1];
            buf_a[buf_idx] = float2(d * kvalues_iq4nl_f[byte0 & 0x0F], d * kvalues_iq4nl_f[byte1 & 0x0F]);
        } else {
            const uint8_t byte0 = qs[2*(loadr - 8)];
            const uint8_t byte1 = qs[2*(loadr - 8) + 1];
            buf_a[buf_idx] = float2(d * kvalues_iq4nl_f[byte0 >> 4], d * kvalues_iq4nl_f[byte1 >> 4]);
        }
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_iq4_xs {
    device const char * src0_row;
    uint num_blocks;
    uint block_idx;
    uint sub_block;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK_K;
        block_idx = 0;
        sub_block = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_iq4_xs_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_idx, sub_block, num_blocks, nb01);
    }

    void advance() {
        sub_block++;
        if (sub_block == 8) {
            sub_block = 0;
            block_idx++;
        }
    }
};

// ============================================================================
// iq2_xxs — 2-bit importance quantization (xxs variant)
// ============================================================================

inline void load_a_iq2_xxs_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;
    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_iq2_xxs * block_ptr = (device const block_iq2_xxs *)(src0_row + loadc * nb01) + block_idx;

        device const uint16_t * q2 = block_ptr->qs + 4 * sub_block;
        const uint32_t aux32_g = q2[0] | (q2[1] << 16);
        const uint32_t aux32_s = q2[2] | (q2[3] << 16);
        thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;

        const float dl = (float)block_ptr->d * (0.5f + (aux32_s >> 28)) * 0.25f;

        const uint elem0 = loadr * 2;
        const uint group = elem0 / 8;

        constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[group]);
        const uint sign_shift = group * 7;
        const uint8_t signs = ksigns_iq2xs[(aux32_s >> sign_shift) & 127];

        const uint pos0 = elem0 % 8;
        const uint pos1 = (elem0 + 1) % 8;

        const float val0 = dl * grid[pos0] * (signs & kmask_iq2xs[pos0] ? -1.f : 1.f);
        const float val1 = dl * grid[pos1] * (signs & kmask_iq2xs[pos1] ? -1.f : 1.f);

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_iq2_xxs {
    device const char * src0_row;
    uint num_blocks;
    uint block_idx;
    uint sub_block;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK_K;
        block_idx = 0;
        sub_block = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_iq2_xxs_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_idx, sub_block, num_blocks, nb01);
    }

    void advance() {
        sub_block++;
        if (sub_block == 8) {
            sub_block = 0;
            block_idx++;
        }
    }
};

// ============================================================================
// iq2_xs — 2-bit importance quantization (xs variant, with per-sub-block scales)
// ============================================================================

inline void load_a_iq2_xs_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;
    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_iq2_xs * block_ptr = (device const block_iq2_xs *)(src0_row + loadc * nb01) + block_idx;

        const uint elem0 = loadr * 2;
        const uint il_half = elem0 / 16;

        const float d = (float)block_ptr->d;
        const float dl = d * (0.5f + ((block_ptr->scales[sub_block] >> 4*il_half) & 0xf)) * 0.25f;

        const uint q2_idx = elem0 / 8;
        device const uint16_t * q2 = block_ptr->qs + 4 * sub_block;

        constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[q2_idx] & 511));
        const uint8_t signs = ksigns_iq2xs[q2[q2_idx] >> 9];

        const uint pos0 = elem0 % 8;
        const uint pos1 = (elem0 + 1) % 8;

        const float val0 = dl * grid[pos0] * (signs & kmask_iq2xs[pos0] ? -1.f : 1.f);
        const float val1 = dl * grid[pos1] * (signs & kmask_iq2xs[pos1] ? -1.f : 1.f);

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_iq2_xs {
    device const char * src0_row;
    uint num_blocks;
    uint block_idx;
    uint sub_block;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK_K;
        block_idx = 0;
        sub_block = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_iq2_xs_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_idx, sub_block, num_blocks, nb01);
    }

    void advance() {
        sub_block++;
        if (sub_block == 8) {
            sub_block = 0;
            block_idx++;
        }
    }
};

// ============================================================================
// iq2_s — 2-bit importance quantization (s variant, with separate signs)
// ============================================================================

inline void load_a_iq2_s_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;
    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_iq2_s * block_ptr = (device const block_iq2_s *)(src0_row + loadc * nb01) + block_idx;

        const uint elem0 = loadr * 2;
        const uint il_half = elem0 / 16;

        const float d = (float)block_ptr->d;
        const float dl = d * (0.5f + ((block_ptr->scales[sub_block] >> 4*il_half) & 0xf)) * 0.25f;

        device const uint8_t * qs = block_ptr->qs + 4 * sub_block + 2 * il_half;
        device const uint8_t * signs = qs + QK_K/8;
        const uint8_t qh = block_ptr->qh[sub_block] >> 4*il_half;

        const uint group8 = elem0 / 8;
        const uint local_group = group8 % 2;

        constant uint8_t * grid = (constant uint8_t *)(iq2s_grid + (qs[local_group] | ((qh << (8-2*local_group)) & 0x300)));

        const uint pos0 = elem0 % 8;
        const uint pos1 = (elem0 + 1) % 8;

        const float val0 = dl * grid[pos0] * select(1.f, -1.f, signs[local_group] & kmask_iq2xs[pos0]);
        const float val1 = dl * grid[pos1] * select(1.f, -1.f, signs[local_group] & kmask_iq2xs[pos1]);

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_iq2_s {
    device const char * src0_row;
    uint num_blocks;
    uint block_idx;
    uint sub_block;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK_K;
        block_idx = 0;
        sub_block = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_iq2_s_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_idx, sub_block, num_blocks, nb01);
    }

    void advance() {
        sub_block++;
        if (sub_block == 8) {
            sub_block = 0;
            block_idx++;
        }
    }
};

// ============================================================================
// iq3_xxs — 3-bit importance quantization (xxs variant)
// ============================================================================

inline void load_a_iq3_xxs_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;
    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_iq3_xxs * block_ptr = (device const block_iq3_xxs *)(src0_row + loadc * nb01) + block_idx;

        device const uint8_t * q3 = block_ptr->qs + 8 * sub_block;
        device const uint16_t * gas = (device const uint16_t *)(block_ptr->qs + QK_K/4) + 2 * sub_block;
        const uint32_t aux32 = gas[0] | (gas[1] << 16);

        const float d = (float)block_ptr->d;
        const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;

        const uint elem0 = loadr * 2;
        const uint il_half = elem0 / 16;
        const uint local_group = (elem0 % 16) / 4;
        const uint q3_idx = 4 * il_half + local_group;

        constant uint8_t * grid = (constant uint8_t *)(iq3xxs_grid + q3[q3_idx]);

        const uint sign_shift = (local_group < 2) ? (14 * il_half) : (14 * il_half + 7);
        const uint8_t signs = ksigns_iq2xs[(aux32 >> sign_shift) & 127];

        const uint kmask_base = (local_group % 2) * 4;
        const uint pos0 = elem0 % 4;
        const uint pos1 = (elem0 + 1) % 4;

        const float val0 = dl * grid[pos0] * (signs & kmask_iq2xs[kmask_base + pos0] ? -1.f : 1.f);
        const float val1 = dl * grid[pos1] * (signs & kmask_iq2xs[kmask_base + pos1] ? -1.f : 1.f);

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_iq3_xxs {
    device const char * src0_row;
    uint num_blocks;
    uint block_idx;
    uint sub_block;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK_K;
        block_idx = 0;
        sub_block = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_iq3_xxs_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_idx, sub_block, num_blocks, nb01);
    }

    void advance() {
        sub_block++;
        if (sub_block == 8) {
            sub_block = 0;
            block_idx++;
        }
    }
};

// ============================================================================
// iq3_s — 3-bit importance quantization (s variant, with separate signs and qh)
// ============================================================================

// iq3_s: 4 grids per half (grid0..3), each grid has 4 values. signs are per 4-element group.
// Reference: dequantize_iq3_s in dequant.metal
inline void load_a_iq3_s_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;
    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_iq3_s * block_ptr = (device const block_iq3_s *)(src0_row + loadc * nb01) + block_idx;

        const float d = (float)block_ptr->d;
        const float dl = d * (1 + 2*((block_ptr->scales[sub_block/2] >> 4*(sub_block%2)) & 0xf));

        device const uint8_t * qs = block_ptr->qs + 8 * sub_block;
        const uint8_t qh_byte = block_ptr->qh[sub_block];
        device const uint8_t * signs = block_ptr->signs + 4 * sub_block;

        const uint elem0 = loadr * 2;
        const uint il = elem0 / 16;  // 0 or 1 — which half
        const uint local_group = (elem0 % 16) / 4;  // 0..3 within half
        const uint q3_idx = 4 * il + local_group;

        // qh shifts: 8, 7, 6, 5 for qs indices 0, 1, 2, 3 within each half
        // Reference: grid1 = qs[4*il+0] | ((qh << 8) & 256)
        //            grid2 = qs[4*il+1] | ((qh << 7) & 256)  etc.
        const uint8_t qh_shifted = (il == 0) ? qh_byte : (qh_byte >> 4);
        const uint qh_shift = 8 - local_group;
        constant uint8_t * grid = (constant uint8_t *)(iq3s_grid + (qs[q3_idx] | ((qh_shifted << qh_shift) & 256)));

        const uint pos0 = elem0 % 4;
        const uint pos1 = (elem0 + 1) % 4;

        // Signs: signs[2*il + 0] for groups 0,1; signs[2*il + 1] for groups 2,3
        // kmask_iq2xs[0..3] for first grid, kmask_iq2xs[4..7] for second
        const uint sign_byte_idx = 2 * il + local_group / 2;
        const uint kmask_base = (local_group % 2) * 4;

        const float val0 = dl * grid[pos0] * select(1.f, -1.f, signs[sign_byte_idx] & kmask_iq2xs[kmask_base + pos0]);
        const float val1 = dl * grid[pos1] * select(1.f, -1.f, signs[sign_byte_idx] & kmask_iq2xs[kmask_base + pos1]);

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_iq3_s {
    device const char * src0_row;
    uint num_blocks;
    uint block_idx;
    uint sub_block;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK_K;
        block_idx = 0;
        sub_block = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_iq3_s_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_idx, sub_block, num_blocks, nb01);
    }

    void advance() {
        sub_block++;
        if (sub_block == 8) {
            sub_block = 0;
            block_idx++;
        }
    }
};

// ============================================================================
// iq1_s — 1-bit importance quantization (s variant)
// ============================================================================

// iq1_s: type4x4 layout = [grid_lo[0..3], grid_hi[0..3], grid2_lo[0..3], grid2_hi[0..3]]
// Reference: dequantize_iq1_s in dequant.metal
inline void load_a_iq1_s_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;
    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_iq1_s * block_ptr = (device const block_iq1_s *)(src0_row + loadc * nb01) + block_idx;

        const float d = (float)block_ptr->d;
        device const uint16_t * qh = block_ptr->qh;

        const float dl = d * (2*((qh[sub_block] >> 12) & 7) + 1);
        const float ml = dl * (qh[sub_block] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);

        const uint elem0 = loadr * 2;
        const uint il_half = elem0 / 16;
        const uint16_t h = qh[sub_block] >> 6*il_half;

        device const uint8_t * qs = block_ptr->qs + 4 * sub_block + 2 * il_half;

        // grid1 = qs[0], shift <<8; grid2 = qs[1], shift <<5
        const uint local_group = (elem0 / 8) % 2;
        constant uint8_t * grid = (constant uint8_t *)(iq1s_grid_gpu + (qs[local_group] | ((h << (8-3*local_group)) & 0x700)));

        // type4x4: elements 0-3 = grid[0..3] & 0xf, elements 4-7 = grid[0..3] >> 4
        const uint pos = elem0 % 8;
        const uint byte0 = pos % 4;
        const bool high0 = pos >= 4;
        const float val0 = dl * (high0 ? (grid[byte0] >> 4) : (grid[byte0] & 0xf)) + ml;

        const uint pos1 = (elem0 + 1) % 8;
        const uint byte1 = pos1 % 4;
        const bool high1 = pos1 >= 4;
        const float val1 = dl * (high1 ? (grid[byte1] >> 4) : (grid[byte1] & 0xf)) + ml;

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_iq1_s {
    device const char * src0_row;
    uint num_blocks;
    uint block_idx;
    uint sub_block;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK_K;
        block_idx = 0;
        sub_block = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_iq1_s_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_idx, sub_block, num_blocks, nb01);
    }

    void advance() {
        sub_block++;
        if (sub_block == 8) {
            sub_block = 0;
            block_idx++;
        }
    }
};

// ============================================================================
// iq1_m — 1-bit importance quantization (m variant, with per-grid deltas)
// ============================================================================

// iq1_m: no .d field, scale reconstructed from scales[] nibbles, qh is uint8_t
// Reference: dequantize_iq1_m in dequant.metal
inline void load_a_iq1_m_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;
    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_iq1_m * block_ptr = (device const block_iq1_m *)(src0_row + loadc * nb01) + block_idx;

        // Reconstruct d from scales[] — iq1_m has no .d field
        device const uint16_t * sc = (device const uint16_t *)block_ptr->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = scale.f16;

        // sub_block = ib32 (0..7), each covers 32 elements
        const uint elem0 = loadr * 2;
        const uint il = elem0 / 16;  // 0 or 1 — which half of the 32-element sub-block

        device const uint8_t * qs = block_ptr->qs + 4 * sub_block + 2 * il;
        device const uint8_t * qh = block_ptr->qh + 2 * sub_block + il;

        const float dl = d * (2*((sc[sub_block/2] >> (6*(sub_block%2) + 3*il)) & 7) + 1);

        // Two groups of 8 per half: group0 uses qs[0]/ml1, group1 uses qs[1]/ml2
        const uint group8 = elem0 / 8;
        const uint local_group = group8 % 2;

        // ml differs per group: bit 3 for group0, bit 7 for group1
        const float ml = dl * (qh[0] & (local_group == 0 ? 0x08 : 0x80) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

        // Grid shifts: <<8 for group0, <<4 for group1 (differs from iq1_s which uses <<8/<<5)
        constant uint8_t * grid = (constant uint8_t *)(iq1s_grid_gpu + (qs[local_group] | ((qh[0] << (8 - 4*local_group)) & 0x700)));

        // type4x4: elements 0-3 = grid[0..3] & 0xf, elements 4-7 = grid[0..3] >> 4
        const uint pos = elem0 % 8;
        const uint byte0 = pos % 4;
        const bool high0 = pos >= 4;
        const float val0 = dl * (high0 ? (grid[byte0] >> 4) : (grid[byte0] & 0xf)) + ml;

        const uint pos1 = (elem0 + 1) % 8;
        const uint byte1 = pos1 % 4;
        const bool high1 = pos1 >= 4;
        const float val1 = dl * (high1 ? (grid[byte1] >> 4) : (grid[byte1] & 0xf)) + ml;

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_iq1_m {
    device const char * src0_row;
    uint num_blocks;
    uint block_idx;
    uint sub_block;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK_K;
        block_idx = 0;
        sub_block = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_iq1_m_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_idx, sub_block, num_blocks, nb01);
    }

    void advance() {
        sub_block++;
        if (sub_block == 8) {
            sub_block = 0;
            block_idx++;
        }
    }
};
