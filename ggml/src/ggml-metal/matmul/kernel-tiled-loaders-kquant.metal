// Tiled matmul A-matrix loaders for QK_K=256 k-quant types (q4_K, q5_K, q6_K, q2_K, q3_K).
// Included by kernel-tiled.metal. Each loader cooperatively fills shared memory with BK=32
// dequantized float2 elements per K-tile. One QK_K=256 block spans 8 K-tiles (256/32=8),
// tracked via block_idx + sub_block counter.

// ============================================================================
// q4_K loader
// ============================================================================

inline void load_a_q4_K_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_q4_K * block_ptr = (device const block_q4_K *)(src0_row + loadc * nb01) + block_idx;

        // Sub-block mapping: 256 elements = 8 sub-blocks of 32 elements each
        // Even sub-blocks (0,2,4,6): low nibbles (mask 0x0F)
        // Odd sub-blocks (1,3,5,7): high nibbles (mask 0xF0)
        // qs bytes: sub-blocks 0-1 use qs[0..31], 2-3 use qs[32..63], etc.

        const uint qs_base = (sub_block / 2) * 32;
        const bool use_high = (sub_block % 2) == 1;
        const uint is = (sub_block / 2) * 2;
        const uint sc_k = use_high ? 1 : 0;
        const uchar2 sc = get_scale_min_k4_just2(is, sc_k, block_ptr->scales);
        const float d = use_high ? (float)block_ptr->d / 16.0f : (float)block_ptr->d;
        const float min_val = (float)block_ptr->dmin;
        const float dl = d * sc[0];
        const float ml = min_val * sc[1];

        device const uint8_t * qs = block_ptr->qs + qs_base;

        // Each loadr (0..15) reads 2 elements
        // loadr 0..7: qs[0..15]
        // loadr 8..15: qs[16..31]
        uint8_t byte0, byte1;
        if (loadr < 8) {
            byte0 = qs[2 * loadr];
            byte1 = qs[2 * loadr + 1];
        } else {
            byte0 = qs[16 + 2 * (loadr - 8)];
            byte1 = qs[16 + 2 * (loadr - 8) + 1];
        }

        float val0, val1;
        if (use_high) {
            // High nibbles: mask 0xF0, value already shifted, compensated by d/16
            val0 = dl * float(byte0 & 0xF0) - ml;
            val1 = dl * float(byte1 & 0xF0) - ml;
        } else {
            // Low nibbles: mask 0x0F
            val0 = dl * float(byte0 & 0x0F) - ml;
            val1 = dl * float(byte1 & 0x0F) - ml;
        }

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_q4_K {
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
        load_a_q4_K_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01,
                             block_idx, sub_block, num_blocks, nb01);
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
// q5_K loader
// ============================================================================

inline void load_a_q5_K_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_q5_K * block_ptr = (device const block_q5_K *)(src0_row + loadc * nb01) + block_idx;

        // Same sub-block layout as q4_K, plus 5th bit from qh array
        const uint qs_base = (sub_block / 2) * 32;
        const bool use_high = (sub_block % 2) == 1;
        const uint is = (sub_block / 2) * 2;
        const uint sc_k = use_high ? 1 : 0;
        const uchar2 sc = get_scale_min_k4_just2(is, sc_k, block_ptr->scales);
        const float d = use_high ? (float)block_ptr->d / 16.0f : (float)block_ptr->d;
        const float min_val = (float)block_ptr->dmin;
        const float dl = d * sc[0];
        const float ml = min_val * sc[1];

        // qh provides 5th bit: ul = 1 << sub_block selects the bit position
        const uint8_t ul = 1 << sub_block;
        const float qh_val = use_high ? 256.0f : 16.0f;

        device const uint8_t * qs = block_ptr->qs + qs_base;
        device const uint8_t * qh = block_ptr->qh;

        // qh layout: first 16 elements use qh[0..15], next 16 use qh[16..31]
        uint qh_byte_idx0, qh_byte_idx1;
        uint8_t byte0, byte1;

        if (loadr < 8) {
            byte0 = qs[2 * loadr];
            byte1 = qs[2 * loadr + 1];
            qh_byte_idx0 = 2 * loadr;
            qh_byte_idx1 = 2 * loadr + 1;
        } else {
            byte0 = qs[16 + 2 * (loadr - 8)];
            byte1 = qs[16 + 2 * (loadr - 8) + 1];
            qh_byte_idx0 = 16 + 2 * (loadr - 8);
            qh_byte_idx1 = 16 + 2 * (loadr - 8) + 1;
        }

        const uint8_t qh0 = qh[qh_byte_idx0];
        const uint8_t qh1 = qh[qh_byte_idx1];
        const float extra0 = (qh0 & ul) ? qh_val : 0.0f;
        const float extra1 = (qh1 & ul) ? qh_val : 0.0f;

        float val0, val1;
        if (use_high) {
            val0 = dl * (float(byte0 & 0xF0) + extra0) - ml;
            val1 = dl * (float(byte1 & 0xF0) + extra1) - ml;
        } else {
            val0 = dl * (float(byte0 & 0x0F) + extra0) - ml;
            val1 = dl * (float(byte1 & 0x0F) + extra1) - ml;
        }

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_q5_K {
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
        load_a_q5_K_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01,
                             block_idx, sub_block, num_blocks, nb01);
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
// q6_K loader
// ============================================================================

inline void load_a_q6_K_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_q6_K * block_ptr = (device const block_q6_K *)(src0_row + loadc * nb01) + block_idx;

        // q6_K: 6 bits per element = 4 bits (ql) + 2 bits (qh)
        // Complex interleaved packing: use element-by-element dequant
        // Map sub_block (0..7) to il parameter (0..15, each il produces 16 elements)
        // sub_block 0 = il 0,1; sub_block 1 = il 2,3; etc.

        const float d = (float)block_ptr->d;
        device const int8_t * scales = (device const int8_t *)block_ptr->scales;

        // Determine which pair of il values this sub_block corresponds to
        const uint il_base = sub_block * 2;
        const bool second_half = (loadr >= 8);
        const uint il = il_base + (second_half ? 1 : 0);

        // Scale index: sc = scales[(il%2) + 2 * (il/2)]
        const float sc = scales[(il % 2) + 2 * (il / 2)];

        // Element indices within this il's 16 elements
        const uint local_idx0 = second_half ? (2 * (loadr - 8)) : (2 * loadr);
        const uint local_idx1 = local_idx0 + 1;

        // Compute element indices within the QK_K=256 block
        // Each il produces 16 elements at different positions
        // From dequant_q6_K: complex bit extraction based on il
        // il/8 selects 32-element group, (il/2)&1 and il&1 select sub-positions

        // Simplified approach: compute byte offsets directly
        // ql: uint16_t access, base = 32*(il/8) + 16*((il/2)&1) + 8*(il&1)
        // qh: uint16_t access, base = 16*(il/8) + 8*(il&1)

        device const uint16_t * ql = (device const uint16_t *)block_ptr->ql;
        device const uint16_t * qh = (device const uint16_t *)block_ptr->qh;

        ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
        qh = qh + 16 * (il / 8) + 8 * (il & 1);

        const uint il_mod = (il / 2) & 3;

        // Extract 6-bit values for elements 0 and 1
        // From dequant_q6_K logic:
        const uint16_t ql0 = ql[local_idx0 / 2];
        const uint16_t ql1 = ql[local_idx1 / 2];
        const uint16_t qh0 = qh[local_idx0 / 2];
        const uint16_t qh1 = qh[local_idx1 / 2];

        // Bit extraction depends on il_mod and element position
        uint q_val0, q_val1;

        if (il_mod == 0) {
            q_val0 = ((ql0 & 0xF) | ((qh0 & 0x03) << 4)) - 32;
            q_val1 = ((ql1 & 0xF) | ((qh1 & 0x03) << 4)) - 32;
        } else if (il_mod == 1) {
            q_val0 = ((ql0 >> 4) | ((qh0 & 0x0c) << 2)) - 32;
            q_val1 = ((ql1 >> 4) | ((qh1 & 0x0c) << 2)) - 32;
        } else if (il_mod == 2) {
            q_val0 = ((ql0 >> 8) | ((qh0 & 0x30) << 0)) - 32;
            q_val1 = ((ql1 >> 8) | ((qh1 & 0x30) << 0)) - 32;
        } else {  // il_mod == 3
            q_val0 = ((ql0 >> 12) | ((qh0 & 0xc0) >> 2)) - 32;
            q_val1 = ((ql1 >> 12) | ((qh1 & 0xc0) >> 2)) - 32;
        }

        const float val0 = d * sc * float(q_val0);
        const float val1 = d * sc * float(q_val1);

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_q6_K {
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
        load_a_q6_K_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01,
                             block_idx, sub_block, num_blocks, nb01);
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
// q2_K loader
// ============================================================================

inline void load_a_q2_K_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_q2_K * block_ptr = (device const block_q2_K *)(src0_row + loadc * nb01) + block_idx;

        // q2_K: 2 bits per element, 64 bytes = 256 elements
        // Each byte holds 4 elements (2 bits each)
        // Map sub_block (0..7) to il parameter (0..15)

        const float d = (float)block_ptr->d;
        const float min_val = (float)block_ptr->dmin;

        const uint il_base = sub_block * 2;
        const bool second_half = (loadr >= 8);
        const uint il = il_base + (second_half ? 1 : 0);

        // Scale and offset: q = qs + 32*(il/8) + 16*(il&1)
        device const uint8_t * q = (device const uint8_t *)block_ptr->qs;
        q = q + 32 * (il / 8) + 16 * (il & 1);

        const uint8_t sc = block_ptr->scales[il];

        // Determine mask and coefficient based on (il/2) % 4
        const uint il_mod = (il / 2) % 4;
        half coef;
        uchar mask;

        if (il_mod == 0) {
            coef = 1.h;
            mask = 3;
        } else if (il_mod == 1) {
            coef = 1 / 4.h;
            mask = 12;
        } else if (il_mod == 2) {
            coef = 1 / 16.h;
            mask = 48;
        } else {  // il_mod == 3
            coef = 1 / 64.h;
            mask = 192;
        }

        const float dl = d * (sc & 0xF) * (float)coef;
        const float ml = min_val * (sc >> 4);

        // Each loadr reads 2 elements (packed in same or adjacent bytes)
        const uint local_idx = second_half ? (loadr - 8) : loadr;
        const uint8_t byte0 = q[2 * local_idx];
        const uint8_t byte1 = q[2 * local_idx + 1];

        const float val0 = dl * float(byte0 & mask) - ml;
        const float val1 = dl * float(byte1 & mask) - ml;

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_q2_K {
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
        load_a_q2_K_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01,
                             block_idx, sub_block, num_blocks, nb01);
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
// q3_K loader
// ============================================================================

// q3_K: 3 bits per element with complex bit packing. Follows dequantize_q3_K exactly.
inline void load_a_q3_K_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr, uint loadc,
    uint idx_m, uint ne01,
    uint block_idx, uint sub_block,
    uint num_blocks, uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_idx < num_blocks) {
        device const block_q3_K * block_ptr = (device const block_q3_K *)(src0_row + loadc * nb01) + block_idx;

        const float d_all = (float)block_ptr->d;
        device const uint8_t * q_base = (device const uint8_t *)block_ptr->qs;
        device const uint8_t * h_base = (device const uint8_t *)block_ptr->hmask;
        device const int8_t * scales = (device const int8_t *)block_ptr->scales;

        // Map (sub_block, loadr) to dequant il parameter
        const uint il = sub_block * 2 + (loadr >= 8 ? 1 : 0);
        const uint local_idx = loadr % 8;

        // Pointer setup from reference
        device const uint8_t * q = q_base + 32 * (il/8) + 16 * (il&1);
        device const uint8_t * h = h_base + 16 * (il&1);
        const uint8_t m = 1 << (il/2);

        // Scale extraction (exact reference replica)
        uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : ((il/4)>0 ? 12 : 3);
        uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
        uint16_t scale_2 = uint16_t(uint8_t(scales[il%8]));
        uint16_t scale_1 = uint16_t(uint8_t(scales[8 + il%4]));
        int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                                   : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
        float dl = il<8 ? d_all * (float(dl_int) - 32.f) : d_all * (float(dl_int) / 16.f - 32.f);
        const float ml = 4.f * dl;

        // Which 2-bit pair within each byte
        uint il2 = (il/2) & 3;
        float coef = il2>1 ? (il2>2 ? 1.0f/64.f : 1.0f/16.f) : (il2>0 ? 1.0f/4.f : 1.0f);
        uint8_t mask = il2>1 ? (il2>2 ? 192 : 48) : (il2>0 ? 12 : 3);
        dl *= coef;

        // Extract 2 elements
        uint i0 = local_idx * 2;
        uint i1 = i0 + 1;
        float val0 = dl * float(q[i0] & mask) - (h[i0] & m ? 0.f : ml);
        float val1 = dl * float(q[i1] & mask) - (h[i1] & m ? 0.f : ml);

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_q3_K {
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
        load_a_q3_K_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01,
                             block_idx, sub_block, num_blocks, nb01);
    }

    void advance() {
        sub_block++;
        if (sub_block == 8) {
            sub_block = 0;
            block_idx++;
        }
    }
};
