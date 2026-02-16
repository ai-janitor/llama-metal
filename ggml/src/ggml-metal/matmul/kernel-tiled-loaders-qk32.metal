// Tiled matmul A-matrix loaders for QK=32 types (f32, bf16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl).
// Included by kernel-tiled.metal. Each loader cooperatively fills shared memory with BK=32
// dequantized float2 elements per K-tile. One quant block = one K-tile for all QK=32 types.
//
// Pattern: block-pointer style (like mxfp4). byte pointer, block_k counter, dequant per element.
// Exception: f32 and bf16 use typed-pointer style (like f16) since they need no dequant.

// ---------------------------------------------------------------------------
// f32: typed pointer, element-based stride (pattern A)
// ---------------------------------------------------------------------------

inline void load_a_f32_to_shmem(
    device const float * data_a,
    threadgroup float2 * buf_a,
    uint pos_a,
    uint loadr,
    uint loadc,
    uint idx_m,
    uint ne01,
    uint block,
    uint end_k,
    uint stride_a
) {
    const uint idx = pos_a + loadc * stride_a + loadr * LOAD_VEC_A;
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block + loadr * LOAD_VEC_A + 1 < end_k) {
        buf_a[buf_idx] = float2(data_a[idx], data_a[idx + 1]);
    } else if (idx_m < ne01 && block + loadr * LOAD_VEC_A < end_k) {
        buf_a[buf_idx] = float2(data_a[idx], 0.0f);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_f32 {
    device const float * data_a;
    uint stride_a;
    uint pos_a;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        data_a = (device const float *)(src0 + offset0 + args.nb01 * (ir * BM));
        stride_a = args.nb01 / sizeof(float);
        pos_a = 0;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_f32_to_shmem(data_a, buf_a, pos_a, loadr, loadc, idx_m, ne01, block, end_k, stride_a);
    }

    void advance() { pos_a += BK; }
};

// ---------------------------------------------------------------------------
// bf16: typed pointer, element-based stride (pattern A)
// ---------------------------------------------------------------------------

#if defined(GGML_METAL_HAS_BF16)

inline void load_a_bf16_to_shmem(
    device const bfloat * data_a,
    threadgroup float2 * buf_a,
    uint pos_a,
    uint loadr,
    uint loadc,
    uint idx_m,
    uint ne01,
    uint block,
    uint end_k,
    uint stride_a
) {
    const uint idx = pos_a + loadc * stride_a + loadr * LOAD_VEC_A;
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block + loadr * LOAD_VEC_A + 1 < end_k) {
        buf_a[buf_idx] = float2(float(data_a[idx]), float(data_a[idx + 1]));
    } else if (idx_m < ne01 && block + loadr * LOAD_VEC_A < end_k) {
        buf_a[buf_idx] = float2(float(data_a[idx]), 0.0f);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_bf16 {
    device const bfloat * data_a;
    uint stride_a;
    uint pos_a;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        data_a = (device const bfloat *)(src0 + offset0 + args.nb01 * (ir * BM));
        stride_a = args.nb01 / sizeof(bfloat);
        pos_a = 0;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_bf16_to_shmem(data_a, buf_a, pos_a, loadr, loadc, idx_m, ne01, block, end_k, stride_a);
    }

    void advance() { pos_a += BK; }
};

#endif // GGML_METAL_HAS_BF16

// ---------------------------------------------------------------------------
// q8_0: block pointer, scale-only dequant (pattern B)
// ---------------------------------------------------------------------------

inline void load_a_q8_0_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr,
    uint loadc,
    uint idx_m,
    uint ne01,
    uint block_k,
    uint num_blocks,
    uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_k < num_blocks) {
        device const block_q8_0 * block_ptr = (device const block_q8_0 *)(src0_row + loadc * nb01) + block_k;

        const float d = float(block_ptr->d);

        // loadr 0..15 → elements 0..31 (2 consecutive int8 values)
        const int8_t val0 = block_ptr->qs[loadr * 2];
        const int8_t val1 = block_ptr->qs[loadr * 2 + 1];

        buf_a[buf_idx] = float2(d * float(val0), d * float(val1));
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_q8_0 {
    device const char * src0_row;
    uint num_blocks;
    uint block_k;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK8_0;
        block_k = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_q8_0_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_k, num_blocks, nb01);
    }

    void advance() { block_k++; }
};

// ---------------------------------------------------------------------------
// q4_0: nibble packing, scale + offset dequant (pattern B)
// ---------------------------------------------------------------------------

inline void load_a_q4_0_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr,
    uint loadc,
    uint idx_m,
    uint ne01,
    uint block_k,
    uint num_blocks,
    uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_k < num_blocks) {
        device const block_q4_0 * block_ptr = (device const block_q4_0 *)(src0_row + loadc * nb01) + block_k;

        const float d = float(block_ptr->d);

        // Nibble packing: byte[i] = element[i] (low nibble) | element[i+16] (high nibble)
        // loadr 0..7: elements 0-15 from low nibbles
        // loadr 8..15: elements 16-31 from high nibbles
        if (loadr < 8) {
            const uint8_t byte0 = block_ptr->qs[2*loadr];
            const uint8_t byte1 = block_ptr->qs[2*loadr + 1];
            const float val0 = d * float(byte0 & 0x0F) - 8.0f * d;
            const float val1 = d * float(byte1 & 0x0F) - 8.0f * d;
            buf_a[buf_idx] = float2(val0, val1);
        } else {
            const uint8_t byte0 = block_ptr->qs[2*(loadr - 8)];
            const uint8_t byte1 = block_ptr->qs[2*(loadr - 8) + 1];
            const float val0 = d * float(byte0 >> 4) - 8.0f * d;
            const float val1 = d * float(byte1 >> 4) - 8.0f * d;
            buf_a[buf_idx] = float2(val0, val1);
        }
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_q4_0 {
    device const char * src0_row;
    uint num_blocks;
    uint block_k;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK4_0;
        block_k = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_q4_0_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_k, num_blocks, nb01);
    }

    void advance() { block_k++; }
};

// ---------------------------------------------------------------------------
// q4_1: nibble packing, scale + min dequant (pattern B)
// ---------------------------------------------------------------------------

inline void load_a_q4_1_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr,
    uint loadc,
    uint idx_m,
    uint ne01,
    uint block_k,
    uint num_blocks,
    uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_k < num_blocks) {
        device const block_q4_1 * block_ptr = (device const block_q4_1 *)(src0_row + loadc * nb01) + block_k;

        const float d = float(block_ptr->d);
        const float m = float(block_ptr->m);

        if (loadr < 8) {
            const uint8_t byte0 = block_ptr->qs[2*loadr];
            const uint8_t byte1 = block_ptr->qs[2*loadr + 1];
            const float val0 = d * float(byte0 & 0x0F) + m;
            const float val1 = d * float(byte1 & 0x0F) + m;
            buf_a[buf_idx] = float2(val0, val1);
        } else {
            const uint8_t byte0 = block_ptr->qs[2*(loadr - 8)];
            const uint8_t byte1 = block_ptr->qs[2*(loadr - 8) + 1];
            const float val0 = d * float(byte0 >> 4) + m;
            const float val1 = d * float(byte1 >> 4) + m;
            buf_a[buf_idx] = float2(val0, val1);
        }
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_q4_1 {
    device const char * src0_row;
    uint num_blocks;
    uint block_k;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK4_1;
        block_k = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_q4_1_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_k, num_blocks, nb01);
    }

    void advance() { block_k++; }
};

// ---------------------------------------------------------------------------
// q5_0: 5-bit values with high bit from qh, scale + offset (pattern B)
// ---------------------------------------------------------------------------

inline void load_a_q5_0_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr,
    uint loadc,
    uint idx_m,
    uint ne01,
    uint block_k,
    uint num_blocks,
    uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_k < num_blocks) {
        device const block_q5_0 * block_ptr = (device const block_q5_0 *)(src0_row + loadc * nb01) + block_k;

        const float d = float(block_ptr->d);

        // Reconstruct 32-bit qh from 4 bytes
        const uint32_t qh32 = ((uint32_t)block_ptr->qh[0])
                            | ((uint32_t)block_ptr->qh[1] << 8)
                            | ((uint32_t)block_ptr->qh[2] << 16)
                            | ((uint32_t)block_ptr->qh[3] << 24);

        // Elements: (loadr*2, loadr*2+1)
        const uint elem0 = loadr * 2;
        const uint elem1 = loadr * 2 + 1;

        uint8_t nibble0, nibble1;

        if (loadr < 8) {
            // Low nibbles
            nibble0 = block_ptr->qs[2*loadr] & 0x0F;
            nibble1 = block_ptr->qs[2*loadr + 1] & 0x0F;
        } else {
            // High nibbles
            nibble0 = block_ptr->qs[2*(loadr - 8)] >> 4;
            nibble1 = block_ptr->qs[2*(loadr - 8) + 1] >> 4;
        }

        // Extract 5th bit from qh32
        const uint8_t bit5_0 = (qh32 >> elem0) & 1;
        const uint8_t bit5_1 = (qh32 >> elem1) & 1;

        // Assemble 5-bit value
        const uint8_t val5bit_0 = nibble0 | (bit5_0 << 4);
        const uint8_t val5bit_1 = nibble1 | (bit5_1 << 4);

        // Dequant: d * val - 16*d
        const float val0 = d * float(val5bit_0) - 16.0f * d;
        const float val1 = d * float(val5bit_1) - 16.0f * d;

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_q5_0 {
    device const char * src0_row;
    uint num_blocks;
    uint block_k;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK5_0;
        block_k = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_q5_0_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_k, num_blocks, nb01);
    }

    void advance() { block_k++; }
};

// ---------------------------------------------------------------------------
// q5_1: 5-bit values with high bit from qh, scale + min (pattern B)
// ---------------------------------------------------------------------------

inline void load_a_q5_1_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr,
    uint loadc,
    uint idx_m,
    uint ne01,
    uint block_k,
    uint num_blocks,
    uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_k < num_blocks) {
        device const block_q5_1 * block_ptr = (device const block_q5_1 *)(src0_row + loadc * nb01) + block_k;

        const float d = float(block_ptr->d);
        const float m = float(block_ptr->m);

        const uint32_t qh32 = ((uint32_t)block_ptr->qh[0])
                            | ((uint32_t)block_ptr->qh[1] << 8)
                            | ((uint32_t)block_ptr->qh[2] << 16)
                            | ((uint32_t)block_ptr->qh[3] << 24);

        const uint elem0 = loadr * 2;
        const uint elem1 = loadr * 2 + 1;

        uint8_t nibble0, nibble1;

        if (loadr < 8) {
            nibble0 = block_ptr->qs[2*loadr] & 0x0F;
            nibble1 = block_ptr->qs[2*loadr + 1] & 0x0F;
        } else {
            nibble0 = block_ptr->qs[2*(loadr - 8)] >> 4;
            nibble1 = block_ptr->qs[2*(loadr - 8) + 1] >> 4;
        }

        const uint8_t bit5_0 = (qh32 >> elem0) & 1;
        const uint8_t bit5_1 = (qh32 >> elem1) & 1;

        const uint8_t val5bit_0 = nibble0 | (bit5_0 << 4);
        const uint8_t val5bit_1 = nibble1 | (bit5_1 << 4);

        // Dequant: d * val + m
        const float val0 = d * float(val5bit_0) + m;
        const float val1 = d * float(val5bit_1) + m;

        buf_a[buf_idx] = float2(val0, val1);
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_q5_1 {
    device const char * src0_row;
    uint num_blocks;
    uint block_k;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK5_1;
        block_k = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_q5_1_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_k, num_blocks, nb01);
    }

    void advance() { block_k++; }
};

// ---------------------------------------------------------------------------
// iq4_nl: nibble packing with LUT dequant (pattern B)
// ---------------------------------------------------------------------------

inline void load_a_iq4_nl_to_shmem(
    device const char * src0_row,
    threadgroup float2 * buf_a,
    uint loadr,
    uint loadc,
    uint idx_m,
    uint ne01,
    uint block_k,
    uint num_blocks,
    uint nb01
) {
    const uint buf_idx = loadc * SHMEM_STRIDE + loadr;

    if (idx_m < ne01 && block_k < num_blocks) {
        device const block_iq4_nl * block_ptr = (device const block_iq4_nl *)(src0_row + loadc * nb01) + block_k;

        const float d = float(block_ptr->d);

        if (loadr < 8) {
            const uint8_t byte0 = block_ptr->qs[2*loadr];
            const uint8_t byte1 = block_ptr->qs[2*loadr + 1];
            const float val0 = d * kvalues_iq4nl_f[byte0 & 0x0F];
            const float val1 = d * kvalues_iq4nl_f[byte1 & 0x0F];
            buf_a[buf_idx] = float2(val0, val1);
        } else {
            const uint8_t byte0 = block_ptr->qs[2*(loadr - 8)];
            const uint8_t byte1 = block_ptr->qs[2*(loadr - 8) + 1];
            const float val0 = d * kvalues_iq4nl_f[byte0 >> 4];
            const float val1 = d * kvalues_iq4nl_f[byte1 >> 4];
            buf_a[buf_idx] = float2(val0, val1);
        }
    } else {
        buf_a[buf_idx] = float2(0.0f);
    }
}

struct tiled_loader_iq4_nl {
    device const char * src0_row;
    uint num_blocks;
    uint block_k;
    uint nb01;

    void init(device const char * src0, uint64_t offset0, uint ir,
              constant ggml_metal_kargs_mul_mm & args) {
        src0_row = src0 + offset0 + args.nb01 * (ir * BM);
        num_blocks = args.ne00 / QK4_NL;
        block_k = 0;
        nb01 = args.nb01;
    }

    void load(threadgroup float2 * buf_a, uint loadr, uint loadc,
              uint idx_m, uint ne01, uint block, uint end_k) {
        load_a_iq4_nl_to_shmem(src0_row, buf_a, loadr, loadc, idx_m, ne01, block_k, num_blocks, nb01);
    }

    void advance() { block_k++; }
};
