#ifndef GGML_METAL_KARGS_MATMUL
#define GGML_METAL_KARGS_MATMUL

// kernel argument structs of the matmul family (included by ggml-metal-impl.h)

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    int32_t  tg_x_offset; // grid X offset for chunked dispatch (Intel timeout avoidance)
} ggml_metal_kargs_mul_mm;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
    int32_t  tg_x_offset; // grid X offset for chunked dispatch (Intel timeout avoidance)
    int32_t  tg_y_offset; // grid Y offset for 2D chunked dispatch
    int32_t  n_tg_x;      // grid X dimension — stride for persistent threadgroup loop (0 = non-persistent)
    int32_t  compute_units; // number of CUs on this GPU (for CU-local row clustering, 0 = disabled)
    int32_t  grid_x;       // total threadgroups in X dimension (needed for kernel-side CU remapping)
} ggml_metal_kargs_mul_mv;

typedef struct {
    uint64_t nb01_0;   // row stride of each matrix, a separate field each: the kernel picks one by compile-time K, never by index
    uint64_t nb01_1;
    uint64_t nb01_2;
    uint64_t nb01_3;
    int32_t  ne00;
    float    eps;
    int32_t  rows0;
    int32_t  rows1;
    int32_t  rows2;
    int32_t  rows3;
    int32_t  start1;   // first group of matrix 1, 2, 3 (groups walk matrix 0, then 1, ...)
    int32_t  start2;
    int32_t  start3;
} ggml_metal_kargs_norm_mul_mv;

typedef struct {
    int32_t  ne0;
} ggml_metal_kargs_add_scale;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    int32_t  tg_x_offset; // grid X offset for chunked dispatch (Intel timeout avoidance)
    int32_t  tg_y_offset; // grid Y offset for 2D chunked dispatch
} ggml_metal_kargs_mul_mv_ext;

typedef struct {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;  // n_expert_used (bcast)
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21; // n_tokens
    int32_t  ne20;  // n_expert_used
    uint64_t nb21;
} ggml_metal_kargs_mul_mm_id_map0;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mm_id;

typedef struct {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
} ggml_metal_kargs_mul_mv_id;

#endif // GGML_METAL_KARGS_MATMUL
