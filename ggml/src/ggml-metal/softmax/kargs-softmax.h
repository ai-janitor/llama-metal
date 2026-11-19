#ifndef GGML_METAL_KARGS_SOFTMAX
#define GGML_METAL_KARGS_SOFTMAX

// kernel argument structs of the softmax family (included by ggml-metal-impl.h)

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
} ggml_metal_kargs_soft_max;

#endif // GGML_METAL_KARGS_SOFTMAX
