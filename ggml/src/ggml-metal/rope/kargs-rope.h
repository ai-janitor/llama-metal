#ifndef GGML_METAL_KARGS_ROPE
#define GGML_METAL_KARGS_ROPE

// kernel argument structs of the rope family (included by ggml-metal-impl.h)

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_past;
    int32_t  n_dims;
    int32_t  n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    int32_t  sect_0;
    int32_t  sect_1;
    int32_t  sect_2;
    int32_t  sect_3;
    int32_t  n_offs; // shared host/kernel rotation offset, op param 15
    bool     src2;
    int32_t  no_trunc; // op param 3: correction dims are not rounded
} ggml_metal_kargs_rope;

#endif // GGML_METAL_KARGS_ROPE
