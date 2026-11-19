#ifndef GGML_METAL_KARGS_ACTIVATION
#define GGML_METAL_KARGS_ACTIVATION

// kernel argument structs of the activation family (included by ggml-metal-impl.h)

typedef struct{
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
} ggml_metal_kargs_glu;

#endif // GGML_METAL_KARGS_ACTIVATION
