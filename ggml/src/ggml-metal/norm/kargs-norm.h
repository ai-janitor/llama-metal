#ifndef GGML_METAL_KARGS_NORM
#define GGML_METAL_KARGS_NORM

// kernel argument structs of the norm family (included by ggml-metal-impl.h)

// NORM
// RMS_NORM
typedef struct {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
    float scale;
} ggml_metal_kargs_norm;

typedef struct {
    int32_t ne0;
    int32_t ne1;
    int32_t ne2;
    int32_t ne3;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_small_gate;

typedef struct {
    ggml_metal_kargs_norm norm[2];
    int32_t ne[2][3];
    uint64_t nbg1;
    uint64_t nbg2;
    uint64_t nbg3;
} ggml_metal_kargs_small_norm;

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
    float    eps;
} ggml_metal_kargs_l2_norm;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ngrp;
    float    eps;
} ggml_metal_kargs_group_norm;

#endif // GGML_METAL_KARGS_NORM
