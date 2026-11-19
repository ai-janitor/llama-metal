#ifndef GGML_METAL_KARGS_REDUCTION
#define GGML_METAL_KARGS_REDUCTION

// kernel argument structs of the reduction family (included by ggml-metal-impl.h)

typedef struct {
    uint64_t np;
} ggml_metal_kargs_sum;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_sum_rows;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  net0;
    int64_t  net1;
    int64_t  net2;
    int64_t  net3;
    uint64_t nbt0;
    uint64_t nbt1;
    uint64_t nbt2;
    uint64_t nbt3;
    bool     outb;
} ggml_metal_kargs_cumsum_blk;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  net0;
    int64_t  net1;
    int64_t  net2;
    int64_t  net3;
    uint64_t nbt0;
    uint64_t nbt1;
    uint64_t nbt2;
    uint64_t nbt3;
} ggml_metal_kargs_cumsum_add;

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
} ggml_metal_kargs_tri;

#endif // GGML_METAL_KARGS_REDUCTION
