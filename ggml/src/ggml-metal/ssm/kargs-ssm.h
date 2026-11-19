#ifndef GGML_METAL_KARGS_SSM
#define GGML_METAL_KARGS_SSM

// kernel argument structs of the ssm family (included by ggml-metal-impl.h)

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    int32_t  has_state_dst;  // 1 if src[2] provides direct conv state write target
} ggml_metal_kargs_ssm_conv;

typedef struct {
    int32_t  nc;        // conv width (taps); the state holds nc - 1 columns
    int32_t  channels;
    uint64_t nbs1;      // state channel stride (bytes)
    uint64_t nbx1;      // input channel stride (bytes)
    uint64_t nbw1;      // weight channel stride (bytes)
} ggml_metal_kargs_conv_state;

// conv_state + the split_l2 of its q and k heads in one launch (hw row conv_l2_fold)
typedef struct {
    ggml_metal_kargs_conv_state cs;
    int32_t  head_dim;   // channels per head (one threadgroup per head)
    int32_t  rows[2];    // q heads, k heads (k heads follow the q heads in the conv output)
    int32_t  ne00[2];    // split_l2's ne00 (the mean's divisor)
    float    eps[2];
    float    scale[2];
    uint64_t nb1[2];     // out row strides, bytes
} ggml_metal_kargs_conv_state_l2;

typedef struct {
    int64_t  d_state;
    int64_t  d_inner;
    int64_t  n_head;
    int64_t  n_group;
    int64_t  n_seq_tokens;
    int64_t  n_seqs;
    uint64_t s_off;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t ns12;
    uint64_t nb13;
    uint64_t nb20;
    uint64_t nb21;
    uint64_t ns21;
    uint64_t nb22;
    int64_t  ne30;
    uint64_t nb31;
    uint64_t nb41;
    uint64_t nb42;
    uint64_t ns42;
    uint64_t nb43;
    uint64_t nb51;
    uint64_t nb52;
    uint64_t ns52;
    uint64_t nb53;
    uint64_t nb0;
} ggml_metal_kargs_ssm_scan;

// Current upstream GDN tensor/stride/snapshot contract (4e5116f47, MIT).
typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne22;
    int32_t  ne23;
    uint64_t nb20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ns02;
    int32_t  ns12;
    int32_t  ns22;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t nb_out; // 0 => snapshots are appended after the attn scores (unfused)
} ggml_metal_kargs_gated_delta_net;

typedef struct {
    int32_t  head_size;
    int32_t  n_head;
    int32_t  n_seq_tokens;
    int32_t  n_seqs;
    float    scale;
} ggml_metal_kargs_gla;

#endif // GGML_METAL_KARGS_SSM
