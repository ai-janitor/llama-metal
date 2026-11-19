#ifndef GGML_METAL_KARGS_FFT
#define GGML_METAL_KARGS_FFT

// kernel argument structs of the fft family (included by ggml-metal-impl.h)

typedef struct {
    int64_t  ne00;    // number of complex pairs per frame (n_fft/2+1)
    int64_t  ne01;    // number of frames (n_codes)
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;     // n_fft (output samples per frame)
    int64_t  ne1;     // number of frames
    uint64_t nb0;
    uint64_t nb1;
} ggml_metal_kargs_irfft;

typedef struct {
    int64_t  ne00;    // n_fft (frame length)
    int64_t  ne01;    // n_codes (number of frames)
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne10;    // window length
    uint64_t nb10;
    int64_t  ne0;     // output length
    uint64_t nb0;
    int32_t  n_hop;
    int32_t  n_pad;
} ggml_metal_kargs_fold;

#endif // GGML_METAL_KARGS_FFT
