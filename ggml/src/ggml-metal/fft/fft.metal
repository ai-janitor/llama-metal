#include "00-common.metal"

// Naive O(n²) scalar IRFFT (Inverse Real FFT)
// Input: complex spectral data as interleaved [real0, imag0, real1, imag1, ...]
//        with N = n_fft/2+1 complex pairs (exploiting conjugate symmetry)
// Output: n_fft real-valued time-domain samples
// Algorithm: Direct DFT computation
//   out[k] = (1/N) * sum_{m=0}^{N-1} (real[m]*cos(2πkm/n) - imag[m]*sin(2πkm/n))
// One threadgroup per frame, each thread computes one or more output samples
kernel void kernel_irfft_f32(
    constant ggml_metal_kargs_irfft & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {

    const int64_t l = tgpig.x;  // frame index (0 to ne1-1)

    if (l >= args.ne1) {
        return;
    }

    const int64_t N = args.ne00 / 2;  // number of complex pairs (n_fft/2 + 1); ne00 is total floats (interleaved real/imag)
    const int64_t n = args.ne0;       // n_fft (output samples per frame)

    device const float * src0_ptr = (device const float *) (src0 + l*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  + l*args.nb1);

    // Each thread computes multiple output samples
    for (int k = tpitg.x; k < n; k += ntg.x) {
        float real_output = 0.0f;
        float imag_output = 0.0f;

        // Sum over all non-redundant complex frequency bins
        for (int m = 0; m < N; ++m) {
            float real_input = src0_ptr[2*m + 0];
            float imag_input = src0_ptr[2*m + 1];

            // Twiddle factor: exp(2πi*k*m/n) = cos(2πkm/n) + i*sin(2πkm/n)
            float angle = 2.0f * M_PI_F * k * m / n;
            float twiddle_real = cos(angle);
            float twiddle_imag = sin(angle);

            // Complex multiplication: (real_input + i*imag_input) * (twiddle_real + i*twiddle_imag)
            real_output += real_input * twiddle_real - imag_input * twiddle_imag;
            imag_output += real_input * twiddle_imag + imag_input * twiddle_real;
        }

        // IRFFT output is the real part, normalized by N
        dst_ptr[k] = real_output / N;
    }
}

// Overlap-add (fold) with Hann² window normalization
// Input src0: windowed frames [n_fft, n_codes]
// Input src1: Hann window [n_fft]
// Output: normalized audio [n_out - 2*n_pad]
// Each output sample is computed by:
//   1. For each overlapping frame: accumulate frame_val, accumulate hann[j]²
//   2. Divide accumulated signal by accumulated envelope
kernel void kernel_fold_f32(
    constant ggml_metal_kargs_fold & args,
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {

    const int64_t n_fft   = args.ne00;     // frame length (n_win)
    const int64_t n_codes = args.ne01;     // number of frames
    const int64_t n_out   = args.ne0;      // output length (after removing padding)
    const int64_t n_hop   = args.n_hop;    // hop size
    const int64_t n_pad   = args.n_pad;    // padding to remove

    device const float * frames = (device const float *) src0;
    device const float * hann   = (device const float *) src1;
    device       float * output = (device       float *) dst;

    // Each thread computes one output sample
    for (int64_t i = tpitg.x; i < n_out; i += ntg.x) {
        float audio_sum = 0.0f;
        float env_sum   = 0.0f;

        // Output position i corresponds to position (i + n_pad) in the padded domain
        int64_t p = i + n_pad;

        // Find all frames that overlap this output position
        // Frame l contributes to positions [l*n_hop, l*n_hop + n_fft)
        // So frame l overlaps position p if: l*n_hop <= p < l*n_hop + n_fft
        // Equivalently: (p - n_fft + 1) <= l*n_hop <= p
        //               (p - n_fft + 1)/n_hop <= l <= p/n_hop

        int64_t l_start = (p >= n_fft - 1) ? (p - n_fft + 1) / n_hop : 0;
        int64_t l_end   = p / n_hop;

        // Clamp to valid frame range
        if (l_start < 0) l_start = 0;
        if (l_end >= n_codes) l_end = n_codes - 1;

        for (int64_t l = l_start; l <= l_end; ++l) {
            // Position within frame l
            int64_t frame_start = l * n_hop;
            int64_t j = p - frame_start;

            // Bounds check
            if (j >= 0 && j < n_fft) {
                device const float * frame_ptr = frames + l*args.nb01/sizeof(float);
                float frame_val = frame_ptr[j];
                float window_val = hann[j];

                audio_sum += frame_val;
                env_sum   += window_val * window_val;
            }
        }

        // Normalize by envelope
        if (env_sum > 0.0f) {
            output[i] = audio_sum / env_sum;
        } else {
            output[i] = 0.0f;
        }
    }
}
