#include "00-common.metal"

// INTENT: apply rotations at the op-specified offset.
// PDL: use offset-relative indices within the rotation interval.
// Reference: ggml-org/llama.cpp 4e5116f47, kernels/rope.metal (MIT).

constant bool FC_rope_is_imrope [[function_constant(FC_ROPE + 0)]];
// ROPE_BACK is the same rotation with the sine negated.
constant bool FC_rope_is_back_v [[function_constant(FC_ROPE + 1)]];
constant bool FC_rope_is_back = is_function_constant_defined(FC_rope_is_back_v) ? FC_rope_is_back_v : false;

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
    if (FC_rope_is_back) {
        *sin_theta *= -1.0f;
    }
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, bool truncate, float dims[2]
) {
    // start and end correction dims
    float start = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base);
    float end   = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base);
    if (truncate) {
        start = floor(start);
        end   = ceil(end);
    }
    dims[0] = max(0.0f,          start);
    dims[1] = min(n_dims - 1.0f, end);
}

// The four rotary layouts, one kernel: the mode picks the position, the pair and the angle at compile time.
enum { rope_mode_norm = 0, rope_mode_neox = 1, rope_mode_multi = 2, rope_mode_vision = 3 };

template<typename T, short MODE>
kernel void kernel_rope(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    // the correction dims are read only when ext_factor != 0
    float corr_dims[2] = { 0.0f, 0.0f };
    if (args.ext_factor != 0.0f) {
        rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, args.no_trunc == 0, corr_dims);
    }

    device const int32_t * pos = (device const int32_t *) src1;

    float theta_base_tok = 0.0f;
    if (MODE == rope_mode_norm || MODE == rope_mode_neox) {
        theta_base_tok = (float) pos[i2];
    }
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        bool rotated;
        if (MODE == rope_mode_vision) {
            rotated = i0 < 2*args.n_dims;
        } else {
            rotated = i0 >= args.n_offs && i0 < args.n_offs + args.n_dims;
        }
        if (rotated) {
            int iw = 0;
            int ic;
            float theta;
            if (MODE == rope_mode_vision) {
                ic = i0/2;

                // mrope theta calculations (only support 2 dimensions)
                const int sect_dims = args.sect_0 + args.sect_1;
                const int sector    = ic % sect_dims;

                float p;
                float theta_base;
                if (sector < args.sect_1) {
                    p = (float) sector;
                    theta_base = (float) pos[i2];
                } else {
                    p = (float) sector - args.sect_0;
                    theta_base = (float) pos[i2 + args.ne02];
                }

                theta = theta_base * pow(args.freq_base, 2.0f * inv_ndims * p);
                // end of mrope
            } else {
                iw = i0 - args.n_offs;
                ic = iw/2;

                float theta_base = theta_base_tok;
                if (MODE == rope_mode_multi) {
                    // mrope theta calculations
                    const int sect_dims = args.sect_0 + args.sect_1 + args.sect_2 + args.sect_3;
                    const int sec_w01   = args.sect_0 + args.sect_1;               // end of section 1
                    const int sec_w012  = args.sect_0 + args.sect_1 + args.sect_2; // end of section 2
                    const int sector    = ic % sect_dims;

                    if (FC_rope_is_imrope) {
                        if (sector % 3 == 1 && sector < 3 * args.sect_1) { // h
                            theta_base = (float) pos[i2 + args.ne02 * 1];
                        } else if (sector % 3 == 2 && sector < 3 * args.sect_2) { // w
                            theta_base = (float) pos[i2 + args.ne02 * 2];
                        } else if (sector % 3 == 0 && sector < 3 * args.sect_0) { // t
                            theta_base = (float) pos[i2 + args.ne02 * 0];
                        } else { // e
                            theta_base = (float) pos[i2 + args.ne02 * 3];
                        }
                    } else {
                        if (sector < args.sect_0) {
                            theta_base = (float) pos[i2];
                        } else if (sector < sec_w01) {
                            theta_base = (float) pos[i2 + args.ne02 * 1];
                        } else if (sector < sec_w012) {
                            theta_base = (float) pos[i2 + args.ne02 * 2];
                        } else {
                            theta_base = (float) pos[i2 + args.ne02 * 3];
                        }
                    }
                    // end of mrope
                }

                theta = theta_base * pow(args.freq_base, inv_ndims*iw);
            }

            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, MODE == rope_mode_vision ? i0 : iw, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            // the first element of the pair and the distance to the second
            int off;
            int d;
            if (MODE == rope_mode_norm) {
                off = i0;
                d   = 1;
            } else if (MODE == rope_mode_vision) {
                off = ic;
                d   = args.n_dims;
            } else {
                off = args.n_offs + ic;
                d   = args.n_dims/2;
            }

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + off*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + off*args.nb0);

            const float x0 = src[0];
            const float x1 = src[d];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[d] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

typedef decltype(kernel_rope<float, rope_mode_norm>) kernel_rope_t;

template [[host_name("kernel_rope_norm_f32")]] kernel kernel_rope_t kernel_rope<float, rope_mode_norm>;
template [[host_name("kernel_rope_norm_f16")]] kernel kernel_rope_t kernel_rope<half,  rope_mode_norm>;

template [[host_name("kernel_rope_neox_f32")]] kernel kernel_rope_t kernel_rope<float, rope_mode_neox>;
template [[host_name("kernel_rope_neox_f16")]] kernel kernel_rope_t kernel_rope<half,  rope_mode_neox>;

template [[host_name("kernel_rope_multi_f32")]] kernel kernel_rope_t kernel_rope<float, rope_mode_multi>;
template [[host_name("kernel_rope_multi_f16")]] kernel kernel_rope_t kernel_rope<half,  rope_mode_multi>;

template [[host_name("kernel_rope_vision_f32")]] kernel kernel_rope_t kernel_rope<float, rope_mode_vision>;
template [[host_name("kernel_rope_vision_f16")]] kernel kernel_rope_t kernel_rope<half,  rope_mode_vision>;
