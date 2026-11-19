#include "00-common.metal"

// FU-MU-40, upstream 4e5116f47: retain channel packing and optional state destination.
constant bool FC_ssm_conv_silu [[function_constant(FC_SSM_CONV + 1)]];

// ref: ggml.c:ggml_compute_forward_ssm_conv_f32
// Grid mapping: each TG handles multiple channels (tgpig.x * ntg.x + tpitg.x).
// On D700/GCN (NW=64), TG=64 fills full wavefront instead of 1-thread TGs.
kernel void kernel_ssm_conv_f32_f32(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        device       float * state_dst, // direct conv state write target (or unused)
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = (int64_t)tgpig.x * ntg.x + tpitg.x;  // channel (packed across TG threads)
    const int64_t i2 = tgpig.y;  // token position
    const int64_t i3 = tgpig.z;  // sequence

    if (ir >= args.ne01) return;  // guard for ceil division padding

    const int64_t nc  = args.ne10;  // d_conv (kernel size)
    const int64_t ncs = args.ne00;  // conv window size (d_conv - 1 + n_tokens)

    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);
    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);
    device       float * x = (device       float *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = FC_ssm_conv_silu ? sumf/(1.0f + exp(-sumf)) : sumf;

    // Write conv state directly to cache if state_dst is provided.
    // The new state is the last (d_conv - 1) elements of the conv window for this channel.
    // Only the last token position writes state (avoid duplicate writes in batched mode).
    if (args.has_state_dst && i2 == args.ne1 - 1) {
        const int64_t state_width = nc - 1;  // d_conv - 1
        // Source: last (d_conv-1) positions in conv window for this channel/seq
        device const float * conv_tail = (device const float *) ((device const char *) src0
            + ir*args.nb01 + (ncs - state_width)*args.nb00 + i3*args.nb02);
        // Destination: state_dst layout is [d_conv-1, channels, n_seqs] contiguous
        device float * sd = state_dst + i3 * (state_width * args.ne01) + ir * state_width;
        for (int64_t i0 = 0; i0 < state_width; ++i0) {
            sd[i0] = conv_tail[i0];
        }
    }
}

// float4 vectorized variant — same channel-packing as above.
kernel void kernel_ssm_conv_f32_f32_4(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        device       float * state_dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = (int64_t)tgpig.x * ntg.x + tpitg.x;  // channel (packed across TG threads)
    const int64_t i2 = tgpig.y;  // token position
    const int64_t i3 = tgpig.z;  // sequence

    if (ir >= args.ne01) return;  // guard for ceil division padding

    const int64_t nc  = args.ne10;  // d_conv
    const int64_t ncs = args.ne00;  // conv window size

    device const float4 * s = (device const float4 *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);
    device const float4 * c = (device const float4 *) ((device const char *) src1 + ir*args.nb11);
    device       float  * x = (device       float  *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    x[0] = FC_ssm_conv_silu ? sumf/(1.0f + exp(-sumf)) : sumf;

    // Write conv state directly to cache if state_dst is provided.
    if (args.has_state_dst && i2 == args.ne1 - 1) {
        const int64_t state_width = nc - 1;
        device const float * conv_tail = (device const float *) ((device const char *) src0
            + ir*args.nb01 + (ncs - state_width)*args.nb00 + i3*args.nb02);
        device float * sd = state_dst + i3 * (state_width * args.ne01) + ir * state_width;
        for (int64_t i0 = 0; i0 < state_width; ++i0) {
            sd[i0] = conv_tail[i0];
        }
    }
}

// One decode token: conv + silu over (state, input) as kernel_ssm_conv_f32_f32[_4], then the state
// shifted by one into its cache cell. A thread owns one channel and reads its state before writing it.
template <bool V4>
static float kernel_ssm_conv_state_impl(
        constant ggml_metal_kargs_conv_state & args,
        device const char * state,
        device const char * input,
        device const char * weight,
        device       float * dst,
        device       float * cell,
        int c) {
    const int nc = args.nc;

    device const float * s = (device const float *) (state  + c*args.nbs1);
    device const float * w = (device const float *) (weight + c*args.nbw1);
    const float xin = *((device const float *) (input + c*args.nbx1));

    float sumf = 0.0f;
    if (V4) {
        for (int i0 = 0; i0 < nc/4; ++i0) {
            float4 win;
            for (short j = 0; j < 4; ++j) {
                const int i = 4*i0 + j;
                win[j] = i < nc - 1 ? s[i] : xin;
            }
            sumf += dot(win, ((device const float4 *) w)[i0]);
        }
    } else {
        for (int i0 = 0; i0 < nc; ++i0) {
            sumf += (i0 < nc - 1 ? s[i0] : xin) * w[i0];
        }
    }

    const float y = sumf/(1.0f + exp(-sumf));
    dst[c] = y;

    // in increasing order: column i is written only after column i + 1 was read
    device float * cs = cell + (int64_t) c*(nc - 1);
    for (int i = 0; i < nc - 2; ++i) {
        cs[i] = s[i + 1];
    }
    cs[nc - 2] = xin;
    return y;
}

kernel void kernel_ssm_conv_state_f32(
        constant ggml_metal_kargs_conv_state & args,
        device const char * state,
        device const char * input,
        device const char * weight,
        device       float * dst,
        device       float * cell,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int c = (int) tgpig.x*ntg.x + tpitg.x;
    if (c >= args.channels) return;
    kernel_ssm_conv_state_impl<false>(args, state, input, weight, dst, cell, c);
}

kernel void kernel_ssm_conv_state_f32_4(
        constant ggml_metal_kargs_conv_state & args,
        device const char * state,
        device const char * input,
        device const char * weight,
        device       float * dst,
        device       float * cell,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int c = (int) tgpig.x*ntg.x + tpitg.x;
    if (c >= args.channels) return;
    kernel_ssm_conv_state_impl<true>(args, state, input, weight, dst, cell, c);
}

// conv_state, then on q and k heads the split_l2 norm of kernel_small_split_l2_f32_4 on the values just
// computed: one threadgroup per head, float4 partials and the same power-of-two tree, same op order.
kernel void kernel_ssm_conv_state_l2_f32_4(
        constant ggml_metal_kargs_conv_state_l2 & args,
        device const char * state,
        device const char * input,
        device const char * weight,
        device       float * dst,
        device       float * cell,
        device       char  * out_q,
        device       char  * out_k,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int tid = tpitg.x;
    const int c = (int) tgpig.x*ntg.x + tid;
    const float yv = kernel_ssm_conv_state_impl<true>(args.cs, state, input, weight, dst, cell, c);

    const int head = tgpig.x;
    const int which = head < args.rows[0] ? 0 : 1;
    if (head >= args.rows[0] + args.rows[1]) {
        return; // v heads: conv only
    }
    const int row = head - (which ? args.rows[0] : 0);
    threadgroup float * xs   = shmem;
    threadgroup float * part = shmem + args.head_dim;
    xs[tid] = yv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int threads = args.head_dim/4;
    float4 xv = 0.0f;
    float sum = 0.0f;
    if (tid < threads) {
        xv = float4(xs[4*tid + 0], xs[4*tid + 1], xs[4*tid + 2], xs[4*tid + 3]);
        sum += dot(xv, xv);
        part[tid] = sum;
    }
    int np2 = 1;
    while (np2 < threads) np2 <<= 1;
    for (int i = threads + tid; i < np2 && tid < threads; i += threads) part[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int step = np2 / 2; step > 0; step >>= 1) {
        if (tid < step) {
            part[tid] += part[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum = part[0];
    const float mean  = sum / args.ne00[which];
    const float scale = 1.0f / sqrt(mean + args.eps[which]);
    if (tid < threads) {
        device float4 * y = (device float4 *) ((which ? out_k : out_q) + row*args.nb1[which]);
        y[tid] = (xv * scale) * args.scale[which];
    }
}

constant short FC_ssm_conv_bs   [[function_constant(FC_SSM_CONV + 0)]];

// Batched version: each threadgroup processes multiple tokens for better efficiency
// Thread layout: each thread handles one token, threadgroup covers BATCH_SIZE tokens
kernel void kernel_ssm_conv_f32_f32_batched(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        device       float * state_dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // tgpig.x = row index (ir)
    // tgpig.y = batch of tokens (i2_base / BATCH_SIZE)
    // tgpig.z = sequence index (i3)
    // tpitg.x = thread within batch (0..BATCH_SIZE-1)
    const short BATCH_SIZE = FC_ssm_conv_bs;

    const int64_t ir      = tgpig.x;
    const int64_t i2_base = tgpig.y * BATCH_SIZE;
    const int64_t i3      = tgpig.z;
    const int64_t i2_off  = tpitg.x;
    const int64_t i2      = i2_base + i2_off;

    const int64_t nc  = args.ne10;  // conv kernel size (typically 4)
    const int64_t n_t = args.ne1;   // number of tokens

    // Bounds check for partial batches at the end
    if (i2 >= n_t) {
        return;
    }

    // Load conv weights (shared across all tokens for this row)
    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);

    // Load source for this specific token
    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);

    // Output location for this token
    device float * x = (device float *) ((device char *) dst + ir*args.nb0 + i2*args.nb1 + i3*args.nb2);

    float sumf = 0.0f;
    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = FC_ssm_conv_silu ? sumf/(1.0f + exp(-sumf)) : sumf;
}

kernel void kernel_ssm_conv_f32_f32_batched_4(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        device       float * state_dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // tgpig.x = row index (ir)
    // tgpig.y = batch of tokens (i2_base / BATCH_SIZE)
    // tgpig.z = sequence index (i3)
    // tpitg.x = thread within batch (0..BATCH_SIZE-1)
    const short BATCH_SIZE = FC_ssm_conv_bs;

    const int64_t ir      = tgpig.x;
    const int64_t i2_base = tgpig.y * BATCH_SIZE;
    const int64_t i3      = tgpig.z;
    const int64_t i2_off  = tpitg.x;
    const int64_t i2      = i2_base + i2_off;

    const int64_t nc  = args.ne10;  // conv kernel size (typically 4)
    const int64_t n_t = args.ne1;   // number of tokens

    // Bounds check for partial batches at the end
    if (i2 >= n_t) {
        return;
    }

    // Load conv weights (shared across all tokens for this row)
    device const float4 * c = (device const float4 *) ((device const char *) src1 + ir*args.nb11);

    // Load source for this specific token
    device const float4 * s = (device const float4 *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);

    // Output location for this token
    device float * x = (device float *) ((device char *) dst + ir*args.nb0 + i2*args.nb1 + i3*args.nb2);

    float sumf = 0.0f;
    for (int64_t i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    x[0] = FC_ssm_conv_silu ? sumf/(1.0f + exp(-sumf)) : sumf;
}

// ref: ggml.c:ggml_compute_forward_ssm_scan_f32, Mamba-2 part
// Optimized version: reduces redundant memory loads by having one thread load shared values
kernel void kernel_ssm_scan_f32(
        constant ggml_metal_kargs_ssm_scan & args,
        device const void * src0,
        device const void * src1,
        device const void * src2,
        device const void * src3,
        device const void * src4,
        device const void * src5,
        device const void * src6,
        device      float * dst,
        threadgroup float * shared [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgptg[[simdgroups_per_threadgroup]],
        uint3    tgpg[[threadgroups_per_grid]],
        ushort simd_width[[threads_per_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const ushort NW = sgptg;

    // Shared memory layout:
    // [0..sgptg*NW-1]: partial sums for reduction (existing)
    // [sgptg*NW..sgptg*NW+sgptg-1]: pre-computed x_dt values for each token in batch
    // [sgptg*NW+sgptg..sgptg*NW+2*sgptg-1]: pre-computed dA values for each token in batch
    threadgroup float * shared_sums = shared;
    threadgroup float * shared_x_dt = shared + sgptg * NW;
    threadgroup float * shared_dA   = shared + sgptg * NW + sgptg;

    for (ushort i = tpitg.x; i < sgptg * NW; i += ntg.x) {
        shared_sums[i] = 0.0f;
    }

    const int32_t i0 = tpitg.x;
    const int32_t i1 = tgpig.x;
    const int32_t ir = tgpig.y; // current head
    const int32_t i3 = tgpig.z; // current seq

    const int32_t nc  = args.d_state;
    const int32_t nr  = args.d_inner;
    const int32_t nh  = args.n_head;
    const int32_t ng  = args.n_group;
    const int32_t n_t = args.n_seq_tokens;

    const int32_t s_off = args.s_off;

    device const int32_t * ids = (device const int32_t *) src6;

    device const float * s0_buff = (device const float *) ((device const char *) src0 + ir*args.nb02 + ids[i3]*args.nb03);
    device       float * s_buff  = (device       float *) ((device       char *) dst  + ir*args.nb02 +      i3*args.nb03 + s_off);

    const int32_t i = i0 + i1*nc;
    const int32_t g = ir / (nh / ng); // repeat_interleave

    float s0 = s0_buff[i];
    float s  = 0.0f;

    device const float * A = (device const float *) ((device const char *) src3 + ir*args.nb31); // {ne30, nh}

    const float A0 = A[i0%args.ne30];

    device const float * x  = (device const float *)((device const char *) src1 + i1*args.nb10  + ir*args.nb11 + i3*args.nb13); // {dim, nh, nt, ns}
    device const float * dt = (device const float *)((device const char *) src2 + ir*args.nb20  + i3*args.nb22);                // {nh, nt, ns}
    device const float * B  = (device const float *)((device const char *) src4 +  g*args.nb41  + i3*args.nb43);                // {d_state, ng, nt, ns}
    device const float * C  = (device const float *)((device const char *) src5 +  g*args.nb51  + i3*args.nb53);                // {d_state, ng, nt, ns}

    device float * y = dst + (i1 + ir*(nr) + i3*(n_t*nh*nr)); // {dim, nh, nt, ns}

    for (int i2 = 0; i2 < n_t; i2 += sgptg) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pre-compute x_dt and dA for this batch of tokens
        // Only first sgptg threads do the loads and expensive math
        if (i0 < sgptg && i2 + i0 < n_t) {
            // ns12 and ns21 are element strides (nb12/nb10, nb21/nb20)
            device const float * x_t  = x  + i0 * args.ns12;
            device const float * dt_t = dt + i0 * args.ns21;

            const float dt0  = dt_t[0];
            const float dtsp = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;
            shared_x_dt[i0] = x_t[0] * dtsp;
            shared_dA[i0]   = dtsp;  // Store dtsp, compute exp(dtsp * A0) per-thread since A0 varies
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < sgptg && i2 + t < n_t; t++) {
            const float x_dt = shared_x_dt[t];
            const float dA   = exp(shared_dA[t] * A0);

            s = (s0 * dA) + (B[i0] * x_dt);

            const float sumf = simd_sum(s * C[i0]);

            if (tiisg == 0) {
                shared_sums[t*NW + sgitg] = sumf;
            }

            // recurse
            s0 = s;

            B  += args.ns42;
            C  += args.ns52;
        }

        // Advance pointers for next batch
        x  += sgptg * args.ns12;
        dt += sgptg * args.ns21;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sumf = 0.0f;
        for (ushort i = tiisg; i < sgptg; i += simd_width) {
            sumf += shared_sums[sgitg*NW + i];
        }
        sumf = simd_sum(sumf);

        if (tiisg == 0 && i2 + sgitg < n_t) {
            y[sgitg*nh*nr] = sumf;
        }

        y += sgptg*nh*nr;
    }

    s_buff[i] = s;
}

kernel void kernel_rwkv_wkv6_f32(
    device const float * k,
    device const float * v,
    device const float * r,
    device const float * tf,
    device const float * td,
    device const float * state_in,
    device       float * dst,
    constant    uint & B,
    constant    uint & T,
    constant    uint & C,
    constant    uint & H,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]])  {

    const uint head_size = 64; // TODO: support head_size = 128
    const uint batch_id = tgpig.x / H;
    const uint head_id = tgpig.x % H;
    const uint tid = tpitg.x;

    if (batch_id >= B || head_id >= H) {
        return;
    }

    const uint state_size = C * head_size;
    const uint n_seq_tokens = T / B;

    threadgroup float _k[head_size];
    threadgroup float _r[head_size];
    threadgroup float _tf[head_size];
    threadgroup float _td[head_size];

    float state[head_size];

    for (uint i = 0; i < head_size; i++) {
        state[i] = state_in[batch_id * state_size + head_id * head_size * head_size
                          + i * head_size + tid];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    _tf[tid] = tf[head_id * head_size + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;
    const uint end_t = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float v_val = v[t];
        float y = 0.0;

        for (uint j = 0; j < head_size; j += 4) {
            float4 k_vec = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            float4 r_vec = float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            float4 tf_vec = float4(_tf[j], _tf[j+1], _tf[j+2], _tf[j+3]);
            float4 td_vec = float4(_td[j], _td[j+1], _td[j+2], _td[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv = k_vec * v_val;

            float4 temp = tf_vec * kv + s_vec;
            y += dot(r_vec, temp);

            s_vec = s_vec * td_vec + kv;
            state[j]   = s_vec[0];
            state[j+1] = s_vec[1];
            state[j+2] = s_vec[2];
            state[j+3] = s_vec[3];
        }

        dst[t] = y;
    }

    for (uint i = 0; i < head_size; i++) {
        dst[T * C + batch_id * state_size + head_id * head_size * head_size
            + i * head_size + tid] = state[i];
    }
}

kernel void kernel_rwkv_wkv7_f32(
    device const float * r,
    device const float * w,
    device const float * k,
    device const float * v,
    device const float * a,
    device const float * b,
    device const float * state_in,
    device       float * dst,
    constant    uint & B,
    constant    uint & T,
    constant    uint & C,
    constant    uint & H,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]])  {

    const uint head_size = 64; // TODO: support head_size = 128
    const uint batch_id = tgpig.x / H;
    const uint head_id = tgpig.x % H;
    const uint tid = tpitg.x;

    if (batch_id >= B || head_id >= H) {
        return;
    }

    const uint state_size = C * head_size;
    const uint n_seq_tokens = T / B;

    threadgroup float _r[head_size];
    threadgroup float _w[head_size];
    threadgroup float _k[head_size];
    threadgroup float _a[head_size];
    threadgroup float _b[head_size];

    float state[head_size];

    for (uint i = 0; i < head_size; i++) {
        state[i] = state_in[batch_id * state_size + head_id * head_size * head_size
                          + tid * head_size + i];
    }

    const uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;
    const uint end_t = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float v_val = v[t];
        float y = 0.0, sa = 0.0;

        float4 sa_vec(0.0);

        for (uint j = 0; j < head_size; j += 4) {
            float4 a_vec = float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa_vec += a_vec * s_vec;
        }
        sa = sa_vec[0] + sa_vec[1] + sa_vec[2] + sa_vec[3];

        for (uint j = 0; j < head_size; j += 4) {
            float4 r_vec = float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            float4 w_vec = float4(_w[j], _w[j+1], _w[j+2], _w[j+3]);
            float4 k_vec = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            float4 b_vec = float4(_b[j], _b[j+1], _b[j+2], _b[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv = k_vec * v_val;

            s_vec = s_vec * w_vec + kv + sa * b_vec;
            y += dot(s_vec, r_vec);

            state[j]   = s_vec[0];
            state[j+1] = s_vec[1];
            state[j+2] = s_vec[2];
            state[j+3] = s_vec[3];
        }

        dst[t] = y;
    }

    for (uint i = 0; i < head_size; i++) {
        dst[T * C + batch_id * state_size + head_id * head_size * head_size
            + tid * head_size + i] = state[i];
    }
}

// INTENT: implement the current q/k/v, gate, beta and K-snapshot GDN contract.
// PDL: preserve upstream recurrence and snapshots; the host derives the split from simd_width.
// Reference: ggml-org/llama.cpp 4e5116f47, kernels/gated_delta_net.metal (MIT).
// FU-MU-14: the upstream nb_out path now binds the cache directly when the host matches GDN + CPY.

constant short FC_gated_delta_net_ne20 [[function_constant(FC_GATED_DELTA_NET + 0)]];
constant short FC_gated_delta_net_ne30 [[function_constant(FC_GATED_DELTA_NET + 1)]];
constant short FC_gated_delta_net_K    [[function_constant(FC_GATED_DELTA_NET + 2)]];

template<short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B (n_seqs)
    const uint i21 = tgpig.y; // H (head)
    const uint i20 = tgpig.x*NSG + ty; // row within S_v

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    // input state layout [S_v, S_v, H, n_seqs] (s0 only): per-seq stride is H*D.
    // state is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous
    const uint state_in_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    device const float * s_ptr = (device const float *) (s) + state_in_base;

    float ls[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        ls[j] = s_ptr[is];
    }

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K, only slots 0..n_tokens-1 are written; older slots are caller-owned.

    // output state base offset: after attention scores
    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;
    // output state per-slot size: S_v * S_v * H * n_seqs
    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;
    // per-(seq,head) offset within a slot
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + attn_size;
    const uint slot_stride = fused ? (uint)args.nb_out : state_size_per_snap;

    for (short t = 0; t < args.ne22; t++) {
        float s_k = 0.0f;

        if (G == 1) {
            const float g_exp = exp(g_ptr[0]);

            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= g_exp;

                s_k += ls[j]*k_ptr[is];
            }
        } else {
            // KDA
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= exp(g_ptr[is]);

                s_k += ls[j]*k_ptr[is];
            }
        }

        s_k = simd_sum(s_k);

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        float y = 0.0f;

        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            ls[j] += k_ptr[is]*d;

            y += ls[j]*q_ptr[is];
        }

        y = simd_sum(y);

        if (tx == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        if (K > 1) {
            const int target_slot = (int)args.ne22 - 1 - (int)t;
            if (target_slot >= 0 && target_slot < (int)K) {
                device float * dst_state = (device float *)state_out + (uint)target_slot * slot_stride + state_out_base;
                FOR_UNROLL (short j = 0; j < NSG; j++) {
                    const short is = tx*NSG + j;
                    dst_state[is] = ls[j];
                }
            }
        }
    }

    if (K == 1) {
        device float * dst_state = (device float *)state_out + state_out_base;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            dst_state[is] = ls[j];
        }
    }

#undef S_v
#undef G
#undef K
}

typedef decltype(kernel_gated_delta_net_impl<4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<4>;

// Gated linear attention, ref: ggml-cpu ggml_compute_forward_gla_f32.
// One threadgroup per (sequence, head), one thread per column j of the head's state matrix.
kernel void kernel_gated_linear_attn_f32(
    constant ggml_metal_kargs_gla & args,
    device const float * k,
    device const float * v,
    device const float * q,
    device const float * g,
    device const float * state_in,
    device       float * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const int hs  = args.head_size;
    const int C   = hs*args.n_head;
    const int seq = tgpig.x / args.n_head;
    const int h   = tgpig.x % args.n_head;
    const int j   = tpitg.x;

    // column j of the state: state[i] = S[i][j]
    float state[128];

    const int s_off = seq*hs*C + h*hs*hs + j;

    for (int i = 0; i < hs; ++i) {
        state[i] = state_in[s_off + i*hs];
    }

    for (int t = 0; t < args.n_seq_tokens; ++t) {
        const int base = (seq*args.n_seq_tokens + t)*C + h*hs;

        const float v_val = v[base + j];

        float y = 0.0f;
        for (int i = 0; i < hs; ++i) {
            const float s = state[i]*g[base + i] + v_val*k[base + i];
            y += s*(q[base + i]*args.scale);
            state[i] = s;
        }

        dst[base + j] = y;
    }

    device float * state_out = dst + args.n_seqs*args.n_seq_tokens*C;

    for (int i = 0; i < hs; ++i) {
        state_out[s_off + i*hs] = state[i];
    }
}
