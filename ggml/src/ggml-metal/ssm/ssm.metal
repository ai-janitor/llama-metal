#include "00-common.metal"

// ref: ggml.c:ggml_compute_forward_ssm_conv_f32
kernel void kernel_ssm_conv_f32_f32(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        device       float * state_dst, // direct conv state write target (or unused)
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;  // channel
    const int64_t i2 = tgpig.y;  // token position
    const int64_t i3 = tgpig.z;  // sequence

    const int64_t nc  = args.ne10;  // d_conv (kernel size)
    const int64_t ncs = args.ne00;  // conv window size (d_conv - 1 + n_tokens)

    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);
    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);
    device       float * x = (device       float *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = sumf;

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

kernel void kernel_ssm_conv_f32_f32_4(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        device       float * state_dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;  // channel
    const int64_t i2 = tgpig.y;  // token position
    const int64_t i3 = tgpig.z;  // sequence

    const int64_t nc  = args.ne10;  // d_conv
    const int64_t ncs = args.ne00;  // conv window size

    device const float4 * s = (device const float4 *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);
    device const float4 * c = (device const float4 *) ((device const char *) src1 + ir*args.nb11);
    device       float  * x = (device       float  *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    x[0] = sumf;

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

    x[0] = sumf;
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

    x[0] = sumf;
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

// Fused gated delta-net recurrence kernel.
// Replaces 16 elementwise graph ops per SSM layer with one GPU dispatch.
// State stays in registers — no intermediate device memory traffic between steps.
//
// Architecture (from arc-b60 kernel_gated_delta_net):
//   1 threadgroup per (state_row, head) — each TG owns one row of the S×S state matrix.
//   32 threads (1 simdgroup) collaborate on the S_v-wide dot products via simd_sum.
//   Per-token loop: decay → dot(state,k) → delta → state update → dot(state,q)
//
// Buffer layout:
//   src0 = k  [S, H, T]       (contiguous f32)
//   src1 = v  [S, H, T]       (contiguous f32)
//   src2 = q  [S, H, T]       (contiguous f32)
//   src3 = gate [1, 1, H, B]  (log decay per head, contiguous f32)
//   src4 = beta [1, 1, H, B]  (gating scalar per head, contiguous f32)
//   src5 = state [S, S, H, B] (contiguous f32, mutated in-place via output packing)
//   dst  = packed [S*H, T + S*B] (output rows then new_state rows)
kernel void kernel_gated_delta_net(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const float * src0,      // k
        device const float * src1,      // v
        device const float * src2,      // q
        device const float * src3,      // gate (log decay)
        device const float * src4,      // beta
        device const float * src5,      // state (input)
        device       float * dst,       // output (+ new_state if no state_dst)
        device       float * state_dst, // direct state write target (or unused)
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint   tx   [[thread_index_in_threadgroup]]) {

    const int S = args.S;           // state dimension
    const int H = args.H;           // value heads
    const int T = args.n_tokens;    // tokens in batch
    const int H_k = args.H_k;      // key heads (GQA)
    const int group_ratio = H / H_k;

    const int row  = (int)tgpig.x;  // which row of the S×S state matrix this TG owns
    const int head = (int)tgpig.y;  // which head

    if (head >= H || row >= S) return;

    // Elements per thread — each thread handles NSG consecutive state columns.
    // With TG=32 and NSG=4: 32*4=128 elements max. S_v for Qwen3.5 is 128.
    constexpr int NSG = 4;

    // Load this state row into registers.
    // State layout: [S_v, S_v, H_v, n_seqs] — row `row` of head `head`:
    //   state_base = (seq * H + head) * S * S + row * S
    // For single-seq (n_seqs=1): (head * S + row) * S
    const int state_row_base = (head * S + row) * S;
    float ls[NSG];
    for (int j = 0; j < NSG; j++) {
        const int col = (int)tx * NSG + j;
        ls[j] = (col < S) ? src5[state_row_base + col] : 0.0f;
    }

    // INCREMENTAL TEST: full computation but NO state update (ls[j] += k*d is skipped)
    for (int t = 0; t < T; t++) {
        const int gh_idx = t * H + head;
        const int k_head = head / group_ratio;
        const int qk_off = (t * H_k + k_head) * S;

        // Step 1: decay
        const float decay = exp(src3[gh_idx]);

        // Step 2: dot(state, k)
        float dot_state_k = 0.0f;
        for (int j = 0; j < NSG; j++) {
            const int col = (int)tx * NSG + j;
            if (col < S) {
                ls[j] *= decay;
                dot_state_k += ls[j] * src0[qk_off + col];
            }
        }
        dot_state_k = simd_sum(dot_state_k);

        // Step 3: delta = (v - dot(state,k)) * beta
        const float v_val = src1[gh_idx * S + row];
        const float beta_val = src4[gh_idx];
        const float delta = (v_val - dot_state_k) * beta_val;

        // Step 4+5: state update + output (FUSED in one loop, like original)
        float dot_state_q = 0.0f;
        for (int j = 0; j < NSG; j++) {
            const int col = (int)tx * NSG + j;
            if (col < S) {
                ls[j] += src0[qk_off + col] * delta;
                dot_state_q += ls[j] * src2[qk_off + col];
            }
        }
        dot_state_q = simd_sum(dot_state_q);

        if (tx == 0) {
            dst[(int64_t)t * S * H + head * S + row] = dot_state_q * args.scale;
        }
    }

    // Write state row back. If state_dst is provided (has_state_dst == 1),
    // write directly to the cache buffer. Otherwise, pack into dst after output rows.
    if (args.has_state_dst) {
        // Direct cache write: state_dst has same layout as src5 [S, S, H, n_seqs]
        for (int j = 0; j < NSG; j++) {
            const int col = (int)tx * NSG + j;
            if (col < S) {
                state_dst[state_row_base + col] = ls[j];
            }
        }
    } else {
        // Legacy packed layout: state goes after T output rows in dst
        const int64_t packed_base = (int64_t)T * S * H + state_row_base;
        for (int j = 0; j < NSG; j++) {
            const int col = (int)tx * NSG + j;
            if (col < S) {
                dst[packed_base + col] = ls[j];
            }
        }
    }
}
