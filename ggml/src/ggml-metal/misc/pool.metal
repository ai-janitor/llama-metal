#include "00-common.metal"

// The fold of a pooling window: the starting value, one step per tap in the window, the result.
struct pool_2d_max_fold {
    float res;
    pool_2d_max_fold(constant ggml_metal_kargs_pool_2d &) : res(-INFINITY) {}
    void step(float cur) { res = MAX(res, cur); }
    float finish() { return res; }
};

struct pool_2d_avg_fold {
    float scale;
    float res;
    // const float scale = 1. / ((eh - bh) * (ew - bw));
    pool_2d_avg_fold(constant ggml_metal_kargs_pool_2d & args) : scale(1. / (args.k0 * args.k1)), res(0) {}
    void step(float cur) { res += cur * scale; }
    float finish() { return res; }
};

template<typename Fold>
kernel void kernel_pool_2d(
        constant    ggml_metal_kargs_pool_2d & args,
        device  const float * src0,
        device        float * dst,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const int idx = gid;
    const int I_HW = args.IH * args.IW;
    const int O_HW = args.OH * args.OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / args.OW;
    const int cur_ow = idx % O_HW % args.OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * args.s1 - args.p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(args.IH, start_h + args.k1);
    const int start_w = cur_ow * args.s0 - args.p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(args.IW, start_w + args.k0);

    Fold fold(args);

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            fold.step(i_ptr[i * args.IW + j]);
        }
    }

    o_ptr[cur_oh * args.OW + cur_ow] = fold.finish();
}

typedef decltype(kernel_pool_2d<pool_2d_max_fold>) kernel_pool_2d_t;

template [[host_name("kernel_pool_2d_max_f32")]] kernel kernel_pool_2d_t kernel_pool_2d<pool_2d_max_fold>;
template [[host_name("kernel_pool_2d_avg_f32")]] kernel kernel_pool_2d_t kernel_pool_2d<pool_2d_avg_fold>;

struct pool_1d_max_fold {
    float acc = -INFINITY;
    void step(float v) { acc = max(acc, v); }
    float finish() { return acc; }
};

// the mean of the taps inside the row; 0 when no tap is
struct pool_1d_avg_fold {
    float acc = 0.0f;
    int   cnt = 0;
    void step(float v) { acc += v; cnt += 1; }
    float finish() { return (cnt > 0) ? (acc / (float)cnt) : 0.0f; }
};

template<typename Fold>
kernel void kernel_pool_1d(
        constant        ggml_metal_kargs_pool_1d & args,
        device  const   float * src,
        device          float * dst,
        uint            gid [[thread_position_in_grid]]
) {

    if (gid >= args.np) {
        return;
    }

    const int ow  = (int)gid % args.OW;
    const int row = (int)gid / args.OW;

    const int base = ow * args.s0 - args.p0;

    Fold fold;

    const int src_off = row * args.IW;
    const int dst_off = row * args.OW;

    for (int ki = 0; ki < args.k0; ++ki) {
        const int j = base + ki;
        if (j < 0 || j >= args.IW) {
            continue;
        }
        fold.step(src[src_off + j]);
    }

    dst[dst_off + ow] = fold.finish();
}

typedef decltype(kernel_pool_1d<pool_1d_max_fold>) kernel_pool_1d_t;

template [[host_name("kernel_pool_1d_max_f32")]] kernel kernel_pool_1d_t kernel_pool_1d<pool_1d_max_fold>;
template [[host_name("kernel_pool_1d_avg_f32")]] kernel kernel_pool_1d_t kernel_pool_1d<pool_1d_avg_fold>;
