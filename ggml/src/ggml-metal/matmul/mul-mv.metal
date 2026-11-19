#ifndef GGML_METAL_MATMUL_MUL_MV_METAL
#define GGML_METAL_MATMUL_MUL_MV_METAL

// Matrix-vector product of a block-quantized weight, written once. Fmt is a format struct (mv-formats.metal); rows
// per simdgroup (NR0), activation columns per launch (NC) and the loop form are template values.

constant bool FC_mul_mv_residual [[function_constant(FC_MUL_MV_RESIDUAL)]];

// Columns per launch of the *_cols entry points (2, 3 or 4) and their loop form (bit 0 = weights held, bit 1 = packed,
// bit 2 = paired halves at 2 or 3 columns).
constant short FC_mul_mv_ncols_raw [[function_constant(FC_MUL_MV + 12)]];
constant short FC_mul_mv_ncols = is_function_constant_defined(FC_mul_mv_ncols_raw) ? FC_mul_mv_ncols_raw : 2;
constant short FC_mul_mv_col_form_raw [[function_constant(FC_MUL_MV + 13)]];
constant short FC_mul_mv_col_form = is_function_constant_defined(FC_mul_mv_col_form_raw) ? FC_mul_mv_col_form_raw : 0;

// One activation column. Lanes share a block in groups of Fmt::threads_per_block; activations are loaded once per
// block and reused over NR0 rows. A threadgroup walks row slices at stride n_tg_x (one pass when n_tg_x is 0).
// residual is read only when the format allows it and FC_mul_mv_residual is set; prep only by a prepared format.
// Format hooks: a lookup table in threadgroup memory (has_table), 32-element units inside a block (units_per_block),
// rows addressed by block count (rows_by_blocks), one row slice per threadgroup (persistent 0), a scale after the row sum.
template<typename Fmt, int NR0, typename args_t, short LANE>
static inline void mul_mv_row(
        args_t args,
        device const typename Fmt::block_t * x,
        int ib,
        short row,
        int ns01,
        int entry,
        thread const typename Fmt::fragment_t & owned,
        thread const typename Fmt::activation_t & prepared,
        thread typename Fmt::acc_t & acc,
        device const char * prep,
        threadgroup char * shmem) {
    const int ibl = ib / Fmt::units_per_block;
    // nb01 is a byte stride between rows; cast before applying it.
    device const typename Fmt::block_t * block = Fmt::rows_by_blocks
        ? x + (row*ns01 + ibl)
        : (device const typename Fmt::block_t *) ((device const char *) (x + ibl) + row*args.nb01);
    Fmt::template accumulate<LANE>(block, owned, prepared, acc, prep, entry, shmem, (short) (ib % Fmt::units_per_block));
}

template<typename Fmt, int NR0, typename args_t, short LANE>
void mul_mv_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const float * residual,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    // With the shared-memory reduce the lane and simdgroup come from the flat thread index.
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const short stride        = Fmt::stride(NW);
    const short block_id      = eff_tiisg / Fmt::threads_per_block;
    const short lane_in_block = eff_tiisg % Fmt::threads_per_block;

    const int nb = args.ne00/Fmt::block_elems;

    const int r1 = tgpig.y + args.tg_y_offset;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    const int rows_per_slice = NSG * NR0;
    const int total_slices = (args.ne01 + rows_per_slice - 1) / rows_per_slice;
    const int grid_stride = (Fmt::persistent && args.n_tg_x > 0) ? args.n_tg_x : total_slices;
    const int ns01 = args.nb01/args.nb00;

    const typename Fmt::fragment_t owned = Fmt::owned_fragment(lane_in_block);
    const bool add_residual = Fmt::has_residual && is_function_constant_defined(FC_mul_mv_residual) && FC_mul_mv_residual;

    if (Fmt::has_table) {
        Fmt::fill_table(shmem, eff_tiisg, eff_sgitg, NW, NSG);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int slice = (int)tgpig.x + args.tg_x_offset; slice < total_slices; slice += grid_stride) {

    const int first_row = (slice * NSG + eff_sgitg) * NR0;
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const typename Fmt::block_t * x = (device const typename Fmt::block_t *) (src0 + offset0);
    device const float                 * y = (device const float                 *) (src1 + offset1);

    typename Fmt::acc_t acc[NR0] = {};

    for (int ib = block_id; ib < nb && (!Fmt::rows_by_blocks || ib < ns01); ib += stride) {
        const typename Fmt::activation_t prepared = Fmt::template prepare_activation<LANE>(owned, y + ib*Fmt::block_elems);
        if (Fmt::unroll_rows) {
            FOR_UNROLL (short row = 0; row < NR0; row++) {
                mul_mv_row<Fmt, NR0, args_t, LANE>(args, x, ib, row, ns01, (first_row + row) * nb + ib, owned, prepared, acc[row], prep, shmem);
            }
        } else {
            for (short row = 0; row < NR0; ++row) {
                mul_mv_row<Fmt, NR0, args_t, LANE>(args, x, ib, row, ns01, (first_row + row) * nb + ib, owned, prepared, acc[row], prep, shmem);
            }
        }
    }

    float sumf[NR0];
    for (short row = 0; row < NR0; ++row) {
        sumf[row] = Fmt::finish(owned, acc[row]);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0;
        if (Fmt::shmem_stored) {
            // All rows written, one barrier, then lane r sums row r in lane order.
            for (int row = 0; row < NR0; ++row) {
                buf[NW * row + ltid] = sumf[row];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int row = ltid; row < NR0; row += NW) {
                if (first_row + row < args.ne0) {
                    float tot = 0.0f;
                    for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                    tot = Fmt::scale(tot);
                    dst_f32[first_row + row] = add_residual ? tot + residual[first_row + row] : tot;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            // One row at a time; the barrier count is the same in every simdgroup unless the format stops at the last row.
            for (int row = 0; row < NR0 && (!Fmt::bound_in_loop || first_row + row < args.ne0); ++row) {
                buf[NW * row + ltid] = sumf[row];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (ltid == 0 && first_row + row < args.ne0) {
                    float tot = 0.0f;
                    for (short i = 0; i < NW; i++) { tot += buf[NW * row + i]; }
                    tot = Fmt::scale(tot);
                    dst_f32[first_row + row] = add_residual ? tot + residual[first_row + row] : tot;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    } else if (Fmt::simd_sum_first) {
        for (int row = 0; row < NR0; ++row) {
            sumf[row] = simd_sum(sumf[row]);
        }
        if (tiisg == 0) {
            for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
                const float tot = Fmt::scale(sumf[row]);
                dst_f32[first_row + row] = add_residual ? tot + residual[first_row + row] : tot;
            }
        }
    } else {
        for (int row = 0; row < NR0; ++row) {
            const float tot = Fmt::scale(simd_sum(sumf[row]));
            if (tiisg == 0 && first_row + row < args.ne0) {
                dst_f32[first_row + row] = add_residual ? tot + residual[first_row + row] : tot;
            }
        }
    }

    } // slice loop
}

// No residual operand and no prepared headers: the form mul_mv_id binds.
template<typename Fmt, int NR0, typename args_t>
void mul_mv_plain(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    mul_mv_impl<Fmt, NR0, args_t, block_dot_lane_fp32>(args, src0, src1, dst, (device const float *) dst, (device const char *) dst, shmem, tgpig, tiisg, sgitg, tidx);
}

// Rows per simdgroup from the pipeline (1, 4 or 8), else the build default of the format.
template<typename Fmt, int NR0_DEFAULT, typename args_t>
void mul_mv_rows(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const float * residual,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if (FC_mul_mv_nr0 == 1) {
        mul_mv_impl<Fmt, 1, args_t, block_dot_lane_fp32>(args, src0, src1, dst, residual, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        mul_mv_impl<Fmt, 4, args_t, block_dot_lane_fp32>(args, src0, src1, dst, residual, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 8) {
        mul_mv_impl<Fmt, 8, args_t, block_dot_lane_fp32>(args, src0, src1, dst, residual, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        mul_mv_impl<Fmt, NR0_DEFAULT, args_t, block_dot_lane_fp32>(args, src0, src1, dst, residual, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

// NC activation columns in one launch: a lane loads each block's weights once for all columns.
// HOLD_W reloads the activations per row instead of keeping them; PACK keeps the weights packed and unpacks per column.
template<typename Fmt, int NR0, int NC, typename args_t, bool HOLD_W, bool PACK>
void mul_mv_cols_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort eff_tiisg = FC_mul_mv_shmem_reduce ? (tidx % NW) : tiisg;
    const ushort eff_sgitg = FC_mul_mv_shmem_reduce ? (tidx / NW) : sgitg;

    const short stride        = Fmt::stride(NW);
    const short block_id      = eff_tiisg / Fmt::threads_per_block;
    const short lane_in_block = eff_tiisg % Fmt::threads_per_block;

    const int nb = args.ne00/Fmt::block_elems;
    const int r1 = (tgpig.y + args.tg_y_offset) * NC;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    const int first_row = (((int) tgpig.x + args.tg_x_offset) * NSG + eff_sgitg) * NR0;
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const typename Fmt::block_t * x = (device const typename Fmt::block_t *) (src0 + offset0);

    float sumf[NR0*NC] = {0.f};

    const typename Fmt::fragment_t owned = Fmt::owned_fragment(lane_in_block);
    for (int ib = block_id; ib < nb; ib += stride) {
        typename Fmt::activation_t held[NC];
        if (!HOLD_W) {
            for (short c = 0; c < NC; ++c) {
                held[c] = Fmt::template prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*Fmt::block_elems);
            }
        }
        for (short row = 0; row < NR0; ++row) {
            device const typename Fmt::block_t * block = (device const typename Fmt::block_t *) ((device const char *) (x + ib) + row*args.nb01);
            const int entry = (first_row + row) * nb + ib;
            if (PACK) {
                const typename Fmt::raw_t w = Fmt::load_raw(block, owned, prep, entry);
                for (short c = 0; c < NC; ++c) {
                    if (HOLD_W) {
                        const typename Fmt::activation_t act = Fmt::template prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*Fmt::block_elems);
                        Fmt::accumulate_raw(w, act, sumf[row*NC + c]);
                    } else {
                        Fmt::accumulate_raw(w, held[c], sumf[row*NC + c]);
                    }
                }
            } else {
                const typename Fmt::weights_t w = Fmt::load_weights(block, owned, prep, entry);
                for (short c = 0; c < NC; ++c) {
                    if (HOLD_W) {
                        const typename Fmt::activation_t act = Fmt::template prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*Fmt::block_elems);
                        Fmt::accumulate_weights(w, act, sumf[row*NC + c]);
                    } else {
                        Fmt::accumulate_weights(w, held[c], sumf[row*NC + c]);
                    }
                }
            }
        }
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    if (FC_mul_mv_shmem_reduce) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort ltid = tidx % NW;
        threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0 * NC;
        for (int k = 0; k < NR0*NC; ++k) {
            buf[NW * k + ltid] = sumf[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Lane k sums value k (row k / NC, column k % NC) in lane order.
        for (int k = ltid; k < NR0*NC; k += NW) {
            const int row = k / NC;
            const int c = k % NC;
            if (first_row + row < args.ne0 && r1 + c < args.ne11) {
                float tot = 0.0f;
                for (short i = 0; i < NW; i++) { tot += buf[NW * k + i]; }
                dst_f32[(int64_t) c*args.ne0 + first_row + row] = tot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (int k = 0; k < NR0*NC; ++k) {
            sumf[k] = simd_sum(sumf[k]);
        }
        if (tiisg == 0) {
            for (int k = 0; k < NR0*NC; ++k) {
                if (first_row + k / NC < args.ne0 && r1 + k % NC < args.ne11) {
                    dst_f32[(int64_t) (k % NC)*args.ne0 + first_row + k / NC] = sumf[k];
                }
            }
        }
    }
}

template<typename Fmt, int NC, typename args_t, bool HOLD_W, bool PACK>
void mul_mv_cols_rows(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if (FC_mul_mv_nr0 == 1) {
        mul_mv_cols_impl<Fmt, 1, NC, args_t, HOLD_W, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 2) {
        mul_mv_cols_impl<Fmt, 2, NC, args_t, HOLD_W, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        mul_mv_cols_impl<Fmt, 4, NC, args_t, HOLD_W, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        mul_mv_cols_impl<Fmt, 8, NC, args_t, HOLD_W, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

template<typename Fmt, int NC, typename args_t>
void mul_mv_cols_form(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if (FC_mul_mv_col_form == 0) {
        mul_mv_cols_rows<Fmt, NC, args_t, false, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_col_form == 1) {
        mul_mv_cols_rows<Fmt, NC, args_t, true, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_col_form == 2) {
        mul_mv_cols_rows<Fmt, NC, args_t, false, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        mul_mv_cols_rows<Fmt, NC, args_t, true, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

// One pass of a paired-half lane over block ib: NC activations, then RH rows of this lane's half.
template<typename Fmt, int RH, int NC, typename args_t, bool PACK>
static inline void mul_mv_half_pass(
        args_t args,
        device const typename Fmt::block_t * x,
        device const char * src1,
        device const char * prep,
        uint64_t offset1,
        int first_row,
        int nb,
        thread const typename Fmt::fragment_t & owned,
        int ib,
        short h,
        thread float * s) {
    typename Fmt::activation_t held[NC];
    for (short c = 0; c < NC; ++c) {
        held[c] = Fmt::template prepare_activation<block_dot_lane_fp32>(owned, (device const float *) (src1 + offset1 + c*args.nb11) + ib*Fmt::block_elems);
    }
    for (short rr = 0; rr < RH; ++rr) {
        const int row = RH*h + rr;
        device const typename Fmt::block_t * block = (device const typename Fmt::block_t *) ((device const char *) (x + ib) + row*args.nb01);
        const int entry = (first_row + row) * nb + ib;
        if (PACK) {
            const typename Fmt::raw_t w = Fmt::load_raw(block, owned, prep, entry);
            for (short c = 0; c < NC; ++c) {
                Fmt::accumulate_raw(w, held[c], s[rr*NC + c]);
            }
        } else {
            const typename Fmt::weights_t w = Fmt::load_weights(block, owned, prep, entry);
            for (short c = 0; c < NC; ++c) {
                Fmt::accumulate_weights(w, held[c], s[rr*NC + c]);
            }
        }
    }
}

// Paired halves (64-lane waves, shared-memory reduce, 16 lanes per block): each half of a wave works RH rows and
// two blocks per pass in two banks, stored at the lane slots of the one-column map so each row sums in the same order.
template<typename Fmt, int RH, int NC, typename args_t, bool PACK>
void mul_mv_half_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    constexpr int NR0 = 2*RH;
    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const ushort ltid = tidx % NW;
    const ushort eff_sgitg = tidx / NW;
    const short h = ltid / 32;
    const short t = ltid % 32;
    const short b = t / 16;

    const int nb = args.ne00/Fmt::block_elems;
    const int r1 = (tgpig.y + args.tg_y_offset) * NC;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    const int first_row = (((int) tgpig.x + args.tg_x_offset) * NSG + eff_sgitg) * NR0;
    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const typename Fmt::block_t * x = (device const typename Fmt::block_t *) (src0 + offset0);

    const typename Fmt::fragment_t owned = Fmt::owned_fragment(t % 16);
    float bank0[RH*NC] = {0.f};
    float bank1[RH*NC] = {0.f};
    for (int ib = b; ib < nb; ib += 4) {
        mul_mv_half_pass<Fmt, RH, NC, args_t, PACK>(args, x, src1, prep, offset1, first_row, nb, owned, ib, h, bank0);
        if (ib + 2 < nb) {
            mul_mv_half_pass<Fmt, RH, NC, args_t, PACK>(args, x, src1, prep, offset1, first_row, nb, owned, ib + 2, h, bank1);
        }
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * buf = (threadgroup float *) shmem + eff_sgitg * NW * NR0 * NC;
    for (short rr = 0; rr < RH; ++rr) {
        for (short c = 0; c < NC; ++c) {
            const int k = (RH*h + rr)*NC + c;
            buf[NW * k + t] = bank0[rr*NC + c];
            buf[NW * k + t + 32] = bank1[rr*NC + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = ltid; k < NR0*NC; k += NW) {
        const int row = k / NC;
        const int c = k % NC;
        if (first_row + row < args.ne0 && r1 + c < args.ne11) {
            float tot = 0.0f;
            for (short i = 0; i < NW; i++) { tot += buf[NW * k + i]; }
            dst_f32[(int64_t) c*args.ne0 + first_row + row] = tot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template<typename Fmt, int NC, typename args_t, bool PACK>
void mul_mv_half_rows(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if (FC_mul_mv_nr0 == 2) {
        mul_mv_half_impl<Fmt, 1, NC, args_t, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_nr0 == 4) {
        mul_mv_half_impl<Fmt, 2, NC, args_t, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        mul_mv_half_impl<Fmt, 4, NC, args_t, PACK>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

// Entry of the column forms: paired halves, or the plain map at 2, 3 or 4 columns.
template<typename Fmt, typename args_t>
void mul_mv_cols(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * prep,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        ushort tidx) {
    if ((FC_mul_mv_col_form & 4) != 0) {
        const bool pack = (FC_mul_mv_col_form & 2) != 0;
        if (FC_mul_mv_ncols == 2) {
            if (pack) { mul_mv_half_rows<Fmt, 2, args_t, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); } else { mul_mv_half_rows<Fmt, 2, args_t, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); }
        } else {
            if (pack) { mul_mv_half_rows<Fmt, 3, args_t, true>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); } else { mul_mv_half_rows<Fmt, 3, args_t, false>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx); }
        }
        return;
    }
    if (FC_mul_mv_ncols == 2) {
        mul_mv_cols_form<Fmt, 2, args_t>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else if (FC_mul_mv_ncols == 3) {
        mul_mv_cols_form<Fmt, 3, args_t>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    } else {
        mul_mv_cols_form<Fmt, 4, args_t>(args, src0, src1, dst, prep, shmem, tgpig, tiisg, sgitg, tidx);
    }
}

#endif
