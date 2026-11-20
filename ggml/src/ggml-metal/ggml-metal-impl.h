#ifndef GGML_METAL_IMPL
#define GGML_METAL_IMPL

// =============================================================================
// Shared conservative matvec geometry.
// Static kernels and host tuning both read these named constants; dynamic basic
// specializations additionally accept the pipeline's function-constant row count.
// UNTIED: baseline rows/simdgroups are compile-time tuning constants, not device gates.

// Dense float (f32, f16, bf16) matvec: N_R0_F rows per threadgroup in the regular entries; a weight of fewer than
// N_F_SHORT_K input columns runs the short entry, N_F_SHORT_ROWS rows per threadgroup. data/matmul.tsv [formats]
// holds the same three numbers (nr0, short_k, short_rows), checked when the host code compiles.
#define N_R0_F 2
#define N_R0_F32 N_R0_F
#define N_R0_F16 N_R0_F
#define N_R0_BF16 N_R0_F
#define N_F_SHORT_K 32
#define N_F_SHORT_ROWS 32

#define N_R0_BASIC 2 // shared baseline specialization for basic quant kernels
#define N_R0_Q4_0 N_R0_BASIC
#define N_SG_Q4_0 2

#define N_R0_Q4_1 N_R0_BASIC
#define N_SG_Q4_1 2

#define N_R0_Q5_0 N_R0_BASIC
#define N_SG_Q5_0 2

#define N_R0_Q5_1 N_R0_BASIC
#define N_SG_Q5_1 2

#define N_R0_Q8_0 4
#define N_SG_Q8_0 2

#define N_R0_MXFP4 2
#define N_SG_MXFP4 2

// Shared activation reuse factor for MUL_MAT_ID k-quant templates.
// Conservative default, shared with host tuning.
#define N_R0_KQUANT 2

#define N_R0_Q2_K N_R0_KQUANT
#define N_SG_Q2_K 2

#define N_R0_Q3_K N_R0_KQUANT
#define N_SG_Q3_K 2

#define N_R0_Q4_K N_R0_KQUANT
#define N_SG_Q4_K 2

#define N_R0_Q5_K N_R0_KQUANT
#define N_SG_Q5_K 2

#define N_R0_Q6_K N_R0_KQUANT
#define N_SG_Q6_K 2

#define N_R0_Q1_0 2
#define N_SG_Q1_0 2

#define N_R0_Q2_0 2
#define N_SG_Q2_0 2

#define N_R0_TQ1_0 2
#define N_SG_TQ1_0 2

#define N_R0_TQ2_0 2
#define N_SG_TQ2_0 2

#define N_R0_NVFP4 2
#define N_SG_NVFP4 2

#define N_R0_F8_E4M3 2
#define N_SG_F8_E4M3 2

#define N_R0_IQ1_S 2
#define N_SG_IQ1_S 2

#define N_R0_IQ1_M 2
#define N_SG_IQ1_M 2

#define N_R0_IQ2_XXS 2
#define N_SG_IQ2_XXS 2

#define N_R0_IQ2_XS 2
#define N_SG_IQ2_XS 2

#define N_R0_IQ2_S 2
#define N_SG_IQ2_S 2

#define N_R0_IQ3_XXS 2
#define N_SG_IQ3_XXS 2

#define N_R0_IQ3_S 2
#define N_SG_IQ3_S 2

#define N_R0_IQ4_NL 2
#define N_SG_IQ4_NL 2

#define N_R0_IQ4_XS 2
#define N_SG_IQ4_XS 2

// function constants offsets
// K/V element formats of the scalar flash attention kernels
#define FA_SCALAR_KV_F16    0
#define FA_SCALAR_KV_F32    1
#define FA_SCALAR_KV_BF16   2
#define FA_SCALAR_KV_Q8_0   3
#define FA_SCALAR_KV_Q4_0   4
#define FA_SCALAR_KV_Q4_1   5
#define FA_SCALAR_KV_Q5_0   6
#define FA_SCALAR_KV_Q5_1   7
#define FA_SCALAR_KV_IQ4_NL 8
#define FA_SCALAR_KV_F8_E4M3 9

#define FC_FLASH_ATTN_EXT_PAD          100
#define FC_FLASH_ATTN_EXT_BLK          200
#define FC_FLASH_ATTN_EXT              300
#define FC_FLASH_ATTN_EXT_VEC          400
#define FC_FLASH_ATTN_EXT_VEC_REDUCE   500
#define FC_DSV4_HC                     2000
#define FC_TOP_K_MASK                  2400
#define FC_FLASH_ATTN_EXT_SCALAR       1750 // +0: K element format, +1: V element format (int32, FA_SCALAR_KV_*)
#define FC_MUL_MV                      600
// FC_MUL_MV + 0: nsg (int16)      — simdgroups per threadgroup
// FC_MUL_MV + 1: nxpsg (int16)    — threads along K per simdgroup (ext kernels only)
// FC_MUL_MV + 2: shmem_reduce (bool) — use barrier-based reduction (NW != 32)
// FC_MUL_MV + 3: src1_trans (bool) — transposed src1 specialization
// FC_MUL_MV + 4: nr0 (int16)      — rows per simdgroup (dynamic, from model/hw profile)
// FC_MUL_MV + 10: q8_load (int16) — Q8_0 weight load width in bytes: 1 or 4 (aligned words)
// FC_MUL_MV + 11: q8_act (int16) — Q8_0 activation load width in bytes: 4 or 16 (two aligned loads per lane)
#define FC_MUL_MV_RESIDUAL          (FC_MUL_MV + 6) // bool: scalar Q4_K residual epilogue
#define FC_MUL_MM                      700
#define FC_ROPE                        800
#define FC_NORM                        1700
#define FC_SSM_CONV                    900
#define FC_GATED_DELTA_NET             950  // +0: elements per thread (NSG); +1: vector transport
#define FC_SOLVE_TRI                   1000
#define FC_COUNT_EQUAL                 1100
#define FC_UNARY                       1200
#define FC_BIN                         1300
#define FC_SUM_ROWS                    1400
#define FC_MUL_MV_GROUPED              1550  // function constants for grouped matvec kernel
#define FC_SIMD_WIDTH                  1500
#define FC_MUL_MV_GROUP                1660  // +0: matrices in one launch (int16)
#define FC_SMALL_ATTN_GATE             1670  // +0: elements per thread (int16)
#define FC_NORM_MUL_MV                 1650  // +0: matrices (int16), +1: mean given (bool), +2: lanes per row (int16)

// op-specific constants
#define OP_FLASH_ATTN_EXT_NQPSG 8
#define OP_FLASH_ATTN_EXT_NCPSG 64

#define OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#define OP_FLASH_ATTN_EXT_VEC_NCPSG 32

#define OP_UNARY_NUM_SCALE      10
#define OP_UNARY_NUM_FILL       11
#define OP_UNARY_NUM_CLAMP      12
#define OP_UNARY_NUM_SQR        13
#define OP_UNARY_NUM_SQRT       14
#define OP_UNARY_NUM_SIN        15
#define OP_UNARY_NUM_COS        16
#define OP_UNARY_NUM_LOG        17
#define OP_UNARY_NUM_LEAKY_RELU 18

#define OP_UNARY_NUM_TANH        100
#define OP_UNARY_NUM_RELU        101
#define OP_UNARY_NUM_SIGMOID     102
#define OP_UNARY_NUM_GELU        103
#define OP_UNARY_NUM_GELU_ERF    104
#define OP_UNARY_NUM_GELU_QUICK  105
#define OP_UNARY_NUM_SILU        106
#define OP_UNARY_NUM_ELU         107
#define OP_UNARY_NUM_NEG         108
#define OP_UNARY_NUM_ABS         109
#define OP_UNARY_NUM_SGN         110
#define OP_UNARY_NUM_STEP        111
#define OP_UNARY_NUM_HARDSWISH   112
#define OP_UNARY_NUM_HARDSIGMOID 113
#define OP_UNARY_NUM_EXP         114
#define OP_UNARY_NUM_SOFTPLUS    115
#define OP_UNARY_NUM_EXPM1       116
#define OP_UNARY_NUM_FLOOR       117
#define OP_UNARY_NUM_CEIL        118
#define OP_UNARY_NUM_ROUND       119
#define OP_UNARY_NUM_TRUNC       120
#define OP_UNARY_NUM_XIELU       121

#define OP_SUM_ROWS_NUM_SUM_ROWS 10
#define OP_SUM_ROWS_NUM_MEAN     11

// kernel argument structs
//
// - element counters (e.g. ne00) typically use int32_t to reduce register usage
//   however, be careful from int overflows when using those in the kernel implementation
//
// - strides (e.g. nb00) use uint64_t
//
// one header per kernel family

#include "activation/kargs-activation.h"
#include "conv/kargs-conv.h"
#include "copy/kargs-copy.h"
#include "fft/kargs-fft.h"
#include "flash-attn/kargs-flash-attn.h"
#include "matmul/kargs-matmul.h"
#include "misc/kargs-misc.h"
#include "norm/kargs-norm.h"
#include "reduction/kargs-reduction.h"
#include "rope/kargs-rope.h"
#include "rows/kargs-rows.h"
#include "softmax/kargs-softmax.h"
#include "spatial/kargs-spatial.h"
#include "ssm/kargs-ssm.h"
#include "unary/kargs-unary.h"

#endif // GGML_METAL_IMPL
