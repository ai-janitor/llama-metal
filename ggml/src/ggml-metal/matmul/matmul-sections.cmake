# The sections of data/matmul.tsv and their header rows, for the generators of this folder.
set(matmul_sections tiles formats matvec_defaults matvec_shapes)
set(matmul_tiles_fields warp threads bm bn bk wm wn wmiter tm tn pad)
set(matmul_tiles_header "card;weight;plan;plan_cols;role;ext_from;tiled_from;wave;group_bytes;${matmul_tiles_fields};route_source;route_test;plan_source;plan_test")
set(matmul_formats_header "weight;nsg;nr0;rows_per;smem;bytes;nr0_from_pipeline;reduce_recompile_below_32;matvec;ext;tiled;w64_fallback;grouped;shape_rows;act_bf16;act_f16;ext_wide;short_k;short_rows")
set(matmul_matvec_defaults_header "card;wave;group_bytes;simd_sum_w64_ok;mv_reduce;f_scalar_w64;mv_group;q8_load_bytes;q8_rows8_min;q8_act_load_bytes;mv_group_red;mv_sum;l1_share_cut;rows_below;rows_from;rows_unknown;f_nsg_max;f_k_per_sg;evidence")
set(matmul_matvec_shapes_header "card;wave;group_bytes;kind;weight;k;m;cols;prep;nr0;form;source")
