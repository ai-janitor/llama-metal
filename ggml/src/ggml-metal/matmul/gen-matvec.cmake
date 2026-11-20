# Turns sections [matvec_defaults] and [matvec_shapes] of data/matmul.tsv into matvec-rows.h.
# Run: cmake -DTABLE=.../data/matmul.tsv -DOUT=... [-DDEVICES=.../data/device.tsv] -P gen-matvec.cmake
# With DEVICES, a row keyed family:<name> must name a family of that table's section [families].
# The policies of CMake 3.14 (the project's minimum), so that a quoted word in if() is a string under CMake 3.x too.
cmake_policy(VERSION 3.14)
include("${CMAKE_CURRENT_LIST_DIR}/../data/tsv-read.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/matmul-sections.cmake")

# the card's shape when the row was measured: both cells are numbers
function(matvec_shape f where)
    list(GET f 1 wave)
    list(GET f 2 group_bytes)
    tsv_number("${wave}" "${where}" "wave" 2147483647 w)
    tsv_number("${group_bytes}" "${where}" "group_bytes" 2147483647 g)
    if (w EQUAL -1 OR g EQUAL -1)
        message(FATAL_ERROR "${where}: wave and group_bytes are the numbers the card reported when the row was measured, not -")
    endif()
endfunction()

# The family names of data/device.tsv [families], when the device table is given (-DDEVICES=...): a row keyed
# family:<name> must name one of them.
set(family_names "")
if (DEFINED DEVICES)
    tsv_section("${DEVICES}" families "devices;families" family_lines)
    set(first 1)
    foreach(line IN LISTS family_lines)
        if (first)
            set(first 0)
            continue()
        endif()
        string(REGEX REPLACE "\t.*$" "" family_name "${line}")
        list(APPEND family_names "${family_name}")
    endforeach()
endif()
# a card cell: any (default rows only) | family:<name> | a Metal device name. Sets ${out} to the C string and ${kind} to any, family or card.
function(matvec_card cell where allow_any out kind)
    if ("${cell}" STREQUAL "any")
        if (NOT allow_any)
            message(FATAL_ERROR "${where}: card any is for the default rows of [matvec_defaults] only")
        endif()
        set(${out} "any" PARENT_SCOPE)
        set(${kind} "any" PARENT_SCOPE)
    elseif ("${cell}" MATCHES "^family:")
        if (NOT "${cell}" MATCHES "^family:[a-z0-9]+$")
            message(FATAL_ERROR "${where}: a family key is family:<name>, the name in lower-case letters and digits, got '${cell}'")
        endif()
        string(REGEX REPLACE "^family:" "" name "${cell}")
        if (DEFINED DEVICES)
            list(FIND family_names "${name}" at)
            if (at EQUAL -1)
                message(FATAL_ERROR "${where}: ${cell} names no family of device.tsv [families]")
            endif()
        endif()
        set(${out} "${name}" PARENT_SCOPE)
        set(${kind} "family" PARENT_SCOPE)
    else()
        tsv_card("${cell}" "${where}" card)
        set(${out} "${card}" PARENT_SCOPE)
        set(${kind} "card" PARENT_SCOPE)
    endif()
endfunction()

set(where "matmul.tsv [matvec_defaults]")
tsv_section("${TABLE}" matvec_defaults "${matmul_sections}" lines)
set(default_rows "")
set(any_rows "")
set(family_default_rows "")
set(seen "")
set(any_any 0)
set(first 1)
foreach(line IN LISTS lines)
    if (first)
        set(first 0)
        tsv_header("${line}" "${matmul_matvec_defaults_header}" "${where}")
        continue()
    endif()
    tsv_text("${line}" shown)
    string(REPLACE "\t" ";" f "${line}")
    list(LENGTH f n)
    if (NOT n EQUAL 19)
        message(FATAL_ERROR "${where}: expected 19 cells, got ${n}: ${shown}")
    endif()
    list(GET f 0 card_cell)
    list(GET f 1 wave_cell)
    list(GET f 2 group_cell)
    list(GET f 3 sum64)
    list(GET f 4 reduce)
    list(GET f 5 fscalar)
    list(GET f 6 group)
    list(GET f 7 load)
    list(GET f 8 rows8)
    list(GET f 9 actload)
    list(GET f 10 groupred)
    list(GET f 11 mvsum)
    list(GET f 12 cut)
    list(GET f 13 below)
    list(GET f 14 from)
    list(GET f 15 unknown)
    list(GET f 16 fnsg)
    list(GET f 17 fkper)
    list(GET f 18 evidence)
    matvec_card("${card_cell}" "${where}" 1 card card_kind)
    if (card_kind STREQUAL "any")
        set(key "any\t${wave_cell}")
        set(row_where "${where} any, wave ${wave_cell}")
    else()
        set(key "${card_cell}")
        set(row_where "${where} ${card}")
    endif()
    list(FIND seen "${key}" at)
    if (NOT at EQUAL -1)
        if (card_kind STREQUAL "any")
            message(FATAL_ERROR "${where}: two default rows (card any) for wave ${wave_cell}")
        endif()
        message(FATAL_ERROR "${where}: two rows for ${card}")
    endif()
    list(APPEND seen "${key}")
    if (card_kind STREQUAL "any")
        # a default row is keyed by the wave alone
        if (NOT "${group_cell}" STREQUAL "any")
            message(FATAL_ERROR "${row_where}: group_bytes of a default row (card any) is any")
        endif()
        if ("${wave_cell}" STREQUAL "any")
            set(v_wave 0)
            set(any_any 1)
        else()
            tsv_number("${wave_cell}" "${row_where}" "wave" 2147483647 v_wave)
            if (v_wave EQUAL -1)
                message(FATAL_ERROR "${row_where}: wave of a default row is any or a number")
            endif()
        endif()
        set(v_groupb 0)
    else()
        matvec_shape("${f}" "${row_where}")
        set(v_wave "${wave_cell}")
        set(v_groupb "${group_cell}")
    endif()
    tsv_word("${sum64}" "${row_where}" "simd_sum_w64_ok" v_sum64 yes=1 no=0 -=-1)
    tsv_word("${reduce}" "${row_where}" "mv_reduce" v_reduce serial=0 tree=1 -=-1)
    tsv_word("${fscalar}" "${row_where}" "f_scalar_w64" v_fscalar on=1 off=0 -=-1)
    tsv_word("${group}" "${row_where}" "mv_group" v_group seq=1 off=0 -=-1)
    tsv_word("${load}" "${row_where}" "q8_load_bytes" v_load 1=1 4=4 -=-1)
    tsv_number("${rows8}" "${row_where}" "q8_rows8_min" 2147483647 v_rows8)
    tsv_word("${actload}" "${row_where}" "q8_act_load_bytes" v_actload 4=4 16=16 -=-1)
    tsv_word("${groupred}" "${row_where}" "mv_group_red" v_groupred on=1 off=0 -=-1)
    tsv_word("${mvsum}" "${row_where}" "mv_sum" v_mvsum wave=0 shared=1 -=-1)
    if ("${cut}" STREQUAL "-")
        set(v_cut "-1.0f")
    elseif ("${cut}" MATCHES "^0\\.[0-9]+$" AND NOT "${cut}" MATCHES "^0\\.0+$")
        set(v_cut "${cut}f")
    else()
        tsv_text("${cut}" cut_shown)
        message(FATAL_ERROR "${row_where}: l1_share_cut is a decimal between 0 and 1 written as 0.<digits>, or -; got '${cut_shown}'")
    endif()
    tsv_word("${below}" "${row_where}" "rows_below" v_below 2=2 4=4 8=8 -=-1)
    tsv_word("${from}" "${row_where}" "rows_from" v_from 2=2 4=4 8=8 -=-1)
        # not 1: the basic quant kernels have instances of 2, 4 and 8 rows, and a request for 1 would run the 4-row one
    tsv_word("${unknown}" "${row_where}" "rows_unknown" v_unknown 2=2 4=4 8=8 -=-1)
    tsv_word("${fnsg}" "${row_where}" "f_nsg_max" v_fnsg 1=1 2=2 3=3 4=4 -=-1)
    tsv_number("${fkper}" "${row_where}" "f_k_per_sg" 2147483647 v_fkper)
    if (v_actload EQUAL 16 AND NOT v_load EQUAL 4)
        message(FATAL_ERROR "${row_where}: q8_act_load_bytes 16 needs q8_load_bytes 4: ${shown}")
    endif()
    if ("${evidence}" STREQUAL "" OR "${evidence}" STREQUAL "-")
        message(FATAL_ERROR "${row_where}: evidence names the measurement behind the cells; it is empty")
    endif()
    if (card_kind STREQUAL "any" AND v_wave EQUAL 0)
        # the row every card starts from: each setting is written
        foreach(pair "mv_reduce=${reduce}" "f_scalar_w64=${fscalar}" "mv_group=${group}" "q8_load_bytes=${load}" "q8_act_load_bytes=${actload}" "mv_group_red=${groupred}" "mv_sum=${mvsum}" "l1_share_cut=${cut}" "rows_below=${below}" "rows_from=${from}" "rows_unknown=${unknown}" "f_nsg_max=${fnsg}" "f_k_per_sg=${fkper}")
            if ("${pair}" MATCHES "=-$")
                string(REGEX REPLACE "=-$" "" cell_name "${pair}")
                message(FATAL_ERROR "${row_where}: the default row writes every setting; ${cell_name} is -")
            endif()
        endforeach()
    endif()
    set(row "    { \"${card}\", ${v_wave}, ${v_groupb}, ${v_sum64}, ${v_reduce}, ${v_fscalar}, ${v_group}, ${v_load}, ${v_rows8}, ${v_actload}, ${v_groupred}, ${v_mvsum}, ${v_cut}, ${v_below}, ${v_from}, ${v_unknown}, ${v_fnsg}, ${v_fkper} },\n")
    if (card_kind STREQUAL "any")
        string(APPEND any_rows "${row}")
    elseif (card_kind STREQUAL "family")
        string(APPEND family_default_rows "${row}")
    else()
        string(APPEND default_rows "${row}")
    endif()
endforeach()
if (first)
    message(FATAL_ERROR "${where}: the section has no header row")
endif()
if (NOT any_any)
    message(FATAL_ERROR "${where}: no default row (card any, wave any, group_bytes any): a card without a row would have no settings")
endif()

# per weight type of [formats]: simdgroups per threadgroup (nsg), whether the pipeline takes a row's nr0, and whether the type joins a grouped launch
tsv_section("${TABLE}" formats "${matmul_sections}" flines)
set(first 1)
foreach(line IN LISTS flines)
    if (first)
        set(first 0)
        tsv_header("${line}" "${matmul_formats_header}" "matmul.tsv [formats]")
        continue()
    endif()
    string(REPLACE "\t" ";" f "${line}")
    list(GET f 0 fw)
    list(GET f 1 fnsg)
    list(GET f 6 fnr0)
    list(GET f 12 fgrouped)
    set(nsg_${fw} "${fnsg}")
    set(nr0_taken_${fw} "${fnr0}")
    set(grouped_${fw} "${fgrouped}")
endforeach()

set(where "matmul.tsv [matvec_shapes]")
tsv_section("${TABLE}" matvec_shapes "${matmul_sections}" lines)
set(shape_rows "")
set(family_shape_rows "")
set(any_seen "")
set(seen "")
set(cards "")
set(last_card "")
set(first 1)
foreach(line IN LISTS lines)
    if (first)
        set(first 0)
        tsv_header("${line}" "${matmul_matvec_shapes_header}" "${where}")
        continue()
    endif()
    tsv_text("${line}" shown)
    string(REPLACE "\t" ";" f "${line}")
    list(LENGTH f n)
    if (NOT n EQUAL 12)
        message(FATAL_ERROR "${where}: expected 12 cells, got ${n}: ${shown}")
    endif()
    list(GET f 0 card_cell)
    list(GET f 3 kind)
    list(GET f 4 weight)
    list(GET f 5 k)
    list(GET f 6 m)
    list(GET f 7 cols)
    list(GET f 8 prep)
    list(GET f 9 nr0)
    list(GET f 10 form)
    list(GET f 11 source)
    matvec_card("${card_cell}" "${where}" 0 card card_kind)
    set(row_where "${where} ${card} ${kind} ${weight} ${k} x ${m}, ${cols} columns")
    matvec_shape("${f}" "${row_where}")
    tsv_word("${kind}" "${row_where}" "kind" v_kind single=0 group=1)
    tsv_word("${weight}" "${row_where}" "weight" v_type q4_K=GGML_TYPE_Q4_K q5_K=GGML_TYPE_Q5_K q6_K=GGML_TYPE_Q6_K)
    set(shape_any 0)
    if ("${k}" STREQUAL "any" AND "${m}" STREQUAL "any")
        # every shape of the weight type: -1 in the generated row
        set(shape_any 1)
        set(v_k -1)
        set(v_m -1)
    elseif ("${k}" STREQUAL "any" OR "${m}" STREQUAL "any")
        message(FATAL_ERROR "${row_where}: k and m are both any (every shape of the weight) or both numbers")
    else()
        tsv_number("${k}" "${row_where}" "k" 9223372036854775807 v_k)
        tsv_number("${m}" "${row_where}" "m" 9223372036854775807 v_m)
        if (v_k EQUAL -1 OR v_m EQUAL -1)
            message(FATAL_ERROR "${row_where}: k and m are the weight's two dimensions, not -")
        endif()
    endif()
    tsv_word("${cols}" "${row_where}" "cols" v_cols 1=1 2=2 3=3 4=4)
    if (shape_any AND NOT v_cols EQUAL 1)
        message(FATAL_ERROR "${row_where}: a row for every shape (k and m any) has one column: the column kernel needs m to be whole threadgroups")
    endif()
    # the first row that fits wins: a row with a written shape after the every-shape row of its card, kind and weight is never read
    set(any_key "${card_cell}\t${kind}\t${weight}")
    if (shape_any)
        list(APPEND any_seen "${any_key}")
    elseif (v_cols EQUAL 1)
        list(FIND any_seen "${any_key}" at)
        if (NOT at EQUAL -1)
            message(FATAL_ERROR "${row_where}: the row stands after the every-shape row (k and m any) of ${card} ${kind} ${weight}, which takes every shape first; put the rows with a written shape first")
        endif()
    endif()
    tsv_word("${prep}" "${row_where}" "prep" v_prep on=1 off=0)
    tsv_word("${nr0}" "${row_where}" "nr0" v_nr0 1=1 2=2 4=4 8=8)
    tsv_word("${form}" "${row_where}" "form" v_form -=0 a=0 w=1 ap=2 wp=3 ha=4 hap=6)
    if (v_prep EQUAL 1 AND NOT weight STREQUAL "q4_K")
        message(FATAL_ERROR "${row_where}: prep on needs weight q4_K")
    endif()
    if ((form STREQUAL "-" AND NOT v_cols EQUAL 1) OR (v_cols EQUAL 1 AND NOT form STREQUAL "-"))
        message(FATAL_ERROR "${row_where}: form is - with one column and one of a, w, ap, wp, ha, hap with 2 to 4")
    endif()
    if ((form STREQUAL "ha" OR form STREQUAL "hap") AND (v_cols EQUAL 4 OR v_nr0 EQUAL 1))
        message(FATAL_ERROR "${row_where}: form ${form} needs 2 or 3 columns and nr0 2, 4 or 8")
    endif()
    if (NOT v_cols EQUAL 1 AND (NOT kind STREQUAL "single" OR weight STREQUAL "q5_K"))
        message(FATAL_ERROR "${row_where}: 2 to 4 columns need kind single and weight q4_K or q6_K")
    endif()
    # the row's nr0 reaches the launch only for a type whose pipeline takes it; otherwise the launch would run another geometry than the row says
    if (NOT "${nr0_taken_${weight}}" STREQUAL "yes")
        message(FATAL_ERROR "${row_where}: the pipeline of ${weight} does not take a row's nr0 ([formats] nr0_from_pipeline is not yes), so the row's nr0 ${v_nr0} would not be the launch's")
    endif()
    # paired halves split a wave at thread 32 and give each half its rows: they exist at wave width 64 only
    list(GET f 1 wave)
    if ((form STREQUAL "ha" OR form STREQUAL "hap") AND NOT wave EQUAL 64)
        message(FATAL_ERROR "${row_where}: form ${form} needs wave 64 (the halves of a 64-thread wave), the row says ${wave}")
    endif()
    # the kernels load every weight row of a threadgroup before they look at the output bound: a written m is whole threadgroups
    if (NOT shape_any)
        if (NOT "${nsg_${weight}}" MATCHES "^[1-9][0-9]*$")
            message(FATAL_ERROR "${row_where}: [formats] gives no simdgroup count for ${weight}")
        endif()
        math(EXPR per_group "${v_nr0} * ${nsg_${weight}}")
        math(EXPR rest "${v_m} % ${per_group}")
        if (NOT rest EQUAL 0)
            message(FATAL_ERROR "${row_where}: m must be a whole number of ${per_group} rows (nr0 ${v_nr0} x ${nsg_${weight}} simdgroups per threadgroup): the kernel reads whole threadgroups of weight rows")
        endif()
    endif()
    # a grouped launch exists only for the types whose [formats] row says grouped yes
    if (kind STREQUAL "group" AND NOT "${grouped_${weight}}" STREQUAL "yes")
        message(FATAL_ERROR "${row_where}: ${weight} has no grouped launch ([formats] grouped is not yes), so a group row is never read")
    endif()
    if ("${source}" STREQUAL "" OR "${source}" STREQUAL "-")
        message(FATAL_ERROR "${row_where}: source names the measurement behind the row; it is empty")
    endif()
    # the rows of one card stand together: the device reads them as one run of the array
    if (NOT "${card_cell}" STREQUAL "${last_card}")
        list(FIND cards "${card_cell}" at)
        if (NOT at EQUAL -1)
            message(FATAL_ERROR "${where}: the rows of ${card} do not stand together: ${shown}")
        endif()
        list(APPEND cards "${card_cell}")
        set(last_card "${card_cell}")
    endif()
    # a second row with the same key could never be reached: the first that fits wins
    set(key "${card_cell}\t${kind}\t${weight}\t${k}\t${m}\t${cols}\t${prep}")
    list(FIND seen "${key}" at)
    if (NOT at EQUAL -1)
        message(FATAL_ERROR "${where}: two rows for ${card} ${kind} ${weight} ${k} x ${m}, ${cols} columns, prep ${prep}")
    endif()
    list(APPEND seen "${key}")
    # the first row that fits wins, and a row without prep fits every tensor: a prep row after it is never read
    if (v_prep EQUAL 1)
        list(FIND seen "${card_cell}\t${kind}\t${weight}\t${k}\t${m}\t${cols}\toff" at)
        if (NOT at EQUAL -1)
            message(FATAL_ERROR "${row_where}: the prep row stands after the row without prep for the same shape, which takes every tensor first; put the prep row first")
        endif()
    endif()
    list(GET f 2 group_bytes)
    set(row "    { \"${card}\", ${v_kind}, ${v_type}, ${v_k}, ${v_m}, ${v_nr0}, ${v_prep}, ${v_cols}, ${v_form}, ${wave}, ${group_bytes} },\n")
    if (card_kind STREQUAL "family")
        string(APPEND family_shape_rows "${row}")
    else()
        string(APPEND shape_rows "${row}")
    endif()
endforeach()
if (first)
    message(FATAL_ERROR "${where}: the section has no header row")
endif()

set(text "// generated from data/matmul.tsv [matvec_defaults] and [matvec_shapes] by matmul/gen-matvec.cmake; do not edit
#pragma once

#include <stdint.h>
#include <string.h>

// One row of data/matmul.tsv [matvec_defaults]. card = the card's Metal device name, compared whole (a family's name in
// the family rows, any in the default rows). -1 = the cell is -: the value of the row before it in the order stays.
struct ggml_metal_matvec_default_row {
    const char * card;
    int32_t      wave;              // the wave the card reported when the row was measured; default rows: the wave the row is for, 0 = every wave
    int32_t      group_bytes;       // the threadgroup memory the card reported; 0 in a default row
    int32_t      simd_sum_w64_ok;
    int32_t      mv_reduce;         // matvec row-sum tail: 0 serial, 1 tree
    int32_t      f_scalar_w64;      // f16 and bf16 scalar matvec at width 64: 1 on, 0 off
    int32_t      mv_group;          // Q8_0 matvecs of one activation in one launch: 1 on, 0 off
    int32_t      q8_load_bytes;     // Q8_0 matvec weight load width in bytes: 1 or 4 (aligned words)
    int32_t      q8_rows8_min;      // smallest row count of a single Q8_0 matvec run at 8 whole rows per simdgroup
    int32_t      q8_act_load_bytes; // Q8_0 matvec activation load width in bytes: 4 or 16 (two aligned loads per lane)
    int32_t      mv_group_red;      // grouped K-quant matvec stores every row before the row sums: 1 on, 0 off
    int32_t      mv_sum;            // a single matvec sums a row with the wave's own reduction (0) or through shared memory (1)
    float        l1_share_cut;      // the cache rule: share of the L1 per compute unit one activation column may take for rows_below
    int32_t      rows_below;        // rows per simdgroup under the cut
    int32_t      rows_from;         // rows per simdgroup from the cut
    int32_t      rows_unknown;      // rows per simdgroup without a model profile or an L1 fact
    int32_t      f_nsg_max;         // dense float matvec: most simdgroups per threadgroup
    int32_t      f_k_per_sg;        // dense float matvec: input columns per simdgroup
};

// the rows of the cards; the last row of each table is the end mark
static const struct ggml_metal_matvec_default_row ggml_metal_matvec_default_rows[] = {
${default_rows}    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

static const int ggml_metal_matvec_default_rows_n = (int) (sizeof(ggml_metal_matvec_default_rows) / sizeof(ggml_metal_matvec_default_rows[0])) - 1;

// the default rows (card any): the row of every wave (wave 0) and the rows of one wave
static const struct ggml_metal_matvec_default_row ggml_metal_matvec_any_rows[] = {
${any_rows}    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

// the family rows (card = the family's name): copies of a measured member's row, sweep candidates, never applied
static const struct ggml_metal_matvec_default_row ggml_metal_matvec_family_default_rows[] = {
${family_default_rows}    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

// The row of the card with this Metal device name, or 0 when it has none.
static inline const struct ggml_metal_matvec_default_row * ggml_metal_matvec_default_row_of(const char * name) {
    for (int i = 0; name != 0 && i < ggml_metal_matvec_default_rows_n; i++) {
        if (strcmp(ggml_metal_matvec_default_rows[i].card, name) == 0) {
            return &ggml_metal_matvec_default_rows[i];
        }
    }
    return 0;
}

// The default row of this wave (0 = the row of every wave), or 0 when there is none.
static inline const struct ggml_metal_matvec_default_row * ggml_metal_matvec_any_row_of(int32_t wave) {
    for (int i = 0; ggml_metal_matvec_any_rows[i].card != 0; i++) {
        if (ggml_metal_matvec_any_rows[i].wave == wave) {
            return &ggml_metal_matvec_any_rows[i];
        }
    }
    return 0;
}

// The candidate row of this family, or 0 when it has none.
static inline const struct ggml_metal_matvec_default_row * ggml_metal_matvec_family_default_row_of(const char * family) {
    for (int i = 0; family != 0 && ggml_metal_matvec_family_default_rows[i].card != 0; i++) {
        if (strcmp(ggml_metal_matvec_family_default_rows[i].card, family) == 0) {
            return &ggml_metal_matvec_family_default_rows[i];
        }
    }
    return 0;
}

// The rows of data/matmul.tsv [matvec_shapes], in file order. See ggml_device_mv_row in ggml-device-profile.h.
// { card, kind (0 single, 1 group), ggml_type, ne00, ne01 (-1 -1 = every shape), nr0, prep, cols, form, wave, group_bytes }; the last row is the end mark
static const struct ggml_device_mv_row ggml_metal_hw_mv_rows[] = {
${shape_rows}    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

static const int ggml_metal_hw_mv_rows_n = (int) (sizeof(ggml_metal_hw_mv_rows) / sizeof(ggml_metal_hw_mv_rows[0])) - 1;

// the family rows (card = the family's name): copies of a measured member's rows, sweep candidates, never applied
static const struct ggml_device_mv_row ggml_metal_family_mv_rows[] = {
${family_shape_rows}    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

// The rows whose card cell is this name in a table that ends with the end mark: the first one, and how many follow it
// (they stand together); 0 rows = none.
static inline const struct ggml_device_mv_row * ggml_metal_mv_rows_in(const struct ggml_device_mv_row * table, const char * name, uint32_t * n) {
    const struct ggml_device_mv_row * first_row = 0;
    *n = 0;
    for (int i = 0; name != 0 && table[i].card != 0; i++) {
        if (strcmp(table[i].card, name) != 0) {
            continue;
        }
        if (first_row == 0) {
            first_row = &table[i];
        }
        (*n)++;
    }
    return first_row;
}

// The rows of the card with this Metal device name.
static inline const struct ggml_device_mv_row * ggml_metal_mv_rows_of(const char * name, uint32_t * n) {
    return ggml_metal_mv_rows_in(ggml_metal_hw_mv_rows, name, n);
}

// The candidate rows of this family.
static inline const struct ggml_device_mv_row * ggml_metal_family_mv_rows_of(const char * family, uint32_t * n) {
    return ggml_metal_mv_rows_in(ggml_metal_family_mv_rows, family, n);
}
")
if (EXISTS "${OUT}")
    file(READ "${OUT}" old)
else()
    set(old "")
endif()
if (NOT old STREQUAL text)
    file(WRITE "${OUT}" "${text}")
endif()
