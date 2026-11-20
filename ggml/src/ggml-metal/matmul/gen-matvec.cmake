# Turns sections [matvec_defaults] and [matvec_shapes] of data/matmul.tsv into matvec-rows.h.
# Run: cmake -DTABLE=.../data/matmul.tsv -DOUT=... -P gen-matvec.cmake
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

set(where "matmul.tsv [matvec_defaults]")
tsv_section("${TABLE}" matvec_defaults "${matmul_sections}" lines)
set(default_rows "")
set(seen "")
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
    if (NOT n EQUAL 12)
        message(FATAL_ERROR "${where}: expected 12 cells, got ${n}: ${shown}")
    endif()
    list(GET f 0 card_cell)
    list(GET f 3 sum64)
    list(GET f 4 reduce)
    list(GET f 5 fscalar)
    list(GET f 6 group)
    list(GET f 7 load)
    list(GET f 8 rows8)
    list(GET f 9 actload)
    list(GET f 10 groupred)
    list(GET f 11 evidence)
    tsv_card("${card_cell}" "${where}" card)
    list(FIND seen "${card_cell}" at)
    if (NOT at EQUAL -1)
        message(FATAL_ERROR "${where}: two rows for ${card}")
    endif()
    list(APPEND seen "${card_cell}")
    set(row_where "${where} ${card}")
    matvec_shape("${f}" "${row_where}")
    tsv_word("${sum64}" "${row_where}" "simd_sum_w64_ok" v_sum64 yes=1 no=0 -=-1)
    tsv_word("${reduce}" "${row_where}" "mv_reduce" v_reduce serial=0 tree=1 -=-1)
    tsv_word("${fscalar}" "${row_where}" "f_scalar_w64" v_fscalar on=1 off=0 -=-1)
    tsv_word("${group}" "${row_where}" "mv_group" v_group seq=1 off=0 -=-1)
    tsv_word("${load}" "${row_where}" "q8_load_bytes" v_load 1=1 4=4 -=-1)
    tsv_number("${rows8}" "${row_where}" "q8_rows8_min" 2147483647 v_rows8)
    tsv_word("${actload}" "${row_where}" "q8_act_load_bytes" v_actload 4=4 16=16 -=-1)
    tsv_word("${groupred}" "${row_where}" "mv_group_red" v_groupred on=1 off=0 -=-1)
    if (v_actload EQUAL 16 AND NOT v_load EQUAL 4)
        message(FATAL_ERROR "${row_where}: q8_act_load_bytes 16 needs q8_load_bytes 4: ${shown}")
    endif()
    if ("${evidence}" STREQUAL "" OR "${evidence}" STREQUAL "-")
        message(FATAL_ERROR "${row_where}: evidence names the measurement behind the cells; it is empty")
    endif()
    string(APPEND default_rows "    { \"${card}\", ${v_sum64}, ${v_reduce}, ${v_fscalar}, ${v_group}, ${v_load}, ${v_rows8}, ${v_actload}, ${v_groupred} },\n")
endforeach()
if (first)
    message(FATAL_ERROR "${where}: the section has no header row")
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
    tsv_card("${card_cell}" "${where}" card)
    set(row_where "${where} ${card} ${kind} ${weight} ${k} x ${m}, ${cols} columns")
    matvec_shape("${f}" "${row_where}")
    tsv_word("${kind}" "${row_where}" "kind" v_kind single=0 group=1)
    tsv_word("${weight}" "${row_where}" "weight" v_type q4_K=GGML_TYPE_Q4_K q5_K=GGML_TYPE_Q5_K q6_K=GGML_TYPE_Q6_K)
    tsv_number("${k}" "${row_where}" "k" 9223372036854775807 v_k)
    tsv_number("${m}" "${row_where}" "m" 9223372036854775807 v_m)
    if (v_k EQUAL -1 OR v_m EQUAL -1)
        message(FATAL_ERROR "${row_where}: k and m are the weight's two dimensions, not -")
    endif()
    tsv_word("${cols}" "${row_where}" "cols" v_cols 1=1 2=2 3=3 4=4)
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
    # the column kernel loads every weight row of a threadgroup before it looks at the output bound: m is whole threadgroups
    if (NOT v_cols EQUAL 1)
        if (NOT "${nsg_${weight}}" MATCHES "^[1-9][0-9]*$")
            message(FATAL_ERROR "${row_where}: [formats] gives no simdgroup count for ${weight}")
        endif()
        math(EXPR per_group "${v_nr0} * ${nsg_${weight}}")
        math(EXPR rest "${v_m} % ${per_group}")
        if (NOT rest EQUAL 0)
            message(FATAL_ERROR "${row_where}: m must be a whole number of ${per_group} rows (nr0 ${v_nr0} x ${nsg_${weight}} simdgroups per threadgroup): the column kernel reads whole threadgroups of weight rows")
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
    string(APPEND shape_rows "    { \"${card}\", ${v_kind}, ${v_type}, ${v_k}, ${v_m}, ${v_nr0}, ${v_prep}, ${v_cols}, ${v_form} },\n")
endforeach()
if (first)
    message(FATAL_ERROR "${where}: the section has no header row")
endif()

set(text "// generated from data/matmul.tsv [matvec_defaults] and [matvec_shapes] by matmul/gen-matvec.cmake; do not edit
#pragma once

#include <stdint.h>
#include <string.h>

// One row of data/matmul.tsv [matvec_defaults]. card = the card's Metal device name, compared whole.
// -1 = the cell is -: the backend's default stays.
struct ggml_metal_matvec_default_row {
    const char * card;
    int32_t      simd_sum_w64_ok;
    int32_t      mv_reduce;         // matvec row-sum tail: 0 serial, 1 tree
    int32_t      f_scalar_w64;      // f16 and bf16 scalar matvec at width 64: 1 on, 0 off
    int32_t      mv_group;          // Q8_0 matvecs of one activation in one launch: 1 on, 0 off
    int32_t      q8_load_bytes;     // Q8_0 matvec weight load width in bytes: 1 or 4 (aligned words)
    int32_t      q8_rows8_min;      // smallest row count of a single Q8_0 matvec run at 8 whole rows per simdgroup
    int32_t      q8_act_load_bytes; // Q8_0 matvec activation load width in bytes: 4 or 16 (two aligned loads per lane)
    int32_t      mv_group_red;      // grouped K-quant matvec stores every row before the row sums: 1 on, 0 off
};

// the last row is the end mark
static const struct ggml_metal_matvec_default_row ggml_metal_matvec_default_rows[] = {
${default_rows}    { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

static const int ggml_metal_matvec_default_rows_n = (int) (sizeof(ggml_metal_matvec_default_rows) / sizeof(ggml_metal_matvec_default_rows[0])) - 1;

// The row of the card with this Metal device name, or 0 when it has none.
static inline const struct ggml_metal_matvec_default_row * ggml_metal_matvec_default_row_of(const char * name) {
    for (int i = 0; name != 0 && i < ggml_metal_matvec_default_rows_n; i++) {
        if (strcmp(ggml_metal_matvec_default_rows[i].card, name) == 0) {
            return &ggml_metal_matvec_default_rows[i];
        }
    }
    return 0;
}

// The rows of data/matmul.tsv [matvec_shapes], in file order. See ggml_device_mv_row in ggml-device-profile.h.
// { card, kind (0 single, 1 group), ggml_type, ne00, ne01, nr0, prep, cols, form }; the last row is the end mark
static const struct ggml_device_mv_row ggml_metal_hw_mv_rows[] = {
${shape_rows}    { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

static const int ggml_metal_hw_mv_rows_n = (int) (sizeof(ggml_metal_hw_mv_rows) / sizeof(ggml_metal_hw_mv_rows[0])) - 1;

// The rows of the card with this Metal device name: the first one, and how many follow it (they stand together); 0 rows = none.
static inline const struct ggml_device_mv_row * ggml_metal_mv_rows_of(const char * name, uint32_t * n) {
    const struct ggml_device_mv_row * first_row = 0;
    *n = 0;
    for (int i = 0; name != 0 && i < ggml_metal_hw_mv_rows_n; i++) {
        if (strcmp(ggml_metal_hw_mv_rows[i].card, name) != 0) {
            continue;
        }
        if (first_row == 0) {
            first_row = &ggml_metal_hw_mv_rows[i];
        }
        (*n)++;
    }
    return first_row;
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
