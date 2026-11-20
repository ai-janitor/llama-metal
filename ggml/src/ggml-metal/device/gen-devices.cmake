# Turns section [devices] of data/device.tsv into device-rows.h.
# Run: cmake -DTABLE=.../data/device.tsv -DOUT=... -P gen-devices.cmake
# The policies of CMake 3.14 (the project's minimum), so that a quoted word in if() is a string under CMake 3.x too.
cmake_policy(VERSION 3.14)
include("${CMAKE_CURRENT_LIST_DIR}/../data/tsv-read.cmake")
set(device_sections devices)
set(device_devices_header "card;chip;architecture;wave;group_bytes;cu;l1_bytes_per_cu;l2_bytes;local_Bps;transfer_Bps;attn_gate_fuse;conv_l2_fold;add_norm_fuse;cb_seq;set_nowait;staging_keep;probe_dispatch;probe_stage;probe_blit;probe_timestamp;evidence")
set(where "device.tsv [devices]")

tsv_section("${TABLE}" devices "${device_sections}" lines)
set(rows "")
set(seen "")
set(first 1)
foreach(line IN LISTS lines)
    if (first)
        set(first 0)
        tsv_header("${line}" "${device_devices_header}" "${where}")
        continue()
    endif()
    tsv_text("${line}" shown)
    string(REPLACE "\t" ";" f "${line}")
    list(LENGTH f n)
    if (NOT n EQUAL 21)
        message(FATAL_ERROR "${where}: expected 21 cells, got ${n}: ${shown}")
    endif()
    list(GET f 0 card_cell)
    list(GET f 1 chip_cell)
    list(GET f 2 arch_cell)
    list(GET f 20 evidence)
    tsv_card("${card_cell}" "${where}" card)
    list(FIND seen "${card_cell}" at)
    if (NOT at EQUAL -1)
        message(FATAL_ERROR "${where}: two rows for ${card}")
    endif()
    list(APPEND seen "${card_cell}")
    set(row_where "${where} ${card}")
    foreach(name chip arch)
        if ("${${name}_cell}" STREQUAL "-")
            set(${name} "")
        else()
            tsv_cstring("${${name}_cell}" "${row_where}" ${name})
        endif()
    endforeach()
    if ("${evidence}" STREQUAL "" OR "${evidence}" STREQUAL "-")
        message(FATAL_ERROR "${row_where}: evidence says where the cells come from; it is empty")
    endif()
    # the sizes: int32_t fields, the two bandwidths int64_t
    set(i 3)
    foreach(name wave group_bytes cu l1_bytes_per_cu l2_bytes)
        list(GET f ${i} cell)
        tsv_number("${cell}" "${row_where}" "${name}" 2147483647 v_${name})
        math(EXPR i "${i} + 1")
    endforeach()
    foreach(name local_Bps transfer_Bps)
        list(GET f ${i} cell)
        tsv_number("${cell}" "${row_where}" "${name}" 9223372036854775807 v_${name})
        math(EXPR i "${i} + 1")
    endforeach()
    foreach(name attn_gate_fuse conv_l2_fold add_norm_fuse cb_seq set_nowait staging_keep probe_dispatch probe_stage probe_blit probe_timestamp)
        list(GET f ${i} cell)
        if (name STREQUAL "cb_seq")
            tsv_word("${cell}" "${row_where}" "${name}" v_${name} 0=0 1=1 2=2 3=3 4=4 5=5 6=6 7=7 8=8 -=-1)
        else()
            tsv_word("${cell}" "${row_where}" "${name}" v_${name} on=1 off=0 -=-1)
        endif()
        math(EXPR i "${i} + 1")
    endforeach()
    string(APPEND rows "    { \"${card}\", \"${chip}\", \"${arch}\", ${v_wave}, ${v_group_bytes}, ${v_cu}, ${v_l1_bytes_per_cu}, ${v_l2_bytes}, ${v_local_Bps}LL, ${v_transfer_Bps}LL, ${v_attn_gate_fuse}, ${v_conv_l2_fold}, ${v_add_norm_fuse}, ${v_cb_seq}, ${v_set_nowait}, ${v_staging_keep}, ${v_probe_dispatch}, ${v_probe_stage}, ${v_probe_blit}, ${v_probe_timestamp} },\n")
endforeach()
if (first)
    message(FATAL_ERROR "${where}: the section has no header row")
endif()

set(text "// generated from data/device.tsv [devices] by device/gen-devices.cmake; do not edit
#pragma once

#include <stdint.h>
#include <string.h>

// One row of data/device.tsv [devices]. card = the card's Metal device name, compared whole.
// -1 or \"\" = the cell is -: the value set before the row is read stays.
struct ggml_metal_device_row {
    const char * card;
    const char * chip;
    const char * arch_family;
    int32_t      wave;            // threads of one wave as recorded; the backend probes the width
    int32_t      group_bytes;     // threadgroup memory as recorded; the backend asks the device
    int32_t      compute_units;
    int32_t      l1_cache_per_cu; // bytes
    int32_t      l2_cache_total;  // bytes
    int64_t      local_bw;        // bytes/s
    int64_t      transfer_bw;     // bytes/s
    int32_t      attn_gate_fuse;  // attention gate copy, sigmoid, copy, multiply in one launch: 1 on, 0 off
    int32_t      conv_l2_fold;    // conv_state also writes the split_l2 norms of q and k: 1 on, 0 off
    int32_t      add_norm_fuse;   // residual add, rms norm of its sum, weight multiply in one launch: 1 on, 0 off
    int32_t      cb_seq;          // command buffers after the first, encoded in order on the calling thread: 0-8
    int32_t      set_nowait;      // input uploads committed without waiting for each: 1 on, 0 off
    int32_t      staging_keep;    // one logits staging buffer kept per context: 1 on, 0 off
    int8_t       probe_dispatch;  // counter sampling worked at this point: 1 yes, 0 no
    int8_t       probe_stage;
    int8_t       probe_blit;
    int8_t       probe_timestamp;
};

// the last row is the end mark
static const struct ggml_metal_device_row ggml_metal_device_rows[] = {
${rows}    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

static const int ggml_metal_device_rows_n = (int) (sizeof(ggml_metal_device_rows) / sizeof(ggml_metal_device_rows[0])) - 1;

// The row of the card with this Metal device name, or 0 when it has none.
static inline const struct ggml_metal_device_row * ggml_metal_device_row_of(const char * name) {
    for (int i = 0; name != 0 && i < ggml_metal_device_rows_n; i++) {
        if (strcmp(ggml_metal_device_rows[i].card, name) == 0) {
            return &ggml_metal_device_rows[i];
        }
    }
    return 0;
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
