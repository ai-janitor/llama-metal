// What a card's row of data/device.tsv [devices] gives the device profile: its sizes (compute units, caches, memory
// rates) and its scheduling settings. One function, called when the device is set up (device/init.m); the host test
// of the matvec plan calls it too. The row is found by the card's Metal device name compared whole. A card without a
// row, and a cell that is -, leave the value 0 (not known): the rules then run their defaults for an unmeasured card.
#pragma once

#include "ggml-device-profile.h"
#include "device-rows.h" // generated from data/device.tsv [devices]

#include <stddef.h>
#include <stdint.h>

// Sets the fields named below from the row of `name`; returns the row, or NULL when the card has none.
static inline const struct ggml_metal_device_row * ggml_metal_device_sizes_load(struct ggml_device_profile * profile, const char * name) {
    profile->compute_units      = 0;
    profile->l1_cache_per_cu    = 0;
    profile->l2_cache_total     = 0;
    profile->local_bandwidth    = 0;
    profile->transfer_bandwidth = 0;
    profile->attn_gate_fuse     = 0;
    profile->conv_l2_fold       = 0;
    profile->add_norm_fuse      = 0;
    profile->cb_seq             = 0;
    profile->set_nowait         = 0;
    profile->staging_keep       = 0;
    const struct ggml_metal_device_row * row = ggml_metal_device_row_of(name);
    if (row == NULL) {
        return NULL;
    }
    if (row->compute_units   >= 0) profile->compute_units      = (uint32_t) row->compute_units;
    if (row->l1_cache_per_cu >= 0) profile->l1_cache_per_cu    = (uint32_t) row->l1_cache_per_cu;
    if (row->l2_cache_total  >= 0) profile->l2_cache_total     = (uint32_t) row->l2_cache_total;
    if (row->local_bw        >= 0) profile->local_bandwidth    = (uint64_t) row->local_bw;
    if (row->transfer_bw     >= 0) profile->transfer_bandwidth = (uint64_t) row->transfer_bw;
    if (row->attn_gate_fuse  >= 0) profile->attn_gate_fuse     = (uint32_t) row->attn_gate_fuse;
    if (row->conv_l2_fold    >= 0) profile->conv_l2_fold       = (uint32_t) row->conv_l2_fold;
    if (row->add_norm_fuse   >= 0) profile->add_norm_fuse      = (uint32_t) row->add_norm_fuse;
    if (row->cb_seq          >= 0) profile->cb_seq             = (uint32_t) row->cb_seq;
    if (row->set_nowait      >= 0) profile->set_nowait         = (uint32_t) row->set_nowait;
    if (row->staging_keep    >= 0) profile->staging_keep       = (uint32_t) row->staging_keep;
    return row;
}
