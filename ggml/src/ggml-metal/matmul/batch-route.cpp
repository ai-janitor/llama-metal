#include "batch-route.h"

#include "batch-routes-table.h"

const struct mv_batch_row * ggml_metal_batch_rows_get(int * n) {
    *n = (int) (sizeof(mv_batch_rows) / sizeof(mv_batch_rows[0])) - 1;
    return mv_batch_rows;
}

const struct mv_batch_row * ggml_metal_batch_row_in(
        const struct mv_batch_row * rows, int n, uint32_t pci_vendor, uint32_t pci_device, enum ggml_type type) {
    for (int i = 0; i < n; ++i) {
        if (rows[i].pci_vendor == pci_vendor && rows[i].pci_device == pci_device && rows[i].type == type) {
            return &rows[i];
        }
    }
    return nullptr;
}

int ggml_metal_batch_rows_count(uint32_t pci_vendor, uint32_t pci_device) {
    int n = 0, count = 0;
    const struct mv_batch_row * rows = ggml_metal_batch_rows_get(&n);
    for (int i = 0; i < n; ++i) {
        if (rows[i].pci_vendor == pci_vendor && rows[i].pci_device == pci_device) {
            count++;
        }
    }
    return count;
}

enum mv_batch_route ggml_metal_batch_route_decide(const struct mv_batch_in * in) {
    const struct mv_format_row * f = in->fmt;
    if (f == nullptr) {
        return MV_BATCH_NONE;
    }
    // The tiled kernel works on 64-row tiles of untransposed operands, on cards without matrix hardware.
    const bool tiled_ok = !in->has_matrix_hw && f->tiled && in->tiled_vendor_ok && in->act_f32 &&
                          !in->trans0 && !in->trans1 && in->m >= 64;

    // A card row sends the op to the tiled kernel ahead of the small-batch kernel.
    if (tiled_ok && in->tiled_from > 0 && in->n >= in->tiled_from) {
        return MV_BATCH_TILED;
    }

    if (in->act_f32 && in->k % 128 == 0) {
        const bool wide = in->n <= GGML_METAL_NE11_MM_MIN || (!in->has_matrix_hw && f->ext_wide);
        if (f->ext == MV_EXT_SMALL && in->n >= (in->ext_from > 0 ? in->ext_from : 2) && wide) {
            return MV_BATCH_EXT;
        }
        // Below the card's column count a K-quant runs once per column, or a column-form kernel takes the batch.
        if (f->ext == MV_EXT_KQUANT && in->n >= (in->ext_from > 0 ? in->ext_from : 4) && wide && (in->n > 4 || !in->cols_takes)) {
            return MV_BATCH_EXT;
        }
    }

    if (tiled_ok && in->n > GGML_METAL_NE11_MM_MIN) {
        return MV_BATCH_TILED;
    }
    return MV_BATCH_NONE;
}
