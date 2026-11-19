#include "decide.h"
#include "lanes/select.h"

#include "act-copy-table.h"

const struct ggml_metal_act_copy_row * ggml_metal_act_copy_rows_get(int * n) {
    *n = (int) (sizeof(ggml_metal_act_copy_rows) / sizeof(ggml_metal_act_copy_rows[0]));
    return ggml_metal_act_copy_rows;
}

const struct ggml_metal_act_copy_row * ggml_metal_act_copy_decide_in(
        const struct ggml_metal_act_copy_row * rows, int n, uint32_t pci_vendor, uint32_t pci_device,
        int family, int route, enum ggml_type precision, enum ggml_type src, int64_t rereads, uint64_t bytes, int * why) {
    int step = GGML_METAL_ACT_COPY_APPLIES;
    const struct ggml_metal_act_copy_row * row = nullptr;
    if (precision != GGML_TYPE_BF16 && precision != GGML_TYPE_F16) {
        step = GGML_METAL_ACT_COPY_NO_PRECISION;
    } else if (src != GGML_TYPE_F32) {
        step = GGML_METAL_ACT_COPY_NOT_F32;
    } else {
        for (int i = 0; i < n; ++i) {
            if (rows[i].family == family && rows[i].route == route && rows[i].pci_vendor == pci_vendor &&
                rows[i].pci_device == pci_device && rows[i].copy == (int) precision) {
                row = &rows[i];
                break;
            }
        }
        if (row == nullptr) {
            step = GGML_METAL_ACT_COPY_NO_ROW;
        } else if (!row->applies) {
            step = GGML_METAL_ACT_COPY_ROW_OFF;
        } else if (rereads < row->min_rereads) {
            step = GGML_METAL_ACT_COPY_FEW_REREADS;
        } else if (bytes > row->max_bytes) {
            step = GGML_METAL_ACT_COPY_TOO_BIG;
        }
    }
    if (why) *why = step;
    return row;
}

const char * ggml_metal_act_copy_why_name(int why) {
    switch (why) {
        case GGML_METAL_ACT_COPY_APPLIES:      return "applies";
        case GGML_METAL_ACT_COPY_NO_PRECISION: return "the model's precision is f32 or not known";
        case GGML_METAL_ACT_COPY_NOT_F32:      return "the activation tensor is not f32";
        case GGML_METAL_ACT_COPY_NO_ROW:       return "no row for this card";
        case GGML_METAL_ACT_COPY_ROW_OFF:      return "the row is off";
        case GGML_METAL_ACT_COPY_FEW_REREADS:  return "too few re-reads";
        case GGML_METAL_ACT_COPY_TOO_BIG:      return "the copy is over the row's size";
        default:                               return "?";
    }
}
