#pragma once

// Which kernel takes a matmul with several activation columns. The per-card rows are matmul/batch-routes.tsv,
// turned into batch-routes-table.h at configure time. No device and no Metal here: the host test compiles this as is.
#include "formats.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum mv_batch_route { MV_BATCH_NONE, MV_BATCH_EXT, MV_BATCH_TILED };

// One row of batch-routes.tsv; 0 in a column count = the cell is "-".
struct mv_batch_row {
    uint32_t       pci_vendor;
    uint32_t       pci_device;
    enum ggml_type type;
    int            ext_from;
    int            tiled_from;
};

struct mv_batch_in {
    const struct mv_format_row * fmt; // row of the weight type, may be NULL
    int  ext_from;                    // the card's row cells, 0 = no row or "-"
    int  tiled_from;
    bool has_matrix_hw;
    bool tiled_vendor_ok;             // the tiled kernel of this type is verified for the card's vendor
    bool act_f32;
    bool trans0;                      // an operand is transposed
    bool trans1;
    int64_t k;                        // ne00
    int64_t m;                        // ne01
    int64_t n;                        // ne11, the activation columns
    bool cols_takes;                  // a column-form kernel takes this K-quant batch of up to 4 columns
};

const struct mv_batch_row * ggml_metal_batch_rows_get(int * n);

// The row of (card, weight type), or NULL.
const struct mv_batch_row * ggml_metal_batch_row_in(
        const struct mv_batch_row * rows, int n, uint32_t pci_vendor, uint32_t pci_device, enum ggml_type type);

// Rows of this card.
int ggml_metal_batch_rows_count(uint32_t pci_vendor, uint32_t pci_device);

// The route of one op. NONE = neither kernel: the matrix, one-column or fallback route decides.
enum mv_batch_route ggml_metal_batch_route_decide(const struct mv_batch_in * in);

#ifdef __cplusplus
}
#endif
