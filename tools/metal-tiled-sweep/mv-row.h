// One row of ggml/src/ggml-metal/data/matmul.tsv [matvec_shapes] as llama-metal-tiled-sweep --matvec prints it: the
// twelve cells of the section's header, in its order, separated by one tab. tests/Makefile check-sweep-row holds a
// row made here to the table's generator (matmul/gen-matvec.cmake).
#pragma once

#include <cstdint>
#include <string>

// kind: 0 single, 1 group. prep: 1 on, 0 off. wave and group_bytes: the card's shape as it reported them.
static inline std::string mv_row_text(const std::string & card, const std::string & wave, const std::string & group_bytes, int kind, const std::string & weight,
                                      int64_t k, int64_t m, int cols, int prep, int nr0, const std::string & form, const std::string & source) {
    return card + "\t" + wave + "\t" + group_bytes + "\t" + (kind == 1 ? "group" : "single") + "\t" + weight + "\t" + std::to_string(k) + "\t" + std::to_string(m) + "\t" +
           std::to_string(cols) + "\t" + (prep ? "on" : "off") + "\t" + std::to_string(nr0) + "\t" + form + "\t" + source;
}
