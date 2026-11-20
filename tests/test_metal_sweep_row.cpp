// Prints one row of data/matmul.tsv [matvec_shapes] through the function llama-metal-tiled-sweep --matvec prints its
// rows with (tools/metal-tiled-sweep/mv-row.h). tests/Makefile check-sweep-row puts it into a copy of the table and
// runs the table's generator on it. The row is the D700's measured 4-column row, so the generated rows must not change.
#include "mv-row.h"

#include <cstdio>

int main() {
    puts(mv_row_text("AMD Radeon HD - FirePro D700", "64", "32768", 0, "q4_K", 9216, 2560, 4, 1, 2, "ap", "a row as the sweep prints it").c_str());
    return 0;
}
