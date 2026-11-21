#include "matmul/mv-launch.h"

#include "mv-launch-rows.h" // generated; this file is its only reader, so a changed number rebuilds this file alone

struct ggml_metal_launch ggml_metal_mv_launch_pick(int id, const char * card, int32_t wave) {
    return ggml_metal_launch_pick_rows(ggml_metal_mv_launch_rows, id, card, wave);
}
