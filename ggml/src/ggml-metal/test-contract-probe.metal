// P11.66: contract-version probe for the offline test payload. Included once by the generated
// wrapper after the embedded MSL; not part of the production library.
#include <metal_stdlib>
using namespace metal;

kernel void lane_contract_version_probe(
        device int * buffer0 [[buffer(0)]],
        uint grid_id [[thread_position_in_grid]]) {
    if (grid_id == 0) {
        buffer0[0] = 3;
    }
}
