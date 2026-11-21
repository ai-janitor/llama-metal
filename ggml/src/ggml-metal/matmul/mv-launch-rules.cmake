# The rules of data/matmul.tsv [launches]: per launch, the cells its code reads and the values that code is known to run.
# device/gen-launches.cmake reads this file and refuses the table when a default row leaves a read cell -, when a row
# writes a cell the launch does not read, and when a written value is outside its rule. A launch with a row and no
# line here is refused, and so is a line for a launch that has no row.
#   launch_rule(<launch> <cell> range <lo> <hi>)    a whole number from lo to hi
#   launch_rule(<launch> <cell> pow2 <lo> <hi>)     a power of two from lo to hi
#   launch_rule(<launch> <cell> oneof <value>...)   one of the values; wave = the cell says wave
# A rule holds what was read in the launch's code and kernel, or what the code ran before the number was a row (a cap:
# every count up to the old cap; a start that doubles: the powers of two from the old start; memory: from the old size
# up to 8192 units = 32768 bytes, the smallest threadgroup memory of data/device.tsv [devices]). To widen a rule: read
# the kernel and the launch's code, or run the op on a card with the new value and the same output; then change the
# line and say where the value comes from. A new launch gets its lines here with its default row.

launch_rule(group_kq units range 2 8)         # 8 = the kernel's member array (matmul/kernel-grouped-mv.metal)
launch_rule(group_q8 units range 2 4)         # the kernel's four buffer slots (matmul/dispatch-scalar.cpp)
launch_rule(q8_rows8 units oneof 4 8)         # the kernel's instances (matmul/kernel-scalar-basic.metal)
launch_rule(ext nsg oneof 2)
# threads along K per simdgroup: a power of two that divides the wave (matmul/kernel-scalar.metal); the old code ran 16, 8, 4
launch_rule(ext_k_wide units pow2 4 16)
launch_rule(ext_k_wide work_min range 1 65536) # K is a multiple of it: a divisor
launch_rule(ext_k_wide cap range 1 65536)      # compared with the launch's columns only
launch_rule(ext_k_mid units pow2 4 16)
launch_rule(ext_k_mid work_min range 1 65536)
launch_rule(ext_k_narrow units pow2 4 16)
# activation columns per threadgroup: the kernel's instances r1_2 to r1_5 (matmul/pipeline.cpp names them)
foreach(n 2 3 4 5 6 7 8 other)
    launch_rule(ext_cols_${n} units oneof 2 3 4 5)
endforeach()
# divisors of the card's threadgroup limit
foreach(l split_tiled split_mv split_ext)
    launch_rule(${l} units range 1 65536)
endforeach()
launch_rule(id_dense nsg range 1 4)           # the launch takes the smaller of this and its count by columns
launch_rule(id_dense units range 1 65536)     # columns per simdgroup: a divisor
