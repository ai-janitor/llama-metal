# The rules of data/device.tsv [launches]: per launch, the cells its code reads and the values that code is known to run.
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

# threads before a threadgroup doubles: the launch already runs every power of two from its start up to the limit
foreach(l l2_norm set_rows tri soft_max argmax top_k_mask sum sum_rows)
    launch_rule(${l} start pow2 32 1024)
endforeach()
# threads that do not grow: only the number the code held is known to run
launch_rule(group_norm start oneof 32)
launch_rule(diag start oneof 32)
launch_rule(norm start oneof wave)            # the code read the card's wave
launch_rule(l2_norm mem_units oneof wave)     # one float per thread of a wave
launch_rule(group_norm mem_units oneof wave)
launch_rule(norm mem_units range 1024 8192)   # one unit per thread the launch may take (1024)
# caps: the launch takes the smaller of the cap and its element count, so every count up to the old cap already runs
foreach(l concat_packed unary glu binary conv_2d cpy act_copy small_gate)
    launch_rule(${l} cap range 1 256)
endforeach()
foreach(l concat rope pad pad_reflect_1d arange timestep_embedding irfft fold argmax)
    launch_rule(${l} cap range 1 1024)
endforeach()
launch_rule(argsort_merge cap range 1 512)
launch_rule(concat start range 1 32)          # rows under this are packed; the old code packed rows under 32
launch_rule(soft_max work_min range 1 1048576) # compared with threads x rows only
launch_rule(soft_max mem_units range 128 8192)
launch_rule(argmax work_min range 1 1048576)
launch_rule(argmax mem_units range 256 8192)
# tokens per threadgroup: the code doubles 8 while it is under the tokens and under cap (ssm/ops-ssm.cpp), so cap is
# a power of two from 8; the old code ran 2 and 8 to 256
launch_rule(ssm_conv_batch start pow2 2 256)
launch_rule(ssm_conv_batch cap pow2 8 256)
launch_rule(cumsum mem_units range 32 8192)
launch_rule(count_equal start oneof 32)       # threads per simdgroup
launch_rule(count_equal nsg pow2 1 32)        # the simdgroups double from 1 up to this (misc/pipeline-misc.cpp)
launch_rule(count_equal mem_units range 32 8192)
launch_rule(solve_tri nsg oneof 8)
launch_rule(ssm_scan start oneof 32)
launch_rule(dsv4_hc nsg range 1 4)            # the launch takes the smaller of this and its tokens or tiles
