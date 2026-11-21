#!/bin/sh
# Host check of data/device.tsv [launches] (make -C tests check-launch-rows).
#   1. the shipped tables generate and compile (data/device.tsv, data/matmul.tsv); the numbers every launch gets (a card
#      without a row at a wave of 32 and of 64, and the three cards with device rows) equal the recorded numbers,
#      tests/launch_rows_expected.tsv and tests/mv_launch_rows_expected.tsv: a changed number is a changed line there,
#      in the same commit.
#   2. a table with test rows: the default row, the row of the wave and the card's row apply in that order.
#   3. each refused table is refused by its text: the form of a row, a default row that leaves a read cell -, a value
#      outside the rule of its launch (device/launch-rules.cmake, matmul/mv-launch-rules.cmake), a launch without a rule.
#   4. data/matmul.tsv with two of its rows in each other's place gives every launch the same numbers by name.
# usage: check_launch_rows.sh <ggml-metal dir> <tests dir> <fresh output dir>
dir="$1"; tests="$2"; out="$3"
gen="$dir/device/gen-launches.cmake"
cxx="c++ -std=c++17 -Wall -Wextra -Werror -I$dir/../../include -I$dir"
mkdir -p "$out/ok" "$out/fix" "$out/refused" || exit 1
fail=0

# shipped NAME TABLE EXPECTED [generator and compiler arguments of another table]
shipped() { name="$1"; table="$2"; expected="$3"; genargs="$4"; def="$5"; src="$6"
    mkdir -p "$out/$name"
    cmake -DTABLE="$table" $genargs -DOUT_IDS="$out/$name/$7launch-ids.h" -DOUT_ROWS="$out/$name/$7launch-rows.h" -P "$gen" || exit 1
    $cxx $def -I"$out/$name" "$tests/test_metal_launch_rows.cpp" "$dir/device/launch-plan.cpp" $src -o "$out/$name/test" || exit 1
    "$out/$name/test" > "$out/$name/numbers.tsv" || { cat "$out/$name/numbers.tsv"; exit 1; }
    if cmp -s "$out/$name/numbers.tsv" "$expected"; then
        echo "launch rows ($name): $(awk -F'\t' '{ print $1 }' "$out/$name/numbers.tsv" | sort -u | wc -l | tr -d ' ') launches, numbers as recorded"
    else
        echo "FAIL launch rows ($name): the numbers differ from $expected"; diff "$expected" "$out/$name/numbers.tsv" | head -20; fail=1
    fi
    # control: one recorded number changed must be reported
    if [ -s "$out/$name/numbers.tsv" ]; then
        awk -F'\t' 'BEGIN { OFS = FS } NR == 1 { $NF = $NF + 1 } { print }' "$expected" > "$out/$name/expected-control.tsv"
        if cmp -s "$out/$name/numbers.tsv" "$out/$name/expected-control.tsv"; then echo "CONTROL number-changed ($name): not reported -- NOT CAUGHT"; fail=1
        else echo "CONTROL number-changed ($name, expected to fail): caught"; fi
    fi; }
# the device table's generated headers are needed by every build of launch-plan.cpp
shipped ok "$dir/data/device.tsv" "$tests/launch_rows_expected.tsv" "" "" "" ""
mvargs="-DLABEL=matmul.tsv -DKNOWN=tiles,formats,matvec_defaults,matvec_shapes,launches -DPREFIX=GGML_METAL_MV_LAUNCH -DSYMBOL=ggml_metal_mv_launch -DRULES=$dir/matmul/mv-launch-rules.cmake"
shipped matmul "$dir/data/matmul.tsv" "$tests/mv_launch_rows_expected.tsv" "$mvargs" \
    "-DLAUNCH_MV -I$out/ok" "$dir/matmul/mv-launch.cpp" "mv-"

# the small-batch matvec's rows per threadgroup (checked inside the matmul test above); control: one wrong expectation
$cxx -DLAUNCH_MV -DEXT_ROWS_W8=1 -I"$out/ok" -I"$out/matmul" "$tests/test_metal_launch_rows.cpp" "$dir/device/launch-plan.cpp" "$dir/matmul/mv-launch.cpp" -o "$out/matmul/test-control" || exit 1
if "$out/matmul/test-control" > /dev/null 2> "$out/matmul/control.txt"; then echo "CONTROL ext-rows-width-8: a K width above the compiled width was given rows -- NOT CAUGHT"; fail=1
else echo "CONTROL ext-rows-width-8 (expected to fail): $(grep -c '^FAIL' "$out/matmul/control.txt") caught"; fi

row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@"; }
fixture() { cat "$dir/data/device.tsv"; row t_reduce any any 32 256 1024 - - 128 "test row"; row t_reduce any 64 wave - - - - wave "test row"
    row t_reduce "Test Card A" 64 - 512 - - - - "test row"; row t_pack any any - - 256 4 8 - "test row"; }
fixture > "$out/fix/device.tsv"
# the rules of the test rows, after the shipped rules
rules="$out/fix/rules.cmake"
{ echo "include(\"$dir/device/launch-rules.cmake\")"
  echo "launch_rule(t_reduce start oneof wave 32)"; echo "launch_rule(t_reduce work_min range 1 1024)"; echo "launch_rule(t_reduce cap range 1 1024)"
  echo "launch_rule(t_reduce mem_units oneof wave 128)"; echo "launch_rule(t_pack cap range 1 256)"; echo "launch_rule(t_pack nsg range 1 8)"
  echo "launch_rule(t_pack units range 1 8)"; } > "$rules"
cmake -DTABLE="$out/fix/device.tsv" -DRULES="$rules" -DOUT_IDS="$out/fix/launch-ids.h" -DOUT_ROWS="$out/fix/launch-rows.h" -P "$gen" || exit 1
$cxx -DLAUNCH_FIXTURE -I"$out/fix" "$tests/test_metal_launch_rows.cpp" "$dir/device/launch-plan.cpp" -o "$out/fix/test" || exit 1
"$out/fix/test" || fail=1
# control: without the wave row the fixture's expectations must fail
fixture | awk -F'\t' '!($1 == "t_reduce" && $2 == "any" && $3 == 64)' > "$out/fix/no-wave-row.tsv"
mkdir -p "$out/fix/c" && cmake -DTABLE="$out/fix/no-wave-row.tsv" -DRULES="$rules" -DOUT_IDS="$out/fix/c/launch-ids.h" -DOUT_ROWS="$out/fix/c/launch-rows.h" -P "$gen" \
    && $cxx -DLAUNCH_FIXTURE -I"$out/fix/c" "$tests/test_metal_launch_rows.cpp" "$dir/device/launch-plan.cpp" -o "$out/fix/c/test" || exit 1
if "$out/fix/c/test" > "$out/fix/c/out.txt"; then echo "CONTROL wave-row-removed: the fixture still passed -- NOT CAUGHT"; fail=1
else echo "CONTROL wave-row-removed (expected to fail): $(grep -c '^FAIL' "$out/fix/c/out.txt") caught"; fi

# refused tables: name, a part of the refusal's text that stands on its first line (CMake wraps a long message), then
# the rows added to the fixture
refused() { name="$1"; text="$2"; shift 2
    { fixture; "$@"; } > "$out/refused/$name.tsv"
    if cmake -DTABLE="$out/refused/$name.tsv" -DRULES="$rules" -DOUT_IDS="$out/refused/$name-ids.h" -DOUT_ROWS="$out/refused/$name-rows.h" -P "$gen" > "$out/refused/$name.txt" 2>&1; then
        echo "CONTROL $name: the table was taken -- NOT CAUGHT"; fail=1
    elif grep -qF "$text" "$out/refused/$name.txt"; then echo "CONTROL $name (expected to fail): refused, '$text'"
    else echo "CONTROL $name: refused without '$text' -- NOT CAUGHT"; tail -3 "$out/refused/$name.txt"; fail=1; fi; }
refused two-defaults      "two rows for card any, wave any"            row t_pack any any - - 128 4 8 - "test row"
refused no-default-first  "the launch's default row (card any, wave any)" row t_new any 64 32 - - - - - "test row"
refused card-without-wave "a card's row writes the wave the card" row t_pack "Test Card A" any - - 128 - - - "test row"
refused start-zero        "start is a whole number from 1, wave, or -" row t_zero any any 0 - - - - - "test row"
refused cap-word          "cap is a whole number from 1 to 1048576"    row t_word any any 32 - wave - - - "test row"
refused cap-below-start   "cap 16 is below start 32"                   row t_low any any 32 - 16 - - - "test row"
refused cell-not-read     "the launch does not read nsg"               row t_reduce any 32 - - - 2 - - "test row"
refused no-rule           "the launch has no launch_rule line"         row t_norule any any 32 - - - - - "test row"
refused card-cap-12       "cap 12 is not a power of two" row ssm_conv_batch "Test Card A" 64 - - 12 - - - "test row"
refused wave-not-taken    "start wave is not a power of two from 32 to 1024" row sum any 64 wave - - - - - "test row"
refused no-number         "the row writes no number"                   row t_none any any - - - - - - "test row"
refused no-evidence       "evidence says where the cells come from"    row t_ev any any 32 - - - - - "-"
refused name-upper        "launch is a name in lower-case letters, digits and" row T_Up any any 32 - - - - - "test row"
refused nine-cells        "expected 10 cells, got 9"                   printf 't_nine\tany\tany\t32\t-\t-\t-\t-\ttest row\n'
together() { row ssm_conv_batch any 64 16 - - - - - "test row"; row ssm_conv_batch "Test Card A" 64 - - 8 - - - "test row"; }
refused merged-cap-below  "cap 8 is below start 16 with the rows"      together
refused two-card-rows     "two rows for card Test Card A, wave 64"     row t_reduce "Test Card A" 64 - 128 - - - - "test row"
# refused_table NAME TEXT GENERATOR-ARGUMENTS COMMAND...: the same for a whole table the command prints
refused_table() { name="$1"; text="$2"; args="$3"; shift 3
    "$@" > "$out/refused/$name.tsv"
    if cmake -DTABLE="$out/refused/$name.tsv" $args -DOUT_IDS="$out/refused/$name-ids.h" -DOUT_ROWS="$out/refused/$name-rows.h" -P "$gen" > "$out/refused/$name.txt" 2>&1; then
        echo "CONTROL $name: the table was taken -- NOT CAUGHT"; fail=1
    elif grep -qF "$text" "$out/refused/$name.txt"; then echo "CONTROL $name (expected to fail): refused, '$text'"
    else echo "CONTROL $name: refused without '$text' -- NOT CAUGHT"; tail -3 "$out/refused/$name.txt"; fail=1; fi; }
# cell TABLE LAUNCH COLUMN VALUE: the table with one cell of the launch's default row changed
cell() { awk -F'\t' -v l="$2" -v c="$3" -v v="$4" 'BEGIN { OFS = FS } /^\[/ { sec = $0 } sec == "[launches]" && $1 == l && $2 == "any" && $3 == "any" { $c = v } { print }' "$1"; }
refused_table default-cell-missing "work_min is - in the default row" "" cell "$dir/data/device.tsv" soft_max 5 -
refused_table default-cap-12       "cap 12 is not a power of two"           "" cell "$dir/data/device.tsv" ssm_conv_batch 6 12
refused_table default-mem-small    "mem_units 64 is not from 128 to 8192"                 "" cell "$dir/data/device.tsv" soft_max 9 64
refused_table mv-divisor-missing   "work_min is - in the default row" "$mvargs" cell "$dir/data/matmul.tsv" ext_k_wide 5 -
refused_table mv-k-three           "units 3 is not a power of two"           "$mvargs" cell "$dir/data/matmul.tsv" ext_k_wide 8 3
refused_table mv-rows-six          "units 6 is not one of 4, 8"                           "$mvargs" cell "$dir/data/matmul.tsv" q8_rows8 8 6
refused_table mv-cols-six          "units 6 is not one of 2, 3, 4, 5"                     "$mvargs" cell "$dir/data/matmul.tsv" ext_cols_6 8 6
echo 'launch_rule(t_gone cap range 1 2)' > "$out/refused/stale-rules.cmake"; cat "$dir/device/launch-rules.cmake" >> "$out/refused/stale-rules.cmake"
refused_table stale-rule           "a rule names launch t_gone, which has no default row"  "-DRULES=$out/refused/stale-rules.cmake" cat "$dir/data/device.tsv"

# rows in each other's place: the launches are picked by name, so the numbers by name do not move
mkdir -p "$out/swap"
awk -F'\t' '$1 == "ext_cols_3" && !a { a = NR } $1 == "ext_cols_5" && !b { b = NR } { line[NR] = $0 } END { t = line[a]; line[a] = line[b]; line[b] = t; for (i = 1; i <= NR; i++) print line[i] }' "$dir/data/matmul.tsv" > "$out/swap/matmul.tsv"
cmake -DTABLE="$out/swap/matmul.tsv" $mvargs -DOUT_IDS="$out/swap/mv-launch-ids.h" -DOUT_ROWS="$out/swap/mv-launch-rows.h" -P "$gen" || exit 1
$cxx -DLAUNCH_MV -I"$out/ok" -I"$out/swap" "$tests/test_metal_launch_rows.cpp" "$dir/device/launch-plan.cpp" "$dir/matmul/mv-launch.cpp" -o "$out/swap/test" || exit 1
"$out/swap/test" > "$out/swap/numbers.tsv" || exit 1
sort "$out/swap/numbers.tsv" > "$out/swap/sorted.tsv"; sort "$tests/mv_launch_rows_expected.tsv" > "$out/swap/expected-sorted.tsv"
if cmp -s "$out/swap/numbers.tsv" "$tests/mv_launch_rows_expected.tsv"; then echo "CONTROL rows-swapped: the table's order did not change -- NOT CAUGHT"; fail=1
elif cmp -s "$out/swap/sorted.tsv" "$out/swap/expected-sorted.tsv"; then echo "launch rows (ext_cols_3 and ext_cols_5 in each other's place): every launch keeps its numbers by name"
else echo "FAIL launch rows: rows in another order change a launch's numbers"; diff "$out/swap/expected-sorted.tsv" "$out/swap/sorted.tsv" | head; fail=1; fi
[ "$fail" -eq 0 ]
