#!/bin/sh
# Host check of data/attention.tsv [launch] (part of make -C tests check-fa-routes).
#   1. the shipped rows give the numbers the routes held in code (written in test_metal_fa_launch.cpp);
#      control: the vec default row with another small_queries cell must fail that test.
#   2. a table with test rows: the row of the wave, then the card's row.
#   3. each refused table is refused by a part of its text that stands on the message's first line.
# usage: check_fa_launch.sh <ggml-metal dir> <tests dir> <fresh output dir>
dir="$1"; tests="$2"; out="$3"
gen="$dir/flash-attn/gen-routes.cmake"; table="$dir/data/attention.tsv"
cxx="c++ -std=c++17 -Wall -Wextra -Werror -I$dir"
fail=0
build() { mkdir -p "$out/$1" && cmake -DROUTES="$2" -DOUT="$out/$1/routes-table.h" -DOUT_LAUNCH="$out/$1/fa-launch-table.h" -P "$gen" \
    && $cxx $3 -I"$out/$1" "$tests/test_metal_fa_launch.cpp" "$dir/flash-attn/launch.cpp" -o "$out/$1/test"; }
build ok "$table" "" || exit 1
"$out/ok/test" || fail=1
mkdir -p "$out/control"
awk -F'\t' 'BEGIN { OFS = FS } /^\[/ { sec = $0 } sec == "[launch]" && $1 == "vec" { $8 = 21 } { print }' "$table" > "$out/control/attention.tsv"
build control "$out/control/attention.tsv" "" || exit 1
if "$out/control/test" > "$out/control/out.txt"; then echo "CONTROL vec-small-queries-21: the shipped numbers still passed -- NOT CAUGHT"; fail=1
else echo "CONTROL vec-small-queries-21 (expected to fail): $(grep -c '^FAIL' "$out/control/out.txt") caught"; fi

row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@"; }
mkdir -p "$out/fix" "$out/refused"
{ cat "$table"; row vec any 64 2 - - - - - "test row"; row vec "Test Card A" 64 1 - - - - - "test row"; row scalar "Test Card A" 64 8 - - - - - "test row"; } > "$out/fix/attention.tsv"
build fix "$out/fix/attention.tsv" -DFA_LAUNCH_FIXTURE || exit 1
"$out/fix/test" || fail=1

refused() { name="$1"; text="$2"; shift 2
    "$@" > "$out/refused/$name.tsv"
    if cmake -DROUTES="$out/refused/$name.tsv" -DOUT="$out/refused/$name-routes.h" -DOUT_LAUNCH="$out/refused/$name-launch.h" -P "$gen" > "$out/refused/$name.txt" 2>&1; then
        echo "CONTROL $name: the table was taken -- NOT CAUGHT"; fail=1
    elif grep -qF "$text" "$out/refused/$name.txt"; then echo "CONTROL $name (expected to fail): refused, '$text'"
    else echo "CONTROL $name: refused without '$text' -- NOT CAUGHT"; tail -3 "$out/refused/$name.txt"; fail=1; fi; }
plus() { cat "$table"; row "$@"; }
without() { awk -F'\t' -v r="$1" '/^\[/ { sec = $0 } !(sec == "[launch]" && $1 == r)' "$table"; }
refused card-workgroups   "workgroups holds for every card"          plus vec "Test Card A" 64 - - - 16 - - "test row"
refused wave-small        "small_queries holds for every card"       plus vec any 64 - - - - 8 - "test row"
refused not-read          "the route does not read threads"          plus scalar any 64 - - - - - 64 "test row"
refused vec-three         "simdgroups 3 is not one of 1, 2, 4"      plus vec any 64 3 - - - - - "test row"
refused scalar-nine       "simdgroups 9 is not from 1 to 8"         plus scalar "Test Card A" 64 9 - - - - - "test row"
refused matrix-six        "wide_simdgroups 6 is not one of 4, 8"    plus matrix any 64 - 6 - - - - "test row"
edit_default() { awk -F'\t' -v r="$1" -v c="$2" -v v="$3" 'BEGIN { OFS = FS } /^\[/ { sec = $0 } sec == "[launch]" && $1 == r && $2 == "any" && $3 == "any" { $c = v } { print }' "$table"; }
refused vec-workgroups-16 "workgroups 16 is not one of 32"          edit_default vec 7 16
refused vec-small-34      "small_queries 34 is not from 1 to 32"    edit_default vec 8 34
refused prepass-64        "threads 64 is not one of 32"             edit_default prepass 9 64
refused card-any-wave     "a card's row writes the wave the card"    plus scalar "Test Card A" any 8 - - - - - "test row"
refused two-defaults      "two rows for card any, wave any"          plus scalar any any 8 - - - - - "test row"
refused unknown-route     "launch_route is scalar, vec, matrix or"   plus tiled any any 8 - - - - - "test row"
refused no-number         "the row writes no number"                 plus scalar any 64 - - - - - - "test row"
refused no-prepass        "route prepass has no default row"         without prepass
refused default-short     "the default row writes wide_head"         awk -F'\t' 'BEGIN { OFS = FS } /^\[/ { sec = $0 } sec == "[launch]" && $1 == "matrix" { $6 = "-" } { print }' "$table"
[ "$fail" -eq 0 ]
