#!/bin/sh
# The tiled kernel is compiled once per plan of section [tiles] of data/matmul.tsv (one plan per distinct set of the
# eleven values; the default first, then in the order a plan first appears). matmul/gen-tiles.cmake turns the section
# into the kernel's plan blocks (tiled-plans.metal) and the host's list (tiled-compiled-table.h). This check runs the
# generator, builds both texts from the section's rows with its own code, and compares, whole text; it also checks
# that matmul/kernel-tiled.metal includes the generated blocks once and holds no plan block written by hand.
# usage: check_tiled_plan_decl.sh GGML_METAL_DIR OUT_DIR [KERNEL_FILE [TABLE]]   (the last two replace
#        matmul/kernel-tiled.metal and data/matmul.tsv, for the controls)
dir="$1"
out="$2"
[ -n "$out" ] || { echo "usage: check_tiled_plan_decl.sh GGML_METAL_DIR OUT_DIR [KERNEL_FILE [TABLE]]"; exit 2; }
kernel="$dir/matmul/kernel-tiled.metal"
[ -n "$3" ] && kernel="$3"
table="$dir/data/matmul.tsv"
[ -n "$4" ] && table="$4"
mkdir -p "$out" || exit 2
if ! cmake -DTABLE="$table" -DOUT_H="$out/tiled-compiled-table.h" -DOUT_METAL="$out/tiled-plans.metal" -P "$dir/matmul/gen-tiles.cmake" 2> "$out/gen.txt"; then
    echo "FAIL the generator refused the table:"; cat "$out/gen.txt"; exit 1
fi
fields=$(sed -n '/plan-fields-begin/,/plan-fields-end/p' "$dir/matmul/tiled-plan.h" | sed -n 's/^ *int32_t \([a-z]*\);.*/\1/p' | tr '\n' ' ')
nf=$(printf "%s" "$fields" | wc -w | tr -d ' ')
if [ "$nf" -eq 0 ]; then echo "FAIL no plan fields found in $dir/matmul/tiled-plan.h"; exit 1; fi
data=$(awk '/^#/ || /^$/ { next } /^\[/ { on = ($0 == "[tiles]"); next } on' "$table")
head=$(printf "%s\n" "$data" | head -1 | cut -f 10-20 | tr '\t' ' ')
if [ "$head " != "$fields" ]; then echo "FAIL columns 10 to 20 of [tiles] are not the struct's fields: $head"; exit 1; fi
# name, then the eleven values, of every plan: the default row's first, then first appearance, each set of values once
rows=$(printf "%s\n" "$data" | tail -n +2 | awk -F '\t' -v OFS='\t' '$3 != "-" { k = $10; for (i = 11; i <= 20; i++) k = k OFS $i; line = $3 OFS k
    if ($5 == "default") { def = line; defk = k } else if (!(k in seen)) { seen[k] = 1; list[++n] = line; keys[n] = k } }
    END { print def; for (i = 1; i <= n; i++) if (keys[i] != defk) print list[i] }')
np=$(printf "%s\n" "$rows" | grep -c .)
want_metal=$(printf "%s\n" "$rows" | awk -F '\t' -v fields="$fields" -v nf="$nf" '
    { n = split(fields, f, " "); id = $2; for (i = 3; i <= nf + 1; i++) id = id "_" $i
      print "namespace tiled_" id " {"; print "#define TILED_PLAN_NAME \"_lay_" id "\""
      for (i = 1; i <= n; i++) print "constant constexpr uint tiled_" f[i] " = " $(i + 1) ";"
      print "#include \"kernel-tiled-body.metal\""; print "#undef TILED_PLAN_NAME"; print "}" }')
have_metal=$(grep -v '^// generated' "$out/tiled-plans.metal")
if [ "$want_metal" != "$have_metal" ]; then echo "FAIL the generated plan blocks differ from the table; expected:"; printf "%s\n" "$want_metal"; exit 1; fi
want_host=$(printf "%s\n" "$rows" | awk -F '\t' -v nf="$nf" '{ s = $2; for (i = 3; i <= nf + 1; i++) s = s ", " $i; print "    { " s " }," }')
have_host=$(grep '^    { ' "$out/tiled-compiled-table.h")
want_ids=$(printf "%s\n" "$rows" | awk -F '\t' '{ print "    \"" $1 "\"," }')
have_ids=$(grep '^    "' "$out/tiled-compiled-table.h")
if [ "$want_ids" != "$have_ids" ]; then echo "FAIL the generated id list differs from the table; expected:"; printf "%s\n" "$want_ids"; exit 1; fi
if [ "$want_host" != "$have_host" ]; then echo "FAIL the generated host list differs from the table; expected:"; printf "%s\n" "$want_host"; exit 1; fi
if grep -q '^ *namespace tiled_' "$kernel"; then echo "FAIL $kernel holds a plan block written by hand: plans are rows of data/matmul.tsv [tiles]"; exit 1; fi
ni=$(grep -c '^#include "tiled-plans.metal"$' "$kernel")
if [ "$ni" -ne 1 ]; then echo "FAIL $kernel must include tiled-plans.metal exactly once (found $ni)"; exit 1; fi
echo "tiled plan declaration: $np compiled plans of $nf values; generated blocks and host list equal to the table; the kernel file includes them"
exit 0
